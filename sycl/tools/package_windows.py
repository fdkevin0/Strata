#!/usr/bin/env python3
"""Package the Windows SYCL (Intel Arc) engine as a ready-made zip (strata-windows-x64-sycl.zip).

    python sycl/tools/package_windows.py --build build-sycl-win --aot bmg-g21 --out dist

sycl/tools/build_windows.bat runs it after the build, in the console setvars.bat and vcvars64.bat prepared (the oneAPI and
Visual C++ folders are found through PATH, ONEAPI_ROOT and VCToolsRedistDir).  The zip holds, all in one folder (Windows
looks for a DLL beside the program first, so no PATH is needed):

    strata.exe, BUILD.json                 backend "sycl", the AOT device (or "spir64": compiled for the card at the
                                           first start), the oneAPI version
    sycl*.dll, ur_loader.dll, umf.dll ...  the SYCL runtime and what it imports, followed through the import tables
    ur_adapter_level_zero*.dll,            loaded BY NAME by the SYCL runtime (in no import table): the Level Zero adapter
    ur_win_proxy_loader.dll                is how the runtime reaches the Arc; the proxy loader is how it finds the adapters
    mkl_sycl_blas*.dll, mkl_core*.dll ...  oneMKL's SYCL BLAS (the prompt's GEMMs) and its core, sequential threading
    mkl_def*.dll, mkl_avx2*.dll, ...       oneMKL's CPU kernels, loaded by name by mkl_core for this CPU
    libmmd.dll, svml_dispmd.dll ...        the Intel compiler's runtime
    msvcp140*.dll, vcruntime140*.dll       the Microsoft C++ runtime
    sycl-ls.exe                            oneAPI's device lister: setup reads it to find the Arc's Level Zero index
    licenses/                              oneAPI's licence files and NOTICE.txt

The Intel graphics driver is the only thing the user installs: ze_loader.dll, ze_intel_gpu64.dll and OpenCL.dll are the
driver's (packaging/binary_deps.py WINDOWS_DRIVER) and are never shipped.  The zip is checked with binary_deps.uncovered
before it is written: an import that is neither shipped nor Windows' nor the driver's refuses the package.

For the per-user component folder (the app and the Terminal), the staged folder goes through packaging/package_engine.py:

    python packaging/package_engine.py --src dist/strata-windows-x64-sycl --backend sycl --platform windows-x86_64 \\
        --version <engine version> --arch bmg-g21 --out <components>/engine
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "hip"))
sys.path.insert(0, str(ROOT / "packaging"))
from dll_closure import pe_imports  # noqa: E402
import binary_deps  # noqa: E402

ASSET = "strata-windows-x64-sycl.zip"
PROGRAMS = ("strata.exe",)
# Loaded with LoadLibrary by the SYCL runtime or by oneMKL, so they are in no import table: shipped by pattern, then their own
# imports are followed like the rest.  A pattern that matches nothing is fine (the names move between oneAPI releases), except
# the Level Zero adapter, without which the runtime sees no GPU at all.
DYNAMIC = ("ur_adapter_level_zero*.dll", "ur_win_proxy_loader*.dll", "pi_win_proxy_loader*.dll",
           "mkl_def*.dll", "mkl_mc3*.dll", "mkl_avx2*.dll", "mkl_avx512*.dll", "mkl_vml_def*.dll", "mkl_vml_avx2*.dll",
           "mkl_vml_avx512*.dll", "mkl_vml_cmpt*.dll")
REQUIRED_DYNAMIC = ("ur_adapter_level_zero*.dll",)
CRT = ("msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "vcruntime140.dll", "vcruntime140_1.dll",
       "msvcp140_atomic_wait.dll", "msvcp140_codecvt_ids.dll", "vcruntime140_threads.dll", "concrt140.dll")
NOTICE = """Strata's Windows SYCL engine ships these Intel oneAPI runtime components unmodified, from the Intel oneAPI Base Toolkit
{oneapi} (https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html):

{dlls}

The SYCL runtime and the Unified Runtime (sycl*.dll, ur_*.dll, umf*.dll) are licensed under Apache 2.0 with LLVM exceptions
(https://github.com/intel/llvm); oneMKL (mkl_*.dll) and the Intel compiler runtime (libmmd.dll, svml_dispmd.dll,
libirngmd.dll, ...) are redistributable under the Intel Simplified Software License.  The licence files of this oneAPI
installation are in this folder.  The Microsoft Visual C++ runtime DLLs (msvcp140*.dll, vcruntime140*.dll) are
redistributable under the Visual Studio license (Distributable Code).
"""


def search_dirs(environ=None) -> list[Path]:
    """Where the oneAPI DLLs are: PATH (setvars.bat put the compiler, MKL, TBB and UMF bin folders on it), then the usual
    folders under ONEAPI_ROOT.  Existing folders only, each once, in that order."""
    environ = os.environ if environ is None else environ
    found: list[Path] = [Path(x) for x in environ.get("PATH", "").split(os.pathsep) if x]
    oneapi = environ.get("ONEAPI_ROOT")
    if oneapi:
        for sub in ("compiler/latest/bin", "compiler/latest/bin/compiler", "mkl/latest/bin", "umf/latest/bin",
                    "tbb/latest/bin", "tcm/latest/bin"):
            found.append(Path(oneapi) / sub)
    out, seen = [], set()
    for d in found:
        key = str(d).lower().rstrip("\\/")
        if key in seen or not d.is_dir():
            continue
        seen.add(key)
        out.append(d)
    return out


def windows_system_dll(name: str) -> bool:
    return binary_deps._match(name, binary_deps.WINDOWS_OS, True) or binary_deps._match(name, binary_deps.WINDOWS_DRIVER, True)


def find_dll(name: str, dirs: list[Path]) -> Path | None:
    low = name.lower()
    for d in dirs:
        hit = d / name
        if hit.is_file():
            return hit
        for c in d.glob("*" + Path(name).suffix):
            if c.name.lower() == low:
                return c
    return None


def find_pattern(pattern: str, dirs: list[Path]) -> list[Path]:
    """Every DLL matching `pattern` in the first folder that has one (a second oneAPI on PATH must not add a second copy)."""
    for d in dirs:
        hits = sorted(c for c in d.glob("*.dll") if fnmatch.fnmatch(c.name.lower(), pattern.lower()))
        if hits:
            return hits
    return []


def closure(roots: list[Path], dirs: list[Path], imports=pe_imports) -> tuple[dict[str, Path], set[str], list[str]]:
    """({name: source} of every non-system DLL the roots need, the C++ runtime names among them, imports found nowhere).
    The roots themselves are not in the first result."""
    need: dict[str, Path] = {}
    crt: set[str] = set()
    missing: list[str] = []
    todo = list(roots)
    seen = {r.name.lower() for r in roots}
    while todo:
        f = todo.pop()
        for name in imports(f):
            low = name.lower()
            if low in seen:
                continue
            seen.add(low)
            if low in CRT:
                crt.add(low)
                continue
            if windows_system_dll(name):
                continue
            hit = find_dll(name, dirs)
            if hit is None:
                missing.append(name)
                continue
            need[hit.name] = hit
            todo.append(hit)
    return need, crt, missing


def crt_dir(environ=None) -> Path | None:
    """The MSVC redist folder with the C++ runtime DLLs (vcvars sets VCToolsRedistDir)."""
    base = (os.environ if environ is None else environ).get("VCToolsRedistDir")
    if not base:
        return None
    hits = sorted(Path(base, "x64").glob("Microsoft.VC*.CRT"))
    return hits[-1] if hits else None


def oneapi_version(environ=None) -> str:
    """The compiler's version (icx --version), else the oneAPI folder name the compiler lives in, else "unknown"."""
    try:
        out = subprocess.run(["icx", "--version"], capture_output=True, text=True, timeout=30).stdout
        m = re.search(r"\b(\d{4}\.\d+\.\d+(?:\.\d+)?)\b", out)
        if m:
            return m.group(1)
    except (OSError, subprocess.SubprocessError):
        pass
    environ = os.environ if environ is None else environ
    m = re.search(r"compiler[\\/](\d{4}\.\d+)", environ.get("CMPLR_ROOT", ""))
    return m.group(1) if m else "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True, type=Path)
    ap.add_argument("--aot", default="", help="the AOT device the engine was compiled for (empty: SPIR-V, JIT)")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    dirs = search_dirs()
    if not dirs:
        sys.exit("no oneAPI folder on PATH: run this from a console where setvars.bat ran (sycl/tools/build_windows.bat does)")
    stage = a.out / "strata-windows-x64-sycl"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    for p in PROGRAMS:
        shutil.copy2(a.build / p, stage / p)

    roots = [stage / p for p in PROGRAMS]
    lister = find_dll("sycl-ls.exe", dirs)               # not a DLL, found the same way: setup reads it to pick the Arc
    if lister is not None:                               # (strata_control/sycl_windows.py, level_zero_index)
        shutil.copy2(lister, stage / "sycl-ls.exe")
        roots.append(stage / "sycl-ls.exe")
    for pattern in DYNAMIC:
        hits = find_pattern(pattern, dirs)
        if not hits and pattern in REQUIRED_DYNAMIC:
            sys.exit(f"no {pattern} in the oneAPI folders ({', '.join(map(str, dirs[:4]))}, ...): the SYCL runtime could not "
                     "reach the Arc through Level Zero; is the oneAPI DPC++ compiler installed?")
        roots += hits
    need, crt, missing = closure(roots, dirs)
    for r in roots:
        if r.parent != stage:
            need.setdefault(r.name, r)
    if missing:
        sys.exit("these DLLs are imported but not in the oneAPI folders, Windows or the Intel driver: " + ", ".join(sorted(missing)))
    for name, src in sorted(need.items()):
        shutil.copy2(src, stage / name)
    cdir = crt_dir()
    if cdir is None:
        sys.exit("VCToolsRedistDir is not set: run this from a Visual Studio x64 developer prompt (vcvars64.bat)")
    for d in sorted(crt):
        if (cdir / d).is_file():
            shutil.copy2(cdir / d, stage / d)

    version = re.search(r"project\(strata VERSION ([\d.]+)", (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")).group(1)
    oneapi = oneapi_version()
    lic = stage / "licenses"
    lic.mkdir()
    oroot = os.environ.get("ONEAPI_ROOT")
    if oroot:
        for sub in ("compiler/latest", "mkl/latest", "umf/latest"):
            for f in sorted(Path(oroot, sub).glob("share/doc/**/licensing/**/*")) + sorted(Path(oroot, sub).glob("licensing/**/*")):
                if f.is_file():
                    shutil.copy2(f, lic / f"{sub.split('/')[0]}-{f.name}")
    shipped = sorted(p.name for p in stage.glob("*.dll"))
    (lic / "NOTICE.txt").write_text(NOTICE.format(oneapi=oneapi, dlls="\n".join("  " + d for d in shipped)), encoding="utf-8")
    meta = {"source": "prebuilt", "backend": "sycl", "platform": "windows-x64", "version": version,
            "archs": [a.aot or "spir64"], "oneapi": oneapi, "vision": "none", "lib_dirs": []}
    (stage / "BUILD.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    left = binary_deps.uncovered(stage)
    if left:
        sys.exit("refusing to package: a clean PC could not start the engine (0xC0000135):\n" +
                 "\n".join(f"  {f} needs {n}" for f, n in sorted(set(left))))
    z = a.out / ASSET
    z.unlink(missing_ok=True)
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as f:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                f.write(p, p.relative_to(stage).as_posix())
    size = sum(p.stat().st_size for p in stage.rglob("*") if p.is_file())
    print(f"{z}: {z.stat().st_size / 2**20:.0f} MiB ({size / 2**20:.0f} MiB unpacked), engine {version}, "
          f"{meta['archs'][0]}, oneAPI {oneapi}, DLLs: {', '.join(shipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
