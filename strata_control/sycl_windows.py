"""The Intel Arc (SYCL) engine on Windows: the settings its Linux launcher gives it, for a native start.

On Linux the SYCL engine runs in a container (sycl/serve/strata-sycl.sh), and that launcher sets the port's run-time switches
and the image sets the SYCL runtime's.  On Windows the engine is strata.exe from sycl/tools/build_windows.bat, started directly
by the server, so the same settings travel in the run config's "env" block (the server puts it in the engine's environment).
Pure functions apart from `sycl_ls_output`, which runs sycl-ls once when it is there.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

#: strata-sycl.sh's defaults (the device-built verify plan, no host handshakes in it, 12 stager threads) and the
#: strata-sycl-dev image's (the persistent JIT cache crashed on Xe2 during the first compile; Sysman for the free-memory query)
DEFAULT_ENV = {"STRATA_VERIFY_DEVICE_PLAN": "1", "STRATA_VERIFY_NO_HOST": "1", "STRATA_STAGER_THREADS": "12",
               "SYCL_CACHE_PERSISTENT": "0", "ZES_ENABLE_SYSMAN": "1"}
#: an Alchemist card (the A-series) has no FP64 hardware: a kernel that declares double is refused at its first launch unless
#: the driver emulates it (docs/INTEL.md); sycl/setup_intel.py sets the same three on Linux
ALCHEMIST_ENV = {"IGC_EnableDPEmulation": "1", "OverrideDefaultFP64Settings": "1", "NEOReadDebugKeys": "1"}

_LZ_LINE = re.compile(r"\[level_zero:gpu\]\s*\[level_zero:(\d+)\]\s*(.*)")


def level_zero_gpus(text: str) -> list[tuple[int, str]]:
    """(Level Zero index, the line's description) of every Level Zero GPU in sycl-ls's output, in its order."""
    found = []
    for line in (text or "").splitlines():
        m = _LZ_LINE.search(line)
        if m:
            found.append((int(m.group(1)), m.group(2).strip()))
    return found


def _model(name: str) -> str:
    """The model part of an Arc's name ("A770", "B580", "PRO B60"), "" when the name has none."""
    m = re.search(r"\b((?:PRO )?[AB]\d{2,3}M?)\b", re.sub(r"\((?:R|TM)\)", "", str(name or "").upper()).replace("  ", " "))
    return m.group(1) if m else ""


def level_zero_index(text: str, card_name: str) -> int | None:
    """The Level Zero index of the Arc called `card_name` in sycl-ls's output, None when it is not there or more than one
    device carries that name (two identical cards: the order then decides, and the caller keeps its own index)."""
    want = _model(card_name)
    if not want:
        return None
    hits = [i for i, desc in level_zero_gpus(text) if _model(desc) == want]
    return hits[0] if len(hits) == 1 else None


def sycl_ls_output(engine_dir: Path | None = None, timeout: float = 20.0) -> str:
    """sycl-ls's output ("" when it is not there or fails): the one beside the engine (the Windows package carries it), else
    the one on PATH (a oneAPI install)."""
    exe = None
    if engine_dir is not None and (Path(engine_dir) / "sycl-ls.exe").is_file():
        exe = str(Path(engine_dir) / "sycl-ls.exe")
    exe = exe or shutil.which("sycl-ls")
    if not exe:
        return ""
    try:
        done = subprocess.run([exe], capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL,
                              env={k: v for k, v in os.environ.items() if k != "ONEAPI_DEVICE_SELECTOR"},
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout if done.returncode == 0 else ""


def engine_env(family: str = "", selector: str = "", split: bool = False) -> dict:
    """The run config's "env" for the Windows SYCL engine: the defaults above, the FP64 emulation on an Alchemist card, and
    the device: `selector` ("level_zero:1") when the caller found the card, else every Level Zero GPU ("level_zero:gpu"; with
    Intel integrated graphics enabled beside the Arc, the engine's device 0 may then be the integrated one - docs/INTEL_ARC.md,
    "Windows"). A layer split also trims each card's copy of the dense weights (STRATA_STAGE_TRIM, as on Linux)."""
    env = dict(DEFAULT_ENV)
    if family == "alchemist":
        env.update(ALCHEMIST_ENV)
    env["ONEAPI_DEVICE_SELECTOR"] = selector or "level_zero:gpu"
    if split:
        env["ONEAPI_DEVICE_SELECTOR"] = "level_zero:gpu"
        env["STRATA_STAGE_TRIM"] = "1"
    return env
