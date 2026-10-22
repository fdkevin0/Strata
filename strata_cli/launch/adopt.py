"""Write a run config from model files that are already prepared, for graphics cards the installer (setup.py) does not set up.

setup.py's Linux paths know NVIDIA (CUDA) and AMD (HIP); an Intel Arc card goes to a different, container-based setup (`--backend sycl`, docs/INTEL_ARC.md)
that this Terminal does not run. When setup says it cannot set this card up, and the model's files AND its prepared pack are already on disk (a box that was
prepared before, "use model files I already have"), the run config is simple enough to write here: the same arguments setup writes for the GSQ-RCO models,
with the engine named as the component (`component:strata-engine`). Nothing is downloaded, prepared or compiled; if a file is missing the person is told which.

Only the GSQ-RCO families (qwen, swift, coder) are adopted this way; images stay off.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

SHARD = re.compile(r"^(?P<stem>.+)-(?P<i>\d{5})-of-(?P<n>\d{5})\.gguf$")
FAMILIES = ("qwen", "swift", "coder")
UNSUPPORTED = ("no nvidia gpu found", "no amd gpu", "nvidia gpu", "not supported", "cannot be used")


def setup_cannot_handle_this_card(message: str) -> bool:
    low = (message or "").lower()
    return ("no nvidia gpu found" in low or "no amd gpu strata can use" in low or "nvidia-smi did not answer" in low
            or "no mac model installer" in low)


def intel_driver() -> str:
    """The kernel driver of the first Intel graphics card: "i915" (an A-series Arc) or "xe" (a B-series Arc), "" when it cannot be read."""
    import glob
    import os
    for card in sorted(glob.glob("/sys/class/drm/card[0-9]/device")):
        try:
            if Path(card, "vendor").read_text().strip() != "0x8086":
                continue
            return os.path.basename(os.path.realpath(os.path.join(card, "driver")))
        except OSError:
            continue
    return ""


def windows_sycl_env() -> dict:
    """The Windows Intel engine's settings (strata_control/sycl_windows.py): on Linux its launcher sets them, on Windows the
    config carries them. The card is the first Intel Arc among the display adapters, its Level Zero index read from sycl-ls
    (beside the installed engine or on PATH) when that names it."""
    from strata_control import engine_component as ec, sycl_windows as sw
    try:
        cards = ec.windows_intel_cards()
    except Exception:                                       # noqa: BLE001 - an unreadable registry leaves the defaults
        cards = []
    card = cards[0] if cards else {}
    engine_dir = None
    try:
        engine_dir = Path(ec.resolve(full=False).root)
    except Exception:                                       # noqa: BLE001 - no engine yet: sycl-ls on PATH, or the default
        pass
    index = sw.level_zero_index(sw.sycl_ls_output(engine_dir), card.get("name", "")) if card else None
    return sw.engine_env(card.get("family", ""), f"level_zero:{index}" if index is not None else "")


def find_shards(folder: Path) -> list[Path]:
    """The shards of one GGUF set in `folder` (-00001-of-0000N ... -0000N-of-0000N), sorted; [] if the set is incomplete."""
    found: dict[tuple[str, int], dict[int, Path]] = {}
    for p in sorted(Path(folder).glob("*.gguf")):
        m = SHARD.match(p.name)
        if m:
            found.setdefault((m.group("stem"), int(m.group("n"))), {})[int(m.group("i"))] = p
    for (_stem, n), items in found.items():
        if set(items) == set(range(1, n + 1)):
            return [items[i] for i in range(1, n + 1)]
    return []


def plan(*, family: str, model: str, model_dir: Path, data_dir: Path, src: Path, context: int, kv: str | None, model_name: str, family_def: dict,
         log: Path, backend: str | None, vram_gb: float = 0.0, notes: list | None = None) -> tuple[dict | None, list[str]]:
    """(config, problems): the config to write, or None with plain-words problems (what is missing)."""
    problems: list[str] = []
    if family not in FAMILIES:
        return None, [f"Strata can write the settings itself only for the Qwen3.8-Flash-Next, Swift and Coder files, not for {family}."]
    shards = find_shards(model_dir)
    if not shards:
        problems.append(f"the model files are not all in {model_dir} (every -0000N-of-0000N.gguf shard is needed)")
    pack = Path(data_dir) / "packs" / model.lower()
    if not (pack / "experts.bin").exists() and not (pack / "index.txt").exists():
        problems.append(f"the prepared pack {pack} is missing (this installer cannot prepare it for this graphics card)")
    tokenizer = pack / "tokenizer"
    if not tokenizer.exists():
        problems.append(f"the tokenizer {tokenizer} is missing")
    profile = src / "data" / (family_def.get("profile") or "expert-profile.bin")
    if not profile.exists():
        problems.append(f"the expert profile {profile} is missing")
    if problems:
        return None, problems
    mtp = Path(data_dir) / "mtp" / "rt"
    args = ["--pack", str(pack), "--native", str(shards[0])]
    if len(shards) > 1:
        args += ["--ple-gguf", str(shards[-1])]
    if backend == "metal":
        # a Mac: the Metal defaults metal/setup_mac.py writes for a model that streams from the SSD (one token per step, no draft layer, the GGUF mapped in place)
        args += ["--expert-profile", str(profile), "--expert-cache", "64", "--prefill", "auto", "--spec", "2", "--suffix-draft", "0",
                 "--max-context", str(context), "--mmap-experts"]
    else:
        args += ["--expert-profile", str(profile), "--expert-cache", "auto", "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5"]
        if mtp.exists():
            args += ["--mtp", str(mtp)]
        args += ["--max-context", str(context), "--kv", kv or ("int8" if context > 8192 else "fp16")]
        if backend == "sycl":
            # an Intel Arc, as sycl/setup_intel.py writes it: an A-series card on i915 loads the experts into a RAM arena and the CPU computes the misses
            # (--ple-io ram); every other Arc (xe: the B70) streams the experts from the GGUF into VRAM (--stream-experts); both keep a VRAM reserve
            reserve = "1024" if context <= 32768 else "2048"
            if vram_gb:                                  # 0.2.2: a card a few hundred MiB short gets a smaller context / reserve (vram_fit)
                from strata_control import vram_fit
                fr = vram_fit.plan(model, vram_gb, context, kv or "int8", int(reserve), "auto", "intel", title=model, ctx_fixed=True)   # the context was settled by max_context or the person
                if fr.changed:
                    args[args.index("--max-context") + 1] = str(fr.ctx)
                    reserve = str(fr.reserve_mib)
                    if fr.prefill:
                        args[args.index("--prefill") + 1] = fr.prefill
                    if notes is not None:
                        notes.append(fr.line)
            if intel_driver() == "i915":
                args += ["--ple-io", "ram", "--vram-reserve-mib", reserve]
            else:
                args += ["--stream-experts", "--vram-reserve-mib", reserve]
    cfg = {"exe": "component:strata-engine", "args": args, "cwd": str(src), "tokenizer": str(tokenizer), "model_name": model_name,
           "log": str(log), "lib_dirs": [], "port": 8080}
    if backend == "metal":
        cfg["env"] = {"STRATA_METAL_STREAM": "1"}
    if backend == "sycl" and os.name == "nt":
        cfg["env"] = windows_sycl_env()
    if backend:
        cfg["backend"] = backend
    return cfg, []


def write(path: Path, cfg: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
