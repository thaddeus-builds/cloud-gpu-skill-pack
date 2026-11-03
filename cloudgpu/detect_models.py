"""Run ON the box: which MiniMax-H3 files this image carries, as the models.json adapter.py reads (docs/cloud-gpu.md).

    python3 detect_models.py [--comfy-root /root/ComfyUI] [--compute-cap 12.0] [--out models.json]

Images keep the same files under different folders (MiniMax/ or minimax/, the vae and the text encoder at the top or in a folder), so a preset per
image does not scale; this reads the ComfyUI models folders (following links: the shared pools are links) and picks, for each key of
`adapter.MODELS`, the first file that matches an ordered list of exact names. Never a guess by substring: an author's merge or an "uncensored"
variant is not picked. The text encoder depends on the card: the nvfp4 file needs a Blackwell GPU (compute capability 12 or more), every other
card takes the int8 one; the capability is asked of nvidia-smi unless --compute-cap says it. Prints a JSON report ({models, missing,
alternatives, compute_cap, tts, nodes}); writes --out only when every key was found; exit 1 otherwise. Stdlib only.

The voice (IndexTTS2) is optional and never makes a box "missing" something: the report's `tts` is {dir, python} when a checkpoints folder (config.yaml and s2mel.pth) and a Python that
can import indextts.infer_v2 were found, else null; --tts-out FILE writes it as JSON only when both were found. CLOUDGPU_TTS_DIR and CLOUDGPU_TTS_PYTHON name them when the search does not.

The audio-lock nodes (0.20.0) are reported too, and never make a box "missing" anything: `nodes` says whether the third-party custom-node pack comfyui-vrgamedevgirl (the one
that holds the audiodrive lock's node) is in ComfyUI's custom_nodes folder and which commit it is on, read from its .git folder without running git. It is a report of what is
installed, not a licence decision: the adapter serves that lock only when the operator sets CLOUDGPU_ALLOW_NODE_VRGDG=1 (docs/cloud-gpu.md). The official lock's node, MiniMaxH3AddGuide,
belongs to ComfyUI itself and is found by the adapter in ComfyUI's /object_info when it runs. --nodes-only prints just this (JSON, or --text for the lines install.sh shows).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

FOLDERS = {"unet": ("diffusion_models", "unet"), "clip": ("text_encoders", "clip"), "vae": ("vae",), "lora": ("loras",)}


def L(*names):
    return [re.compile("^" + re.escape(n) + "$") for n in names]


def wanted(blackwell: bool) -> dict:
    """key -> (kind of folder, ordered exact basenames, steps or None)."""
    clip = ("qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "qwen3vl_32b_minimax_h3_int8_convrot.safetensors") if blackwell else \
           ("qwen3vl_32b_minimax_h3_int8_convrot.safetensors",)
    return {
        "unet_ref": ("unet", L("minimax_h3_ref2va_pruned_int8_convrot.safetensors", "minimax_h3_ref2va_pruned_fp8_scaled.safetensors",
                               "minimax_h3_ref2va_int8_convrot.safetensors")),
        "unet_fl": ("unet", L("minimax_h3_fl2va_pruned_int8_convrot.safetensors", "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
                              "minimax_h3_fl2va_int8_convrot.safetensors")),
        "lora_ref": ("lora", L("minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",         # the ref2va turbo lora when the image has it,
                               "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors")),   # else the fl2v one (it worked: docs/cloud-gpu.md)
        "lora_fl": ("lora", L("minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors", "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
                              "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors")),
        "clip": ("clip", L(*clip)),
        "vae_video": ("vae", L("minimax_h3_video_vae_fp16.safetensors")),
        "vae_audio": ("vae", L("minimax_h3_audio_vae_fp32.safetensors")),
    }


STEPS = {"steps_ref": 4, "steps_fl": 6}       # what the pairs above were run at (adapter.py MODELS)


def listing(models: Path, kind: str) -> list[str]:
    """Relative posix paths of the files under the folders of `kind` that ComfyUI would offer (links followed)."""
    out = []
    for folder in FOLDERS[kind]:
        root = models / folder
        for dirpath, _, files in os.walk(root, followlinks=True):
            for f in files:
                if f.endswith(".safetensors"):
                    out.append(Path(dirpath, f).relative_to(root).as_posix())
    return sorted(out)


def detect(comfy_root: Path, compute_cap: float) -> dict:
    models = comfy_root / "models"
    have = {k: listing(models, k) for k in FOLDERS}
    chosen, missing, alts = {}, [], {}
    for key, (kind, pats) in wanted(compute_cap >= 12.0).items():
        found = None
        for pat in pats:
            hits = [p for p in have[kind] if pat.match(Path(p).name)]
            if hits:
                found = sorted(hits, key=lambda p: (p.count("/"), p))[0]          # the shallowest, then the first
                alts[key] = [p for p in hits if p != found]
                break
        if found:
            chosen[key] = found
        else:
            missing.append(key)
    chosen.update({k: v for k, v in STEPS.items()})
    return {"models": chosen, "missing": missing, "alternatives": {k: v for k, v in alts.items() if v}, "compute_cap": compute_cap}


TTS_FILES = ("config.yaml", "s2mel.pth")          # a folder is IndexTTS2's checkpoints when it holds both (IndexTTS 1.x has no s2mel)
TTS_ROOTS = ("index-tts", "IndexTTS2", "IndexTTS-2", "index-tts2")


def find_tts_dir(comfy_root: Path, home: Path, env: dict) -> Path | None:
    """The named folder (CLOUDGPU_TTS_DIR) if it holds the checkpoints, else the first folder with them under ComfyUI's models folder or a checkout in the home folder (depth 4, links followed)."""
    named = env.get("CLOUDGPU_TTS_DIR")
    if named:
        d = Path(named).expanduser()
        return d if all((d / f).is_file() for f in TTS_FILES) else None
    for root in [comfy_root / "models"] + [home / n for n in TTS_ROOTS]:
        base = len(root.parts)
        for dirpath, dirs, files in os.walk(root, followlinks=True):
            if len(Path(dirpath).parts) - base >= 4:
                dirs[:] = []
            if all(f in files for f in TTS_FILES):
                return Path(dirpath)
    return None


def imports_indextts(python: str) -> bool:
    try:
        r = subprocess.run([python, "-c", "import indextts.infer_v2"], capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0


def find_tts(comfy_root: Path, home: Path, env: dict, can_import=imports_indextts) -> dict | None:
    """{dir, python} of an installed voice, or None. The Python is CLOUDGPU_TTS_PYTHON, else a virtual environment next to the checkout, else the ones on this box: the first that imports indextts.infer_v2."""
    d = find_tts_dir(comfy_root, home, env)
    if d is None:
        return None
    cands = [env.get("CLOUDGPU_TTS_PYTHON")] if env.get("CLOUDGPU_TTS_PYTHON") else \
        [str(p / ".venv" / "bin" / "python") for p in (d.parent, d.parent.parent)] + [str(comfy_root / ".venv" / "bin" / "python"), sys.executable, "python3", "python"]
    for c in cands:
        if c and can_import(c):
            return {"dir": str(d), "python": c}
    return None


NODE_PACKS = {"audiodrive": ("comfyui-vrgamedevgirl", "VRGDG_MiniMaxH3AudioDrive")}    # lock -> (the custom_nodes folder, the node it holds); the folder name is what the repository is called
VRGDG_NOTE = "third-party, licence-restricted (VRGameDevGirl Community and Commercial License v1.0): off unless the operator sets CLOUDGPU_ALLOW_NODE_VRGDG=1"


def git_head(folder: Path) -> str | None:
    """The commit a folder's git checkout is on (40 hex), from .git/HEAD and its ref or packed-refs without running git; None when it is not a checkout or cannot be read. (adapter.py has the same
    ten lines: the adapter is copied alone onto the box; tests/test_audio_lock.py keeps the two equal.)"""
    git = folder / ".git"
    try:
        head = (git / "HEAD").read_text().strip()
        if re.fullmatch(r"[0-9a-f]{40}", head):
            return head
        if not head.startswith("ref: "):
            return None
        ref = head[5:].strip()
        if ".." in ref or ref.startswith("/"):
            return None
        try:
            val = (git / ref).read_text().strip()
        except OSError:
            val = next((ln.split()[0] for ln in (git / "packed-refs").read_text().splitlines() if ln.endswith(" " + ref)), "")
        return val if re.fullmatch(r"[0-9a-f]{40}", val) else None
    except (OSError, ValueError, IndexError):          # ValueError: an undecodable file or a NUL in a path
        return None


def find_nodes(comfy_root: Path) -> dict:
    """{lock: {pack, node, present, commit, disabled, note}} for each node pack in NODE_PACKS: whether ComfyUI's custom_nodes folder holds it (a folder named like the repository, in any case; one
    ComfyUI-Manager disabled is `<name>.disabled` and counts as not present) and the commit it is on. Nothing in the pack is read but .git."""
    base = comfy_root / "custom_nodes"
    try:
        names = sorted(p.name for p in base.iterdir() if p.is_dir())
    except OSError:
        names = []
    out = {}
    for lock, (pack, node) in NODE_PACKS.items():
        hit = next((n for n in names if n.lower() == pack), None)
        out[lock] = {"pack": pack, "node": node, "present": hit is not None, "folder": hit, "commit": git_head(base / hit) if hit else None,
                     "disabled": any(n.lower() == pack + ".disabled" for n in names), "note": VRGDG_NOTE}
    return out


def nodes_text(nodes: dict) -> str:
    """The lines install.sh shows about the lock nodes."""
    lines = ["audio lock: the official lock (MiniMaxH3AddGuide) is a ComfyUI core node; the adapter looks for it in ComfyUI when it starts and says in /health (audio_lock)"]
    for lock, n in nodes.items():
        if n["present"]:
            lines.append(f"audio lock {lock}: pack {n['pack']} is installed ({'commit ' + n['commit'] if n['commit'] else 'commit unknown: not a git checkout'}); {n['note']}")
        else:
            lines.append(f"audio lock {lock}: pack {n['pack']} is not installed" + (" (a disabled copy is there)" if n["disabled"] else "") + f"; {n['note']}")
    return "\n".join(lines)


def gpu_cap() -> float:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout
        return float(out.split()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return 0.0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--comfy-root", default="/root/ComfyUI")
    ap.add_argument("--compute-cap", type=float)
    ap.add_argument("--out")
    ap.add_argument("--tts-out", help="write the voice found ({dir, python}) here as JSON")
    ap.add_argument("--tts-only", action="store_true", help="look for the voice only (no H3 models, no GPU query)")
    ap.add_argument("--nodes-only", action="store_true", help="report the audio-lock node packs only (no H3 models, no GPU query)")
    ap.add_argument("--text", action="store_true", help="with --nodes-only: the lines install.sh shows instead of JSON")
    a = ap.parse_args(argv)
    try:
        nodes = find_nodes(Path(a.comfy_root))
    except Exception:                      # a report about a pack never stops the detection of the models or the voice
        nodes = {}
    if a.nodes_only:
        print(nodes_text(nodes) if a.text else json.dumps({"nodes": nodes}, indent=1))
        return 0
    tts = find_tts(Path(a.comfy_root), Path.home(), dict(os.environ))
    if a.tts_out and tts:
        Path(a.tts_out).write_text(json.dumps(tts) + "\n")
    if a.tts_only:
        print(json.dumps({"tts": tts}, indent=1))
        return 0
    cap = a.compute_cap if a.compute_cap is not None else gpu_cap()
    rep = {**detect(Path(a.comfy_root), cap), "tts": tts, "nodes": nodes}
    if a.out and not rep["missing"]:
        Path(a.out).write_text(json.dumps(rep["models"], indent=1) + "\n")
    print(json.dumps(rep, indent=1))
    return 1 if rep["missing"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
