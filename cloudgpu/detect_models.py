"""Run ON the box: which MiniMax-H3 files this image carries, as the models.json adapter.py reads (docs/cloud-gpu.md).

    python3 detect_models.py [--comfy-root /root/ComfyUI] [--compute-cap 12.0] [--out models.json]

Images keep the same files under different folders (MiniMax/ or minimax/, the vae and the text encoder at the top or in a folder), so a preset per
image does not scale; this reads the ComfyUI models folders (following links: the shared pools are links) and picks, for each key of
`adapter.MODELS`, the first file that matches an ordered list of exact names. Never a guess by substring: an author's merge or an "uncensored"
variant is not picked. The text encoder depends on the card: the nvfp4 file needs a Blackwell GPU (compute capability 12 or more), every other
card takes the int8 one; the capability is asked of nvidia-smi unless --compute-cap says it. Prints a JSON report ({models, missing,
alternatives}); writes --out only when every key was found; exit 1 otherwise. Stdlib only.
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
    a = ap.parse_args(argv)
    cap = a.compute_cap if a.compute_cap is not None else gpu_cap()
    rep = detect(Path(a.comfy_root), cap)
    if a.out and not rep["missing"]:
        Path(a.out).write_text(json.dumps(rep["models"], indent=1) + "\n")
    print(json.dumps(rep, indent=1))
    return 1 if rep["missing"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
