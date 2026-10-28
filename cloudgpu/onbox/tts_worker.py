"""Run ON the GPU box, in the Python environment IndexTTS2 is installed in (not the adapter's): make one voice take and exit.

    <IndexTTS2 python> tts_worker.py JOB.json

The adapter (adapter.py, `run_tts`) writes JOB.json and starts this once per voice, so the voice model's GPU memory is given back when it exits. The job:
{"dir": the checkpoints folder (config.yaml in it), "text", "spk": the voice sample, "out": the wav to write, "emo": "same"|"audio"|"vector",
 "vector": 8 floats in IndexTTS2's order (happy, angry, sad, afraid, disgusted, melancholic, surprised, calm) or null, "alpha": 0..1 or null,
 "random": bool, "emo_audio": the emotion sample or null}. The calls are IndexTTS2's own (indextts.infer_v2.IndexTTS2.infer, as its README shows them).
Exit 0 and a wav at "out", or a non-zero exit and one last line on stderr that says why. Nothing is printed on success."""

from __future__ import annotations

import json
import os
import sys


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: tts_worker.py JOB.json", file=sys.stderr)
        return 2
    with open(argv[0], encoding="utf-8") as f:
        job = json.load(f)
    try:
        from indextts.infer_v2 import IndexTTS2
    except ImportError as e:
        print(f"IndexTTS2 cannot be imported by this Python ({e}): --tts-python must be the Python of its environment", file=sys.stderr)
        return 3
    d = job["dir"]
    tts = IndexTTS2(cfg_path=os.path.join(d, "config.yaml"), model_dir=d, use_fp16=True, use_cuda_kernel=False, use_deepspeed=False)
    kw = {"spk_audio_prompt": job["spk"], "text": job["text"], "output_path": job["out"], "use_random": bool(job.get("random")), "verbose": False}
    if job["emo"] == "audio":
        kw["emo_audio_prompt"] = job["emo_audio"]
    elif job["emo"] == "vector":
        kw["emo_vector"] = [float(x) for x in job["vector"]]
    if job.get("alpha") is not None:
        kw["emo_alpha"] = float(job["alpha"])
    tts.infer(**kw)
    if not os.path.isfile(job["out"]):
        print("IndexTTS2 returned without writing the wav", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
