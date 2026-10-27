# cloud-gpu-skill-pack

Run ComfyUI workflows (MiniMax-H3 video first) on a GPU you rent from **any vendor that gives SSH**, from any project, with the same few calls as autodl.art's workflow API.
A box is a rented GPU with ComfyUI and the weights on it plus a small token-checked adapter; this package sets it up, reaches it (an SSH tunnel by key, no password), powers it
(the cloud account's API as a vendor: autodl built in and the default, `none`, or the next cloud you buy as one module), finds the model files by exact name in any image layout, and prices a second of video from
measured runs. Stdlib only, Python 3.9+.

```
cloudgpu/           the package: Box (reach, tunnel, power), Pool (boxes in order of preference), vendors/ (the cloud accounts' APIs: autodl built in), power backends, the adapter (runs on the box), detect_models.py, price, status, bench
cloudgpu/onbox/       install.sh, tunnel.sh, the two verified model presets
cloudgpu/data/      runs.jsonl: the takes measured on a 4090 48G and an RTX PRO 6000
docs/cloud-gpu.md   what it costs, how to set up a box, the settings, the calls, safety
skills/cloud-gpu/   an agent's order of work and what only a person does
tests/              python3 tests/run.py (offline: a fake ComfyUI stands in for the card)
```

```
python3 -m cloudgpu status          # where the box stands, read only
python3 -m cloudgpu instances list  # the cloud account (autodl): list, status, snapshot; power, create, release only with --yes
python3 -m cloudgpu bundle          # cloudgpu-box.tgz: what to copy onto a box
python3 -m cloudgpu price --box rtx4090 --format tsv
```

```python
from cloudgpu import Box
box = Box()                         # CLOUDGPU_* from the environment
box.ensure_up()                     # tunnel by key, power on if off (by design, no confirmation: see docs, "Who may power a box")
api, token = box.api_base(), box.token()
```

Use it from another project by installing it (`pip install git+https://github.com/thaddeus-builds/cloud-gpu-skill-pack.git`) or by vendoring the `cloudgpu/` folder as a plain copy
(never edit the copy: fix it here and take it again). The first consumer is [adcut-skill-pack](https://git.coco.xyz/ming.liu/adcut-skill-pack) (its `h3host` avatar provider).

Measured 2026-10-23: a 4090 48G (¥2.88/h) makes a second of 480p video for about ¥0.005 to 0.007, a PRO 6000 (¥6.98/h) for about ¥0.014 to 0.016, hosted is ¥0.03 to 0.04; at 768p the
4090 is ¥0.018 to 0.029, the PRO 6000 ¥0.04 to 0.063, hosted ¥0.04 to 0.05. Full use, one take at a time; an idle hour is paid on top. See docs/cloud-gpu.md before relying on a number.
