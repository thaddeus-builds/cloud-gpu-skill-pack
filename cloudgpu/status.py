"""Where a box stands, read only: the settings (names, set or unset), the box's /health, the power backend's account view. Sends no take, powers
nothing on or off, writes nothing, costs nothing, prints no value of a setting. `ready` is true when a take could go to the box now; `next` says
what a person must do otherwise, one line each. A consumer adds its own checks (prices, a project's switch) to `next`."""

from __future__ import annotations

import time
from collections.abc import Mapping

from . import config
from .box import Box
from .errors import CloudGpuError
from .power import PASS_BILLING


def _norm(text: str) -> str:
    return "".join(c for c in text.lower() if c.isalnum())


def status(env: Mapping | None = None, envfile: str | None = None) -> dict:
    """`envfile`: the file the settings were read from, if any; the status then warns when other users can read it."""
    box = Box(env)
    out: dict = {"vars": config.present(env)}
    nxt: list[str] = []
    warnings: list[str] = []
    if envfile and config.env_file_warning(envfile):
        warnings.append(config.env_file_warning(envfile))
    has_reach = bool(box.get("BASE") or box.get("SSH_KEY"))
    reach_ok = has_reach and bool(box.get("TOKEN"))
    health, impostor = None, False
    if reach_ok:
        try:
            health = box.health()
        except CloudGpuError as e:
            health, impostor = None, e.kind == "http"
            if impostor:
                nxt.append(e.message)
    out["box"] = {"answers": health is not None, "comfy": bool(health and health.get("comfy")), "busy": (health or {}).get("busy"),
                  "worker": (health or {}).get("worker"), "card": (health or {}).get("card")}
    card, name = out["box"]["card"], box.get("BOX")
    if card and card != "unknown" and name and _norm(name) not in _norm(card):
        nxt.append(f"the box says its card is {card!r} but CLOUDGPU_BOX is {name!r}: prices are per box, check which one this is")
    if health and health.get("worker") == "dead":
        nxt.append("the adapter answers but its worker is dead: no take will run; restart it on the box (start.sh)")
    if not reach_ok:
        nxt.append("set CLOUDGPU_TOKEN and CLOUDGPU_BASE (or CLOUDGPU_SSH_KEY: the tunnel) in the environment; a person pipes them, never in chat")
    elif impostor:
        pass
    elif health is None:
        nxt.append("the box does not answer: a tunnel that is down, or the box is off"
                   + ("; the box opens the tunnel by key" if box.get("SSH_KEY") else "; set CLOUDGPU_SSH_KEY to let it open the tunnel"))
    elif not health.get("comfy"):
        nxt.append("the adapter answers but ComfyUI does not yet: wait a minute after a power-on, then run this again")
    pw = None
    try:
        pw = box.power
    except CloudGpuError as e:
        nxt.append(e.message)
    if pw is not None and pw.configured():
        try:
            acc = {"backend": pw.name, **pw.pass_info()}
        except CloudGpuError as e:
            acc = {"backend": pw.name, "error": e.message[:200]}
        out["account"] = acc
        if acc.get("billing") in PASS_BILLING:
            if acc.get("hours_left", 1) < 0:
                nxt.append("the pass has ended: renew it at the vendor")
            if config.flag("AUTOOFF", box.env):
                nxt.append("CLOUDGPU_AUTOOFF is set on a pass: powering off saves nothing there (a pass is left on), unset it")
    elif pw is not None and health is None and reach_ok:
        nxt.append(f"the {pw.name} power backend is not configured, so the box is not powered on for you: start it at its vendor, or set its variables "
                   "(docs/cloud-gpu.md, 'Power'), or CLOUDGPU_POWER=none when it is always on")
    if not box.get("BOX"):
        nxt.append("set CLOUDGPU_BOX to this box's name (rtx4090, rtx-pro6000, ...): prices are per box")
    out["ready"] = bool(reach_ok and health and health.get("comfy"))
    out["next"] = nxt
    out["warnings"] = warnings
    out["checked_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return out
