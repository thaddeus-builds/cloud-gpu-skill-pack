"""Where a box stands, read only: the settings (names, set or unset), the box's /health, the power backend's account view. Sends no take, powers
nothing on or off, writes nothing, costs nothing, lists the active holds, prints no value of a setting. `ready` is true when a take could go to the box now; `next` says
what a person must do otherwise, one line each. `services` says per process the box needs (adapter, comfy, voxcpm) {state: up | starting | down | absent | unknown, reason}: the adapter's view of the box's own processes (0.21.0). `line` is the box's own line of callers (who runs and who waits, any machine); `queue` is this machine's. A consumer adds its own checks (prices, a project's switch) to `next`."""

from __future__ import annotations

import time
from collections.abc import Mapping

from . import config, hold, turns
from .box import Box
from .errors import CloudGpuError
from .power import PASS_BILLING


def _norm(text: str) -> str:
    return "".join(c for c in text.lower() if c.isalnum())


def services_view(health: dict | None, reach_ok: bool, impostor: bool) -> dict:
    """{adapter, comfy, voxcpm: {state, reason}}: state is up | starting | down | absent | unknown. The box's /health says comfy and voxcpm (0.21.0); an older adapter says only
    whether comfy answers; with no adapter answering, the other two are not known."""
    if health is None:
        if not reach_ok:
            why = "no CLOUDGPU_TOKEN and CLOUDGPU_BASE (or CLOUDGPU_SSH_KEY) to ask with"
            return {n: {"state": "unknown", "reason": why} for n in ("adapter", "comfy", "voxcpm")}
        a = ("something else answers at the box's local address and does not prove it is the adapter" if impostor
             else "does not answer: a tunnel that is down, or the box is off or its pass has ended")
        return {"adapter": {"state": "down", "reason": a},
                "comfy": {"state": "unknown", "reason": "the adapter does not answer, so the box cannot be asked"},
                "voxcpm": {"state": "unknown", "reason": "the adapter does not answer, so the box cannot be asked"}}
    out = {"adapter": {"state": "up" if health.get("ok") is not False else "down",
                       "reason": "answers /health" if health.get("ok") is not False else "answers but its worker is dead: no take will run"}}
    svc = health.get("services") if isinstance(health.get("services"), dict) else {}
    for n in ("comfy", "voxcpm"):
        e = svc.get(n)
        if isinstance(e, dict) and e.get("state") in ("up", "starting", "down", "absent", "unknown"):
            out[n] = {k: e[k] for k in ("state", "reason", "port", "log", "loaded", "last_line") if k in e}
        elif n == "comfy":
            out[n] = {"state": "up" if health.get("comfy") else "down",
                      "reason": "answers" if health.get("comfy") else "does not answer (an adapter older than 0.21.0 does not say why)"}
        else:
            out[n] = {"state": "unknown", "reason": "an adapter older than 0.21.0 does not report it"}
    return out


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
                  "worker": (health or {}).get("worker"), "card": (health or {}).get("card"),
                  "voice": (health or {}).get("voice"),           # True | False | None (an adapter older than 0.6.0 does not say)
                  "audio_lock": (health or {}).get("audio_lock")}  # the `lock` values minimax_h3_image_audio_locked serves now (["official","addguide"]; [] none); None: an adapter older than 0.20.0
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
        nxt.append("the box does not answer: a tunnel that is down, or the box is off or its pass has ended"
                   + ("; the box opens the tunnel by key" if box.get("SSH_KEY") else "; set CLOUDGPU_SSH_KEY to let it open the tunnel"))
    elif not health.get("comfy"):
        c = (health.get("services") or {}).get("comfy") if isinstance(health.get("services"), dict) else None
        if isinstance(c, dict) and c.get("state") == "starting":
            nxt.append(f"ComfyUI is starting on the box ({c.get('reason')}): it takes 2 to 3 minutes after a power-on; run this again then")
        elif isinstance(c, dict) and c.get("state") in ("down", "absent"):
            nxt.append(f"ComfyUI is {c['state']} on the box: {c.get('reason')}. start.sh on the box starts it (it starts only what is down); its log is {c.get('log') or 'logs/comfyui.log'} in the adapter's folder")
        else:
            nxt.append("the adapter answers but ComfyUI does not yet: wait a minute after a power-on, then run this again (an adapter older than 0.21.0 does not say why)")
    vox = (health.get("services") or {}).get("voxcpm") if health and isinstance(health.get("services"), dict) else None
    if isinstance(vox, dict) and vox.get("state") in ("down", "starting"):
        nxt.append(f"VoxCPM (the voice service) is {vox['state']} on the box: {vox.get('reason')}; a Make with a VoxCPM voice needs it, an H3 take does not")
    out["services"] = services_view(health, reach_ok, impostor)
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
            ex = acc.get("expiry")
            if ex and ex["level"] != "ok":
                nxt.append(ex["message"])
            if config.flag("AUTOOFF", box.env):
                nxt.append("CLOUDGPU_AUTOOFF is set on a pass: powering off saves nothing there (a pass is left on), unset it")
    elif pw is not None and health is None and reach_ok and pw.name == "none":
        nxt.append("this setup cannot tell whether the box is off, its pass has ended, or it was released: it has no view of the cloud account. Set AUTODL_DEV_TOKEN and CLOUDGPU_INSTANCE "
                   "(the account holder has them and gives them to whoever runs this, never in chat) and unset CLOUDGPU_POWER: `status` then says which, and a box that is off is started")
    elif pw is not None and health is None and reach_ok:
        nxt.append(f"the {pw.name} power backend is not configured, so the box is not powered on for you: start it at its vendor, or set its variables "
                   "(docs/cloud-gpu.md, 'Power'), or CLOUDGPU_POWER=none when it is always on")
    if not box.get("BOX"):
        nxt.append("set CLOUDGPU_BOX to this box's name (rtx4090, rtx-pro6000, ...): prices are per box")
    try:
        holds = hold.active(hold.state_dir(env))
    except OSError:
        holds = []
    try:
        line = turns.tickets(hold.state_dir(env), clean=False)
    except OSError:
        line = []
    out["queue"] = [{k: t[k] for k in ("state", "box", "pid", "label", "since")} for t in line]
    for t in line:
        nxt.append(turns.describe(t) + " (a take of this machine)")
    out["line"] = []                         # the box's own line: callers of every machine (None from an adapter older than 0.7.0 reads as empty)
    if health is not None and reach_ok and not impostor:
        try:
            out["line"] = box.line_list() or []
        except CloudGpuError:
            out["line"] = []
    if out["line"]:
        running = [t for t in out["line"] if t.get("state") == "go"]
        waiting = len(out["line"]) - len(running)
        nxt.append((f"{running[0].get('label') or 'a take'} is running on the box" if running else "the box has callers in its line") + f" (the box's line: {waiting} waiting, callers of every machine)")
    out["holds"] = [{**h, "until_local": hold.clock(h["until"])} for h in holds]
    for h in holds:
        nxt.append(f"{h['box']} is held by {h['who']} until {hold.clock(h['until'])}" + (f" ({h['why']})" if h["why"] else "") + ": takes through the Pool wait for it")
    out["ready"] = bool(reach_ok and health and health.get("comfy"))
    out["next"] = nxt
    out["warnings"] = warnings
    out["checked_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return out
