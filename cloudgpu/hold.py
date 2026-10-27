"""A person's hold on a box: "do not start takes on this box until HH:MM". One small JSON file per box in the state directory
(`CLOUDGPU_STATE_DIR`, else `~/.cloudgpu`), so it works across processes and restarts, and ends by itself at its time.

    python3 -m cloudgpu hold BOX --minutes 60 --who NAME --why "manual test"
    python3 -m cloudgpu release BOX

`Pool.take` treats a held box as busy and `Pool.power_off_if_idle` leaves it on. A hold governs callers that go through `Pool`: someone who talks to ComfyUI directly is not stopped.
BOX is the name in `CLOUDGPU_POOL`, or `default` for a pool of one. Nothing here names a caller; `who` and `why` are free text for the next person to read."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Mapping
from pathlib import Path

from . import config
from .errors import CloudGpuError

DEFAULT_BOX = "default"
MAX_MINUTES = 480                      # a typo ("600" for "60") must not hold a box all day
NAME = re.compile(r"[a-z0-9]+")
TEXT_MAX = 200


def box_key(name: str) -> str:
    """The key a pool's box name is held under: the name, lower case; a pool of one (no name) is `default`."""
    return (name or DEFAULT_BOX).lower()


def state_dir(env: Mapping | None = None) -> Path:
    v = config.get("STATE_DIR", env)
    return Path(v).expanduser() if v else Path.home() / ".cloudgpu"


def _file(sd: Path, box: str) -> Path:
    key = box_key(box)
    if not NAME.fullmatch(key):
        raise CloudGpuError("key_missing", f"{box!r} is not a box name (letters and digits)")
    return Path(sd) / "holds" / (key + ".json")


def _read(path: Path, now: float) -> dict | None:
    """The hold in `path`, or None when there is none, it has ended, or the file is not a hold (a file nobody can read must not block a box for ever)."""
    try:
        h = json.loads(path.read_text(encoding="utf-8"))
        until = float(h["until"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if until <= now:
        return None
    return {"box": path.stem, "who": str(h.get("who") or ""), "why": str(h.get("why") or ""), "since": h.get("since"), "until": until}


def get(sd: Path, box: str, now: float | None = None, *, clean: bool = True) -> dict | None:
    """The active hold on `box`, else None. An ended or unreadable hold file is removed (`clean`); the read-only status passes clean=False."""
    path, now = _file(sd, box), time.time() if now is None else now
    if not path.exists():
        return None
    h = _read(path, now)
    if h is None and clean:
        try:
            path.unlink()
        except OSError:
            pass
    return h


def active(sd: Path, now: float | None = None) -> list[dict]:
    """Every active hold, soonest end first. Reads only."""
    now = time.time() if now is None else now
    d = Path(sd) / "holds"
    if not d.is_dir():
        return []
    out = [h for p in sorted(d.glob("*.json")) if (h := _read(p, now))]
    return sorted(out, key=lambda h: h["until"])


def place(sd: Path, box: str, minutes: float, who: str, why: str = "", now: float | None = None) -> dict:
    """Hold `box` for `minutes` from now. Someone else's active hold is refused (release it first, or ask them); your own is replaced, so a hold can be extended."""
    if not (0 < minutes <= MAX_MINUTES):
        raise CloudGpuError("key_missing", f"--minutes must be above 0 and at most {MAX_MINUTES}", ["hold for a shorter time and hold again when it runs out"])
    who, why = str(who).strip()[:TEXT_MAX], str(why).strip()[:TEXT_MAX]
    if not who:
        raise CloudGpuError("key_missing", "say who holds the box (--who NAME): the next person to find it held needs to know whom to ask")
    now = time.time() if now is None else now
    path = _file(sd, box)
    cur = get(sd, box, now)
    if cur and cur["who"] != who:
        raise CloudGpuError("key_missing", f"{box_key(box)} is already held by {cur['who']} until {clock(cur['until'])}",
                            [f"ask {cur['who']}, or release it (python3 -m cloudgpu release {box_key(box)}) if it is stale"])
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"who": who, "why": why, "since": now, "until": now + minutes * 60}
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(doc), encoding="utf-8")
    os.replace(tmp, path)                                       # a reader sees the old hold or the new one, never half of one
    return {"box": box_key(box), **doc}


def release(sd: Path, box: str) -> bool:
    """End the hold on `box`; True when there was a file to remove (releasing nothing is not an error)."""
    try:
        _file(sd, box).unlink()
        return True
    except FileNotFoundError:
        return False


def clock(t: float) -> str:
    """HH:MM in the local time zone."""
    return time.strftime("%H:%M", time.localtime(t))


def waiting_message(h: dict) -> str:
    return f"waiting for the hold on {h['box']}" + (f" by {h['who']}" if h["who"] else "") + f" until {clock(h['until'])}"
