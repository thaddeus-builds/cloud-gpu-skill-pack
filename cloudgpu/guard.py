"""A hard cap on how long a box stays on: `instances guard ID --minutes N`, and `--max-minutes N` on `instances power-on` and `instances ensure`, which start one in the background.

The guard waits until the deadline (by the clock on the wall, so a laptop that slept still shuts the box at once when it wakes), shuts the box down, and reads the status until the vendor says it is off; a shutdown that
is refused or does not take is asked again a few times. It prints one JSON line: `off` true only when the vendor's own status says so. It never releases anything. Limits, said plainly: it is a process on this machine, so a
machine that is switched off, or a guard that is killed, does not shut anything (the vendor's balance is the other stop); and it shuts the box down even if a take is running there, which is what a hard cap is.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from .errors import CloudGpuError

OFF = ("shutdown", "off")
TICK_S = 30                      # the longest single sleep: the clock is looked at again after each
CONFIRM_S = 240                  # after the shutdown request, how long the status is read for "off"
CONFIRM_POLL_S = 15
MAX_MINUTES = 24 * 60


def minutes_ok(value) -> float:
    """The cap in minutes, or CloudGpuError(unknown): a finite number above 0 and at most a day."""
    try:
        m = float(value)
    except (TypeError, ValueError):
        m = float("nan")
    if not (0 < m <= MAX_MINUTES):
        raise CloudGpuError("unknown", f"--max-minutes / --minutes must be a number above 0 and at most {MAX_MINUTES} (a day)")
    return m


def off_at(minutes: float, now: float | None = None) -> str:
    t = (time.time() if now is None else now) + minutes * 60
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds")


def run(vendor, instance_id: str, minutes: float, *, now=time.time, sleep=time.sleep, confirm_s: float = CONFIRM_S, poll_s: float = CONFIRM_POLL_S, on_off=None) -> dict:
    """Wait `minutes`, power `instance_id` off, and read the status until it is off. `on_off()` is called once the shutdown request went through (the caller's ledger line)."""
    deadline = now() + minutes * 60
    while now() < deadline:
        sleep(max(0.0, min(TICK_S, deadline - now())))
    out: dict = {"instance": instance_id, "minutes": minutes, "off": False, "status": None, "asked": 0, "errors": []}
    waited = 0.0
    while True:
        try:
            st = vendor.status(instance_id)
        except CloudGpuError as e:
            st = None
            out["errors"].append(e.message[:160])
        out["status"] = st
        if st in OFF:
            out["off"] = True
            break
        if waited > confirm_s:
            break
        if out["asked"] == 0 or (st is not None and waited >= out["asked"] * 3 * poll_s and out["asked"] < 4):       # ask again now and then, not every poll
            try:
                vendor.power_off(instance_id)
                out["asked"] += 1
                if on_off and out["asked"] == 1:
                    on_off()
            except CloudGpuError as e:
                out["asked"] += 1
                out["errors"].append(e.message[:160])
        sleep(poll_s)
        waited += poll_s
    return out
