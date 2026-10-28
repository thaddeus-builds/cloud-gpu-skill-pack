"""A queue of takes across the processes of one machine: every take that goes through `Pool.take` takes a ticket and waits its turn, oldest first.

A ticket is two files in `<state dir>/turns/`: `<id>.lock`, which its owner holds with flock for as long as the take lives (the OS lets go of it when the process dies, however it dies,
so a dead take never blocks the line), and `<id>.json`, replaced whole when the take's state changes (`waiting`, then `running` on a box). The id starts with the wall clock in
nanoseconds, so names sort in arrival order. A ticket whose lock can be taken is dead and is removed by whoever finds it.

A take may claim a box when nothing else is running on it and no older live ticket that may use the same box is still waiting. A take pinned to one box (a result to fetch) is eligible
for that box only. This orders the takes of ONE machine; callers on different machines keep their line on the box itself (line.py, the adapter's /cloudgpu/line). Without flock (Windows) or a state directory that
cannot be written there is no queue and a take goes as before: the in-process lock still holds. `python3 -m cloudgpu status` shows who runs and who waits."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

try:
    import fcntl
except ImportError:                    # Windows: no flock, no queue
    fcntl = None


def _dir(sd) -> Path:
    return Path(sd) / "turns"


def _alive(lock: Path, *, clean: bool) -> bool:
    """True when some process holds the ticket's lock. A lock nobody holds is a dead ticket (removed when `clean`)."""
    if fcntl is None:
        return False
    try:
        fd = os.open(lock, os.O_RDWR)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return True
    else:
        if clean:
            for p in (lock, lock.with_suffix(".json")):
                try:
                    p.unlink()
                except OSError:
                    pass
        return False
    finally:
        os.close(fd)


def tickets(sd, *, clean: bool = True) -> list[dict]:
    """The live tickets, oldest first: {id, state, box, boxes, pid, label, since}. `boxes` empty means any box (a ticket whose json is not written yet)."""
    d = _dir(sd)
    if fcntl is None or not d.is_dir():
        return []
    out = []
    for lock in sorted(d.glob("*.lock")):
        if not _alive(lock, clean=clean):
            continue
        try:
            doc = json.loads(lock.with_suffix(".json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            doc = {}
        out.append({"id": lock.stem, "state": doc.get("state", "waiting"), "box": doc.get("box"), "boxes": list(doc.get("boxes") or []),
                    "pid": doc.get("pid"), "label": doc.get("label", ""), "since": doc.get("since")})
    return out


class Ticket:
    """This take's place in the line. `open` returns None when there is no queue to join (no flock, or the state directory cannot be written)."""

    def __init__(self, sd, boxes, label, now):
        self.sd, self.boxes, self.label, self.now = sd, list(boxes), str(label)[:80], now
        d = _dir(sd)
        d.mkdir(parents=True, exist_ok=True)
        self.id = f"{time.time_ns():020d}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self._lock = d / (self.id + ".lock")
        self._fd = os.open(self._lock, os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        self.state, self.box, self.since = "waiting", None, now()
        self._write()

    @classmethod
    def open(cls, sd, boxes, label="", now=time.time):
        if fcntl is None:
            return None
        try:
            return cls(sd, boxes, label, now)
        except OSError:
            return None

    def _write(self) -> None:
        doc = {"state": self.state, "box": self.box, "boxes": self.boxes, "pid": os.getpid(), "label": self.label, "since": self.since}
        path = self._lock.with_suffix(".json")
        tmp = path.with_name(path.name + ".tmp")
        try:
            tmp.write_text(json.dumps(doc), encoding="utf-8")
            os.replace(tmp, path)                 # a reader sees the old state or the new one, never half
        except OSError:
            pass                                  # an unwritable state costs only what `status` shows; the lock still keeps the ticket alive

    def my_turn(self, box: str) -> bool:
        """Nothing else runs on `box` and no older live ticket that may use it waits."""
        for t in tickets(self.sd):
            if t["id"] == self.id:
                continue
            if t["state"] == "running" and t["box"] == box:
                return False
            if t["id"] < self.id and t["state"] == "waiting" and (not t["boxes"] or box in t["boxes"]):
                return False
        return True

    def run(self, box: str) -> bool:
        """Mark this take as running on `box`. False (and back to waiting) when an older ticket got there at the same moment."""
        self.state, self.box = "running", box
        self._write()
        if any(t["id"] < self.id and t["state"] == "running" and t["box"] == box for t in tickets(self.sd)):
            self.wait()
            return False
        return True

    def wait(self) -> None:
        self.state, self.box = "waiting", None
        self._write()

    def blockers(self, box: str | None = None) -> list[dict]:
        """The live tickets ahead of this one (running on a box it may use, or waiting and older), for the message of a take that waits."""
        return [t for t in tickets(self.sd, clean=False) if t["id"] != self.id and ((t["state"] == "running" and (box is None or t["box"] == box or t["box"] in self.boxes))
                                                                                  or (t["state"] == "waiting" and t["id"] < self.id))]

    def close(self) -> None:
        for p in (self._lock.with_suffix(".json"), self._lock):
            try:
                p.unlink()
            except OSError:
                pass
        try:
            os.close(self._fd)
        except OSError:
            pass


def describe(t: dict) -> str:
    """One line for a take in the line: 'take-3 (pid 4121) is running on a'."""
    who = (t.get("label") or "a take") + (f" (pid {t['pid']})" if t.get("pid") else "")
    return f"{who} is running on {t['box']}" if t["state"] == "running" else f"{who} is waiting"
