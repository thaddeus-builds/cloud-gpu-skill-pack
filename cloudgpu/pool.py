"""A pool of boxes in order of preference: a take waits for the first box that is not making another take and comes up.

`CLOUDGPU_POOL=a,b` names the boxes; each box's settings are `CLOUDGPU_<A>_<NAME>` (`CLOUDGPU_A_TOKEN`, `CLOUDGPU_A_BASE`, ...: one local tunnel port of its own in its BASE),
and every setting that is not a box's is shared. With no pool one box on the plain `CLOUDGPU_<NAME>` settings is a pool of one and nothing changes for its caller.

`Pool.take(out_dir)` is the one call a consumer makes around a take:

    with pool.take(out_dir) as box:        # the box this take runs on; pool.box is the same inside this thread
        ... send the take, wait for it, fetch it ...

A box makes one take at a time (the adapter queues behind ComfyUI), so the pool holds a lock per box. A box that fails to come up is tried again after `retry_down` seconds, and
every box down with none busy raises the last box's error instead of waiting for ever. The choice is written to the take's folder (`pin_file`): when the consumer says a take was
sent earlier (`sent`), the take runs on the box that holds it and on no other, because only that box can give its result back.
"""

from __future__ import annotations

import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from .box import Box
from .errors import CloudGpuError

POOL_VAR = "CLOUDGPU_POOL"
PIN_FILE = "cloudgpu_box"
RETRY_DOWN_S = 60.0
POLL_S = 2.0


def boxes_from(env, box_cls=Box) -> list:
    """[(name, Box)] from the environment: one unnamed box without CLOUDGPU_POOL, else one per listed name, in that order (the first is preferred)."""
    names = [n.strip() for n in str(env.get(POOL_VAR) or "").split(",") if n.strip()]
    if not names:
        return [("", box_cls(env))]
    out = []
    for n in names:
        if not re.fullmatch(r"[A-Za-z0-9]+", n) or n.lower() in [o for o, _ in out]:
            raise CloudGpuError("key_missing", f"{POOL_VAR}: {n!r} is not a box name (letters and digits, each once)")
        pre = f"CLOUDGPU_{n.upper()}_"
        sub = {k: v for k, v in env.items() if not k.startswith(("CLOUDGPU_", "H3HOST_"))}
        sub.update({"CLOUDGPU_" + k[len(pre):]: v for k, v in env.items() if k.startswith(pre)})
        out.append((n.lower(), box_cls(sub)))
    return out


class _Skip(Exception):
    pass


class Pool:
    def __init__(self, boxes: list, *, pin_file: str = PIN_FILE, retry_down: float = RETRY_DOWN_S, poll: float = POLL_S, clock=time.monotonic, sleep=time.sleep):
        self.boxes = boxes
        self.pin_file, self.retry_down, self.poll, self._clock, self._sleep = pin_file, retry_down, poll, clock, sleep
        self._free = {n: threading.Lock() for n, _ in boxes}      # a box makes one take at a time
        self._here = threading.local()                           # the box this thread's take runs on
        self._down: dict[str, float] = {}                        # box name -> when it last failed to come up

    @classmethod
    def from_env(cls, env, **kw) -> "Pool":
        return cls(boxes_from(env), **kw)

    @property
    def box(self) -> Box:
        """The box of this thread's take, else the first box (a pool of one is simply its box)."""
        return getattr(self._here, "box", None) or self.boxes[0][1]

    @box.setter
    def box(self, value) -> None:
        self.boxes[0] = (self.boxes[0][0], value)

    def _recently_down(self, name: str) -> bool:
        t = self._down.get(name)
        return t is not None and self._clock() - t < self.retry_down

    @contextmanager
    def take(self, out_dir, sent=lambda out_dir: False):
        """Run a take on a box of the pool: yields the box. `sent(out_dir)` says an earlier run sent this take (its task is on the box written in `out_dir`)."""
        if len(self.boxes) == 1:
            yield self.boxes[0][1]
            return
        out_dir = Path(out_dir)
        names = [n for n, _ in self.boxes]
        pin = out_dir / self.pin_file
        pinned = pin.read_text(encoding="utf-8").strip() if pin.is_file() and sent(out_dir) else None
        last: CloudGpuError | None = None
        while True:
            for name, box in self.boxes:
                if pinned in names and name != pinned:
                    continue
                if not self._free[name].acquire(blocking=False):
                    continue
                try:
                    if pinned is None and self._recently_down(name):
                        raise _Skip()
                    box.ensure_up()
                except _Skip:
                    self._free[name].release()
                    continue
                except CloudGpuError as e:
                    self._down[name] = self._clock()
                    self._free[name].release()
                    if pinned is not None:                  # the box that holds the sent take: no other box can give its result back
                        raise
                    last = e
                    continue
                except BaseException:
                    self._free[name].release()
                    raise
                self._down.pop(name, None)
                self._here.box = box
                try:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    pin.write_text(name, encoding="utf-8")
                    yield box
                finally:
                    self._here.box = None
                    self._free[name].release()
                return
            if last is not None and all(self._recently_down(n) for n in names) and not any(self._free[n].locked() for n in names):
                raise last                                  # every box is down and none is busy: nothing will free up
            self._sleep(self.poll)

    def power_off_if_idle(self) -> bool:
        return any([b.power_off_if_idle() for _, b in self.boxes])
