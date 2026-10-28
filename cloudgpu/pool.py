"""A pool of boxes in order of preference: a take waits for the first box that is not making another take and comes up.

`CLOUDGPU_POOL=a,b` names the boxes; each box's settings are `CLOUDGPU_<A>_<NAME>` (`CLOUDGPU_A_TOKEN`, `CLOUDGPU_A_BASE`, ...: one local tunnel port of its own in its BASE),
and every setting that is not a box's is shared. With no pool one box on the plain `CLOUDGPU_<NAME>` settings is a pool of one and nothing changes for its caller.

`Pool.take(out_dir)` is the one call a consumer makes around a take:

    with pool.take(out_dir) as box:        # the box this take runs on; pool.box is the same inside this thread
        ... send the take, wait for it, fetch it ...

A box makes one take at a time (the adapter queues behind ComfyUI), so the pool holds a lock per box. A box that fails to come up is tried again after `retry_down` seconds, and
every box down with none busy raises the last box's error instead of waiting for ever. The choice is written to the take's folder (`pin_file`): when the consumer says a take was
sent earlier (`sent`), the take runs on the box that holds it and on no other, because only that box can give its result back.

A box a person holds (cloudgpu/hold.py) counts as busy: the take goes to another box, and when every box it may use is held it waits, saying so once through `on_wait` (stderr by
default), until a hold ends. That includes a pool of one and a take sent earlier (a hold means stay off the box). `power_off_if_idle` leaves a held box on.

Takes of one machine (processes sharing the state directory) also keep a line (turns.py): the oldest waiting take for a free box goes first, whichever process it is in.

Takes of DIFFERENT machines keep a line on the box itself (line.py, the adapter's /cloudgpu/line): once a take has the box to itself on this machine and the box is up, it takes a ticket there
and goes when the box says "go". While it runs a thread asks about the ticket every `beat` seconds (the adapter drops a ticket nobody asked about for 60 s, so a machine that died never blocks the
line). With several boxes a take that waits in one box's line tries the next box, keeps its place where it waits, and gives up the others when it goes. A box whose adapter has no line (older than
0.7.0), or that does not answer the line, is gone to as before: the line orders callers, it never stops one. The ticket's label (this machine's name and the take's folder name) goes to the box.
"""

from __future__ import annotations

import os
import re
import socket
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from . import hold, turns
from .box import Box
from .errors import CloudGpuError

POOL_VAR = "CLOUDGPU_POOL"
PIN_FILE = "cloudgpu_box"
RETRY_DOWN_S = 60.0
POLL_S = 2.0
BEAT_S = 20.0                      # a third of the adapter's lease (60 s)


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
    def __init__(self, boxes: list, *, pin_file: str = PIN_FILE, retry_down: float = RETRY_DOWN_S, poll: float = POLL_S, clock=time.monotonic, sleep=time.sleep,
                 state_dir=None, now=time.time, on_wait=None, beat: float = BEAT_S):
        self.boxes = boxes
        self.pin_file, self.retry_down, self.poll, self._clock, self._sleep = pin_file, retry_down, poll, clock, sleep
        self._state_dir, self._now = state_dir, now              # the holds' directory (None: CLOUDGPU_STATE_DIR, else ~/.cloudgpu) and the wall clock they are read against
        self._on_wait = on_wait or (lambda msg: print(msg, file=sys.stderr, flush=True))
        self._said: dict[str, str] = {}                          # what was already announced, by kind: a wait is said once, not every poll
        self._free = {n: threading.Lock() for n, _ in boxes}      # a box makes one take at a time
        self._here = threading.local()                           # the box this thread's take runs on
        self._down: dict[str, float] = {}                        # box name -> when it last failed to come up
        self.beat = beat
        self._noline: set = set()                                # boxes whose adapter has no line (older than 0.7.0)

    @classmethod
    def from_env(cls, env, **kw) -> "Pool":
        kw.setdefault("state_dir", hold.state_dir(env))
        return cls(boxes_from(env), **kw)

    def _held(self, name: str) -> dict | None:
        sd = self._state_dir if self._state_dir is not None else hold.state_dir(os.environ)
        try:
            return hold.get(sd, name, self._now())
        except (CloudGpuError, OSError):                         # a state directory nobody can read is no hold
            return None

    def _announce(self, h: dict) -> None:
        self._say("hold:" + h["box"], hold.waiting_message(h))

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

    def _sd(self):
        return self._state_dir if self._state_dir is not None else hold.state_dir(os.environ)

    def _say(self, key: str, text: str) -> None:
        if self._said.get(key) != text:
            self._said[key] = text
            self._on_wait(text)

    @contextmanager
    def take(self, out_dir, sent=lambda out_dir: False):
        """Run a take on a box of the pool: yields the box. `sent(out_dir)` says an earlier run sent this take (its task is on the box written in `out_dir`).
        The take joins the line of this machine (turns.py) and goes when it is oldest for a box that is free, not held and comes up."""
        out_dir = Path(out_dir)
        names = [n for n, _ in self.boxes]
        pin = out_dir / self.pin_file
        pinned = pin.read_text(encoding="utf-8").strip() if len(names) > 1 and pin.is_file() and sent(out_dir) else None
        eligible = [pinned] if pinned in names else names
        tk = turns.Ticket.open(self._sd(), [hold.box_key(n) for n in eligible], out_dir.name, self._now)
        lines: dict = {}                                         # box name -> {box, id, label}: this take's ticket in that box's line
        label = f"{socket.gethostname().split('.')[0]}/{out_dir.name}"
        try:
            with self._take(out_dir, pin, names, pinned, tk, lines, label) as box:
                yield box
        finally:
            self._leave(lines)
            if tk is not None:
                tk.close()

    def _ahead(self, tk, name: str) -> bool:
        """True (and it says so, once) when another take of this machine is ahead of this one for `name`."""
        if tk is None or tk.my_turn(hold.box_key(name)):
            return False
        ahead = tk.blockers(hold.box_key(name))
        self._say("turn", "waiting for its turn: " + (turns.describe(ahead[0]) if ahead else "another take is ahead"))
        return True

    def _line(self, box, name: str, lines: dict, label: str) -> bool:
        """True when the take may use `box`: it has no line to wait in, or the box says "go". False (and it says so) while callers of other machines are ahead. The ticket is kept in `lines`
        and asked about on every call, which renews its lease; one the adapter forgot is joined again. Every failure to ask is True: the line orders callers, it does not stop one."""
        if name in self._noline or not hasattr(box, "line_join"):
            return True
        try:
            ent = lines.get(name)
            st = box.line_poll(ent["id"]) if ent else None
            if st is None:
                st = box.line_join(label)
                if st is None:
                    self._noline.add(name)
                    lines.pop(name, None)
                    return True
        except CloudGpuError:
            return True
        lines[name] = {"box": box, "id": st["ticket"], "label": label}
        if st["state"] == "go":
            return True
        who = f" ({st['running']} is running)" if st.get("running") else ""
        self._say("line:" + name, f"waiting for its turn on {name or 'the box'}: {st['ahead']} take(s) of other machines ahead{who}")
        return False

    @staticmethod
    def _drop(lines: dict, name: str) -> None:
        """Give up this take's ticket on `name`, best effort: one that is not given up ends by itself with its lease."""
        ent = lines.pop(name, None)
        if ent is not None:
            try:
                ent["box"].line_leave(ent["id"])
            except CloudGpuError:
                pass

    def _leave(self, lines: dict, keep: str | None = None) -> None:
        for n in [n for n in lines if n != keep]:
            self._drop(lines, n)

    @contextmanager
    def _running(self, name: str, lines: dict):
        """While the take runs on `name`: its other tickets are given up and a thread keeps this one's lease alive."""
        self._leave(lines, keep=name)
        ent = lines.get(name)
        stop = threading.Event()
        th = None
        if ent is not None:
            def beat():
                while not stop.wait(self.beat):
                    try:
                        if ent["box"].line_poll(ent["id"]) is None:         # forgotten: take a place again (the front, if no one else is waiting)
                            st = ent["box"].line_join(ent["label"])
                            if st is not None:
                                ent["id"] = st["ticket"]
                    except CloudGpuError:
                        pass
            th = threading.Thread(target=beat, daemon=True)
            th.start()
        try:
            yield
        finally:
            stop.set()
            if th is not None:
                th.join(timeout=1.0)

    @contextmanager
    def _take(self, out_dir, pin, names, pinned, tk, lines, label):
        if len(self.boxes) == 1:
            name, box = self.boxes[0]
            while True:
                h = self._held(name)
                if h is not None:
                    self._announce(h)
                elif not self._ahead(tk, name) and self._line(box, name, lines, label) and (tk is None or tk.run(hold.box_key(name))):
                    break
                self._sleep(self.poll)
            try:
                with self._running(name, lines):
                    yield box
            finally:
                if tk is not None:
                    tk.wait()
            return
        last: CloudGpuError | None = None
        while True:
            held, queued = [], False
            for name, box in self.boxes:
                if pinned in names and name != pinned:
                    continue
                h = self._held(name)
                if h is not None:
                    held.append(h)
                    continue
                if self._ahead(tk, name):
                    queued = True
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
                if not self._line(box, name, lines, label):                    # callers of other machines are ahead on this box: try the next, keep the place
                    self._free[name].release()
                    queued = True
                    continue
                if tk is not None and not tk.run(hold.box_key(name)):          # an older take of another process got there first
                    self._free[name].release()
                    self._drop(lines, name)                                    # and the place it was given on the box goes to the next in the line
                    queued = True
                    continue
                self._down.pop(name, None)
                self._here.box = box
                try:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    pin.write_text(name, encoding="utf-8")
                    with self._running(name, lines):
                        yield box
                finally:
                    self._here.box = None
                    if tk is not None:
                        tk.wait()
                    self._free[name].release()
                return
            if held:
                self._announce(min(held, key=lambda h: h["until"]))
            elif queued:
                pass
            elif last is not None and all(self._recently_down(n) for n in names) and not any(self._free[n].locked() for n in names):
                raise last                                  # every box is down and none is busy: nothing will free up
            self._sleep(self.poll)

    def power_off_if_idle(self) -> bool:
        return any([b.power_off_if_idle() for n, b in self.boxes if self._held(n) is None])
