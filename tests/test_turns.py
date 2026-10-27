"""cloudgpu.turns and its use in Pool.take: takes of one machine keep a line, oldest first; a take that dies leaves nothing behind (the OS lets go of its lock);
a ticket is alive while its lock is held, so a second Ticket in this process stands in for another process. Fake boxes, no network."""

import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import _t

from cloudgpu import turns
from cloudgpu.pool import Pool
from cloudgpu.status import status


class FakeBox:
    def __init__(self, env=None):
        self.ups = 0

    def ensure_up(self):
        self.ups += 1

    def power_off_if_idle(self):
        return False


def pool(sd, names="ab", **kw):
    return Pool([(n, FakeBox()) for n in names], poll=0.01, state_dir=sd, on_wait=kw.pop("on_wait", lambda m: None), **kw)


def later(fn, delay=0.15):
    t = threading.Timer(delay, fn)
    t.start()
    return t


def test_an_older_waiting_take_goes_first_and_a_running_take_blocks_only_its_box():
    with _t.tmpdir() as sd, _t.tmpdir() as d:
        older = turns.Ticket.open(sd, ["a", "b"], "older")
        got, said = [], []
        p = pool(sd, on_wait=said.append)

        def go():
            with p.take(Path(d) / "t") as box:
                got.append(time.time())
        th = threading.Thread(target=go)
        th.start()
        time.sleep(0.2)
        assert not got and any("waiting for its turn: older" in m for m in said), said
        t0 = time.time()
        older.close()
        th.join(5)
        assert got and got[0] >= t0
        r = turns.Ticket.open(sd, ["a", "b"], "running")                         # another process runs on a
        assert r.my_turn("a") is True and r.run("a")
        with p.take(Path(d) / "t2") as box:
            assert box is p.boxes[1][1], "a is taken by the other process: this one goes to b"
        r.close()


def test_a_pool_of_one_keeps_the_line_in_arrival_order_across_pools():
    with _t.tmpdir() as sd, _t.tmpdir() as d:
        first = pool(sd, "x")
        order, ready = [], threading.Event()

        def make(label, p, hold_s=0):
            def go():
                with p.take(Path(d) / label):
                    order.append(label)
                    ready.set()
                    time.sleep(hold_s)
            return threading.Thread(target=go)
        t1 = make("one", first, 0.3)
        t1.start()
        ready.wait(5)
        t2, t3 = make("two", pool(sd, "x")), make("three", pool(sd, "x"))
        t2.start()
        time.sleep(0.1)
        t3.start()
        for t in (t1, t2, t3):
            t.join(10)
        assert order == ["one", "two", "three"], order


def test_a_dead_take_leaves_nothing_behind_and_a_ticket_with_no_owner_is_removed():
    with _t.tmpdir() as sd:
        code = ("import sys, time; sys.path.insert(0, %r); from cloudgpu import turns; t = turns.Ticket.open(%r, ['a'], 'child'); print('up', flush=True); time.sleep(60)"
                % (_t.ROOT, sd))
        c = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
        try:
            assert c.stdout.readline().strip() == "up"
            assert [t["label"] for t in turns.tickets(sd)] == ["child"], "alive while its process lives"
        finally:
            c.kill()
            c.wait()
        assert turns.tickets(sd) == [] and not list((Path(sd) / "turns").glob("*")), "killed: the OS let go of the lock, and the files are removed"
        (Path(sd) / "turns" / "00000000000000000001-1-x.lock").write_text("")      # a lock file nobody holds
        assert turns.tickets(sd) == []


def test_without_flock_or_a_writable_state_directory_a_take_goes_as_before():
    with _t.tmpdir() as d:
        ro = Path(d) / "ro"
        ro.write_text("a file, so no directory can be made under it")
        p = pool(str(ro))
        with p.take(Path(d) / "t") as box:
            assert box is p.boxes[0][1]
        old, turns.fcntl = turns.fcntl, None
        try:
            with pool(tempfile.mkdtemp()).take(Path(d) / "t2") as box:
                assert box is not None
        finally:
            turns.fcntl = old


def test_status_shows_who_runs_and_who_waits_without_removing_anything():
    with _t.tmpdir() as sd:
        r, w = turns.Ticket.open(sd, ["a"], "take-1"), turns.Ticket.open(sd, ["a"], "take-2")
        r.run("a")
        doc = status({"CLOUDGPU_STATE_DIR": sd})
        assert [(t["label"], t["state"]) for t in doc["queue"]] == [("take-1", "running"), ("take-2", "waiting")], doc["queue"]
        assert any("take-1" in n and "running on a" in n for n in doc["next"]) and len(list((Path(sd) / "turns").glob("*.lock"))) == 2
        r.close()
        w.close()
        assert status({"CLOUDGPU_STATE_DIR": sd})["queue"] == []


if __name__ == "__main__":
    _t.main(globals())
