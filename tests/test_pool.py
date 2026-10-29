"""cloudgpu.pool: boxes in order of preference. A take goes to the first box that is free and comes up; a box that fails to come up is tried again after a pause; every box
down with none busy is an error, not a wait; a take sent earlier stays on the box that holds it; a pool of one is just its box. Fake boxes, no network."""

import tempfile
import threading
import time
from pathlib import Path

import _t

from cloudgpu import hold
from cloudgpu.errors import CloudGpuError
from cloudgpu.pool import PIN_FILE, Pool, boxes_from


class FakeBox:
    def __init__(self, env=None, up=True):
        self.env, self.up, self.ups = env or {}, up, 0

    def ensure_up(self):
        self.ups += 1
        if not self.up:
            raise CloudGpuError("unreachable", "the box does not answer")

    def power_off_if_idle(self):
        return False


def pool_of(*ups, **kw):
    boxes = [(n, FakeBox(up=u)) for n, u in zip("ab", ups)]
    kw.setdefault("state_dir", tempfile.mkdtemp(prefix="cloudgpu-state-"))      # never the real ~/.cloudgpu
    return Pool(boxes, poll=0.01, **kw), boxes


def test_a_take_goes_to_the_first_free_box_that_comes_up_and_is_written_to_its_folder():
    with _t.tmpdir() as d:
        pool, boxes = pool_of(True, True)
        with pool.take(Path(d) / "t1") as box:
            assert box is boxes[0][1] and pool.box is boxes[0][1] and pool._free["a"].locked() and not pool._free["b"].locked()
        assert not pool._free["a"].locked() and (Path(d) / "t1" / PIN_FILE).read_text() == "a"
        pool._free["a"].acquire()                                           # the first box is making another take
        with pool.take(Path(d) / "t2") as box:
            assert box is boxes[1][1]
        pool._free["a"].release()


def test_a_box_that_does_not_come_up_is_skipped_tried_again_later_and_all_down_is_an_error():
    with _t.tmpdir() as d:
        pool, boxes = pool_of(False, True)
        with pool.take(Path(d) / "t1") as box:
            assert box is boxes[1][1]
        ups = boxes[0][1].ups
        with pool.take(Path(d) / "t2") as box:                              # a is down and was tried a moment ago: not tried again at once
            assert box is boxes[1][1]
        assert boxes[0][1].ups == ups
        pool.retry_down = 0.0
        boxes[0][1].up = True
        with pool.take(Path(d) / "t3") as box:
            assert box is boxes[0][1]
        both, _ = pool_of(False, False)
        with _t.raises("unreachable"):
            with both.take(Path(d) / "t4"):
                pass
        assert not any(l.locked() for l in both._free.values())


def test_a_take_sent_earlier_stays_on_its_box_and_waits_for_it():
    with _t.tmpdir() as d:
        pool, boxes = pool_of(True, True)
        out = Path(d) / "t"
        out.mkdir()
        (out / PIN_FILE).write_text("b")
        with pool.take(out, sent=lambda o: True) as box:
            assert box is boxes[1][1]
        with pool.take(out, sent=lambda o: False) as box:                   # nothing was sent: the pin is only a note, the first box takes it
            assert box is boxes[0][1]
        (out / PIN_FILE).write_text("b")
        boxes[1][1].up = False
        with _t.raises("unreachable"):                                      # the box that holds it is down: no other box can fetch it
            with pool.take(out, sent=lambda o: True):
                pass
        boxes[1][1].up = True
        pool._free["b"].acquire()
        got = []

        def wait():
            with pool.take(out, sent=lambda o: True) as box:
                got.append(box)
        t = threading.Thread(target=wait)
        t.start()
        time.sleep(0.1)
        assert not got, "it waits for b, it does not take a"
        pool._free["b"].release()
        t.join(5)
        assert got == [boxes[1][1]]


def test_the_pool_is_read_from_the_environment_and_one_box_is_a_pool_of_one():
    env = {"CLOUDGPU_POOL": "a,b", "CLOUDGPU_A_TOKEN": "ta", "CLOUDGPU_B_TOKEN": "tb", "CLOUDGPU_B_BASE": "http://127.0.0.1:16010", "AUTODL_DEV_TOKEN": "dev"}
    pool = boxes_from(env, FakeBox)
    assert [n for n, _ in pool] == ["a", "b"]
    assert pool[0][1].env["CLOUDGPU_TOKEN"] == "ta" and pool[1][1].env["CLOUDGPU_BASE"] == "http://127.0.0.1:16010"
    assert "CLOUDGPU_BASE" not in pool[0][1].env and pool[0][1].env["AUTODL_DEV_TOKEN"] == "dev" and "CLOUDGPU_A_TOKEN" not in pool[0][1].env
    one = Pool(boxes_from({"CLOUDGPU_TOKEN": "t"}, FakeBox))
    with _t.tmpdir() as d:
        with one.take(Path(d) / "x") as box:
            assert box is one.box and not (Path(d) / "x").exists(), "a pool of one writes nothing"
    for bad in ("a,a", "bad name", "a,,b,a"):
        with _t.raises("key_missing"):
            boxes_from({"CLOUDGPU_POOL": bad}, FakeBox)


def test_a_held_box_is_busy_and_another_box_takes_the_take():
    with _t.tmpdir() as d:
        pool, boxes = pool_of(True, True)
        hold.place(pool._state_dir, "a", 30, "pat", "manual test")
        with pool.take(Path(d) / "t1") as box:
            assert box is boxes[1][1] and boxes[0][1].ups == 0, "a is held: it is not even brought up"
        hold.release(pool._state_dir, "a")
        with pool.take(Path(d) / "t2") as box:
            assert box is boxes[0][1]


def test_when_every_box_is_held_the_take_waits_says_so_once_and_goes_when_the_hold_ends():
    with _t.tmpdir() as d:
        said = []
        now = [1000.0]
        pool, boxes = pool_of(True, True, now=lambda: now[0], on_wait=said.append)
        pool._sleep = lambda s: now.__setitem__(0, now[0] + 60)                  # each wait of the pool is a minute of the wall clock
        hold.place(pool._state_dir, "a", 3, "pat", now=now[0])
        hold.place(pool._state_dir, "b", 10, "sam", now=now[0])
        with pool.take(Path(d) / "t1") as box:
            assert box is boxes[0][1] and now[0] >= 1180, "a's hold ended first and a took it"
        assert len(said) == 1 and "waiting for the hold on a by pat until" in said[0], said


def test_a_pool_of_one_a_pinned_take_and_a_held_only_box_wait_too_and_no_hold_is_no_wait():
    with _t.tmpdir() as d:
        now = [1000.0]
        one = Pool([("", FakeBox())], poll=0.01, state_dir=tempfile.mkdtemp(), now=lambda: now[0], on_wait=lambda m: None)
        one._sleep = lambda s: now.__setitem__(0, now[0] + 60)
        hold.place(one._state_dir, "", 2, "pat", now=now[0])
        with one.take(Path(d) / "x") as box:
            assert box is one.box and now[0] >= 1120
        pool, boxes = pool_of(True, True, now=lambda: now[0], on_wait=lambda m: None)
        pool._sleep = lambda s: now.__setitem__(0, now[0] + 60)
        out = Path(d) / "t"
        out.mkdir()
        (out / PIN_FILE).write_text("b")
        t0 = now[0]
        hold.place(pool._state_dir, "b", 2, "pat", now=t0)
        with pool.take(out, sent=lambda o: True) as box:
            assert box is boxes[1][1] and now[0] >= t0 + 120, "the box that holds the sent take is held: no other box can serve it, so it waits"


def test_an_all_down_pool_still_raises_and_a_held_box_is_not_powered_off():
    with _t.tmpdir() as d:
        pool, boxes = pool_of(False, False)
        with _t.raises("unreachable"):
            with pool.take(Path(d) / "t"):
                pass
        calls = []
        pool, boxes = pool_of(True, True)
        for n, b in boxes:
            b.power_off_if_idle = (lambda n=n: calls.append(n) or True)
        hold.place(pool._state_dir, "a", 30, "pat")
        assert pool.power_off_if_idle() is True and calls == ["b"]


def test_a_hold_file_is_atomic_expires_by_itself_is_capped_and_is_not_taken_over():
    with _t.tmpdir() as sd:
        h = hold.place(sd, "A", 60, "pat", "x" * 500, now=1000.0)
        assert h["until"] == 4600.0 and len(h["why"]) == hold.TEXT_MAX and hold.get(sd, "a", 1500.0)["who"] == "pat"
        assert hold.get(sd, "a", 4600.0) is None and not (Path(sd) / "holds" / "a.json").exists(), "ended: gone and the file removed"
        hold.place(sd, "a", 60, "pat", now=1000.0)
        with _t.raises("key_missing") as c:                                       # someone else's hold is not replaced
            hold.place(sd, "a", 10, "sam", now=1100.0)
        assert "pat" in c.err.message
        assert hold.place(sd, "a", 90, "pat", now=1100.0)["until"] == 6500.0, "your own hold is extended"
        for bad in (0, -5, hold.MAX_MINUTES + 1):
            with _t.raises("key_missing"):
                hold.place(sd, "b", bad, "pat")
        with _t.raises("key_missing"):
            hold.place(sd, "b", 5, " ")
        with _t.raises("key_missing"):
            hold.place(sd, "../x", 5, "pat")
        (Path(sd) / "holds" / "c.json").write_text("not json")
        assert hold.get(sd, "c") is None and not (Path(sd) / "holds" / "c.json").exists(), "a file that is no hold never blocks a box"
        assert [x["box"] for x in hold.active(sd, 1200.0)] == ["a"]
        assert hold.release(sd, "a") is True and hold.release(sd, "a") is False


class LineBox(FakeBox):
    """A box whose adapter has the line (0.7.0): it always says go and counts the tickets given out."""

    def __init__(self):
        super().__init__()
        self.joins, self.leaves = 0, 0

    def line_join(self, label):
        self.joins += 1
        return {"ticket": f"T{self.joins}", "state": "go", "ahead": 0}

    def line_poll(self, ticket):
        return {"ticket": ticket, "state": "go", "ahead": 0}

    def line_leave(self, ticket):
        self.leaves += 1


def running_takes(pool, n, d, hold_s=5.0):
    """Start `n` takes in threads that stay inside `pool.take` until `go` is set; returns (threads, which box each got, the event that lets them go)."""
    got, go, threads = {}, threading.Event(), []

    def one(i):
        with pool.take(Path(d) / f"t{i}") as box:
            got[i] = box
            go.wait(hold_s)

    for i in range(n):
        th = threading.Thread(target=one, args=(i,), daemon=True)
        th.start()
        threads.append(th)
        time.sleep(0.1)                                         # in arrival order
    return threads, got, go


def test_with_a_depth_of_two_a_box_takes_a_second_take_and_the_least_loaded_box_goes_first():
    with _t.tmpdir() as d:
        pool, boxes = pool_of(True, True, depth=2)
        a, b = boxes[0][1], boxes[1][1]
        threads, got, go = running_takes(pool, 4, d)
        try:
            assert got[0] is a and got[1] is b, "the second take goes to the empty box, not to the first box's second place"
            assert {got[2], got[3]} == {a, b} and len(got) == 4 and pool._free["a"].locked() and pool._free["b"].locked()
            extra = []
            th = threading.Thread(target=lambda: [extra.append(1) for _ in [pool.take(Path(d) / "t9").__enter__()]], daemon=True)
            th.start()
            time.sleep(0.3)
            assert not extra, "a fifth take waits: both boxes have their two"
        finally:
            go.set()
        for t in threads:
            t.join(5)
        th.join(5)
        assert extra == [1] and not pool._free["a"].locked()


def test_a_depth_of_one_is_the_old_lock_and_the_setting_is_read_from_the_environment():
    with _t.tmpdir() as d:
        pool, boxes = pool_of(True, True)
        assert pool.depth == 1
        threads, got, go = running_takes(pool, 2, d)
        try:
            assert got[0] is boxes[0][1] and got[1] is boxes[1][1] and pool._free["a"].locked() and pool._free["b"].locked()
        finally:
            go.set()
        for t in threads:
            t.join(5)
    base = {"CLOUDGPU_STATE_DIR": tempfile.mkdtemp(prefix="cloudgpu-state-")}
    assert Pool.from_env({**base}).depth == 1
    assert Pool.from_env({**base, "CLOUDGPU_QUEUE_DEPTH": "2"}).depth == 2
    assert Pool.from_env({**base, "CLOUDGPU_QUEUE_DEPTH": "9"}).depth == 4, "more than four is four"
    with _t.raises("key_missing"):
        Pool.from_env({**base, "CLOUDGPU_QUEUE_DEPTH": "many"})


def test_only_the_first_take_on_a_box_takes_a_place_in_its_line_and_the_cross_process_queue_lets_depth_takes_run():
    with _t.tmpdir() as d:
        box = LineBox()
        pool = Pool([("a", box), ("b", FakeBox())], poll=0.01, depth=2, state_dir=tempfile.mkdtemp(prefix="cloudgpu-state-"))
        pool.boxes[1] = ("b", FakeBox(up=False))                  # only box a comes up: both takes land on it
        pool.retry_down = 1000.0
        threads, got, go = running_takes(pool, 2, d)
        try:
            from cloudgpu import turns
            assert got[0] is box and got[1] is box and box.joins == 1, (got, box.joins)       # the second rides on the first's place
            run = [t for t in turns.tickets(pool._state_dir) if t["state"] == "running"]
            assert len(run) == 2 and {t["box"] for t in run} == {hold.box_key("a")}, run
        finally:
            go.set()
        for t in threads:
            t.join(5)
        assert box.leaves >= 1


def test_the_state_directory_comes_from_the_setting():
    assert hold.state_dir({"CLOUDGPU_STATE_DIR": "/x/y"}) == Path("/x/y") and hold.state_dir({}) == Path.home() / ".cloudgpu"
    assert Pool.from_env({"CLOUDGPU_STATE_DIR": "/x/y"})._state_dir == Path("/x/y")


if __name__ == "__main__":
    _t.main(globals())
