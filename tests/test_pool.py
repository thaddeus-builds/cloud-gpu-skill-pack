"""cloudgpu.pool: boxes in order of preference. A take goes to the first box that is free and comes up; a box that fails to come up is tried again after a pause; every box
down with none busy is an error, not a wait; a take sent earlier stays on the box that holds it; a pool of one is just its box. Fake boxes, no network."""

import threading
import time
from pathlib import Path

import _t

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


if __name__ == "__main__":
    _t.main(globals())
