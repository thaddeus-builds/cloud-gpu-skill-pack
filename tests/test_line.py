"""The line across machines: Pool joins a box's line (line.py, the adapter's /cloudgpu/line) before it uses the box. Scripted boxes for the rules (wait while others are ahead, a
place kept while another box is tried, an old adapter or a line that does not answer is gone past, the lease kept alive while a take runs); a real adapter, over HTTP, for two
pools of two "machines" taking one box in turn; `status` lists who runs and who waits."""

import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import _t
import test_adapter as ta

from cloudgpu import line
from cloudgpu.box import Box as RealBox
from cloudgpu.errors import CloudGpuError
from cloudgpu.pool import Pool
from cloudgpu.status import status


class LineBox:
    """A box that is always up and whose line is scripted: `script` is the list of states poll/join answer with, in order (the last one repeats)."""

    def __init__(self, script=("go",), join=True):
        self.script, self.can_join, self.calls, self.n = list(script), join, [], 0

    def ensure_up(self):
        pass

    def power_off_if_idle(self):
        return False

    def _answer(self):
        st = self.script[min(self.n, len(self.script) - 1)]
        self.n += 1
        if isinstance(st, Exception):
            raise st
        return {"ticket": "t1", "state": st, "ahead": 0 if st == "go" else 2, "running": None if st == "go" else "other/take", "lease_s": 60}

    def line_join(self, label=""):
        self.calls.append(("join", label))
        return self._answer() if self.can_join else None

    def line_poll(self, ticket):
        self.calls.append(("poll", ticket))
        return self._answer()

    def line_leave(self, ticket):
        self.calls.append(("leave", ticket))


def pool_of(*boxes, said=None, **kw):
    import tempfile
    kw.setdefault("state_dir", tempfile.mkdtemp(prefix="cloudgpu-state-"))
    kw.setdefault("beat", 0.02)
    return Pool([(n, b) for n, b in zip("ab", boxes)] if len(boxes) > 1 else [("", boxes[0])], poll=0.01, on_wait=(said.append if said is not None else (lambda m: None)), **kw)


def test_a_take_waits_while_callers_of_other_machines_are_ahead_says_so_once_and_gives_up_its_place_at_the_end():
    with _t.tmpdir() as d:
        box, said = LineBox(["waiting", "waiting", "go"]), []
        pool = pool_of(box, said=said)
        with pool.take(Path(d) / "t1") as b:
            assert b is box and box.n == 3
        assert said == ["waiting for its turn on the box: 2 take(s) of other machines ahead (other/take is running)"], said
        kinds = [c[0] for c in box.calls]
        assert kinds[0] == "join" and kinds[1:3] == ["poll", "poll"] and kinds[-1] == "leave" and box.calls[0][1].endswith("/t1"), box.calls


def test_the_ticket_is_asked_about_while_the_take_runs_so_its_lease_is_kept():
    with _t.tmpdir() as d:
        box = LineBox(["go"])
        pool = pool_of(box)
        with pool.take(Path(d) / "t1"):
            time.sleep(0.3)
            assert sum(1 for c in box.calls if c[0] == "poll") >= 3, box.calls
        n = len(box.calls)
        time.sleep(0.1)
        assert len(box.calls) == n and box.calls[-1][0] == "leave"                  # the thread stops with the take


def test_an_adapter_without_a_line_or_a_line_that_does_not_answer_is_gone_past_never_waited_on():
    with _t.tmpdir() as d:
        old = LineBox(join=False)
        pool = pool_of(old)
        with pool.take(Path(d) / "t1") as b:
            assert b is old
        with pool.take(Path(d) / "t2"):
            pass
        assert [c[0] for c in old.calls] == ["join"]                                 # asked once, then remembered: no call per take
        broken = LineBox([CloudGpuError("http", "the box's line did not answer")])
        with pool_of(broken).take(Path(d) / "t3") as b:
            assert b is broken
        forgotten = LineBox(["go"])                                                   # a ticket the adapter forgot is joined again
        orig = forgotten._answer
        answers = iter([None, "again"])

        def poll(ticket):
            forgotten.calls.append(("poll", ticket))
            return None if next(answers, "again") is None else orig()
        forgotten.line_poll = poll
        with pool_of(forgotten, beat=0.01).take(Path(d) / "t4"):
            time.sleep(0.15)
        assert [c[0] for c in forgotten.calls].count("join") >= 2, forgotten.calls


def test_with_two_boxes_a_take_that_waits_on_the_first_uses_the_second_and_gives_the_first_back():
    with _t.tmpdir() as d:
        a, b = LineBox(["waiting"]), LineBox(["go"])
        pool = pool_of(a, b)
        with pool.take(Path(d) / "t1") as box:
            assert box is b and (Path(d) / "t1" / "cloudgpu_box").read_text() == "b"
            assert ("leave", "t1") in a.calls                                         # its place on a was given up when it went
        assert b.calls[-1] == ("leave", "t1")
        a2, b2 = LineBox(["waiting", "go"]), LineBox(["waiting"])                     # both busy: it keeps both places and goes where the box says go first
        with pool_of(a2, b2).take(Path(d) / "t2") as box:
            assert box is a2
        assert ("leave", "t1") in b2.calls


class OldAdapter(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _no(self):
        self.send_response(404)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    do_GET = do_POST = _no


class StubBox:
    def __init__(self, base):
        self.base = base

    def root(self):
        return self.base

    def token(self):
        return "t" * 32


def test_line_py_reads_a_404_as_no_line_a_dead_listener_as_an_error_and_a_full_line_as_429():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), OldAdapter)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        box = StubBox(f"http://127.0.0.1:{srv.server_address[1]}")
        assert line.join(box, "x") is None and line.poll(box, "abc") is None and line.listing(box) is None
        line.leave(box, "abc")                                                         # an old adapter answers 501 or 404: ignored
    finally:
        srv.shutdown()
    with _t.raises("http"):
        line.join(StubBox("http://127.0.0.1:1"), "x")


class Machine:
    """A box seen from one machine: the real client (cloudgpu.box.Box) against the adapter at `base`, with the proof of the token checked as it is on a tunnel."""

    @staticmethod
    def box(base):
        return RealBox({"CLOUDGPU_BASE": base, "CLOUDGPU_TOKEN": ta.TOKEN, "CLOUDGPU_POWER": "none"})


def test_two_machines_take_one_box_in_turn_and_status_says_who_runs_and_who_waits():
    with _t.tmpdir() as d:
        adapter_box = ta.Box(d)
        try:
            m1, m2 = Machine.box(adapter_box.base), Machine.box(adapter_box.base)
            m1.ensure_up = m2.ensure_up = lambda: None
            p1 = Pool([("", m1)], poll=0.02, beat=0.05, state_dir=str(Path(d) / "s1"), on_wait=lambda m: None)
            said = []
            p2 = Pool([("", m2)], poll=0.02, beat=0.05, state_dir=str(Path(d) / "s2"), on_wait=said.append)
            order, second_in = [], threading.Event()

            def second():
                with p2.take(Path(d) / "m2-take"):
                    order.append("m2 in")
                    second_in.set()

            with p1.take(Path(d) / "m1-take"):
                order.append("m1 in")
                th = threading.Thread(target=second)
                th.start()
                for _ in range(100):
                    if said:
                        break
                    time.sleep(0.02)
                assert said and "1 take(s) of other machines ahead" in said[0] and "/m1-take is running" in said[0], said
                assert not second_in.is_set()                                          # the box is m1's until it leaves
                doc = status({"CLOUDGPU_BASE": adapter_box.base, "CLOUDGPU_TOKEN": ta.TOKEN, "CLOUDGPU_POWER": "none", "CLOUDGPU_STATE_DIR": str(Path(d) / "s3")})
                assert [(x["state"], x["label"].split("/", 1)[1]) for x in doc["line"]] == [("go", "m1-take"), ("waiting", "m2-take")], doc["line"]
                assert any("m1-take" in n and "running" in n for n in doc["next"]), doc["next"]
                assert json.loads(urllib.request.urlopen(adapter_box.base + "/health").read())["busy"] == 2
                time.sleep(0.3)                  # m1 keeps its lease alive while it runs
                assert not second_in.is_set()
            th.join(timeout=_t.patient(10))
            assert order == ["m1 in", "m2 in"] and second_in.is_set()
            assert adapter_box.host.line.count() == 0                                  # both gave up their places
            assert json.loads(urllib.request.urlopen(adapter_box.base + "/health").read())["busy"] == 0
        finally:
            adapter_box.close()


if __name__ == "__main__":
    _t.main(globals())
