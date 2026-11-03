"""cloudgpu.guard and `instances guard` / `--max-minutes`, offline: the box is shut down at the deadline by the wall clock and the vendor's own status is read until it says off; a refused or slow shutdown is asked
again; a bad cap is refused; power-on and ensure start one guard per box they switch on, and nothing before --yes."""

import io
import json
import os
from contextlib import redirect_stdout

import _t

from cloudgpu import __main__ as cli
from cloudgpu import guard, vendors
from cloudgpu.errors import CloudGpuError
from cloudgpu.vendors import Instance, Vendor


class Clock:
    def __init__(self):
        self.t = 1000.0
        self.slept = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


class Box(Vendor):
    name = "fakebox"

    def __init__(self, off_after=1, refuse=0):
        super().__init__({})
        self.status_now, self.calls, self.asks, self.off_after, self.refuse = "running", [], 0, off_after, refuse

    def status(self, i):
        return self.status_now

    def power_off(self, i):
        self.calls.append(("power_off", i))
        self.asks += 1
        if self.asks <= self.refuse:
            raise CloudGpuError("http", "ServerBusy", status="ServerBusy")
        if self.asks >= self.off_after:
            self.status_now = "shutdown"


def test_the_box_is_shut_down_at_the_deadline_by_the_wall_clock_and_confirmed_by_its_status():
    c, b, noted = Clock(), Box(), []
    r = guard.run(b, "pro-1", 2, now=c.now, sleep=c.sleep, on_off=lambda: noted.append(1))
    assert c.t >= 1000 + 120 and b.calls == [("power_off", "pro-1")]                 # nothing was asked before the 2 minutes were up
    assert all(s <= guard.TICK_S for s in c.slept)                                  # the clock is looked at again at least every tick
    assert r["off"] is True and r["status"] == "shutdown" and r["asked"] == 1 and noted == [1]


def test_a_box_that_is_already_off_is_not_asked_and_a_refused_shutdown_is_asked_again():
    c, b = Clock(), Box()
    b.status_now = "shutdown"
    r = guard.run(b, "pro-1", 1, now=c.now, sleep=c.sleep)
    assert r["off"] is True and b.calls == [] and r["asked"] == 0
    c, b = Clock(), Box(refuse=1)                                                    # the first request is refused (ServerBusy): the guard keeps asking
    r = guard.run(b, "pro-1", 1, now=c.now, sleep=c.sleep, poll_s=10, confirm_s=120)
    assert r["off"] is True and r["asked"] >= 2 and any("ServerBusy" in e for e in r["errors"])


def test_a_box_that_never_goes_off_is_reported_not_off():
    c, b = Clock(), Box(off_after=99)
    r = guard.run(b, "pro-1", 1, now=c.now, sleep=c.sleep, poll_s=10, confirm_s=60)
    assert r["off"] is False and r["status"] == "running" and 1 <= r["asked"] <= 4


def test_a_bad_cap_is_refused():
    for bad in (0, -1, float("nan"), float("inf"), 24 * 60 + 1, "x", None):
        with _t.raises("unknown"):
            guard.minutes_ok(bad)
    assert guard.minutes_ok("21") == 21.0 and guard.minutes_ok(24 * 60) == 24 * 60


def _run(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(["instances", "--vendor", "fakebox", *argv])
    return rc, json.loads(buf.getvalue())


def test_power_on_starts_one_guard_only_with_yes_and_a_bad_cap_changes_nothing():
    box, spawned = Box(), []
    reg = vendors.registry
    vendors.registry = lambda: {**reg(), "fakebox": lambda env=None: box}
    box.configured = lambda: True
    box.power_on = lambda i, start_command=None: box.calls.append(("power_on", i))
    real = cli._spawn_guard
    cli._spawn_guard = lambda v, i, m, agent: spawned.append((i, m, agent)) or {"pid": 1, "off_at": "x", "log": "y"}
    with _t.tmpdir() as d:
        os.environ["CLOUDGPU_STATE_DIR"] = d
        try:
            rc, doc = _run(["power-on", "pro-1", "--max-minutes", "21"])
            assert rc == 0 and doc["dry_run"] is True and "21" in doc["would_guard"] and spawned == [] and box.calls == []
            rc, doc = _run(["power-on", "pro-1", "--max-minutes", "0", "--yes"])
            assert rc == 1 and doc["kind"] == "unknown" and box.calls == [] and spawned == []
            rc, doc = _run(["power-on", "pro-1", "--max-minutes", "21", "--yes"])
            assert rc == 0 and doc["done"] == "power-on" and doc["guard"]["pid"] == 1 and spawned == [("pro-1", 21.0, False)] and box.calls == [("power_on", "pro-1")]
            rc, doc = _run(["power-on", "pro-2", "--yes"])                                   # no cap asked: no guard, as before
            assert rc == 0 and "guard" not in doc and len(spawned) == 1
        finally:
            cli._spawn_guard = real
            vendors.registry = reg
            os.environ.pop("CLOUDGPU_STATE_DIR", None)


def test_the_guard_verb_shuts_the_box_and_an_agent_may_only_guard_its_own():
    box = Box()
    reg = vendors.registry
    vendors.registry = lambda: {**reg(), "fakebox": lambda env=None: box}
    box.configured = lambda: True
    real = guard.run
    guard.run = lambda v, i, m, on_off=None, **k: {"instance": i, "minutes": m, "off": True, "status": "shutdown", "asked": 1, "errors": []}
    with _t.tmpdir() as d:
        os.environ["CLOUDGPU_STATE_DIR"] = d
        try:
            rc, doc = _run(["guard", "pro-1", "--minutes", "5"])
            assert rc == 0 and doc["off"] is True
            rc, doc = _run(["guard", "pro-1", "--minutes", "5", "--agent"])
            assert rc == 1 and doc["kind"] == "not_allowed"
        finally:
            guard.run = real
            vendors.registry = reg
            os.environ.pop("CLOUDGPU_STATE_DIR", None)


def test_the_background_guard_is_a_detached_process_with_its_own_log_and_no_token_on_the_command_line():
    import subprocess
    seen = {}

    class P:
        pid = 4242
    real = subprocess.Popen
    subprocess.Popen = lambda cmd, **kw: seen.update(cmd=cmd, kw=kw) or P()
    with _t.tmpdir() as d:
        os.environ["CLOUDGPU_STATE_DIR"] = d
        try:
            r = cli._spawn_guard(Box(), "pro-9", 21, True)
        finally:
            subprocess.Popen = real
            os.environ.pop("CLOUDGPU_STATE_DIR", None)
        assert r["pid"] == 4242 and r["log"] == os.path.join(d, "guard-pro-9.log") and os.path.exists(r["log"])
    assert seen["cmd"][1:] == ["-m", "cloudgpu", "instances", "--vendor", "fakebox", "guard", "pro-9", "--minutes", "21", "--agent"]
    assert seen["kw"]["start_new_session"] is True and seen["kw"]["stdin"] == subprocess.DEVNULL
    assert not any("TOKEN" in x.upper() for x in seen["cmd"])


if __name__ == "__main__":
    _t.main(globals())
