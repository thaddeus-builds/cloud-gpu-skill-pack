"""cloudgpu.util: the utilisation report. The statistics are a pure function of task records (made-up ones here, numbers worked out by hand); the adapter lists its records
(`/cloudgpu/tasks`, token-checked, no ids) and `openmax-cloudgpu util` reads them through the real client. Read only throughout."""

import io
import json
import os
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import _t
import test_adapter as ta

from cloudgpu import util
from cloudgpu.__main__ import main
from cloudgpu.box import Box as RealBox

B = 1000.0


def rec(c, s, d, status, wf="w"):
    return {"workflow": wf, "status": status, "created": B + c, "started": None if s is None else B + s, "done": None if d is None else B + d}


RECS = [rec(0, 0, 40, "SUCCESS"), rec(100, 100, 160, "SUCCESS"), rec(250, 250, 330, "SUCCESS"), rec(300, 330, 340, "FAILED"),
        rec(0, 0, 100, "SUCCESS", "v"), rec(50, 50, 150, "SUCCESS", "v"), rec(900, None, None, "QUEUED", "v"),
        rec(0, 0, 100, "SUCCESS", "x"), rec(50, 50, 150, "SUCCESS", "x"), rec(200, 200, 250, "SUCCESS", "x")]


def test_the_report_counts_times_gaps_the_busy_share_and_the_recent_takes_per_workflow():
    rows = {r["workflow"]: r for r in util.report(RECS, now=B + 1900)}
    w = rows["w"]
    assert w["takes"] == 4 and w["failed"] == 1
    assert w["created_to_done_s"] == {"median": 60.0, "slowest": 80.0}, "the three that went well: 40, 60, 80 s (the failed one is not a time to expect)"
    assert w["gap_s"] == {"median": 100.0, "mean": 100.0}, "creations at 0, 100, 250, 300: gaps 100, 150, 50"
    assert w["busy_share"] == round(190 / 340, 3), "40 + 60 + 80 + 10 s of a 340 s span"
    assert w["last_30_min"] == 3, "now is 1900 s: the last 30 minutes reach back to 100, so the take made at 0 is out"
    v = rows["v"]
    assert v["takes"] == 3 and v["created_to_done_s"] == {"median": 100.0, "slowest": 100.0} and v["busy_share"] == 1.0, "two takes overlapping 50 s are busy 150 s of 150 s"
    assert rows["x"]["busy_share"] == 0.8, "0-100 and 50-150 overlap, then 200-250: 200 s busy of 250, not the 250 s the three takes add up to"
    assert v["last_30_min"] == 1, "only the queued one, made at 900, is in the last 30 minutes of 1900"


def test_a_take_still_running_counts_to_now_one_take_has_no_gap_and_since_cuts_the_records():
    one = util.report([rec(0, 10, None, "RUNNING")], now=B + 100)[0]
    assert one["takes"] == 1 and one["gap_s"] is None and one["created_to_done_s"] is None and one["busy_share"] == 0.9, one
    assert util.report([], now=B) == []
    late = {r["workflow"]: r for r in util.report(RECS, now=B + 1900, since=B + 250)}
    assert late["w"]["takes"] == 2 and late["w"]["gap_s"]["median"] == 50.0 and "v" in late and late["v"]["takes"] == 1
    assert "failed" in util.line_of("a", "w", late["w"]) and "busy" in util.line_of("a", "w", late["w"])


def test_since_is_a_number_an_age_or_a_local_date():
    now = 2_000_000.0
    assert util.parse_since(None, now) == 0.0 and util.parse_since("1234.5", now) == 1234.5
    assert util.parse_since("90m", now) == now - 5400 and util.parse_since("2h", now) == now - 7200 and util.parse_since("1d", now) == now - 86400
    assert util.parse_since("2026-10-29T14:30", now) == time.mktime((2026, 10, 29, 14, 30, 0, 0, 0, -1)) and util.parse_since("2026-10-29", now) < util.parse_since("2026-10-29 00:01", now)
    with _t.raises("key_missing"):
        util.parse_since("yesterday-ish", now)


def machine(box):
    b = RealBox({"CLOUDGPU_BASE": box.base, "CLOUDGPU_TOKEN": ta.TOKEN, "CLOUDGPU_POWER": "none"})
    return b


def test_the_adapter_lists_its_records_with_the_token_only_and_no_task_id_and_the_client_reads_them():
    with _t.tmpdir() as d:
        b = ta.Box(d)
        try:
            for _ in range(2):
                assert b.wait(b.call("POST", f"{ta.API}/minimax_h3_lightx2v", ta.FL)[1]["data"]["task_id"])["status"] == "SUCCESS"
            assert b.call("GET", "/cloudgpu/tasks", token=None)[0] == 401 and b.call("GET", "/cloudgpu/tasks", token="wrong")[0] == 401
            assert b.call("GET", "/cloudgpu/tasks?since=abc")[0] == 400
            code, doc = b.call("GET", "/cloudgpu/tasks")
            tasks = doc["data"]["tasks"]
            assert code == 200 and len(tasks) == 2 and all(set(t) == {"workflow", "status", "created", "started", "done"} for t in tasks), tasks
            assert all(t["workflow"] == "minimax_h3_lightx2v" and t["status"] == "SUCCESS" and t["done"] >= t["started"] >= t["created"] for t in tasks)
            assert b.call("GET", f"/cloudgpu/tasks?since={time.time() + 60}")[1]["data"]["tasks"] == []
            got = util.fetch(machine(b), 0.0)
            assert len(got["tasks"]) == 2 and abs(got["now"] - time.time()) < 60 and got["ttl_s"] == 86400
            rows, lines = util.run([("", machine(b))], None)
            assert len(rows) == 1 and rows[0]["box"] == "default" and rows[0]["takes"] == 2 and rows[0]["failed"] == 0 and len(lines) == 1 and lines[0].startswith("default minimax_h3_lightx2v: 2 takes"), (rows, lines)
            names, _ = util.run([("a", machine(b)), ("b", machine(b))], None, ["b"])
            assert [r["box"] for r in names] == ["b"]
            with _t.raises("key_missing"):
                util.run([("a", machine(b))], None, ["zzz"])
            assert not list((b.root / "input").iterdir()) and len(list((Path(d) / "state" / "tasks").glob("*.json"))) == 2, "it only read"
        finally:
            b.close()


def test_a_box_that_does_not_answer_is_a_row_with_an_error_and_an_old_adapter_says_to_install_again():
    dead = RealBox({"CLOUDGPU_BASE": "http://127.0.0.1:9", "CLOUDGPU_TOKEN": ta.TOKEN, "CLOUDGPU_POWER": "none"})
    rows, lines = util.run([("", dead)], None)
    assert rows[0]["box"] == "default" and "does not answer" in rows[0]["error"] and "nothing was powered on" in lines[0], rows
    old = RealBox({"CLOUDGPU_BASE": "http://127.0.0.1:9", "CLOUDGPU_TOKEN": ta.TOKEN, "CLOUDGPU_POWER": "none"})
    with mock.patch.object(util, "fetch", return_value=None), mock.patch.object(RealBox, "health", return_value={"ok": True}):
        rows, _ = util.run([("", old)], None)
    assert "older than 0.12.0" in rows[0]["error"], rows


def test_the_verb_prints_json_on_stdout_one_line_per_box_and_workflow_on_stderr_and_reads_only():
    with _t.tmpdir() as d:
        b = ta.Box(d)
        try:
            assert b.wait(b.call("POST", f"{ta.API}/minimax_h3_lightx2v", ta.FL)[1]["data"]["task_id"])["status"] == "SUCCESS"
            env = {"CLOUDGPU_BASE": b.base, "CLOUDGPU_TOKEN": ta.TOKEN, "CLOUDGPU_POWER": "none", "CLOUDGPU_STATE_DIR": str(Path(d) / "s")}
            out, err = io.StringIO(), io.StringIO()
            with mock.patch.dict(os.environ, env, clear=True), redirect_stdout(out), redirect_stderr(err):
                assert main(["util", "--since", "1h"]) == 0
            doc = json.loads(out.getvalue())
            assert doc["since"] == "1h" and doc["util"][0]["takes"] == 1 and doc["util"][0]["busy_share"] is not None, doc
            assert err.getvalue().count("\n") == 1 and "minimax_h3_lightx2v: 1 takes" in err.getvalue(), err.getvalue()
            out = io.StringIO()
            with mock.patch.dict(os.environ, env, clear=True), redirect_stdout(out):
                assert main(["util", "nope"]) == 2
            assert json.loads(out.getvalue())["kind"] == "key_missing"
        finally:
            b.close()


if __name__ == "__main__":
    _t.main(globals())
