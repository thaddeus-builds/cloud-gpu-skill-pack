"""cloudgpu.expiry and where it is used (instances list|status|expiry, status, ensure), offline against a fake vendor: a pass that ends in 1 day, in 3 days, in 4 days, one that has ended, one
that is switched off but runs on, one with no end time, and a box paid by the hour. No AutoDL, no money, no real clock (a fixed `now`, or times made from the real one)."""

import contextlib
import io
import json
import os
from datetime import datetime, timedelta, timezone

import _t
from test_box import autodl_api
from test_steward import FakeCloud, allow, stew

from cloudgpu import __main__ as cli
from cloudgpu import expiry, status, vendors
from cloudgpu.vendors import Instance

NOW = datetime(2026, 11, 1, 12, 0, 0, tzinfo=timezone(timedelta(hours=8))).timestamp()


def at(days=0.0, hours=0.0, now=NOW, tz=8):
    """The ISO end time of a pass that ends `days`/`hours` after `now`, written in the vendor's offset."""
    return datetime.fromtimestamp(now + days * 86400 + hours * 3600, timezone(timedelta(hours=tz))).isoformat()


def inst(iid, billing, ends, status="running", name=None):
    return Instance(id=iid, status=status, name=name, billing=billing, ends=ends)


def real_now_ends(days):
    return (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()


def test_a_pass_ending_in_less_than_the_warning_days_warns_and_a_later_one_does_not():
    r1 = expiry.row(inst("pro-1", "weekly", at(days=1), name="box B"), NOW)
    assert (r1["level"], r1["state"], r1["days_left"]) == ("warn", "expiring", 1.0)
    assert "pro-1" in r1["message"] and "box B" in r1["message"] and r1["ends"] in r1["message"] and "a person renews it" in r1["message"] and "console" in r1["message"]
    assert "person" in r1["who"] and "pay-as-you-go" in r1["who"]
    r3 = expiry.row(inst("pro-3", "weekly", at(days=3, hours=-1)), NOW)               # 2 days 23 hours: still inside the 3 days
    assert (r3["level"], r3["state"]) == ("warn", "expiring")
    assert expiry.row(inst("pro-3", "weekly", at(days=3)), NOW)["level"] == "ok"        # exactly 3 days: not yet "less than"
    r4 = expiry.row(inst("pro-4", "daily", at(days=4)), NOW)
    assert (r4["level"], r4["state"], r4["who"]) == ("ok", "active", None)
    assert expiry.row(inst("pro-4", "daily", at(days=4)), NOW, days=5)["level"] == "warn"      # the number is a parameter


def test_an_ended_pass_is_an_error_that_says_which_box_when_it_ended_and_who_renews_it():
    r = expiry.row(inst("pro-9", "weekly", at(hours=-5), status="shutdown", name="box B"), NOW, vendor="autodl")
    assert (r["level"], r["state"], r["hours_left"]) == ("error", "expired", -5.0)
    assert "pro-9" in r["message"] and "ENDED" in r["message"] and r["ends"] in r["message"] and "5 hours ago" in r["message"] and "the autodl console" in r["message"] and "renews" in r["message"]
    assert "person" in r["who"]


def test_a_box_that_is_off_with_its_pass_running_is_told_apart_from_one_that_ended():
    off = expiry.row(inst("pro-5", "monthly", at(days=20), status="shutdown"), NOW)
    assert (off["level"], off["state"], off["who"]) == ("info", "off", None) and "power it on" in off["message"] and "nothing to renew" in off["message"]
    ended = expiry.row(inst("pro-5", "monthly", at(days=-1), status="shutdown"), NOW)
    assert ended["state"] == "expired" and "power it on" not in ended["message"]
    soon_off = expiry.row(inst("pro-5", "monthly", at(days=1), status="shutdown"), NOW)       # off AND ending: both are said
    assert soon_off["state"] == "expiring" and "renews" in soon_off["message"] and "power it on" in soon_off["message"]


def test_a_box_paid_by_the_hour_has_no_row_and_a_pass_with_no_end_time_warns():
    assert expiry.row(inst("pro-p", "payg", None), NOW) is None
    assert expiry.row(inst("pro-q", None, None), NOW) is None and expiry.row(inst("pro-r", "hourly", None), NOW) is None
    n = expiry.row(inst("pro-n", "weekly", None), NOW)
    assert (n["level"], n["state"]) == ("warn", "no_end") and "no end time" in n["message"]
    assert expiry.row(inst("pro-n", "weekly", "not a time"), NOW)["state"] == "no_end"


def test_the_end_time_is_read_with_its_own_offset_not_the_machines():
    same = [expiry.hours_left(at(hours=30, tz=tz), NOW) for tz in (8, 0, -5)]
    assert same == [30.0, 30.0, 30.0], same                                            # one instant, three spellings
    z = datetime.fromtimestamp(NOW + 36000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert expiry.hours_left(z, NOW) == 10.0
    naive = datetime.fromtimestamp(NOW + 36000, timezone(timedelta(hours=8))).strftime("%Y-%m-%dT%H:%M:%S")      # no offset: the vendor's clock, UTC+8
    assert expiry.hours_left(naive, NOW) == 10.0


def test_a_pass_that_ended_a_minute_ago_has_ended_and_a_fraction_of_a_second_is_read():
    r = expiry.row(inst("pro-1", "weekly", at(hours=-0.03), status="shutdown"), NOW)
    assert r["state"] == "expired" and r["level"] == "error"
    for frac in (".1", ".12", ".1234", ".123456789"):
        assert abs(expiry.hours_left(at(hours=10).replace("+08:00", frac + "+08:00"), NOW) - 10) < 0.01, frac


def test_the_warning_days_are_a_setting_and_a_bad_value_is_the_default():
    assert expiry.warn_days({}) == 3.0 and expiry.warn_days({"CLOUDGPU_PASS_WARN_DAYS": "5"}) == 5.0 and expiry.warn_days({"H3HOST_PASS_WARN_DAYS": "1.5"}) == 1.5
    for bad in ("soon", "0", "-2", "nan", "inf", ""):
        assert expiry.warn_days({"CLOUDGPU_PASS_WARN_DAYS": bad}) == 3.0, bad


def test_rows_are_sorted_soonest_first_and_alerts_leave_out_what_is_well():
    rs = expiry.rows([inst("a", "weekly", at(days=10)), inst("b", "weekly", at(days=1)), inst("c", "payg", None), inst("d", "daily", at(hours=-2), status="shutdown"),
                      inst("e", "monthly", at(days=9), status="shutdown")], NOW)
    assert [r["id"] for r in rs] == ["d", "b", "e", "a"]
    assert [(r["id"], r["level"]) for r in expiry.alerts(rs)] == [("d", "error"), ("b", "warn"), ("e", "info")] and expiry.worst(rs) == "error"
    assert expiry.worst([]) == "ok" and [r["id"] for r in expiry.rows([inst("a", "weekly", at(days=1)), inst("b", "weekly", at(days=1))], NOW, only={"b"})] == ["b"]


def test_the_account_check_needs_the_token_and_lists_once():
    cloud = FakeCloud(instances=[inst("pro-1", "weekly", real_now_ends(1)), inst("pro-2", "payg", None)])
    assert [r["id"] for r in expiry.account_rows(cloud)] == ["pro-1"]
    cloud.env = {}
    with _t.raises("key_missing"):
        expiry.account_rows(cloud)


def run_cli(cloud, *args):
    reg = vendors.registry
    vendors.registry = lambda: {"fakecloud": lambda env=None: cloud}
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(["instances", "--vendor", "fakecloud", *args])
    finally:
        vendors.registry = reg
    return code, out.getvalue(), err.getvalue()


def test_instances_list_keeps_its_shape_adds_the_pass_and_warns_on_stderr():
    cloud = FakeCloud(instances=[inst("pro-1", "weekly", real_now_ends(1), name="box B"), inst("pro-2", "payg", None, status="shutdown"), inst("pro-3", "weekly", real_now_ends(-1), status="shutdown")])
    code, out, err = run_cli(cloud, "list")
    rows = json.loads(out)                                                             # stdout is still one JSON list of instances
    assert code == 0 and [r["id"] for r in rows] == ["pro-1", "pro-2", "pro-3"] and rows[0]["billing"] == "weekly"
    assert rows[0]["pass"]["state"] == "expiring" and "pass" not in rows[1] and rows[2]["pass"]["state"] == "expired"
    assert "WARN: pro-1" in err and "ERROR: pro-3" in err and "pro-2" not in err
    _, _, err2 = run_cli(cloud, "list", "--warn-days", "0.5")
    assert "pro-1" not in err2 and "ERROR: pro-3" in err2


def test_instances_status_carries_the_pass_of_that_instance_only():
    cloud = FakeCloud(instances=[inst("pro-1", "daily", real_now_ends(0.5)), inst("pro-2", "payg", None)])
    cloud.rows["pro-1"].status = "running"
    code, out, err = run_cli(cloud, "status", "pro-1")
    assert code == 0 and json.loads(out)["pass"]["state"] == "expiring" and "WARN: pro-1" in err
    assert "pass" not in json.loads(run_cli(cloud, "status", "pro-2")[1])


def test_instances_expiry_exits_0_1_2_and_lists_only_the_boxes_asked_for():
    well = FakeCloud(instances=[inst("pro-1", "weekly", real_now_ends(6)), inst("pro-2", "weekly", real_now_ends(0.2)), inst("pro-3", "payg", None)])
    code, out, _ = run_cli(well, "expiry", "pro-1")
    doc = json.loads(out)
    assert code == 0 and doc["level"] == "ok" and doc["alerts"] == [] and [p["id"] for p in doc["passes"]] == ["pro-1"] and doc["warn_days"] == 3.0
    code, out, _ = run_cli(well, "expiry")
    doc = json.loads(out)
    assert code == 1 and doc["level"] == "warn" and len(doc["alerts"]) == 1 and "pro-2" in doc["alerts"][0]
    code, out, _ = run_cli(well, "expiry", "pro-1", "pro-typo", "pro-3")                # a named id that is no pass is said, and is not "all well"
    doc = json.loads(out)
    assert code == 1 and doc["not_passes"] == ["pro-3", "pro-typo"] and doc["level"] == "warn" and any("pro-typo" in m for m in doc["alerts"])
    for bad in ("0", "-1", "nan"):
        code, out, _ = run_cli(well, "expiry", "--warn-days", bad)
        assert code == 1 and "above 0" in json.loads(out)["error"], bad
    well.rows["pro-2"].ends = real_now_ends(-0.1)
    assert run_cli(well, "expiry")[0] == 2
    assert run_cli(well, "expiry", "--warn-days", "9")[0] == 2 and json.loads(run_cli(well, "expiry", "--warn-days", "9")[1])["warn_days"] == 9.0


def test_the_status_names_the_pass_that_ends_soon_and_the_one_that_ended():
    def listing(billing, ends, st="running"):
        return {"list": [{"uuid": "pro-1", "status": st, "charge_type": billing, "expired_at": {"Time": ends, "Valid": True}}]}
    env = {"CLOUDGPU_INSTANCE": "pro-1", "AUTODL_DEV_TOKEN": "dev-secret-value", "CLOUDGPU_BOX": "rtx4090"}
    with autodl_api(lambda m, p, b, q: listing("weekly", real_now_ends(1))):
        doc = status.status(env)
    assert doc["account"]["expiry"]["state"] == "expiring" and any("pro-1" in n and "renews it" in n for n in doc["next"]), doc["next"]
    assert abs(doc["account"]["hours_left"] - 24) < 1 and "dev-secret-value" not in json.dumps(doc)
    with autodl_api(lambda m, p, b, q: listing("weekly", real_now_ends(-2), "shutdown")):
        doc = status.status(env)
    assert doc["account"]["expiry"]["state"] == "expired" and any("ENDED" in n and "pro-1" in n for n in doc["next"])
    with autodl_api(lambda m, p, b, q: listing("weekly", real_now_ends(30))):
        doc = status.status({**env, "CLOUDGPU_PASS_WARN_DAYS": "40"})                  # the setting is read here too
    assert doc["account"]["expiry"]["state"] == "expiring"
    with autodl_api(lambda m, p, b, q: listing("payg", "0001-01-01T00:00:00Z")):
        doc = status.status(env)
    assert "expiry" not in doc["account"] and not any("pass" in n for n in doc["next"])


def test_ensure_powers_on_a_pass_that_is_off_but_runs_and_leaves_an_ended_one_to_a_person():
    with _t.tmpdir() as d:
        allow(d, instances=0, cny=0)
        cloud = FakeCloud(instances=[inst("pro-run", "weekly", real_now_ends(1), name="box A"), inst("pro-off", "weekly", real_now_ends(5), status="shutdown", name="box B"),
                                     inst("pro-dead", "weekly", real_now_ends(-1), status="shutdown", name="box C"), inst("pro-hour", "payg", None, status="shutdown")])
        env = {"CLOUDGPU_INSTANCE": "pro-run", "CLOUDGPU_B_INSTANCE": "pro-off", "CLOUDGPU_C_INSTANCE": "pro-dead", "CLOUDGPU_D_INSTANCE": "pro-hour"}
        s = stew(d, cloud, env=env)
        dry = s.ensure(2, {}, specs=["v-48g"], centers=["westDC3"], yes=False)
        assert dry["would_power_on"] == ["pro-off"] and dry["missing"] == 0                  # the pass first: it costs nothing more; the hourly box is not needed
        r = s.ensure(2, {}, specs=["v-48g"], centers=["westDC3"], yes=True, boot_wait=0)
        assert r["running"] == ["pro-run"] and r["powered_on"] == ["pro-off"] and r["missing"] == 0
        assert ("power_on", "pro-dead") not in cloud.calls and ("power_on", "pro-hour") not in cloud.calls
        states = {p["id"]: p["state"] for p in r["passes"]}
        assert states == {"pro-run": "expiring", "pro-off": "off", "pro-dead": "expired"}, states      # the hourly box has no row
        assert any("pro-run" in n and "renews" in n for n in r["next"]) and any("pro-dead" in n and "ENDED" in n for n in r["next"]) and not any("pro-hour" in n for n in r["next"])


def test_ensure_needing_more_than_the_passes_give_goes_on_to_the_hourly_boxes():
    with _t.tmpdir() as d:
        allow(d, instances=0, cny=0)
        cloud = FakeCloud(instances=[inst("pro-off", "weekly", real_now_ends(5), status="shutdown"), inst("pro-hour", "payg", None, status="shutdown")])
        s = stew(d, cloud, env={"CLOUDGPU_INSTANCE": "pro-off", "CLOUDGPU_B_INSTANCE": "pro-hour"})
        r = s.ensure(2, {}, specs=["v-48g"], centers=["westDC3"], yes=True, boot_wait=0)
        assert r["powered_on"] == ["pro-off", "pro-hour"] and r["missing"] == 0


if __name__ == "__main__":
    _t.main(globals())
