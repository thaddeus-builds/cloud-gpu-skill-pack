"""The budget gate and `ensure --instance`, offline against a fake vendor: `--budget-cny` counts the hours the ledger shows run today (each set aside to its `--max-minutes` cap) and refuses a power-on or a create
that would pass it, saying the numbers; `ensure --instance` powers on a box nobody wired in the settings before it creates one; `power-on` prints one line of JSON an agent can read."""

import io
import json
import os
import time
from contextlib import redirect_stdout

import _t

from test_steward import TEMPLATE, FakeCloud, allow, env_of, stew

from cloudgpu import __main__ as cli
from cloudgpu import quota, vendors
from cloudgpu.errors import CloudGpuError
from cloudgpu.vendors import Instance

H = 3600.0
NOON = time.mktime((2026, 11, 3, 12, 0, 0, 0, 0, -1))


def row(action, instance, at, **f):
    return {"action": action, "instance": instance, "ts": at, **f}


def test_a_run_costs_its_hours_at_its_price_from_the_power_on_to_the_power_off():
    rows = [row("power-on", "a", NOON - 2 * H, price_cny_per_hour=3.0), row("power-off", "a", NOON - H)]
    assert abs(quota.run_spend_today(rows, NOON) - 3.0) < 1e-6
    assert abs(quota.run_spend_today([row("power-on", "a", NOON - 2 * H, price_cny_per_hour=3.0)], NOON) - 6.0) < 1e-6          # still on: until now
    midnight = quota.day_start(NOON)
    assert abs(quota.run_spend_today([row("power-on", "a", midnight - 5 * H, price_cny_per_hour=2.0)], NOON) - 2.0 * (NOON - midnight) / H) < 1e-6      # since midnight only


def test_a_run_with_a_deadline_is_set_aside_to_it_unless_the_guard_failed():
    on = row("power-on", "a", NOON - 3 * H, price_cny_per_hour=3.0)
    deadline = row("deadline", "a", NOON - 3 * H, until=NOON - 2 * H)
    assert abs(quota.run_spend_today([on, deadline], NOON) - 3.0) < 1e-6                 # taken to have ended at its deadline
    assert abs(quota.run_spend_today([on, deadline, row("guard-failed", "a", NOON - 2 * H)], NOON) - 9.0) < 1e-6      # still on: counted to now
    assert abs(quota.run_spend_today([on, deadline], NOON - 2.5 * H) - 3.0) < 1e-6       # a deadline not reached yet is set aside in full


def test_the_gate_adds_this_run_to_the_ledgers_hours_and_says_the_numbers():
    with _t.tmpdir() as d:
        s = stew(d, FakeCloud(), now=lambda: NOON)
        s.ledger.now = lambda: NOON - H
        s.ledger.append("power-on", vendor="fakecloud", instance="pro-y", price_cny_per_hour=3.0)          # 1 h on already: CNY 3.00
        s.ledger.now = lambda: NOON
        g = s.gate(3.0, 60, 6.5, "powering on pro-a")
        assert g["spent_today_cny"] == 3.0 and g["would_add_cny"] == 3.0 and g["budget_cny"] == 6.5
        with _t.raises("over_budget") as c:
            s.gate(3.0, 90, 6.5, "powering on pro-a")
        assert "3.00" in c.err.message and "4.5" in c.err.message and "6.5" in c.err.message
        with _t.raises("over_budget"):
            s.gate(3.0, None, 6.5, "x")                       # no --max-minutes: the gate cannot know how long
        with _t.raises("over_budget"):
            s.gate(None, 30, 6.5, "x")                        # no price known
        assert s.gate(None, None, None, "x") == {}            # no budget asked: nothing to check


def cli_run(cloud, d, argv):
    reg, os.environ["CLOUDGPU_STATE_DIR"] = vendors.registry, d
    vendors.registry = lambda: {"fakecloud": lambda env=None: cloud}
    real = cli._spawn_guard
    cli._spawn_guard = lambda v, i, m, agent=False: {"pid": 4242, "off_at": f"in {m:g} min", "log": "x"}
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cli.main(["instances", "--vendor", "fakecloud", *argv])
        return rc, [json.loads(x) for x in buf.getvalue().strip().splitlines()]
    finally:
        cli._spawn_guard, vendors.registry = real, reg
        os.environ.pop("CLOUDGPU_STATE_DIR", None)


def test_a_power_on_prints_one_line_with_the_price_and_the_stop_and_the_gate_counts_it_next_time():
    with _t.tmpdir() as d:
        cloud = FakeCloud(instances=[Instance(id="pro-a", status="shutdown", billing="payg", gpu_spec="v-48g")])
        rc, lines = cli_run(cloud, d, ["power-on", "--yes", "--max-minutes", "21", "--budget-cny", "5", "pro-a"])
        o = lines[0]
        assert rc == 0 and len(lines) == 1 and o["done"] == "power-on" and o["id"] == "pro-a" and o["price_cny_per_hour"] == 3.0 and o["max_minutes"] == 21
        assert o["started_at"] and o["auto_off_at"] and o["budget_cny"] == 5 and o["spent_today_cny"] == 0 and o["guard"]["pid"] == 4242
        led = quota.Ledger(quota.ledger_path(env_of(d))).rows()
        assert [x["action"] for x in led] == ["power-on", "deadline"] and led[0]["price_cny_per_hour"] == 3.0
        cloud.rows["pro-a"].status = "shutdown"
        rc, lines = cli_run(cloud, d, ["power-on", "--yes", "--max-minutes", "120", "--budget-cny", "5", "pro-a"])          # 21 min set aside (1.05) + 6.00 is over 5
        assert rc == 1 and lines[0]["kind"] == "over_budget" and cloud.rows["pro-a"].status == "shutdown"
        rc, lines = cli_run(cloud, d, ["power-on", "--yes", "--budget-cny", "5", "pro-a"])                                   # a budget with no cap: refused before any call
        assert rc == 2 and "--max-minutes" in lines[0]["error"]


def test_a_pass_is_neither_gated_nor_capped():
    with _t.tmpdir() as d:
        cloud = FakeCloud(instances=[Instance(id="pro-p", status="shutdown", billing="daily")])
        rc, lines = cli_run(cloud, d, ["power-on", "--yes", "--max-minutes", "10", "--budget-cny", "0.01", "pro-p"])
        assert rc == 0 and "pass" in lines[0]["guard"] and "auto_off_at" not in lines[0]


def test_ensure_powers_on_a_named_instance_nobody_wired_before_it_creates():
    with _t.tmpdir() as d:
        allow(d, instances=3, cny=500)
        cloud = FakeCloud(instances=[Instance(id="pro-793376", status="shutdown", billing="payg", gpu_spec="pro6000-p"), Instance(id="pro-other", status="shutdown", billing="payg")],
                          stock={("pro6000-p", "westDC3"): 1})
        cloud.power_errors["pro-793376"] = CloudGpuError("http", "当前算力规格暂无库存 库存不足")          # the sample of 2026-11-03: a free card in the zone, still no power-on
        r = stew(d, cloud).ensure(1, TEMPLATE, specs=["pro6000-p"], centers=["westDC3"], yes=True, instances=["pro-793376"], max_minutes=21, budget=5, boot_wait=60)
        assert ("power_on", "pro-793376") in cloud.calls and ("power_on", "pro-other") not in cloud.calls          # tried first; the other is nobody's
        assert [c["created"] for c in r["created"]] == ["pro-new1"] and r["missing"] == 0 and r["runs"]["pro-new1"]["budget_cny"] == 5
        cloud2 = FakeCloud(instances=[Instance(id="pro-793376", status="shutdown", billing="payg", gpu_spec="pro6000-p")], stock={("pro6000-p", "westDC3"): 1})
        r2 = stew(d, cloud2).ensure(1, TEMPLATE, specs=["pro6000-p"], centers=["westDC3"], yes=True, instances=["pro-793376"], boot_wait=60)      # able to open: reused, no card spent
        assert r2["powered_on"] == ["pro-793376"] and r2["created"] == [] and r2["not_in_account"] == []
        r3 = stew(d, FakeCloud()).ensure(1, TEMPLATE, specs=["v-48g"], centers=["westDC3"], yes=False, instances=["pro-gone"])
        assert r3["not_in_account"] == ["pro-gone"] and any("pro-gone" in n for n in r3["next"])


def test_ensure_stops_at_the_budget_and_says_so():
    with _t.tmpdir() as d:
        allow(d, instances=3, cny=500, prices={"v-48g": 3.0})
        cloud = FakeCloud(instances=[Instance(id="pro-a", status="shutdown", billing="payg", gpu_spec="v-48g")], stock={("v-48g", "westDC3"): 5})
        r = stew(d, cloud).ensure(2, TEMPLATE, specs=["v-48g"], centers=["westDC3"], yes=True, instances=["pro-a"], max_minutes=60, budget=4.0, patience=0, boot_wait=0)
        assert r["powered_on"] == ["pro-a"] and r["created"] == [] and r["missing"] == 1               # 3.00 set aside for the first hour; a second would be 6.00 > 4.00
        assert any(f["kind"] == "over_budget" for f in r["failures"]) and any("budget" in n for n in r["next"])
        assert not any(c[0] == "create" for c in cloud.calls)


def test_the_cli_ensure_names_instances_and_caps_a_wired_box_without_the_agent_flag():
    with _t.tmpdir() as d:
        allow(d, instances=2, cny=500)
        cloud = FakeCloud(instances=[Instance(id="pro-793376", status="shutdown", billing="payg", gpu_spec="v-48g")])
        spec = os.path.join(d, "t.json")
        open(spec, "w").write(json.dumps(TEMPLATE))
        seen = []
        reg, os.environ["CLOUDGPU_STATE_DIR"] = vendors.registry, d
        vendors.registry = lambda: {"fakecloud": lambda env=None: cloud}
        real = cli._spawn_guard
        cli._spawn_guard = lambda v, i, m, agent=False: seen.append((i, m, agent)) or {"pid": 1, "off_at": "t", "log": "x"}
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cli.main(["instances", "--vendor", "fakecloud", "ensure", "--count", "1", "--spec", spec, "--specs", "v-48g", "--data-centers", "westDC3",
                               "--instance", "pro-793376", "--max-minutes", "21", "--budget-cny", "5", "--patience", "0", "--yes"])
            o = json.loads(buf.getvalue())
            assert rc == 0 and o["powered_on"] == ["pro-793376"] and seen == [("pro-793376", 21.0, False)] and o["runs"]["pro-793376"]["auto_off_at"] == "t"       # not the agent's own box: the guard is not asked to refuse it
        finally:
            cli._spawn_guard, vendors.registry = real, reg
            os.environ.pop("CLOUDGPU_STATE_DIR", None)


def test_the_dry_run_shows_the_three_numbers_or_the_refusal_before_yes():
    with _t.tmpdir() as d:
        cloud = FakeCloud(instances=[Instance(id="pro-a", status="shutdown", billing="payg", gpu_spec="v-48g")])
        rc, lines = cli_run(cloud, d, ["power-on", "--max-minutes", "21", "--budget-cny", "5", "pro-a"])
        o = lines[0]
        assert rc == 0 and o["dry_run"] is True and o["price_cny_per_hour"] == 3.0 and o["spent_today_cny"] == 0 and o["would_add_cny"] == 1.05 and o["budget_cny"] == 5 and "would_refuse" not in o
        assert not any(c[0] == "power_on" for c in cloud.calls)
        os.environ["CLOUDGPU_STATE_DIR"] = d
        try:
            with redirect_stdout(io.StringIO()):
                assert cli.main(["ledger", "add", "--cny", "4.65", "--by", "yongshen", "--note", "outside the package"]) == 0
        finally:
            os.environ.pop("CLOUDGPU_STATE_DIR", None)
        rc, lines = cli_run(cloud, d, ["power-on", "--max-minutes", "21", "--budget-cny", "5", "pro-a"])
        assert rc == 0 and "4.65" in lines[0]["would_refuse"] and "5" in lines[0]["would_refuse"] and "refuse" in lines[0]["next"][0]


def test_ledger_add_counts_money_spent_outside_the_package_and_only_today():
    now = NOON
    rows = [row("spend", None, now - 2 * H, cny=4.65), row("spend", None, quota.day_start(now) - H, cny=100.0)]
    assert abs(quota.run_spend_today(rows, now) - 4.65) < 1e-6                     # yesterday's does not count
    with _t.tmpdir() as d:
        os.environ["CLOUDGPU_STATE_DIR"] = d
        try:
            for argv, want in ((["ledger", "add", "--cny", "0", "--by", "x"], 2), (["ledger", "add", "--cny", "nan", "--by", "x"], 2), (["ledger", "add", "--cny", "2.5", "--by", "x"], 0)):
                with redirect_stdout(io.StringIO()):
                    assert cli.main(argv) == want
            buf = io.StringIO()
            with redirect_stdout(buf):
                assert cli.main(["ledger", "today"]) == 0
            o = json.loads(buf.getvalue())
            assert o["spent_today_cny"] == 2.5 and o["outside_the_package"][0]["actor"] == "x"
        finally:
            os.environ.pop("CLOUDGPU_STATE_DIR", None)


def test_the_ensure_dry_run_says_the_cap_the_budget_and_where_it_would_create_if_the_power_on_is_refused():
    with _t.tmpdir() as d:
        allow(d, instances=3, cny=500)
        cloud = FakeCloud(instances=[Instance(id="pro-a", status="shutdown", billing="payg", gpu_spec="v-48g")])
        r = stew(d, cloud).ensure(1, TEMPLATE, specs=["v-48g"], centers=["westDC3", "beijingDC2"], yes=False, instances=["pro-a"], max_minutes=21, budget=5)
        assert r["would_power_on"] == ["pro-a"] and "21" in r["would_guard"] and r["budget_checks"]["pro-a"]["would_add_cny"] == 1.05
        assert [o["data_center"] for o in r["order"]] == ["westDC3", "beijingDC2"] and r["if_a_power_on_is_refused_for_no_stock"] and not any(c[0] in ("power_on", "create") for c in cloud.calls)
        r2 = stew(d, cloud).ensure(1, TEMPLATE, specs=["v-48g"], centers=["westDC3"], yes=False, instances=["pro-a"], max_minutes=21, budget=0.5)
        assert "would_refuse" in r2["budget_checks"]["pro-a"]
        assert "budget gate would refuse pro-a" in r2["next"][0] and "pass --yes" not in r2["next"][0]
        assert r["next"][0] == "pass --yes to do it"


if __name__ == "__main__":
    _t.main(globals())
