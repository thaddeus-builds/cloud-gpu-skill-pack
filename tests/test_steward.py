"""cloudgpu.quota and cloudgpu.steward, offline against a fake vendor (no AutoDL, no box, no money): the allowance is 0 until a person sets it (who and when are recorded); the agent creates only
inside it and releases only what the ledger says it made; every create, release and power-on is a ledger line; `ensure` powers on, then creates by spec and data center in order, tries again
when a card is not in stock, tells that apart from every other failure, and reports how many are up, how many are missing and why."""

import io
import json
import os
import time
from contextlib import redirect_stdout

import _t

from cloudgpu import __main__ as cli
from cloudgpu import power, quota, status
from cloudgpu.errors import CloudGpuError
from cloudgpu.steward import Steward, pool_instances
from cloudgpu.vendors import Instance, Vendor
from cloudgpu.vendors import autodl as va

TEMPLATE = {"req_gpu_amount": 1, "expand_system_disk_by_gb": 0, "application_uuid": "app1", "application_version": "latest"}


class FakeCloud(Vendor):
    """An account: `stock[(spec, dc)]` cards left; `errors[(spec, dc)]` an error to raise instead; `power_errors[id]`."""
    name, token_env = "fakecloud", "FAKE_TOKEN"

    def __init__(self, env=None, instances=None, stock=None):
        super().__init__(env or {"FAKE_TOKEN": "t"})
        self.rows = {i.id: i for i in (instances or [])}
        self.stock, self.errors, self.power_errors, self.calls, self.n = stock or {}, {}, {}, [], 0
        self.boot_after = 0                                          # status polls before a new instance says running

    def list(self, *, page=1, size=50):
        return list(self.rows.values()), 1

    def status(self, i):
        self.calls.append(("status", i))
        r = self.rows[i]
        if r.status == "booting":
            self.boot_after -= 1
            if self.boot_after <= 0:
                r.status = "running"
        return r.status

    def snapshot(self, i):
        return {"price_cny_per_hour": 3.0, "ssh": {"host": "h.example", "port": 2222}}

    def create_problems(self, spec):
        return []

    def create(self, spec):
        key = (spec["gpu_spec_uuid"], (spec.get("data_center_list") or [None])[0])
        self.calls.append(("create", key))
        if key in self.errors:
            raise self.errors[key]
        if self.stock.get(key, 0) <= 0:
            raise CloudGpuError("http", "autodl api create: InstanceStockNotEnough 库存不足", status="InstanceStockNotEnough", detail="库存不足")
        self.stock[key] -= 1
        self.n += 1
        iid = f"pro-new{self.n}"
        self.rows[iid] = Instance(id=iid, status="booting", billing="payg", gpu_spec=key[0])
        return iid

    def power_on(self, i, start_command=None):
        self.calls.append(("power_on", i))
        if i in self.power_errors:
            raise self.power_errors[i]
        self.rows[i].status = "running"

    def release(self, i):
        self.calls.append(("release", i))
        del self.rows[i]

    def is_no_stock(self, err):
        return "库存" in err.message or "StockNotEnough" in err.message


def allow(d, instances=2, cny=100, prices=None):
    p = quota.Quota(max_instances=instances, max_cny_per_day=cny, prices=prices or {"v-48g": 3.0, "pro6000-p": 7.0}, set_by="boyuan", set_at="now")
    os.environ["CLOUDGPU_STATE_DIR"] = d
    quota.save(p, {"CLOUDGPU_STATE_DIR": d})


def env_of(d, **more):
    return {"CLOUDGPU_STATE_DIR": d, **more}


def stew(d, cloud, agent=True, **kw):
    return Steward(cloud, env_of(d, **kw.pop("env", {})), agent=agent, sleep=kw.pop("sleep", lambda s: None), **kw)


def spec(s="v-48g", dc="westDC3"):
    return {**TEMPLATE, "gpu_spec_uuid": s, "data_center_list": [dc]}


def test_the_allowance_is_zero_by_default_and_the_agent_creates_nothing_then():
    with _t.tmpdir() as d:
        cloud = FakeCloud(stock={("v-48g", "westDC3"): 1})
        with _t.raises("over_quota"):
            stew(d, cloud).create(spec())
        assert cloud.calls == [] and quota.Ledger(quota.ledger_path(env_of(d))).rows() == []         # nothing was called, nothing to record
        q = quota.load(env_of(d))
        assert q.allows_create is False and q.max_instances == 0
        # a person's own create (no --agent) is as before, and is in the ledger as the person's
        r = stew(d, cloud, agent=False).create(spec())
        rows = quota.Ledger(quota.ledger_path(env_of(d))).rows()
        assert r["created"] == "pro-new1" and [x["action"] for x in rows] == ["create-attempt", "create"] and rows[1]["actor"] == "person" and rows[1]["agent"] is False


def test_the_quota_records_who_set_it_and_when_and_a_file_others_can_write_is_not_believed():
    with _t.tmpdir() as d:
        os.environ["CLOUDGPU_STATE_DIR"] = d
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cli.main(["quota", "set", "--by", "boyuan", "--max-instances", "2", "--max-cny-per-day", "80", "--price", "v-48g=3.2", "--note", "pilot"])
            q = quota.load(env_of(d))
            assert rc == 0 and (q.max_instances, q.max_cny_per_day, q.prices, q.set_by) == (2, 80.0, {"v-48g": 3.2}, "boyuan") and q.set_at and q.allows_create
            led = quota.Ledger(quota.ledger_path(env_of(d))).rows()
            assert led[-1]["action"] == "quota-set" and led[-1]["actor"] == "boyuan"
            os.chmod(quota.quota_path(env_of(d)), 0o666)
            q = quota.load(env_of(d))
            assert q.allows_create is False and "written by group or others" in q.problem
            with redirect_stdout(io.StringIO()):
                assert cli.main(["quota", "set", "--by", "x", "--max-instances", "9", "--max-cny-per-day", "9"]) == 2         # an agent cannot quietly replace it either
            assert quota.load(env_of(d)).max_instances == 0
        finally:
            os.environ.pop("CLOUDGPU_STATE_DIR", None)


def test_inside_the_allowance_the_agent_creates_and_the_ledger_has_spec_price_and_who():
    with _t.tmpdir() as d:
        allow(d, instances=2, cny=100)
        cloud = FakeCloud(stock={("v-48g", "westDC3"): 5})
        s = stew(d, cloud, env={"CLOUDGPU_ACTOR": "emma-agent"})
        a, b = s.create(spec()), s.create(spec())
        assert a["created"] == "pro-new1" and a["price_cny_per_hour"] == 3.0 and a["price_source"] == "snapshot"
        rows = [r for r in s.ledger.rows() if r["action"] == "create"]
        assert [(r["instance"], r["spec"], r["data_center"], r["price_cny_per_hour"], r["actor"], r["agent"]) for r in rows] == [
            ("pro-new1", "v-48g", "westDC3", 3.0, "emma-agent", True), ("pro-new2", "v-48g", "westDC3", 3.0, "emma-agent", True)]
        with _t.raises("over_quota") as c:
            s.create(spec())                                                                           # the third: two at once is the allowance
        assert "2 instance" in c.err.message and cloud.n == 2
        quota.save(quota.Quota(max_instances=5, max_cny_per_day=5, prices={"v-48g": 3.0}, set_by="p"), env_of(d))
        with _t.raises("over_quota") as c:
            stew(d, cloud).create(spec())                                                              # 2 h of one more (CNY 6) set aside is over the day's 5
        assert "day's allowance" in c.err.message
        with _t.raises("over_quota") as c:
            stew(d, cloud).create(spec("pro6000-p"))                                                   # no price for this card: what it spends is not known
        assert "no price" in c.err.message


def test_the_days_spend_is_an_upper_bound_from_the_ledger_and_yesterday_does_not_count():
    now = time.mktime((2026, 10, 31, 12, 0, 0, 0, 0, -1))
    rows = [{"action": "create", "instance": "a", "ts": now - 3 * 3600, "price_cny_per_hour": 2.0},
            {"action": "create", "instance": "b", "ts": now - 40 * 3600, "price_cny_per_hour": 1.0},           # made yesterday, still alive: counts from midnight (12 h)
            {"action": "create", "instance": "c", "ts": now - 5 * 3600, "price_cny_per_hour": 4.0},
            {"action": "release", "instance": "c", "ts": now - 4 * 3600}]
    assert round(quota.spent_today(rows, now), 2) == round(2.0 * 3 + 1.0 * 12 + 4.0 * 1, 2)


def test_the_agent_releases_only_what_it_made_that_is_shut_down_and_not_wired_to_a_box():
    with _t.tmpdir() as d:
        allow(d)
        cloud = FakeCloud(instances=[Instance(id="pro-person", status="shutdown", billing="payg")], stock={("v-48g", "westDC3"): 3})
        s = stew(d, cloud)
        mine = s.create(spec())["created"]
        cloud.rows[mine].status = "shutdown"
        with _t.raises("not_allowed") as c:
            s.check_release("pro-person", cloud.all())                                                  # a person's box: never
        assert "not created by the agent" in c.err.message
        with _t.raises("not_allowed"):
            stew(d, cloud, env={"CLOUDGPU_B_INSTANCE": mine}).check_release(mine, cloud.all())         # wired into a pool box: a take may be on it
        s.check_release(mine, cloud.all())
        assert s.release(mine) == {"done": "release", "id": mine} and mine not in cloud.rows
        assert [r["action"] for r in s.ledger.rows()][-1] == "release" and s.ledger.created("fakecloud") == {}
        assert pool_instances({"CLOUDGPU_INSTANCE": "pro-1", "CLOUDGPU_B_INSTANCE": "pro-2", "H3HOST_INSTANCE": "pro-3", "CLOUDGPU_POOL": "a,b", "CLOUDGPU_B_TOKEN": "x"}) == {"pro-1", "pro-2", "pro-3"}


def test_the_cli_release_of_the_agent_checks_the_ledger_then_the_state_of_the_instance():
    with _t.tmpdir() as d:
        allow(d)
        cloud = FakeCloud(instances=[Instance(id="pro-person", status="shutdown", billing="payg")])
        os.environ["CLOUDGPU_STATE_DIR"] = d
        from cloudgpu import vendors
        reg = vendors.registry
        vendors.registry = lambda: {"fakecloud": lambda env=None: cloud}
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = cli.main(["instances", "--vendor", "fakecloud", "release", "--agent", "--yes", "pro-person"])
            out = json.loads(buf.getvalue())
            assert rc == 1 and out["kind"] == "not_allowed" and ("release", "pro-person") not in cloud.calls and "pro-person" in cloud.rows
        finally:
            vendors.registry = reg
            os.environ.pop("CLOUDGPU_STATE_DIR", None)


def test_ensure_counts_what_runs_powers_on_what_is_off_by_the_hour_and_creates_the_rest():
    with _t.tmpdir() as d:
        allow(d, instances=3, cny=500)
        cloud = FakeCloud(instances=[Instance(id="pro-a", status="running", billing="payg"), Instance(id="pro-b", status="shutdown", billing="payg"),
                                     Instance(id="pro-pass", status="shutdown", billing="daily"), Instance(id="pro-other", status="shutdown", billing="payg")],
                          stock={("v-48g", "westDC3"): 1})
        s = stew(d, cloud, env={"CLOUDGPU_INSTANCE": "pro-a", "CLOUDGPU_B_INSTANCE": "pro-b", "CLOUDGPU_C_INSTANCE": "pro-pass"})
        dry = s.ensure(3, TEMPLATE, specs=["v-48g"], centers=["westDC3"], yes=False)
        assert dry["dry_run"] and dry["would_power_on"] == ["pro-b"] and dry["would_create"] == 1 and ("power_on", "pro-b") not in cloud.calls and not any(c[0] == "create" for c in cloud.calls)
        r = s.ensure(3, TEMPLATE, specs=["v-48g"], centers=["westDC3"], yes=True, boot_wait=60)
        assert r["running"] == ["pro-a"] and r["powered_on"] == ["pro-b"] and [c["created"] for c in r["created"]] == ["pro-new1"] and r["missing"] == 0 and r["failures"] == []
        assert ("power_on", "pro-other") not in cloud.calls and ("power_on", "pro-pass") not in cloud.calls                # not wired to us / a pass: not touched
        assert any("pass" in n and "pro-pass" in n for n in r["next"]) and set(r["now_running"]) == {"pro-b", "pro-new1"} and r["doors"]["pro-new1"]["ssh"]["port"] == 2222
        assert [x["action"] for x in s.ledger.rows() if x["action"] in ("power-on", "create")] == ["power-on", "create"]


def test_ensure_tries_the_specs_and_data_centers_in_order_and_retries_only_when_there_is_no_stock():
    with _t.tmpdir() as d:
        allow(d, instances=4, cny=1000)
        slept = []
        cloud = FakeCloud(stock={("pro6000-p", "beijingDC2"): 1})
        s = stew(d, cloud, sleep=slept.append)
        r = s.ensure(1, TEMPLATE, specs=["v-48g", "pro6000-p"], centers=["westDC3", "beijingDC2"], yes=True, delays=(5, 10, 20), patience=100)
        creates = [c[1] for c in cloud.calls if c[0] == "create"]
        assert creates == [("v-48g", "westDC3"), ("v-48g", "beijingDC2"), ("pro6000-p", "westDC3"), ("pro6000-p", "beijingDC2")]     # the order asked for
        assert r["missing"] == 0 and r["created"][0]["spec"] == "pro6000-p" and r["created"][0]["data_center"] == "beijingDC2" 
        assert {f["kind"] for f in r["failures"]} == {"no_stock"} and len(r["failures"]) == 3


def test_ensure_backs_off_and_reports_how_many_are_missing_and_each_reason_when_nothing_comes():
    with _t.tmpdir() as d:
        allow(d, instances=4, cny=1000)
        slept = []
        cloud = FakeCloud(stock={})
        r = stew(d, cloud, sleep=slept.append).ensure(2, TEMPLATE, specs=["v-48g"], centers=["westDC3", "beijingDC2"], yes=True, delays=(5, 10, 20), patience=30)
        assert slept == [5, 10] and r["waited_s"] == 15 and r["missing"] == 2 and r["created"] == []                       # the third pause (20) would pass the patience of 30
        f = {x["where"]: x for x in r["failures"]}
        assert set(f) == {"v-48g@westDC3", "v-48g@beijingDC2"} and all(x["kind"] == "no_stock" and x["tries"] == 3 for x in f.values())
        assert any("2 short" in n and "no free card" in n for n in r["next"])


def test_a_failure_that_is_not_stock_is_reported_apart_and_not_retried():
    with _t.tmpdir() as d:
        allow(d, instances=4, cny=1000)
        slept = []
        cloud = FakeCloud(stock={("v-48g", "beijingDC2"): 1})
        cloud.errors[("v-48g", "westDC3")] = CloudGpuError("http", "autodl api create: AccountBalanceNotEnough 余额不足", status="AccountBalanceNotEnough")
        r = stew(d, cloud, sleep=slept.append).ensure(1, TEMPLATE, specs=["v-48g"], centers=["westDC3", "beijingDC2"], yes=True)
        assert r["missing"] == 0 and r["created"][0]["data_center"] == "beijingDC2" and slept == []
        assert [(f["kind"], f["tries"]) for f in r["failures"]] == [("http", 1)] and "余额" in r["failures"][0]["reason"]
        cloud2 = FakeCloud(stock={})
        cloud2.errors[("v-48g", "westDC3")] = CloudGpuError("http", "autodl api create: AccountBalanceNotEnough 余额不足", status="AccountBalanceNotEnough")
        slept2 = []
        r2 = stew(d, cloud2, sleep=slept2.append).ensure(1, TEMPLATE, specs=["v-48g"], centers=["westDC3"], yes=True)
        assert r2["missing"] == 1 and slept2 == [] and any("not stock" in n for n in r2["next"])                 # only non-stock failures: no waiting


def test_ensure_stops_creating_at_the_allowance_and_says_so():
    with _t.tmpdir() as d:
        allow(d, instances=1, cny=1000)
        cloud = FakeCloud(stock={("v-48g", "westDC3"): 9})
        r = stew(d, cloud).ensure(3, TEMPLATE, specs=["v-48g"], centers=["westDC3"], yes=True)
        assert len(r["created"]) == 1 and r["missing"] == 2 and [f["kind"] for f in r["failures"]] == ["over_quota"] and any("allowance" in n for n in r["next"])
        assert cloud.n == 1


def test_a_power_on_with_no_free_card_is_retried_like_a_create():
    with _t.tmpdir() as d:
        cloud = FakeCloud(instances=[Instance(id="pro-b", status="shutdown", billing="payg")])
        cloud.power_errors["pro-b"] = CloudGpuError("http", "autodl api power_on: 库存不足", status="X", detail="库存不足")
        slept = []
        r = stew(d, cloud, env={"CLOUDGPU_B_INSTANCE": "pro-b"}, sleep=slept.append).ensure(1, TEMPLATE, yes=True, delays=(5, 5), patience=60)
        assert r["missing"] == 1 and slept == [5, 5] and r["failures"][0]["what"] == "power-on" and r["failures"][0]["kind"] == "no_stock" and r["failures"][0]["tries"] == 3


def test_autodl_no_stock_is_told_from_a_missing_balance_or_a_bad_token():
    v = va.AutoDL({"AUTODL_DEV_TOKEN": "x"})
    stock = CloudGpuError("http", "autodl api create: X 当前地区GPU库存不足", status="X")
    assert v.is_no_stock(stock) is True
    assert v.is_no_stock(CloudGpuError("http", "autodl api create: X 账户余额不足", status="X")) is False
    assert v.is_no_stock(CloudGpuError("http", "autodl api create: HTTP 401", status=401)) is False
    assert v.is_no_stock(CloudGpuError("http", "autodl api create: timed out", status="network")) is False
    assert v.is_no_stock(CloudGpuError("key_missing", "stock")) is False


def test_a_box_with_no_account_view_says_it_may_be_off_or_ended_and_who_can_help():
    try:
        power.Power().power_on("/x")
        assert False
    except CloudGpuError as e:
        text = e.message + " ".join(e.next)
        assert e.kind == "down" and "switched off" in e.message and "pass may have ended" in e.message
        assert "AUTODL_DEV_TOKEN" in e.message and "CLOUDGPU_INSTANCE" in e.message and "account holder" in text and "never in chat" in e.message
    s = status.status({"CLOUDGPU_POWER": "none", "CLOUDGPU_BASE": "http://127.0.0.1:1", "CLOUDGPU_TOKEN": "t" * 30, "CLOUDGPU_BOX": "x"})
    assert any("cannot tell whether the box is off, its pass has ended" in n and "AUTODL_DEV_TOKEN" in n for n in s["next"]), s["next"]


if __name__ == "__main__":
    _t.main(globals())
