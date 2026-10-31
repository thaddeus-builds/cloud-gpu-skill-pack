"""cloudgpu.vendors and `python3 -m cloudgpu instances`, offline: the autodl vendor against the documented answers (autodl.art/docs/app_api): the list, one instance, the status,
the snapshot WITHOUT its password and token, the create request checked against the documented limits; and the CLI's rule: reading is free, everything that changes the
account (power, create, release) is a dry run until `--yes`, and release refuses an instance that is not shut down. A vendor taken in later gets a power backend for free."""

import io
import json
from contextlib import contextmanager, redirect_stdout

import _t

from cloudgpu import __main__ as cli
from cloudgpu import power, vendors
from cloudgpu.vendors import Instance, Vendor
from cloudgpu.vendors import autodl as va

LIST = {"list": [
    {"created_at": "2026-10-22T23:45:34+08:00", "uuid": "pro-1", "status": "running", "charge_type": "daily", "req_gpu_amount": 1, "name": "",
     "expired_at": {"Time": "2026-10-24T01:01:32+08:00", "Valid": True}, "started_at": {"Time": "2026-10-23T00:44:40+08:00", "Valid": True},
     "stopped_at": {"Time": "0001-01-01T00:00:00Z", "Valid": False}, "gpu_spec_uuid": "v-48g", "region_name": "西北B区"},
    {"uuid": "pro-2", "status": "shutdown", "charge_type": "payg", "expired_at": {"Time": "0001-01-01T00:00:00Z", "Valid": False}, "gpu_spec_uuid": "pro6000-p"}],
    "page_index": 1, "page_size": 50, "max_page": 1}
SNAP = {"region_sign": "bj-B1", "payg_price": 6980, "origin_pay_price": 7350, "snapshot_gpu_alias_name": "NVIDIA RTX PRO 6000", "proxy_host": "connect.example.com",
        "root_password": "SECRET-PASSWORD", "ssh_port": 34222, "jupyter_token": "SECRET-JUPYTER", "jupyter_domain": "a1.example.com:8443",
        "service_6006_domain": "u1.example.com:8443", "service_6006_port_protocol": "http", "service_6008_domain": "uu1.example.com:8443", "service_6008_port_protocol": "http",
        "cg_application_info": {"application_uuid": "vbxo", "current_version": "v2", "application_name": "ComfyUI"}}
CREATE = {"req_gpu_amount": 1, "expand_system_disk_by_gb": 0, "gpu_spec_uuid": "pro6000-p", "application_uuid": "J1c7lsvbq5", "application_version": "latest"}


@contextmanager
def api(answer):
    calls = []
    real = va.AutoDL._call

    def call(self, method, path, body=None, query=None):
        calls.append((method, path, body, query))
        return answer(method, path, body, query)
    va.AutoDL._call = call
    try:
        yield calls
    finally:
        va.AutoDL._call = real


def answers(method, path, body, query):
    return {"list": LIST, "snapshot": SNAP, "status": "running", "create": "pro-new"}.get(path)


def test_the_list_and_one_instance_are_normalised_and_a_pass_has_its_end_time():
    with api(answers):
        v = vendors.get("autodl", {"AUTODL_DEV_TOKEN": "x"})
        rows = v.all()
        assert [i.id for i in rows] == ["pro-1", "pro-2"] and isinstance(rows[0], Instance)
        a, b = rows
        assert (a.billing, a.ends, a.gpu_spec, a.gpus, a.region, a.name) == ("daily", "2026-10-24T01:01:32+08:00", "v-48g", 1, "西北B区", None)
        assert (b.billing, b.ends, b.status) == ("payg", None, "shutdown")
        assert v.get("pro-2").gpu_spec == "pro6000-p" and v.get("pro-nope") is None
        assert v.status("pro-1") == "running"


def test_the_snapshot_has_the_price_the_ssh_door_and_the_services_and_never_a_password_or_a_token():
    with api(answers) as calls:
        snap = vendors.get("autodl", {"AUTODL_DEV_TOKEN": "x"}).snapshot("pro-1")
    assert calls[0][:2] == ("GET", "snapshot") and calls[0][3] == {"instance_uuid": "pro-1"}          # the query string form, not a body
    assert snap["price_cny_per_hour"] == 6.98 and snap["list_price_cny_per_hour"] == 7.35 and snap["ssh"] == {"host": "connect.example.com", "port": 34222}
    assert snap["services"] == {6006: {"domain": "u1.example.com:8443", "protocol": "http"}, 6008: {"domain": "uu1.example.com:8443", "protocol": "http"}}
    assert snap["application"]["name"] == "ComfyUI" and snap["gpu_name"] == "NVIDIA RTX PRO 6000"
    assert "SECRET-PASSWORD" not in json.dumps(snap) and "SECRET-JUPYTER" not in json.dumps(snap)


def test_the_requests_are_the_documented_ones_and_a_missing_token_is_told_what_to_set():
    with api(answers) as calls:
        v = vendors.get("autodl", {"AUTODL_DEV_TOKEN": "x"})
        v.power_on("pro-1", "/root/h3host/start.sh")
        v.power_on("pro-1")
        v.power_off("pro-1")
        v.release("pro-1")
        assert v.create(CREATE) == "pro-new"
    assert [(c[0], c[1]) for c in calls] == [("POST", "power_on"), ("POST", "power_on"), ("POST", "power_off"), ("POST", "release"), ("POST", "create")]
    assert calls[0][2] == {"instance_uuid": "pro-1", "payload": "gpu", "start_command": "/root/h3host/start.sh"} and calls[1][2] == {"instance_uuid": "pro-1", "payload": "gpu"}
    assert calls[3][2] == {"instance_uuid": "pro-1"} and calls[4][2] == CREATE
    with _t.raises("key_missing") as c:
        vendors.get("autodl", {}).status("pro-1")
    assert "AUTODL_DEV_TOKEN" in c.err.message


def test_a_create_request_is_checked_against_the_documented_limits_before_anything_is_sent():
    v = vendors.get("autodl", {"AUTODL_DEV_TOKEN": "x"})
    assert v.create_problems(CREATE) == [] and v.create_problems({**CREATE, "data_center_list": ["westDC3"], "instance_name": "n", "start_command": "sleep 1"}) == []
    bad = {"req_gpu_amount": 5, "expand_system_disk_by_gb": 501, "gpu_spec_uuid": "nope", "data_center_list": ["mars"], "extra": 1}
    problems = " | ".join(v.create_problems(bad))
    for part in ("req_gpu_amount must be", "expand_system_disk_by_gb must be", "gpu_spec_uuid 'nope'", "application_uuid is required", "application_version is required", "data_center_list", "unknown fields: extra"):
        assert part in problems, (part, problems)
    assert any("must be an integer 1 to 4" in p for p in v.create_problems({**CREATE, "req_gpu_amount": True}))        # a bool is not a number of cards
    with api(answers) as calls:
        with _t.raises("key_missing"):
            v.create(bad)
        assert calls == []                                                                                          # nothing was sent


def run(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(argv)
    return rc, json.loads(buf.getvalue().strip().splitlines()[-1]) if buf.getvalue().strip() else None


def test_the_cli_reads_freely_and_every_change_is_a_dry_run_until_yes():
    import os
    os.environ["AUTODL_DEV_TOKEN"] = "x"
    os.environ["CLOUDGPU_STATE_DIR"] = _state = _t.tempfile.mkdtemp(prefix="cloudgpu-test-")          # the ledger of a person's --yes goes here, not to the home directory
    try:
        with api(answers) as calls:
            rc, d = run(["instances", "list"])
            assert rc == 0 and [i["id"] for i in d] == ["pro-1", "pro-2"]
            rc, d = run(["instances", "snapshot", "pro-1"])
            assert rc == 0 and "SECRET" not in json.dumps(d)
            rc, d = run(["instances", "specs"])
            assert rc == 0 and d["v-48g"] == "4090-48G"
            before = len(calls)
            for argv in (["power-on", "pro-1"], ["power-off", "pro-1"]):
                rc, d = run(["instances", *argv])
                assert rc == 0 and d["dry_run"] is True and "--yes" in d["next"][0], d
            with_spec = _t.tmpdir()
            with with_spec as dd:
                f = f"{dd}/spec.json"
                open(f, "w").write(json.dumps(CREATE))
                rc, d = run(["instances", "create", "--spec", f])
                assert rc == 0 and d["dry_run"] is True and "STARTS BILLING" in d["next"][0] and d["request"] == CREATE
                open(f, "w").write(json.dumps({"req_gpu_amount": 9}))
                rc, d = run(["instances", "create", "--spec", f])
                assert rc == 2 and d["problems"]
            assert [c[1] for c in calls[before:]] == []                                       # not one state-changing call went out
            rc, d = run(["instances", "release", "pro-1"])                                  # running: refused even before the dry run
            assert rc == 2 and "not shutdown" in d["error"]
            assert [c[1] for c in calls[before:]] == ["status"]
        with api(lambda m, path, b, q: "shutdown" if path == "status" else None) as calls:
            rc, d = run(["instances", "release", "pro-2"])
            assert rc == 0 and d["dry_run"] is True and "DESTROYS" in d["next"][0] and [c[1] for c in calls] == ["status"]
            rc, d = run(["instances", "release", "pro-2", "--yes"])
            assert rc == 0 and d == {"done": "release", "id": "pro-2"} and [c[1] for c in calls] == ["status", "status", "release"]
            rc, d = run(["instances", "power-on", "pro-2", "--yes"])
            assert rc == 0 and d["done"] == "power-on" and calls[-1][1] == "power_on"
    finally:
        os.environ.pop("AUTODL_DEV_TOKEN", None)
        os.environ.pop("CLOUDGPU_STATE_DIR", None)
        _t.shutil.rmtree(_state, ignore_errors=True)
    rc, d = run(["instances", "list"])
    assert rc == 1 and d["kind"] == "key_missing" and "AUTODL_DEV_TOKEN" in d["error"]            # no token: told what to set, nothing called


class Fake(Vendor):
    """A vendor taken in later: only what it documents."""
    name, token_env = "fakecloud", "FAKE_TOKEN"

    def status(self, instance_id):
        return "running"

    def power_off(self, instance_id):
        self.off = instance_id


def test_a_vendor_taken_in_later_gets_its_power_backend_for_free_and_a_call_it_lacks_says_so():
    real = vendors.registry
    vendors.registry = lambda: {**real(), "fakecloud": Fake}
    try:
        env = {"CLOUDGPU_POWER": "fakecloud", "CLOUDGPU_INSTANCE": "i-1", "FAKE_TOKEN": "t"}
        pw = power.backend(env)
        assert isinstance(pw, power.VendorPower) and pw.name == "fakecloud" and pw.configured() and pw.status() == "running" and pw.power_off() is True
        with _t.raises("unknown"):
            pw.power_on("/x")                                                              # the vendor documents no power-on: said, not guessed
        assert power.backend({"CLOUDGPU_POWER": "fakecloud", "CLOUDGPU_INSTANCE": "i-1"}).configured() is False          # no token: not configured
    finally:
        vendors.registry = real


if __name__ == "__main__":
    _t.main(globals())
