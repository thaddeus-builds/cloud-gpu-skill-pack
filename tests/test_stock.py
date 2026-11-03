"""`openmax-cloudgpu instances stock`, offline: the free cards of the autodl market page per data center (one request each, no token), the filter by spec, the exit code
that gates a power-on (0 a card is free, 3 none is), and the two refusals autodl alternates between when no card is free both counting as no stock."""

import io
import json
from contextlib import contextmanager, redirect_stdout

import _t

from cloudgpu import __main__ as cli
from cloudgpu import vendors
from cloudgpu.errors import CloudGpuError
from cloudgpu.vendors import autodl as va

WEST = [{"gpu_type": "RTX PRO 6000", "user_define_name": "PRO6000-96G", "gpu_idle_num": 0, "gpu_total_number": 693},
        {"gpu_type": "vGPU-48GB", "user_define_name": "4090-48G", "gpu_idle_num": 3, "gpu_total_number": 330},
        {"gpu_type": "NEW CARD", "user_define_name": "", "gpu_idle_num": 0, "gpu_total_number": 8}]
BEIJING = [{"gpu_type": "RTX PRO 6000", "user_define_name": "PRO6000-96G", "gpu_idle_num": 0, "gpu_total_number": 90},
           {"gpu_type": "vGPU-48GB", "user_define_name": "4090-48G", "gpu_idle_num": 0, "gpu_total_number": 0}]


@contextmanager
def market(answers):
    calls = []
    real = va.AutoDL._market

    def call(self, path, body):
        calls.append((path, body))
        return answers[body["data_center_list"][0]]
    va.AutoDL._market = call
    try:
        yield calls
    finally:
        va.AutoDL._market = real


def run(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cli.main(["instances", "--vendor", "autodl", "stock", *argv])
    return rc, json.loads(buf.getvalue())


def test_one_request_per_data_center_and_the_rows_carry_the_spec_id_the_account_uses():
    with market({"westDC3": WEST, "beijingDC2": BEIJING}) as calls:
        rc, doc = run([])
    assert [c[1] for c in calls] == [{"data_center_list": ["westDC3"]}, {"data_center_list": ["beijingDC2"]}]          # several in one request come back added together
    assert rc == 0 and doc["any_idle"] is True and len(doc["rows"]) == 5
    west48 = next(r for r in doc["rows"] if r["data_center"] == "westDC3" and r["spec"] == "v-48g")
    assert (west48["card"], west48["idle"], west48["total"]) == ("4090-48G", 3, 330)
    assert next(r for r in doc["rows"] if r["gpu_type"] == "NEW CARD")["spec"] is None                  # a card this package does not know is shown, with no spec id


def test_the_exit_code_is_3_when_no_card_is_free_and_a_filter_narrows_the_rows():
    with market({"westDC3": WEST, "beijingDC2": BEIJING}) as calls:
        rc, doc = run(["--data-center", "beijingDC2"])
        assert rc == 3 and doc["any_idle"] is False and {r["data_center"] for r in doc["rows"]} == {"beijingDC2"} and len(calls) == 1
        rc, doc = run(["--spec", "pro6000-p"])
        assert rc == 3 and [r["spec"] for r in doc["rows"]] == ["pro6000-p", "pro6000-p"]               # the 48G card is free in the west, but not the one asked for
        rc, doc = run(["--spec", "v-48g", "--data-center", "westDC3"])
        assert rc == 0 and [r["idle"] for r in doc["rows"]] == [3]


def test_an_unknown_data_center_or_spec_is_refused_and_no_token_is_needed_or_sent():
    with market({"westDC3": WEST, "beijingDC2": BEIJING}) as calls:
        rc, doc = run(["--spec", "nope"])
        assert rc == 1 and doc["kind"] == "unknown" and "nope" in doc["error"]
        rc, doc = run(["--data-center", "mars"])
        assert rc == 1 and doc["kind"] == "unknown" and "mars" in doc["error"] and len(calls) == 2          # nothing was asked for mars
    assert not vendors.get("autodl", {}).configured()                                                      # the run above had no token set: reading stock needs none


def test_a_vendor_with_no_stock_read_says_so_and_an_answer_that_is_not_success_is_an_error():
    class Bare(vendors.Vendor):
        name = "bare"
    with _t.raises("unknown"):
        Bare({}).stock()
    real = va.AutoDL._market
    va.AutoDL._market = lambda self, path, body: (_ for _ in ()).throw(CloudGpuError("http", "autodl market gpu_type: AuthorizeFailed", status="AuthorizeFailed"))
    try:
        rc, doc = run([])
    finally:
        va.AutoDL._market = real
    assert rc == 1 and doc["kind"] == "http" and doc["status"] == "AuthorizeFailed"


def test_both_refusals_autodl_gives_when_no_card_is_free_are_no_stock_and_a_bad_balance_still_is_not():
    v = va.AutoDL({"AUTODL_DEV_TOKEN": "x"})
    assert v.is_no_stock(CloudGpuError("http", "autodl api pro/power_on: InternalError 当前算力规格暂无库存", status="InternalError", detail="当前算力规格暂无库存")) is True
    assert v.is_no_stock(CloudGpuError("http", "autodl api pro/power_on: ServerBusy 服务正忙", status="ServerBusy", detail="服务正忙")) is True
    assert v.is_no_stock(CloudGpuError("http", "autodl api pro/power_on: X 账户余额不足,服务正忙", status="X")) is False


if __name__ == "__main__":
    _t.main(globals())
