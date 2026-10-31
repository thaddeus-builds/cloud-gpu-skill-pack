"""autodl.art: the application-instance API (https://www.autodl.art/docs/app_api/, read 2026-10-22 and again 2026-10-23). Base https://www.autodl.art, header
`Authorization: <the account's developer token>` (console -> account -> settings -> developer token; NOT the workflow token), variable AUTODL_DEV_TOKEN.

    POST /api/v1/adl_dev/dev/instance/pro/create     {data_center_list?, req_gpu_amount 1..4, expand_system_disk_by_gb 0..500, gpu_spec_uuid, application_uuid,
                                                     application_version, instance_name?, start_command?}  -> data: the new instance's id (pay as you go: the only mode)
    GET  .../pro/snapshot?instance_uuid=             -> payg_price, origin_pay_price, snapshot_gpu_alias_name, proxy_host, ssh_port, service_6006/6008_domain, ...
                                                        (it also carries `root_password` and `jupyter_token`: dropped here, never returned or logged)
    GET  .../pro/status?instance_uuid=               -> data "running" | "shutdown" | ...
    POST .../pro/list {page_index, page_size}        -> data.list[], data.max_page; each: uuid, status, charge_type (payg | daily | weekly | monthly), expired_at {Time, Valid},
                                                        gpu_spec_uuid, req_gpu_amount, region_name, started_at, stopped_at, name
    POST .../pro/power_on {instance_uuid, payload: "gpu", start_command?}   (no card-less start by API)
    POST .../pro/power_off {instance_uuid}
    POST .../pro/release {instance_uuid}             shut it down first

`status`, `snapshot` and `list` are called with a query string (the body form the page shows answers RequestParameterIsWrong; checked 2026-10-22). Prices: `payg_price` 6980 is
¥6.98 per hour (checked against the console 2026-10-23: PRO6000 ¥6.98, list ¥7.35 = origin_pay_price 7350), so the unit is 0.001 CNY per hour. A pass instance's snapshot says `payg_price` 0 (checked on a day pass,
2026-10-23): the pass's price is not in it, so a pass's cost is the console's, not this API's.
"""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.parse
import urllib.request

from .. import net
from ..errors import CloudGpuError
from . import Instance, Vendor

API = "https://www.autodl.art/api/v1/adl_dev/dev/instance/pro"
# the appendix's table (page name -> API spec id), plus a spec seen on a real instance (6000d) that the appendix does not list
GPU_SPECS = {"h800": "H800-80G", "v-48g": "4090-48G", "pro6000-p": "PRO6000-96G", "v-32g-p": "4080(S)-32G", "v-48g-350w": "3090-48G", "5090-p": "5090-32G",
             "6000d": "6000D-84G (seen on an instance; not in the appendix)"}
DATA_CENTERS = {"westDC3": "northwest", "beijingDC2": "beijing"}
PASS = ("daily", "weekly", "monthly")
NO_STOCK_WORDS = ("stocknotenough", "out of stock", "no stock", "sold out", "soldout", "no available gpu", "no gpu available", "no free gpu",
                  "库存不足", "库存", "资源不足", "没有空闲", "无空闲", "无可用", "没有可用", "资源紧张", "售罄", "卡不够")
NOT_STOCK_WORDS = ("balance", "余额", "欠费", "funds", "quota", "配额", "arrears", "unauthor", "forbidden", "rights", "token", "permission", "权限", "disk", "磁盘")      # a missing balance or right is not a missing card


def _time(cell) -> str | None:
    """autodl's nullable time: {Time, Valid}."""
    return cell.get("Time") if isinstance(cell, dict) and cell.get("Valid") else None


class AutoDL(Vendor):
    name = "autodl"
    token_env = "AUTODL_DEV_TOKEN"

    def _call(self, method: str, path: str, body: dict | None = None, query: dict | None = None):
        if not self.configured():
            raise CloudGpuError("key_missing", "the autodl vendor needs AUTODL_DEV_TOKEN (the account's developer token, console -> account -> settings, not the workflow token)",
                                ["set it, or use CLOUDGPU_POWER=none when the box is always on or a pass"])
        token = self.token()
        if not net.header_safe(token):
            raise CloudGpuError("key_missing", f"{self.token_env} has a character that cannot be sent in an HTTP header (a newline inside it?)")
        url = f"{API}/{path}" + ("?" + urllib.parse.urlencode(query) if query else "")
        r = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None, method=method,
                                   headers={"Authorization": token, "Content-Type": "application/json"})
        try:
            with net.urlopen(r, timeout=60) as resp:         # a redirect to another host is refused: the token would follow it
                doc = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise CloudGpuError("http", f"autodl api {path} -> HTTP {e.code}", status=e.code, detail=e.read().decode(errors="replace")[:300]) from None
        except (urllib.error.URLError, OSError) as e:
            raise CloudGpuError("http", f"autodl api {path}: {e}", status="network", detail=str(e)[:200]) from None
        except (ValueError, http.client.HTTPException):          # http.client's text can carry the header value, the token: say nothing of it
            raise CloudGpuError("http", f"autodl api {path}: the request failed or the answer was not JSON", status="network") from None
        if doc.get("code") != "Success":
            raise CloudGpuError("http", f"autodl api {path}: {doc.get('code')} {doc.get('msg')}", status=doc.get("code"), detail=str(doc.get("msg", "")))
        return doc.get("data")

    # ---- reading ----
    @staticmethod
    def _instance(row: dict) -> Instance:
        return Instance(id=row.get("uuid"), status=row.get("status"), name=row.get("name") or None, billing=row.get("charge_type"), ends=_time(row.get("expired_at")),
                        gpu_spec=row.get("gpu_spec_uuid"), gpus=row.get("req_gpu_amount"), region=row.get("region_name"), created_at=row.get("created_at"),
                        started_at=_time(row.get("started_at")), stopped_at=_time(row.get("stopped_at")))

    def list(self, *, page=1, size=50):
        d = self._call("POST", "list", {"page_index": page, "page_size": size}) or {}
        return [self._instance(r) for r in d.get("list", [])], int(d.get("max_page") or 1)

    def status(self, instance_id):
        return self._call("GET", "status", query={"instance_uuid": instance_id})

    def snapshot(self, instance_id):
        d = self._call("GET", "snapshot", query={"instance_uuid": instance_id}) or {}
        services = {}
        for port in (6006, 6008):
            if d.get(f"service_{port}_domain"):
                services[port] = {"domain": d[f"service_{port}_domain"], "protocol": d.get(f"service_{port}_port_protocol", "http")}
        app = d.get("cg_application_info") or {}
        out = {"gpu_name": d.get("snapshot_gpu_alias_name"), "region": d.get("region_sign"),
               "price_cny_per_hour": d["payg_price"] / 1000 if d.get("payg_price") is not None else None,
               "list_price_cny_per_hour": d["origin_pay_price"] / 1000 if d.get("origin_pay_price") is not None else None,
               "ssh": {"host": d.get("proxy_host"), "port": int(d["ssh_port"])} if d.get("proxy_host") and d.get("ssh_port") else None,
               "services": services, "jupyter_domain": d.get("jupyter_domain"),
               "application": {"uuid": app.get("application_uuid"), "version": app.get("current_version"), "name": app.get("application_name")} if app else None}
        return out                                      # `root_password` and `jupyter_token` are in `d` and stay there

    # ---- power ----
    def power_on(self, instance_id, start_command=None):
        self._call("POST", "power_on", {"instance_uuid": instance_id, "payload": "gpu", **({"start_command": start_command} if start_command else {})})

    def power_off(self, instance_id):
        self._call("POST", "power_off", {"instance_uuid": instance_id})

    # ---- money and data ----
    def create_problems(self, spec):
        out = []
        for k in ("req_gpu_amount", "expand_system_disk_by_gb", "gpu_spec_uuid", "application_uuid", "application_version"):
            if spec.get(k) in (None, ""):
                out.append(f"{k} is required")
        n = spec.get("req_gpu_amount")
        if n is not None and not (isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= 4):
            out.append("req_gpu_amount must be an integer 1 to 4")
        e = spec.get("expand_system_disk_by_gb")
        if e is not None and not (isinstance(e, int) and not isinstance(e, bool) and 0 <= e <= 500):
            out.append("expand_system_disk_by_gb must be an integer 0 to 500")
        if spec.get("gpu_spec_uuid") not in (None, "") and spec["gpu_spec_uuid"] not in GPU_SPECS:
            out.append(f"gpu_spec_uuid {spec['gpu_spec_uuid']!r} is not one of {', '.join(GPU_SPECS)}")
        dc = spec.get("data_center_list")
        if dc is not None and not (isinstance(dc, list) and dc and all(x in DATA_CENTERS for x in dc)):
            out.append(f"data_center_list must be a non-empty list of {', '.join(DATA_CENTERS)}")
        extra = set(spec) - {"data_center_list", "req_gpu_amount", "expand_system_disk_by_gb", "gpu_spec_uuid", "application_uuid", "application_version", "instance_name", "start_command"}
        if extra:
            out.append(f"unknown fields: {', '.join(sorted(extra))}")
        return out

    def create(self, spec):
        problems = self.create_problems(spec)
        if problems:
            raise CloudGpuError("key_missing", "the create request is not one autodl takes: " + "; ".join(problems))
        return self._call("POST", "create", dict(spec))

    def release(self, instance_id):
        self._call("POST", "release", {"instance_uuid": instance_id})

    def is_no_stock(self, err):
        """The API page lists no error codes (read 2026-10-31), so this is a reading of the words a refusal carries, not a documented code: a code or message that talks of stock,
        shortage, or no free/available GPU or resource. An error that has none of them (a bad request, a missing token, a network failure) is not no-stock and is not retried."""
        if err.kind != "http" or err.status == "network":
            return False
        text = f"{err.status} {err.message} {err.detail}".lower()
        return not any(w in text for w in NOT_STOCK_WORDS) and any(w in text for w in NO_STOCK_WORDS)

    def gpu_specs(self):
        return dict(GPU_SPECS)
