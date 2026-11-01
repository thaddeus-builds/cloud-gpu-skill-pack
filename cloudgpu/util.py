"""How much a box works, read from its task records (`python3 -m cloudgpu util [--since TIME] [BOX...]`), read only.

Whether a box is the bottleneck cannot be told from a consumer's own wall clock: the adapter's records say how long a take took on the box and how often one arrived. The adapter lists them
(`GET /cloudgpu/tasks`, 0.12.0, token-checked; it keeps a record for 24 h, so a report reaches back a day at most); `fetch` reads that over the box's tunnel and `report` is the statistics, a pure
function of the records, so a test feeds it made-up ones. It sends no take and touches no running one. A box that does not answer is a row with an `error`, never powered on or tunnelled to here.

Per box and workflow: `takes` (all that were created), `failed`, `created_to_done_s` (median and slowest, of the takes that finished well), `gap_s` (median and mean between consecutive
creations), `busy_share` (the share of the span, first creation to last finish, in which at least one take was running on the box: from `started`, else from `created`, to `done`; a take still
running counts to `now`), `last_30_min` (takes created in the last 30 minutes before the box's `now`). A number that cannot be told (one take has no gap) is null.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from statistics import mean, median

from . import net
from .errors import CloudGpuError

PATH = "/cloudgpu/tasks"
TIMEOUT_S = 30
RECENT_S = 30 * 60


def parse_since(text: str | None, now: float) -> float:
    """`--since` as seconds since 1970: a number of those, `90m` / `2h` / `1d` ago, or a local date and time (`2026-10-29`, `2026-10-29T14:30`, `2026-10-29 14:30`). None is the start of time."""
    if not text:
        return 0.0
    t = text.strip()
    try:
        return float(t)
    except ValueError:
        pass
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd])", t.lower())
    if m:
        return now - float(m.group(1)) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return time.mktime(time.strptime(t, fmt))
        except ValueError:
            pass
    raise CloudGpuError("key_missing", f"--since {text!r} is not a time: use 90m, 2h, 1d, 2026-10-29, 2026-10-29T14:30 or seconds since 1970")


def _busy(spans: list, now: float) -> float:
    """Seconds covered by at least one of the (start, end) spans (end None: still running, to `now`)."""
    cover, edge = 0.0, None
    for a, b in sorted((a, now if b is None else b) for a, b in spans):
        if b <= a:
            continue
        if edge is None or a > edge:
            cover, edge = cover + (b - a), b
        elif b > edge:
            cover, edge = cover + (b - edge), b
    return cover


def _group(rows: list, now: float, recent_s: float) -> dict:
    rows = sorted(rows, key=lambda r: r["created"])
    made = [r["created"] for r in rows]
    gaps = [b - a for a, b in zip(made, made[1:])]
    num = lambda v: isinstance(v, (int, float))                         # noqa: E731
    took = [r["done"] - r["created"] for r in rows if r.get("status") == "SUCCESS" and num(r.get("done")) and r["done"] >= r["created"]]
    spans = [(r["started"] if num(r.get("started")) else r["created"], r["done"] if num(r.get("done")) else None)
             for r in rows if num(r.get("done")) or r.get("status") == "RUNNING"]           # one still running has no end yet: it counts to now
    ends = [e if e is not None else now for _, e in spans]
    span = max(ends) - made[0] if ends else 0.0
    return {
        "takes": len(rows),
        "failed": sum(1 for r in rows if r.get("status") == "FAILED"),
        "created_to_done_s": {"median": round(median(took), 1), "slowest": round(max(took), 1)} if took else None,
        "gap_s": {"median": round(median(gaps), 1), "mean": round(mean(gaps), 1)} if gaps else None,
        "busy_share": round(min(1.0, _busy(spans, now) / span), 3) if span > 0 else None,
        "last_30_min": sum(1 for c in made if c >= now - recent_s),
    }


def line_of(box: str, workflow: str, g: dict) -> str:
    """The one line a person reads for a box and workflow."""
    def pair(d, a, b):
        return f"{d[a]:.0f}/{d[b]:.0f} s" if d else "-"
    share = "-" if g["busy_share"] is None else f"{g['busy_share'] * 100:.0f}%"
    return (f"{box} {workflow}: {g['takes']} takes ({g['failed']} failed), created to done median/slowest {pair(g['created_to_done_s'], 'median', 'slowest')}, "
            f"gap between takes median/mean {pair(g['gap_s'], 'median', 'mean')}, busy {share} of the span, {g['last_30_min']} in the last 30 min")


def report(records: list, now: float, since: float = 0.0, recent_s: float = RECENT_S) -> list:
    """[{workflow, takes, failed, created_to_done_s, gap_s, busy_share, last_30_min}] for the records created at or after `since`, one per workflow (sorted by name); records without a
    creation time are ignored. Pure: no clock, no file, no network."""
    by: dict = {}
    for r in records:
        if isinstance(r, dict) and isinstance(r.get("created"), (int, float)) and r["created"] >= since:
            by.setdefault(str(r.get("workflow") or "?"), []).append(r)
    return [{"workflow": w, **_group(rows, now, recent_s)} for w, rows in sorted(by.items())]


def fetch(box, since: float = 0.0) -> dict | None:
    """The box's task records: {"tasks": [...], "now": EPOCH, "ttl_s": N}, or None for an adapter older than 0.12.0 (it has no such path). Another failure is CloudGpuError(http)."""
    req = urllib.request.Request(f"{box.root()}{PATH}?since={urllib.parse.quote(repr(float(since)))}", headers={"Authorization": box.token()})
    try:
        with net.urlopen(req, timeout=TIMEOUT_S) as r:
            doc = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code in (404, 405, 501):
            return None
        raise CloudGpuError("http", f"the box's task list answered HTTP {e.code}", status=e.code) from None
    except (urllib.error.URLError, OSError, ValueError):
        raise CloudGpuError("http", "the box's task list did not answer") from None
    d = doc.get("data") if isinstance(doc, dict) and doc.get("code") == "Success" else None
    if not isinstance(d, dict) or not isinstance(d.get("tasks"), list) or not isinstance(d.get("now"), (int, float)):
        raise CloudGpuError("http", "the box's task list gave an answer this version does not know")
    return d


def run(boxes: list, since_text: str | None, names: list | None = None, now=time.time) -> tuple:
    """([row per box and workflow, or per box with an `error`], [the human lines]) for the pool's `boxes` [(name, Box)], limited to `names` (`default` is a pool of one)."""
    from .hold import box_key
    want = [box_key(n) for n in names or []]
    unknown = [n for n in want if n not in [box_key(b) for b, _ in boxes]]
    if unknown:
        raise CloudGpuError("key_missing", f"no box named {unknown[0]!r} in the settings (CLOUDGPU_POOL), the names are: " + ", ".join(box_key(b) for b, _ in boxes))
    rows, lines = [], []
    for name, box in boxes:
        key = box_key(name)
        if want and key not in want:
            continue
        try:
            if box.health() is None:
                raise CloudGpuError("down", "does not answer (a tunnel that is down, or the box is off): nothing was powered on or opened")
            since = parse_since(since_text, now())
            got = fetch(box, since)
            if got is None:
                raise CloudGpuError("http", "its adapter is older than 0.12.0 and has no task list: install the adapter on the box again")
        except CloudGpuError as e:
            if e.kind == "key_missing":
                raise
            rows.append({"box": key, "error": e.message})
            lines.append(f"{key}: {e.message}")
            continue
        for g in report(got["tasks"], got["now"], since):
            rows.append({"box": key, **g, "records_kept_s": got.get("ttl_s")})
            lines.append(line_of(key, g["workflow"], g))
        if not any(r["box"] == key for r in rows):
            rows.append({"box": key, "workflow": None, "takes": 0})
            lines.append(f"{key}: no takes in the records" + (f" since {since_text}" if since_text else ""))
    return rows, lines
