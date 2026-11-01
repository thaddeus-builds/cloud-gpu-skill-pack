"""When a pass ends: the warning a person gets before a rented (daily, weekly, monthly) box expires, and the difference between a pass that is switched off and one that has ended.

A pass is bought and renewed by a person in the vendor's console; this package cannot do either (the API only creates pay-as-you-go instances). So the job here is only to NOTICE and to
NAME the box, the time and who acts. Used by `instances list|status|expiry`, `status`, and `ensure`; a consumer's health check calls `account_rows` and shows the `message` of each row.

A row (`row`) per pass instance: id, name, billing, status, ends (the vendor's ISO time), hours_left, days_left, `level` and `state`:

    state "expired"   level "error"  the pass has ended: a person renews it (rents again) in the vendor's console; powering on will not help
    state "expiring"  level "warn"   it ends in less than `warn_days` (default 3, CLOUDGPU_PASS_WARN_DAYS): a person renews it before then
    state "no_end"    level "warn"   the vendor gave no end time: the end cannot be watched, look in the console
    state "off"       level "info"   switched off but the pass runs on: power it on (`instances power-on ID --yes`, or `ensure`); nothing to renew yet
    state "active"    level "ok"     nothing to say

A pay-as-you-go instance never has a row: it has no end.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone

from . import config
from .errors import CloudGpuError
from .vendors import Instance, Vendor

PASS_BILLING = ("daily", "weekly", "monthly")      # autodl's charge_type of a pass: paid whether the card works or not
DEFAULT_WARN_DAYS = 3.0
OFF_STATUS = ("shutdown", "off")                   # the vendor's words for an instance that is clearly off (as in steward.py)
LEVELS = ("ok", "info", "warn", "error")
WHO = "a person renews it in the vendor's console (this package cannot: the API only makes pay-as-you-go instances)"


def warn_days(env: Mapping | None = None) -> float:
    """CLOUDGPU_PASS_WARN_DAYS as days (may be fractional); the default when unset, not a number, or not above 0."""
    try:
        v = float(config.get("PASS_WARN_DAYS", env) or DEFAULT_WARN_DAYS)
    except ValueError:
        return DEFAULT_WARN_DAYS
    return v if v > 0 and v == v and v != float("inf") else DEFAULT_WARN_DAYS


def parse_time(text: str | None) -> float | None:
    """Seconds since 1970 of an ISO time with its own offset (`2026-10-24T01:01:32+08:00`, or `...Z`); a time with no offset is read as UTC+8, the vendor's clock. None when it is not a time."""
    if not text:
        return None
    try:
        t = datetime.fromisoformat(re.sub(r"(\.\d+)", lambda m: (m.group(1) + "000000")[:7], str(text).strip().replace("Z", "+00:00"), count=1))      # python 3.9 takes 3 or 6 fraction digits only
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone(timedelta(hours=8)))
    return t.timestamp()


def _hours(ends: str | None, now: float | None = None) -> float | None:
    t = parse_time(ends)
    return None if t is None else (t - (time.time() if now is None else now)) / 3600


def hours_left(ends: str | None, now: float | None = None) -> float | None:
    h = _hours(ends, now)
    return None if h is None else round(h, 1)


def _left(h: float) -> str:
    return f"{abs(h) / 24:.1f} days" if abs(h) >= 48 else f"{abs(h):.0f} hours"


def row(i: Instance, now: float | None = None, days: float = DEFAULT_WARN_DAYS, vendor: str = "") -> dict | None:
    """The row for one instance, or None for one that is not a pass."""
    if i.billing not in PASS_BILLING:
        return None
    who = f"the {vendor} console" if vendor else "the vendor's console"
    label = f"{i.id}" + (f" ({i.name})" if i.name else "") + f", a {i.billing} pass"
    h = _hours(i.ends, now)                     # not rounded: a pass that ended two minutes ago has ended
    off = i.status in OFF_STATUS
    if h is None:
        level, state = "warn", "no_end"
        msg = f"{label}, reports no end time, so its end cannot be watched: a person looks at it in {who}"
    elif h < 0:
        level, state = "error", "expired"
        msg = f"{label}, ENDED {i.ends} ({_left(h)} ago): a person renews it in {who} (this package cannot: the API only makes pay-as-you-go instances); until then it takes no work"
    elif h < days * 24:
        level, state = "warn", "expiring"
        msg = f"{label}, ends {i.ends} (in {_left(h)}): a person renews it in {who} before then" + ("; it is off, power it on meanwhile (nothing extra to pay)" if off else "")
    elif off:
        level, state = "info", "off"
        msg = f"{label}, is off but the pass runs until {i.ends} (in {_left(h)}): power it on (`instances power-on {i.id} --yes`); nothing to renew yet"
    else:
        level, state = "ok", "active"
        msg = f"{label}, runs until {i.ends} (in {_left(h)})"
    return {"id": i.id, "name": i.name, "billing": i.billing, "status": i.status, "ends": i.ends, "hours_left": None if h is None else round(h, 1),
            "days_left": None if h is None else round(h / 24, 1), "level": level, "state": state, "message": msg, "who": WHO if level in ("warn", "error") and state != "off" else None}


def rows(instances, now: float | None = None, days: float = DEFAULT_WARN_DAYS, vendor: str = "", only=None) -> list[dict]:
    """Rows of the passes among `instances` (only those whose id is in `only`, when given), the ones that end soonest first."""
    out = [r for i in instances if (only is None or i.id in only) for r in [row(i, now, days, vendor)] if r]
    return sorted(out, key=lambda r: (r["hours_left"] is None, r["hours_left"] or 0))


def alerts(rs) -> list[dict]:
    """The rows that say something (everything but `ok`), worst first."""
    return sorted((r for r in rs if r["level"] != "ok"), key=lambda r: -LEVELS.index(r["level"]))


def worst(rs) -> str:
    return max((r["level"] for r in rs), key=LEVELS.index, default="ok")


def account_rows(vendor: Vendor, env: Mapping | None = None, *, only=None, now: float | None = None, days: float | None = None) -> list[dict]:
    """One listing of the account, as rows. `only`: the instance ids to look at (a health check passes the ones wired in the settings, so a pass nobody uses does not nag).
    Raises CloudGpuError when the account cannot be read: the caller says so, it is not 'no alerts'."""
    if not vendor.configured():
        raise CloudGpuError("key_missing", f"the {vendor.name} vendor needs {vendor.token_env} to watch a pass's end")
    return rows(vendor.all(), now, warn_days(env) if days is None else days, vendor.name, only)
