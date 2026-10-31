"""The allowance a person gives an agent to rent boxes by itself, and the ledger of every create and release.

The allowance is one small JSON file in the state directory (`CLOUDGPU_QUOTA_FILE`, else `<state dir>/quota.json`; the state directory is `CLOUDGPU_STATE_DIR`, else `~/.cloudgpu`).
**No file, or a zero in it, means the agent may not create anything**: that is the default, and the only way the package ever lets an agent start billing is a person writing numbers here:

    python3 -m cloudgpu quota set --by NAME --max-instances 2 --max-cny-per-day 100 --price v-48g=3.2 [--note ...]
    python3 -m cloudgpu quota show

`--by` is recorded with the time (in the file and in the ledger). A quota file that group or others can write is not believed (it is the agent's own account to change, then): the allowance is 0
until `chmod go-w` is done. Keep the file out of the agent's reach (another user's file, mode 0644, or read only) for the allowance to be a limit and not a habit; see docs, "An allowance for an agent".

The ledger (`ledger.jsonl` beside it, `CLOUDGPU_LEDGER_FILE`) is append only, one JSON object a line: time, action (create, release, power-on, quota-set, gone), the instance, the card spec, the data
center, the price per hour, who triggered it (`actor`: CLOUDGPU_ACTOR, else `agent` or `person`) and whether it was the agent's own call (`agent`). The agent releases only an instance that
the ledger shows it created itself and that has not been released since.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .errors import CloudGpuError

RESERVE_HOURS = 2                       # a new instance is counted as this many hours of its price against the day's budget before it has run at all


def state_dir(env: Mapping | None = None) -> Path:
    v = config.get("STATE_DIR", env)
    return Path(v).expanduser() if v else Path.home() / ".cloudgpu"


def quota_path(env: Mapping | None = None) -> Path:
    v = config.get("QUOTA_FILE", env)
    return Path(v).expanduser() if v else state_dir(env) / "quota.json"


def ledger_path(env: Mapping | None = None) -> Path:
    v = config.get("LEDGER_FILE", env)
    return Path(v).expanduser() if v else state_dir(env) / "ledger.jsonl"


@dataclass
class Quota:
    max_instances: int = 0                                    # at most this many instances the agent created, existing at once (0: none)
    max_cny_per_day: float = 0.0                              # at most this much spent today, counted as an upper bound (see `spent_today`); 0: none
    prices: dict = field(default_factory=dict)                # card spec -> CNY per hour the person assumes (a spec without a price is not created)
    reserve_hours: float = RESERVE_HOURS
    set_by: str = ""
    set_at: str = ""
    note: str = ""
    problem: str = ""                                         # why a file that exists is not believed ("" when it is fine)

    @property
    def allows_create(self) -> bool:
        return self.max_instances > 0 and self.max_cny_per_day > 0 and not self.problem

    def as_dict(self) -> dict:
        d = {"max_instances": self.max_instances, "max_cny_per_day": self.max_cny_per_day, "prices_cny_per_hour": self.prices, "reserve_hours": self.reserve_hours,
             "set_by": self.set_by, "set_at": self.set_at, "note": self.note, "agent_may_create": self.allows_create}
        if self.problem:
            d["problem"] = self.problem
        return d


def load(env: Mapping | None = None) -> Quota:
    """The allowance on disk: all zero when there is no file, it cannot be read, or others can write it."""
    p = quota_path(env)
    try:
        st = p.stat()
    except OSError:
        return Quota()
    if st.st_mode & 0o022:
        return Quota(problem=f"{p.name} can be written by group or others (mode {st.st_mode & 0o777:03o}): not believed; a person must chmod go-w it")
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        prices = {str(k): float(v) for k, v in (d.get("prices_cny_per_hour") or {}).items() if float(v) > 0}
        return Quota(max_instances=max(0, int(d.get("max_instances") or 0)), max_cny_per_day=max(0.0, float(d.get("max_cny_per_day") or 0)), prices=prices,
                     reserve_hours=max(0.0, float(d.get("reserve_hours", RESERVE_HOURS))), set_by=str(d.get("set_by") or ""), set_at=str(d.get("set_at") or ""),
                     note=str(d.get("note") or "")[:200])
    except (OSError, ValueError, TypeError, AttributeError):
        return Quota(problem=f"{p.name} is not a quota file (JSON with max_instances and max_cny_per_day): the allowance is 0")


def save(q: Quota, env: Mapping | None = None) -> Path:
    """Write the allowance (mode 0644, in a 0700 state directory) and a ledger line saying who and when."""
    p = quota_path(env)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps({k: v for k, v in q.as_dict().items() if k not in ("agent_may_create", "problem")}, ensure_ascii=False, indent=1), encoding="utf-8")
    os.chmod(tmp, 0o644)
    os.replace(tmp, p)
    return p


class Ledger:
    """Append only. `rows()` skips a line it cannot read."""

    def __init__(self, path: Path, *, now=time.time):
        self.path, self.now = Path(path), now

    def append(self, action: str, **fields) -> dict:
        row = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(self.now())), "ts": round(self.now(), 1), "action": action, **fields}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
        finally:
            os.close(fd)
        return row

    def rows(self) -> list[dict]:
        out = []
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError:
            return out
        for line in text.splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict):
                out.append(d)
        return out

    def created(self, vendor: str | None = None) -> dict[str, dict]:
        """{instance id: its create row} for every instance the ledger shows created and not released or gone since (the last create of an id wins)."""
        live: dict[str, dict] = {}
        for r in self.rows():
            if vendor and r.get("vendor") not in (None, vendor):
                continue
            i = r.get("instance")
            if r.get("action") == "create" and i:
                live[i] = r
            elif r.get("action") in ("release", "gone") and i:
                live.pop(i, None)
        return live

    def all_creates(self) -> list[dict]:
        return [r for r in self.rows() if r.get("action") == "create"]


def day_start(now: float) -> float:
    t = time.localtime(now)
    return time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, -1))


def spent_today(rows: list[dict], now: float) -> float:
    """CNY spent since local midnight, an UPPER bound: every instance the ledger shows created counts at its price per hour from its create (or midnight) until its release (or now), as if it
    ran all that time: a box that was powered off in between costs less than this says. An instance with no known price counts at the price the quota assumed when it was created."""
    start, total = day_start(now), 0.0
    open_: dict[str, dict] = {}

    def close(row: dict, end: float) -> float:
        t0 = max(float(row.get("ts") or 0), start)
        return float(row.get("price_cny_per_hour") or 0) * (end - t0) / 3600 if end > t0 else 0.0

    for r in rows:
        a, i = r.get("action"), r.get("instance")
        if not i:
            continue
        if a == "create":
            if i in open_:                                      # the same id again: the earlier life ended
                total += close(open_.pop(i), float(r.get("ts") or now))
            open_[i] = r
        elif a in ("release", "gone") and i in open_:
            total += close(open_.pop(i), float(r.get("ts") or now))
    return total + sum(close(r, now) for r in open_.values())


def parse_price(items: list[str]) -> dict:
    out = {}
    for it in items:
        spec, _, val = it.partition("=")
        try:
            v = float(val)
        except ValueError:
            v = 0
        if not spec or v <= 0:
            raise CloudGpuError("key_missing", f"--price must be SPEC=CNY_PER_HOUR (v-48g=3.2), not {it!r}")
        out[spec] = v
    return out
