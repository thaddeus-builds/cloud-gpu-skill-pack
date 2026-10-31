"""An agent managing its own boxes inside the allowance a person set (quota.py): create and release with the checks, and `ensure`, the verb "have N boxes up".

    python3 -m cloudgpu instances create --agent --yes --spec request.json           # inside the allowance, else refused with the reason
    python3 -m cloudgpu instances release --agent --yes ID                           # only an instance the ledger shows this agent created, shut down, not in a pool
    python3 -m cloudgpu instances ensure --count 2 --spec template.json --yes        # power on what is off, create what is missing, say what could not be had

Everything is the vendor's account API (vendors/); nothing here talks to a box. Every create, release and power-on is a line in the ledger. A person's own `--yes` without `--agent` is as before
(no allowance asked) but is ledgered too, and an instance a person made is never released by the agent.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping

from . import config, quota
from .errors import CloudGpuError
from .vendors import Vendor

OFF_STATUS = ("shutdown", "off")                          # the vendor's words for an instance that is clearly off (as in box.py)
DEFAULT_SPECS = ("v-48g",)                                # 4090 48G
DEFAULT_CENTERS = ("westDC3", "beijingDC2")
START_COMMAND = "/root/h3host/start.sh"                   # as in box.py: written by onbox/install.sh, run by the vendor after boot
BACKOFF_S = (20, 40, 80, 160, 300)                        # waits between rounds when a card is not in stock
PATIENCE_S = 600                                          # the most `ensure` waits in all for stock
BOOT_WAIT_S = 300                                         # how long a new instance is waited for to say `running`
INSTANCE_VAR = re.compile(r"^(?:CLOUDGPU|H3HOST)_(?:[A-Za-z0-9]+_)?INSTANCE$")


def pool_instances(env: Mapping) -> set[str]:
    """The instance ids the settings wire to a box (CLOUDGPU_INSTANCE, CLOUDGPU_<NAME>_INSTANCE): boxes a take may be running on, which the agent never releases."""
    return {str(v).strip() for k, v in env.items() if INSTANCE_VAR.fullmatch(k) and str(v).strip()}


def csv(text: str | None, default: tuple) -> list[str]:
    items = [x.strip() for x in (text or "").split(",") if x.strip()]
    return items or list(default)


class Steward:
    def __init__(self, vendor: Vendor, env: Mapping, *, agent: bool = True, now=time.time, sleep=time.sleep):
        self.vendor, self.env, self.agent, self.now, self.sleep = vendor, env, agent, now, sleep
        self.quota = quota.load(env)
        self.ledger = quota.Ledger(quota.ledger_path(env), now=now)
        self.actor = config.get("ACTOR", env) or ("agent" if agent else "person")

    # ---- what the ledger and the account say ----
    def sync(self, listing) -> dict[str, dict]:
        """{id: create row} of the instances this ledger created that the account still lists. One it no longer lists is written to the ledger as gone, so it stops counting."""
        alive = {i.id for i in listing}
        mine = self.ledger.created(self.vendor.name)
        for i in list(mine):
            if i not in alive:
                self.ledger.append("gone", vendor=self.vendor.name, instance=i, actor="steward")
                del mine[i]
        return mine

    def _log(self, action: str, **f) -> dict:
        return self.ledger.append(action, vendor=self.vendor.name, actor=self.actor, agent=self.agent, **f)

    def check_create(self, spec: dict, mine: dict) -> float:
        """The assumed price per hour when the allowance covers one more instance of `spec`; CloudGpuError(over_quota) saying which limit otherwise."""
        q = self.quota
        if not q.allows_create:
            raise CloudGpuError("over_quota", "the agent may not create boxes: the allowance is 0" + (f" ({q.problem})" if q.problem else ""),
                                ["a person sets it: python3 -m cloudgpu quota set --by NAME --max-instances N --max-cny-per-day CNY --price SPEC=CNY_PER_HOUR"])
        sp = spec.get("gpu_spec_uuid")
        price = q.prices.get(sp)
        if not price:
            raise CloudGpuError("over_quota", f"the allowance has no price for card spec {sp!r}, so what it would spend is not known",
                                [f"a person adds it: python3 -m cloudgpu quota set ... --price {sp}=CNY_PER_HOUR"])
        if len(mine) + 1 > q.max_instances:
            raise CloudGpuError("over_quota", f"the allowance is {q.max_instances} instance(s) at once and the agent already has {len(mine)}",
                                ["release one the agent made (when nothing runs on it), or a person raises max_instances"])
        spent, need = quota.spent_today(self.ledger.rows(), self.now()), price * q.reserve_hours
        if spent + need > q.max_cny_per_day:
            raise CloudGpuError("over_quota", f"the day's allowance is CNY {q.max_cny_per_day:g}; about {spent:.2f} is spent (an upper bound) and one more instance needs {need:.2f} set aside",
                                ["wait for tomorrow, or a person raises max_cny_per_day"])
        return price

    # ---- create ----
    def create(self, spec: dict, listing=None) -> dict:
        """Create one instance. For the agent: only inside the allowance. The attempt is in the ledger before the call (so a ledger that cannot be written stops it, not a bill
        with no record), the instance and its price after. CloudGpuError(no_stock) when the vendor has no card of that spec there now."""
        problems = self.vendor.create_problems(spec)
        if problems:
            raise CloudGpuError("key_missing", "the request is not one the vendor takes: " + "; ".join(problems))
        listing = self.vendor.all() if listing is None else listing
        mine = self.sync(listing)
        assumed = self.check_create(spec, mine) if self.agent else self.quota.prices.get(spec.get("gpu_spec_uuid"), 0.0)
        centers = spec.get("data_center_list")
        try:
            self._log("create-attempt", spec=spec.get("gpu_spec_uuid"), data_center=",".join(centers or []))
        except OSError as e:
            raise CloudGpuError("key_missing", f"the ledger cannot be written ({e.strerror}): nothing was created") from None
        try:
            iid = self.vendor.create(spec)
        except CloudGpuError as e:
            if self.vendor.is_no_stock(e):
                raise CloudGpuError("no_stock", f"no {spec.get('gpu_spec_uuid')} in stock" + (f" at {','.join(centers)}" if centers else ""), status=e.status, detail=e.detail) from None
            raise
        price, source = assumed or None, "quota"
        try:
            snap = self.vendor.snapshot(iid) or {}
            if snap.get("price_cny_per_hour"):
                price, source = snap["price_cny_per_hour"], "snapshot"
        except CloudGpuError:
            pass
        row = {"instance": iid, "spec": spec.get("gpu_spec_uuid"), "data_center": ",".join(centers or []), "price_cny_per_hour": price, "price_source": source}
        try:
            self._log("create", **row)
        except OSError as e:
            raise CloudGpuError("key_missing", f"instance {iid} WAS created but the ledger cannot be written ({e.strerror}): record it by hand, it is billing") from None
        out = {"created": iid, **{k: row[k] for k in ("spec", "data_center", "price_cny_per_hour", "price_source")}}
        if self.agent and price and assumed and price > assumed:
            out["warning"] = f"the real price {price} is above the {assumed} the allowance assumed: a person should raise it"
        return out

    # ---- release ----
    def check_release(self, instance_id: str, listing) -> None:
        """For the agent: only an instance the ledger shows it created, still listed, and not wired to a box in the settings. CloudGpuError(not_allowed) otherwise."""
        if not self.agent:
            return
        mine = self.sync(listing)
        row = mine.get(instance_id)
        if row is None or not row.get("agent"):
            raise CloudGpuError("not_allowed", f"{instance_id} was not created by the agent (the ledger has no such line): a person's box is released by a person",
                                ["tell the person; the agent never releases a box it did not make"])
        if instance_id in pool_instances(self.env):
            raise CloudGpuError("not_allowed", f"{instance_id} is wired to a box in the settings (CLOUDGPU_INSTANCE): a take may be running on it",
                                ["a person unwires it first, then releases it"])

    def release(self, instance_id: str) -> dict:
        self.vendor.release(instance_id)
        self._log("release", instance=instance_id)
        return {"done": "release", "id": instance_id}

    # ---- ensure ----
    def ensure(self, count: int, template: dict, *, specs=None, centers=None, yes: bool = False, patience: float = PATIENCE_S, delays=BACKOFF_S, boot_wait: float = BOOT_WAIT_S) -> dict:
        """Have `count` of the boxes this setup knows (the ledger's, and the ones wired in the settings) running. Order: instances already running count; then those that are off and
        billed by the hour are powered on (a pass that is off is a person's to renew); then, inside the allowance, new ones are created from `template` for each card spec in `specs`,
        in order, at each data center in `centers`, in order. A card that is not in stock is tried again after a pause (`delays`, at most `patience` seconds in all); every other
        failure is reported once and not retried. Without `yes` nothing is changed: the plan is returned. The result says how many are up, how many are missing, and each failure's reason."""
        specs = list(specs or DEFAULT_SPECS)
        centers = list(centers or [])
        listing = self.vendor.all()
        mine = self.sync(listing)
        known = pool_instances(self.env) | set(mine)
        have = [i for i in listing if i.id in known]
        running = [i.id for i in have if i.status == "running"]
        off = [i for i in have if i.status in OFF_STATUS]
        pass_off = [i.id for i in off if i.billing != "payg"]
        power_cands = [i for i in off if i.billing == "payg"]
        unclear = [{"id": i.id, "status": i.status} for i in have if i.status != "running" and i.status not in OFF_STATUS]
        need = max(0, count - len(running))
        combos = [(s, c) for s in specs for c in (centers or [None])]
        out: dict = {"wanted": count, "running": running, "powered_on": [], "created": [], "failures": [], "unclear": unclear}
        notes: list[str] = []
        if pass_off:
            notes.append(f"{', '.join(pass_off)} is off and billed as a pass (not by the hour): it may have ended; a person renews it at the vendor")
        if unclear:
            notes.append("an instance in a state this package does not know is neither counted nor touched: look at it with `instances snapshot ID`")
        if not yes:
            plan: dict = {"dry_run": True, "would_power_on": [i.id for i in power_cands[:need]]}
            short = need - len(plan["would_power_on"])
            plan["would_create"] = short
            if short > 0:
                plan["order"] = [{"gpu_spec_uuid": s, "data_center": c} for s, c in combos]
                try:
                    self.check_create({**template, "gpu_spec_uuid": specs[0]}, mine)
                    plan["allowance"] = "covers one more"
                except CloudGpuError as e:
                    plan["allowance"] = e.message
            return {**out, **plan, "missing": short if short > 0 else 0, "next": ["pass --yes to do it"] + notes}
        fails: dict[tuple, dict] = {}

        def fail(what: str, where: str, e: CloudGpuError) -> None:
            f = fails.setdefault((what, where, e.kind), {"what": what, "where": where, "kind": e.kind, "reason": e.message[:240], "tries": 0})
            f["tries"] += 1

        pending, creating, waited, rnd = list(power_cands), True, 0.0, 0
        while need > 0:
            stock_miss = False
            for i in list(pending):
                if need <= 0:
                    break
                try:
                    self.vendor.power_on(i.id, START_COMMAND)
                except CloudGpuError as e:
                    if self.vendor.is_no_stock(e):
                        e = CloudGpuError("no_stock", f"no free card to power {i.id} on now", status=e.status, detail=e.detail)
                        stock_miss = True
                    else:
                        pending.remove(i)
                    fail("power-on", i.id, e)
                    continue
                pending.remove(i)
                self._log("power-on", instance=i.id)
                out["powered_on"].append(i.id)
                need -= 1
            for s, c in combos if creating else []:
                while need > 0:
                    spec = {**template, "gpu_spec_uuid": s, **({"data_center_list": [c]} if c else {})}
                    spec.setdefault("start_command", START_COMMAND)
                    try:
                        r = self.create(spec, listing=listing)
                    except CloudGpuError as e:
                        fail("create", f"{s}@{c or 'any'}", e)
                        if e.kind == "no_stock":
                            stock_miss = True
                        elif e.kind in ("over_quota", "key_missing"):
                            creating = False                # the allowance, or a request the vendor refuses outright, is the same for every spec and place
                        break
                    out["created"].append(r)
                    listing = self.vendor.all()
                    need -= 1
                if not creating or need <= 0:
                    break
            if need <= 0 or not stock_miss or rnd >= len(delays) or waited + delays[rnd] > patience:
                break
            self.sleep(delays[rnd])
            waited += delays[rnd]
            rnd += 1
        out["waited_s"] = waited
        out["failures"] = list(fails.values())
        out["missing"] = need
        self._await_running(out, boot_wait)
        out["next"] = notes + self._next(out)
        return out

    def _await_running(self, out: dict, boot_wait: float) -> None:
        """A new or powered-on instance is waited for to say `running` (at most `boot_wait` seconds in all), then its SSH door is read. A secret is never in the snapshot."""
        ids = [c["created"] for c in out["created"]] + out["powered_on"]
        waited, up = 0.0, set()
        while ids and len(up) < len(ids):
            for i in ids:
                if i not in up:
                    try:
                        if self.vendor.status(i) == "running":
                            up.add(i)
                    except CloudGpuError:
                        pass
            if len(up) == len(ids) or waited >= boot_wait:
                break
            self.sleep(10)
            waited += 10
        out["now_running"] = sorted(up)
        out["not_yet_running"] = [i for i in ids if i not in up]
        doors = {}
        for i in up:
            try:
                s = self.vendor.snapshot(i) or {}
                doors[i] = {"ssh": s.get("ssh"), "price_cny_per_hour": s.get("price_cny_per_hour")}
            except CloudGpuError:
                pass
        if doors:
            out["doors"] = doors

    @staticmethod
    def _next(out: dict) -> list[str]:
        nxt = []
        if out["created"]:
            nxt.append("a new instance carries only the image: the adapter and the voice service are set up on it (docs/cloud-gpu.md, 'A new box: what is automatic and what is not') "
                       "before it takes a take; then wire it in the settings (CLOUDGPU_<NAME>_INSTANCE, _SSH, _BASE, _TOKEN) and run `python3 -m cloudgpu status`")
        if out["missing"]:
            kinds = {f["kind"] for f in out["failures"]}
            if "no_stock" in kinds:
                nxt.append(f"{out['missing']} short: the vendor had no free card at the places and specs tried; run `ensure` again later, add another spec or data center, or a person rents one")
            if "over_quota" in kinds:
                nxt.append("the allowance stopped further creation: a person raises it (quota set) or releases an instance the agent made")
            if kinds - {"no_stock", "over_quota"}:
                nxt.append("some failures are not stock: read their reasons; retrying will not change them")
        return nxt
