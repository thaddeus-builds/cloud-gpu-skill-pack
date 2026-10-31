"""python3 -m cloudgpu COMMAND: status (read only), hold / release (a person's hold on a box), instances (the cloud account: list, status, snapshot read; power, create, release only with --yes), price, bench (a paid take), bundle (what to copy onto a box)."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE_FILES = {"adapter.py": HERE / "adapter.py", "tts_worker.py": HERE / "onbox" / "tts_worker.py", "detect_models.py": HERE / "detect_models.py", "install.sh": HERE / "onbox" / "install.sh",
                "tunnel.sh": HERE / "onbox" / "tunnel.sh", "models.4090.json": HERE / "onbox" / "models.4090.json",
                "models.pro6000.json": HERE / "onbox" / "models.pro6000.json"}


def bundle(out: str) -> list[str]:
    """A flat tar.gz of exactly the files a box needs: `tar xzf` it there and `sh install.sh auto 6009 < token-file`."""
    with tarfile.open(out, "w:gz") as t:
        for name, src in BUNDLE_FILES.items():
            info = t.gettarinfo(str(src), arcname=name)
            info.mode = 0o755 if name.endswith(".sh") else 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with open(src, "rb") as f:
                t.addfile(info, io.BytesIO(f.read()))
    return list(BUNDLE_FILES)


def instances(a, out=print) -> int:
    """The cloud account's instances. Reading (list, status, snapshot, specs) is free. Anything that changes the account (power-on, power-off, create, release) prints what it
    would do and stops unless `--yes` is given: a person gives it; create STARTS BILLING and release DESTROYS THE INSTANCE'S DATA. No secret is ever printed."""
    from . import config, vendors
    from .errors import CloudGpuError
    name = a.vendor or config.get("VENDOR") or (config.get("POWER") if config.get("POWER") not in (None, "none") else None) or "autodl"
    try:
        v = vendors.get(name)
        c = a.icmd
        if c == "specs":
            out(json.dumps(v.gpu_specs(), ensure_ascii=False))
            return 0
        if not v.configured() and c != "create":
            raise CloudGpuError("key_missing", f"the {name} vendor needs {v.token_env}")
        if c == "list":
            out(json.dumps([i.as_dict() for i in v.all()], ensure_ascii=False))
        elif c == "status":
            out(json.dumps({"id": a.id, "status": v.status(a.id)}))
        elif c == "snapshot":
            out(json.dumps(v.snapshot(a.id), ensure_ascii=False))
        elif c == "ensure":
            return ensure_cmd(a, v, out)
        elif c in ("power-on", "power-off"):
            if a.agent:
                st0 = _steward(v, a)
                mine = st0.ledger.created(v.name)
                from .steward import pool_instances
                if a.id not in mine and not (c == "power-on" and a.id in pool_instances(os.environ)):
                    raise CloudGpuError("not_allowed", f"{a.id} is not a box this agent made" + (" or one wired in the settings" if c == "power-on" else "") + f": a person does {c} on it",
                                        ["tell the person; the agent touches only its own boxes"])
            if not a.yes:
                out(json.dumps({"dry_run": True, "would": c, "id": a.id, "next": [f"pass --yes (a person does) to {c}" + (": it starts the hourly billing" if c == "power-on" else "")]}))
                return 0
            v.power_on(a.id) if c == "power-on" else v.power_off(a.id)
            st, done = _steward(v, a), {"done": c, "id": a.id}
            try:
                st.ledger.append(c, vendor=v.name, instance=a.id, actor=st.actor, agent=bool(a.agent))
            except OSError:
                done["ledger"] = "not written"
            out(json.dumps(done))
        elif c == "create":
            spec = json.loads(sys.stdin.read() if a.spec == "-" else Path(a.spec).read_text())
            problems = v.create_problems(spec)
            if problems:
                out(json.dumps({"error": "the request is not one the vendor takes", "problems": problems}, ensure_ascii=False))
                return 2
            st = _steward(v, a)
            if not a.yes:
                verdict = "the allowance covers it"
                if a.agent:
                    try:
                        st.check_create(spec, st.sync(v.all()))
                    except CloudGpuError as e:
                        verdict = e.message
                out(json.dumps({"dry_run": True, "would": "create", "request": spec, **({"allowance": verdict} if a.agent else {}),
                                "next": ["pass --yes to create: it STARTS BILLING at once" + (" (inside the allowance)" if a.agent else " (a person does)")]}, ensure_ascii=False))
                return 0
            out(json.dumps({**st.create(spec), "next": ["python3 -m cloudgpu instances snapshot ID for the price and the SSH door"]}, ensure_ascii=False))
        elif c == "release":
            st = v.status(a.id)
            if st != "shutdown":
                out(json.dumps({"error": f"the instance is {st}, not shutdown: power it off first", "id": a.id}))
                return 2
            st = _steward(v, a)
            st.check_release(a.id, v.all() if a.agent else [])
            if not a.yes:
                out(json.dumps({"dry_run": True, "would": "release", "id": a.id, "next": ["pass --yes to release: it DESTROYS THE INSTANCE'S DATA" + ("" if a.agent else " (a person does)")]}))
                return 0
            out(json.dumps(st.release(a.id)))
        return 0
    except CloudGpuError as e:
        out(json.dumps({"error": e.message, "kind": e.kind, "next": e.next, **({"status": e.status} if e.status is not None else {})}, ensure_ascii=False))
        return 1


def _steward(v, a):
    import os
    from .steward import Steward
    return Steward(v, os.environ, agent=bool(getattr(a, "agent", False)))


def ensure_cmd(a, v, out) -> int:
    """`instances ensure --count N --spec TEMPLATE`: up to N of the boxes this setup knows running, powered on or created inside the allowance; the report says how many and why not."""
    import os
    from . import config
    from .steward import DEFAULT_CENTERS, DEFAULT_SPECS, Steward, csv
    template = json.loads(sys.stdin.read() if a.spec == "-" else Path(a.spec).read_text())
    template.pop("gpu_spec_uuid", None)
    template.pop("data_center_list", None)
    template.setdefault("req_gpu_amount", 1)
    specs = csv(a.specs or config.get("ENSURE_SPECS"), DEFAULT_SPECS)
    centers = csv(a.data_centers or config.get("ENSURE_DATA_CENTERS"), DEFAULT_CENTERS)
    for s in specs:
        problems = v.create_problems({**template, "gpu_spec_uuid": s, "data_center_list": centers})
        if problems:
            out(json.dumps({"error": "the request is not one the vendor takes", "problems": problems}, ensure_ascii=False))
            return 2
    st = Steward(v, os.environ, agent=True)
    r = st.ensure(a.count, template, specs=specs, centers=centers, yes=a.yes, patience=a.patience)
    out(json.dumps(r, ensure_ascii=False))
    return 0 if not r["missing"] else 1


def quota_cmd(a) -> int:
    import os
    from . import quota
    from .errors import CloudGpuError
    try:
        if a.qcmd == "show":
            st = _quota_view(os.environ)
            print(json.dumps(st, ensure_ascii=False))
            return 0
        q = quota.load(os.environ)
        if q.problem and not a.force:
            print(json.dumps({"error": q.problem, "next": ["chmod go-w the file, or pass --force to write a new one"]}))
            return 2
        import math
        if any(x is not None and not math.isfinite(x) for x in (a.max_cny_per_day, a.reserve_hours)):
            print(json.dumps({"error": "the numbers must be finite"}))
            return 2
        for field_, val in (("max_instances", a.max_instances), ("max_cny_per_day", a.max_cny_per_day), ("note", a.note), ("reserve_hours", a.reserve_hours)):
            if val is not None:
                setattr(q, field_, val)
        if a.price:
            q.prices = {**q.prices, **quota.parse_price(a.price)}
        q.problem, q.set_by, q.set_at = "", a.by, time_now()
        if q.max_instances < 0 or q.max_cny_per_day < 0:
            print(json.dumps({"error": "the numbers cannot be negative"}))
            return 2
        p = quota.save(q, os.environ)
        quota.Ledger(quota.ledger_path(os.environ)).append("quota-set", actor=a.by, agent=False, **{k: v for k, v in q.as_dict().items() if k != "agent_may_create"})
        print(json.dumps({**q.as_dict(), "file": str(p), "next": [f"chmod go-w {p}; keep it out of the agent's reach (docs, 'An allowance for an agent')"]}, ensure_ascii=False))
        return 0
    except CloudGpuError as e:
        print(json.dumps({"error": e.message, "kind": e.kind}, ensure_ascii=False))
        return 2


def time_now() -> str:
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _quota_view(env) -> dict:
    import time
    from . import quota
    q = quota.load(env)
    led = quota.Ledger(quota.ledger_path(env))
    rows = led.rows()
    return {**q.as_dict(), "instances_counted_against_the_allowance": sorted(led.created()), "create_outcomes_unknown": len(led.unresolved()), "spent_today_upper_bound_cny": round(quota.spent_today(rows, time.time()), 2),
            "quota_file": str(quota.quota_path(env)), "ledger_file": str(quota.ledger_path(env))}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(prog="python3 -m cloudgpu", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("status", help="where the box stands (read only)")
    st.add_argument("--env-file", help="read the settings from this KEY=value file instead of the environment (it warns when other users can read it)")
    ins = sub.add_parser("instances", help="the cloud account: list, status, snapshot, specs (read); power-on, power-off, create, release (only with --yes)")
    ins.add_argument("--vendor", help="a cloud vendor (default: CLOUDGPU_VENDOR, CLOUDGPU_POWER, else autodl)")
    isub = ins.add_subparsers(dest="icmd", required=True)
    isub.add_parser("list")
    isub.add_parser("specs")
    for c in ("status", "snapshot", "power-on", "power-off", "release"):
        q = isub.add_parser(c)
        q.add_argument("id")
        if c in ("power-on", "power-off", "release"):
            q.add_argument("--yes", action="store_true")
            q.add_argument("--agent", action="store_true", help="an agent's call: release only touches an instance this agent created (ledger)")
    q = isub.add_parser("create")
    q.add_argument("--spec", required=True, help="a JSON file of the vendor's create request, or - for stdin")
    q.add_argument("--yes", action="store_true")
    q.add_argument("--agent", action="store_true", help="an agent's call: refused unless the allowance (quota set) covers it")
    q = isub.add_parser("ensure", help="have N boxes up: power on the ones that are off, create what is missing inside the allowance, retry when no card is free, report what failed")
    q.add_argument("--count", type=int, required=True)
    q.add_argument("--spec", required=True, help="a JSON file of the create request WITHOUT the card spec and data center (they come from --specs and --data-centers), or - for stdin")
    q.add_argument("--specs", help="card specs in order of preference, comma separated (default: CLOUDGPU_ENSURE_SPECS, else v-48g = 4090 48G)")
    q.add_argument("--data-centers", help="data centers in order, comma separated (default: CLOUDGPU_ENSURE_DATA_CENTERS, else westDC3,beijingDC2)")
    q.add_argument("--patience", type=float, default=600, help="seconds to keep trying when no card is in stock (default 600)")
    q.add_argument("--yes", action="store_true", help="do it; without it the plan is printed and nothing changes")
    qs = sub.add_parser("quota", help="the allowance an agent may rent boxes inside: show (free), set (a person; records who and when)")
    qsub = qs.add_subparsers(dest="qcmd", required=True)
    qsub.add_parser("show")
    qq = qsub.add_parser("set")
    qq.add_argument("--by", required=True, help="who sets it (recorded with the time)")
    qq.add_argument("--max-instances", type=int)
    qq.add_argument("--max-cny-per-day", type=float)
    qq.add_argument("--price", action="append", default=[], help="SPEC=CNY_PER_HOUR the allowance assumes, repeatable (v-48g=3.2)")
    qq.add_argument("--reserve-hours", type=float)
    qq.add_argument("--note")
    qq.add_argument("--force", action="store_true")
    ho = sub.add_parser("hold", help="hold a box for a time so no take starts on it (Pool.take waits or uses another box); ends by itself")
    ho.add_argument("box", help="the box's name in CLOUDGPU_POOL, or default for a pool of one")
    ho.add_argument("--minutes", type=float, required=True)
    ho.add_argument("--who", required=True, help="who holds it, so the next person knows whom to ask")
    ho.add_argument("--why", default="")
    rl = sub.add_parser("release", help="end a hold now")
    rl.add_argument("box")
    p = sub.add_parser("price", help="price rows from the recorded runs")
    p.add_argument("--box")
    p.add_argument("--runs")
    p.add_argument("--provider", default="h3host")
    p.add_argument("--format", choices=("json", "tsv"), default="json")
    sub.add_parser("bench", help="time one take (a paid take); arguments after: ENVFILE NAME WORKFLOW RES SECONDS IMAGE [AUDIO]")
    b = sub.add_parser("bundle", help="make the tar.gz to copy onto a box")
    b.add_argument("out", nargs="?", default="cloudgpu-box.tgz")
    a, rest = ap.parse_known_args(argv)
    if a.cmd == "instances":
        return instances(a)
    if a.cmd == "quota":
        return quota_cmd(a)
    if a.cmd == "status":
        from . import config
        from .status import status
        env = None
        if a.env_file:
            try:
                env = config.read_env_file(a.env_file)
            except OSError as e:
                print(json.dumps({"error": f"cannot read the settings file: {e.strerror}"}))
                return 2
        print(json.dumps(status(env, a.env_file), ensure_ascii=False))
        return 0
    if a.cmd in ("hold", "release"):
        from . import hold
        from .errors import CloudGpuError
        try:
            sd = hold.state_dir()
            if a.cmd == "release":
                print(json.dumps({"released": hold.release(sd, a.box), "box": hold.box_key(a.box)}))
            else:
                h = hold.place(sd, a.box, a.minutes, a.who, a.why)
                print(json.dumps({**h, "until_local": hold.clock(h["until"]), "next": [f"python3 -m cloudgpu release {h['box']} when you are done; it ends by itself at {hold.clock(h['until'])}"]}, ensure_ascii=False))
            return 0
        except CloudGpuError as e:
            print(json.dumps({"error": e.message, "kind": e.kind, "next": e.next}, ensure_ascii=False))
            return 2
        except OSError as e:
            print(json.dumps({"error": f"cannot write the state directory: {e.strerror}"}))
            return 2
    if a.cmd == "price":
        from . import price
        obs = price.rows(price.read_runs(a.runs or price.RUNS), a.box, provider=a.provider)
        if not obs:
            print(json.dumps({"error": "no warm runs for that box"}))
            return 2
        if a.format == "tsv":
            for r in obs:
                print("\t".join(str(r[k]) for k in ("provider", "kind", "model", "unit", "value")) + "\tpending\t" + r["source"] + "\t" + r["read_on"])
        else:
            print(json.dumps(obs, ensure_ascii=False))
        return 0
    if a.cmd == "bench":
        from .bench import main as bench_main
        return bench_main(rest)
    names = bundle(a.out)
    print(json.dumps({"bundle": a.out, "files": names, "next": [f"scp {a.out} user@box:~", "on the box: tar xzf " + Path(a.out).name + " && sh install.sh auto 6009 < token-file"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
