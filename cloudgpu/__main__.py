"""python3 -m cloudgpu COMMAND: status (read only), instances (the cloud account: list, status, snapshot read; power, create, release only with --yes), price, bench (a paid take), bundle (what to copy onto a box)."""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUNDLE_FILES = {"adapter.py": HERE / "adapter.py", "detect_models.py": HERE / "detect_models.py", "install.sh": HERE / "onbox" / "install.sh",
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
        elif c in ("power-on", "power-off"):
            if not a.yes:
                out(json.dumps({"dry_run": True, "would": c, "id": a.id, "next": [f"pass --yes (a person does) to {c}" + (": it starts the hourly billing" if c == "power-on" else "")]}))
                return 0
            v.power_on(a.id) if c == "power-on" else v.power_off(a.id)
            out(json.dumps({"done": c, "id": a.id}))
        elif c == "create":
            spec = json.loads(sys.stdin.read() if a.spec == "-" else Path(a.spec).read_text())
            problems = v.create_problems(spec)
            if problems:
                out(json.dumps({"error": "the request is not one the vendor takes", "problems": problems}, ensure_ascii=False))
                return 2
            if not a.yes:
                out(json.dumps({"dry_run": True, "would": "create", "request": spec, "next": ["pass --yes (a person does) to create: it STARTS BILLING at once"]}, ensure_ascii=False))
                return 0
            out(json.dumps({"created": v.create(spec), "next": ["python3 -m cloudgpu instances snapshot ID for the price and the SSH door"]}))
        elif c == "release":
            st = v.status(a.id)
            if st != "shutdown":
                out(json.dumps({"error": f"the instance is {st}, not shutdown: power it off first", "id": a.id}))
                return 2
            if not a.yes:
                out(json.dumps({"dry_run": True, "would": "release", "id": a.id, "next": ["pass --yes (a person does) to release: it DESTROYS THE INSTANCE'S DATA"]}))
                return 0
            v.release(a.id)
            out(json.dumps({"done": "release", "id": a.id}))
        return 0
    except CloudGpuError as e:
        out(json.dumps({"error": e.message, "kind": e.kind, "next": e.next, **({"status": e.status} if e.status is not None else {})}, ensure_ascii=False))
        return 1


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
    q = isub.add_parser("create")
    q.add_argument("--spec", required=True, help="a JSON file of the vendor's create request, or - for stdin")
    q.add_argument("--yes", action="store_true")
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
