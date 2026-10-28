"""cloudgpu.price: a box's price rows come from its recorded runs (cloudgpu/data/runs.jsonl): the worst warm run of each workflow and resolution at
full use; nothing is written here; the CLI prints json or tsv rows."""

import json
import subprocess
import sys
from pathlib import Path

import _t

from cloudgpu import price


def run(box, wf, res, secs, machine_s, hourly=3.6, **more):
    return {"box": box, "card": "X", "hourly_cny": hourly, "workflow": wf, "resolution": res, "seconds": secs, "machine_s": machine_s, "date": "2026-10-23", **more}


def test_the_cost_is_machine_seconds_per_video_second_at_the_hourly_price_and_the_worst_warm_run_counts():
    runs = [run("b1", "w", "480p", 8, 400.0, cold=True),                  # the first take loads the models: left out
            run("b1", "w", "480p", 8, 80.0), run("b1", "w", "480p", 15, 300.0),           # 10 and 20 machine s per video s: the worst is 20
            run("b1", "w", "768p", 8, 160.0), run("b2", "w", "480p", 8, 80.0, hourly=7.2)]
    got = {r["model"]: r["value"] for r in price.rows(runs)}
    assert got == {"w@480p#b1": 0.02, "w@768p#b1": 0.02, "w@480p#b2": 0.02}, got        # 20 s/s * 3.6 CNY/h / 3600 = 0.02; 10 * 7.2 / 3600 = 0.02
    assert [r["model"] for r in price.rows(runs, "b2")] == ["w@480p#b2"]
    assert all(r["provider"] == "h3host" and r["kind"] == "avatar" and r["unit"] == "per_second" for r in price.rows(runs))
    assert price.rows(runs, provider="mine")[0]["provider"] == "mine"
    assert price.rows([run("b1", "w", "480p", 8, 80.0, cold=True)]) == []                 # only a cold run: no row at all


def test_the_shipped_runs_are_well_formed_and_every_box_has_a_warm_run():
    runs = price.read_runs()
    assert runs and all({"box", "card", "hourly_cny", "workflow", "resolution", "seconds", "machine_s", "date"} <= set(r) for r in runs)
    assert {r["box"] for r in runs} == {r["box"] for r in runs if not r.get("cold")}
    assert {r["model"].split("#")[1] for r in price.rows(runs)} == {"rtx4090", "rtx-pro6000"}


def test_the_cli_prints_json_or_tsv_and_refuses_a_box_with_no_runs():
    cmd = [sys.executable, "-m", "cloudgpu", "price", "--box", "rtx4090"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=_t.ROOT)
    assert r.returncode == 0 and all(x["model"].endswith("#rtx4090") for x in json.loads(r.stdout))
    r = subprocess.run(cmd + ["--format", "tsv"], capture_output=True, text=True, cwd=_t.ROOT)
    assert r.returncode == 0 and all(line.split("\t")[5] == "pending" for line in r.stdout.strip().splitlines())
    assert subprocess.run([sys.executable, "-m", "cloudgpu", "price", "--box", "nobody"], capture_output=True, text=True, cwd=_t.ROOT).returncode == 2


def test_the_bundle_holds_exactly_what_a_box_needs():
    import tarfile
    with _t.tmpdir() as d:
        out = Path(d) / "b.tgz"
        r = subprocess.run([sys.executable, "-m", "cloudgpu", "bundle", str(out)], capture_output=True, text=True, cwd=_t.ROOT)
        assert r.returncode == 0, r.stderr
        with tarfile.open(out) as t:
            names = sorted(t.getnames())
            assert names == sorted(["adapter.py", "tts_worker.py", "detect_models.py", "install.sh", "tunnel.sh", "models.4090.json", "models.pro6000.json"]), names
            assert t.getmember("install.sh").mode & 0o111 and not t.getmember("adapter.py").mode & 0o111


if __name__ == "__main__":
    _t.main(globals())
