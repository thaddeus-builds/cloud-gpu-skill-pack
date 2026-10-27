"""What a second of video costs on a box, from its recorded runs: `python3 -m cloudgpu price [--box ID] [--format json|tsv]`.

`cloudgpu/data/runs.jsonl` holds one line per measured take: box, card, hourly_cny (the box's price per hour), workflow, resolution, seconds (of
video), machine_s (seconds the machine took: the adapter's own task record, `started` to `finished`), `cold: true` on the first take after a
start (it loads the models and is left out), date. For every (box, workflow, resolution) the WORST warm run counts: cost per video second =
machine_s / seconds * hourly_cny / 3600, at full use (an idle hour is paid on top). The row's model is `<workflow>@<resolution>#<box>`. This module
writes nothing: a consumer turns the rows into its own price table (as pending: a person confirms a price).
"""

from __future__ import annotations

import json
from pathlib import Path

RUNS = Path(__file__).resolve().parent / "data" / "runs.jsonl"


def read_runs(path: str | Path = RUNS) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def rows(runs: list[dict], box: str | None = None, *, provider: str = "h3host", kind: str = "avatar") -> list[dict]:
    """The observed price rows, one per (box, workflow, resolution), from the worst warm run of each."""
    worst: dict[tuple, dict] = {}
    for r in runs:
        if r.get("cold") or (box and r["box"] != box):
            continue
        key = (r["box"], r["workflow"], r["resolution"])
        per_s = r["machine_s"] / r["seconds"]
        if key not in worst or per_s > worst[key]["per_s"]:
            worst[key] = {**r, "per_s": per_s}
    out = []
    for (b, wf, res), r in sorted(worst.items()):
        out.append({"provider": provider, "kind": kind, "model": f"{wf}@{res}#{b}", "unit": "per_second",
                    "value": round(r["per_s"] * r["hourly_cny"] / 3600, 4), "read_on": r["date"],
                    "source": f"measured {r['date']} on {r['card']} at {r['hourly_cny']:g} CNY/h: {r['per_s']:.1f} machine s per video s (the worst "
                              f"warm run: {r['seconds']} s of video in {r['machine_s']:g} s); full use, an idle hour is paid on top; the owner confirms"})
    return out
