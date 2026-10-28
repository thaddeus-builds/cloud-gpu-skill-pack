"""The client of a box's line (adapter.py, `/cloudgpu/line`): the order in which callers on DIFFERENT machines take the box, one at a time.

`join` takes a place, `poll` asks where it is (and renews its lease: the adapter drops a ticket nobody asked about for 60 s, so a caller that died never blocks the line), `leave` gives it up,
`listing` says who runs and who waits. Each returns what the adapter said (`ticket`, `state` "go" or "waiting", `ahead`, `running`), or None when the adapter has no such thing: `join` None means
an adapter older than 0.7.0 with no line (the caller goes as before), `poll` None means the ticket was forgotten (its lease ran out, or the adapter restarted: join again). A box that does not
answer is CloudGpuError(http). The token goes only through `Box.token()`, which checks the local end of the tunnel first."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from . import net
from .errors import CloudGpuError

PATH = "/cloudgpu/line"
TIMEOUT_S = 15


def _call(box, method: str, path: str, body: dict | None = None):
    req = urllib.request.Request(box.root() + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": box.token(), "Content-Type": "application/json"})
    try:
        with net.urlopen(req, timeout=TIMEOUT_S) as r:
            doc = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code in (404, 405, 501):                 # no such ticket, or an adapter that has no line
            return None
        if e.code == 429:
            raise CloudGpuError("http", "the box's line is full (64 callers)", status=429) from None
        raise CloudGpuError("http", f"the box's line answered HTTP {e.code}", status=e.code) from None
    except (urllib.error.URLError, OSError, ValueError):
        raise CloudGpuError("http", "the box's line did not answer") from None
    if not isinstance(doc, dict) or doc.get("code") != "Success":
        raise CloudGpuError("http", "the box's line gave an answer this version does not know")
    return doc.get("data") if method != "DELETE" else {}


def join(box, label: str = "") -> dict | None:
    return _call(box, "POST", PATH, {"label": label})


def poll(box, ticket: str) -> dict | None:
    return _call(box, "GET", f"{PATH}/{ticket}")


def leave(box, ticket: str) -> None:
    _call(box, "DELETE", f"{PATH}/{ticket}")


def listing(box) -> list | None:
    d = _call(box, "GET", PATH)
    return d.get("line") if isinstance(d, dict) else None
