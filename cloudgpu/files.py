"""The client of a box's file store (adapter.py, `/cloudgpu/files`, 0.10.0): a picture sent once and named by its sha256 in every take after it.

A take's body carries its pictures inline as `data:` URIs, so the same character still travels with every take (a 1.3 MB base64 string, twice for a first-and-last-frame take): on a link of 20 KB/s that
is more than a minute of every take, and the box idles while it comes. `ensure(box, path)` asks the box whether it has the file (`GET /cloudgpu/files/<sha256>`), sends the raw bytes once when it does not
(`PUT`, no base64) and returns `sha256:<hex>`, which a take's body may carry wherever it carries a picture or an audio. `supported(box)` is the box's own word (`"files": 1` in /health), so an adapter older
than 0.10.0 is never asked; a consumer then sends the data inline as before. A file the box dropped meanwhile (unused for a week, or the store over its size) is answered `file_unknown`: `ensure` again.
A box that does not answer is CloudGpuError(http). The token goes only through `Box.token()`."""

from __future__ import annotations

import hashlib
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

from . import net
from .errors import CloudGpuError

PATH = "/cloudgpu/files"
TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".wav": "audio/wav", ".mp3": "audio/mpeg"}
ASK_TIMEOUT_S = 15
MIN_PUT_S = 60
PUT_KB_S = 5                                   # the slowest link the wait allows for: a file of N KB waits N / 5 seconds (and at least a minute)
_sums: dict = {}                               # (path, mtime_ns, size) -> sha256 hex
_lock = threading.Lock()


def sha256_of(path) -> str:
    p = Path(path)
    st = p.stat()
    key = (str(p), st.st_mtime_ns, st.st_size)
    with _lock:
        if key not in _sums:
            _sums[key] = hashlib.sha256(p.read_bytes()).hexdigest()
        return _sums[key]


def supported(box) -> bool:
    """True when the box says it keeps files (an adapter of 0.10.0 or newer). The answer is the box's /health, asked each time: it is a small call and a restarted adapter may be an older one."""
    try:
        h = box.health()
    except CloudGpuError:
        return False
    return isinstance(h, dict) and h.get("files") == 1


def _call(box, method: str, sha: str, data: bytes | None = None, ctype: str = "application/json", timeout: float = ASK_TIMEOUT_S):
    req = urllib.request.Request(f"{box.root()}{PATH}/{sha}", method=method, data=data, headers={"Authorization": box.token(), "Content-Type": ctype})
    try:
        with net.urlopen(req, timeout=timeout) as r:
            doc = json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 404 and method == "GET":
            return None
        try:
            why = json.loads(e.read()).get("msg", "")
        except (ValueError, OSError):
            why = ""
        raise CloudGpuError("http", f"the box's file store answered HTTP {e.code}" + (f": {why}" if why else ""), status=e.code) from None
    except (urllib.error.URLError, OSError, ValueError):
        raise CloudGpuError("http", "the box's file store did not answer") from None
    if not isinstance(doc, dict) or doc.get("code") != "Success":
        raise CloudGpuError("http", "the box's file store gave an answer this version does not know")
    return doc.get("data")


def has(box, sha: str) -> bool:
    return _call(box, "GET", sha) is not None


def put(box, path) -> str:
    """Send the file's bytes to the box under their sha256; returns the sha."""
    p = Path(path)
    ctype = TYPES.get(p.suffix.lower())
    if ctype is None:
        raise CloudGpuError("http", f"{p.name}: only png, jpeg, webp, wav and mp3 go to the file store")
    data = p.read_bytes()
    sha = sha256_of(p)
    _call(box, "PUT", sha, data, ctype, timeout=max(MIN_PUT_S, len(data) / 1024 / PUT_KB_S))
    return sha


def ensure(box, path) -> str:
    """`sha256:<hex>` of the file, after making sure the box has it (sent once, never twice)."""
    sha = sha256_of(path)
    if not has(box, sha):
        put(box, path)
    return f"sha256:{sha}"
