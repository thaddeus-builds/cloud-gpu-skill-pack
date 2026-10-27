"""The tiny test harness every tests/test_*.py ends with: `if __name__ == "__main__": _t.main(globals())`. `main` runs each `test_*` function the file
defines, in file order; a failure prints `FAIL <name>` and its traceback; a file where every test passed prints `RESULT: N passed` and exits 0.
Importing this puts the repo root on sys.path. Helpers: tmpdir(), raises(kind) (the block must raise CloudGpuError of that kind), skip(why), patient(s)."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@contextmanager
def tmpdir():
    d = tempfile.mkdtemp(prefix="cloudgpu-test-")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


class _Caught:
    err = None


@contextmanager
def raises(kind: str):
    from cloudgpu.errors import CloudGpuError
    caught = _Caught()
    try:
        yield caught
    except CloudGpuError as e:
        assert e.kind == kind, f"expected CloudGpuError kind {kind!r}, got {e.kind!r}: {e}"
        caught.err = e
        return
    raise AssertionError(f"expected CloudGpuError kind {kind!r}, nothing was raised")


def skip(why: str) -> None:
    print(f"SKIP: {why}")


def patient(seconds: float) -> float:
    return seconds * float(os.environ.get("CLOUDGPU_TEST_PATIENCE", "1"))


def main(g: dict) -> None:
    if not __debug__:
        print("FAILED: python -O skips every assert; run without -O")
        sys.exit(1)
    tests = [(n, f) for n, f in g.items() if n.startswith("test_") and callable(f) and getattr(f, "__module__", None) == g.get("__name__")]
    if not tests:
        print("FAILED: no test_* function found")
        sys.exit(1)
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    if failed:
        print(f"FAILED: {failed} of {len(tests)}")
        sys.exit(1)
    print(f"RESULT: {len(tests)} passed")
