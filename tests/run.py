"""Run every tests/test_*.py as its own process (python3 tests/run.py [NAME ...]): prints one line per file and the total; exit 1 if any failed."""

import glob
import os
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
names = sys.argv[1:]
files = sorted(f for f in glob.glob(os.path.join(here, "test_*.py")) if not names or any(n in os.path.basename(f) for n in names))
bad = 0
for f in files:
    r = subprocess.run([sys.executable, f], capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
    ok = r.returncode == 0 and last.startswith("RESULT")
    print(("ok   " if ok else "FAIL ") + os.path.basename(f).ljust(28) + last)
    if not ok:
        bad += 1
        print((r.stdout + r.stderr)[-1500:])
print("FAILED: %d of %d files" % (bad, len(files)) if bad else "ALL PASSED: %d files" % len(files))
sys.exit(1 if bad else 0)
