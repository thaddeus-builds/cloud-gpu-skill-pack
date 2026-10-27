"""cloudgpu hold / release and the holds in `status`: the commands a person types, JSON out, the state directory from CLOUDGPU_STATE_DIR. No network."""

import contextlib
import io
import json
import os

import _t

from cloudgpu.__main__ import main
from cloudgpu.status import status


def run(argv, sd):
    out = io.StringIO()
    old = os.environ.get("CLOUDGPU_STATE_DIR")
    os.environ["CLOUDGPU_STATE_DIR"] = sd
    try:
        with contextlib.redirect_stdout(out):
            code = main(argv)
    finally:
        os.environ.pop("CLOUDGPU_STATE_DIR") if old is None else os.environ.__setitem__("CLOUDGPU_STATE_DIR", old)
    return code, json.loads(out.getvalue())


def test_hold_and_release_from_the_command_line():
    with _t.tmpdir() as sd:
        code, doc = run(["hold", "a", "--minutes", "30", "--who", "pat", "--why", "manual test"], sd)
        assert code == 0 and doc["box"] == "a" and doc["who"] == "pat" and "release a" in doc["next"][0], doc
        code, doc = run(["hold", "a", "--minutes", "5", "--who", "sam"], sd)
        assert code == 2 and "held by pat" in doc["error"], doc                     # not taken over
        code, doc = run(["hold", "a", "--minutes", "9999", "--who", "pat"], sd)
        assert code == 2 and "at most" in doc["error"], doc
        assert run(["release", "a"], sd) == (0, {"released": True, "box": "a"})
        assert run(["release", "a"], sd) == (0, {"released": False, "box": "a"})
        assert run(["hold", "--minutes", "5", "--who", "pat", "default"], sd)[1]["box"] == "default"


def test_status_lists_the_active_holds_without_writing_or_removing_anything():
    with _t.tmpdir() as sd:
        run(["hold", "a", "--minutes", "30", "--who", "pat", "--why", "manual test"], sd)
        doc = status({"CLOUDGPU_STATE_DIR": sd})
        assert [h["box"] for h in doc["holds"]] == ["a"] and any("held by pat" in n and "manual test" in n for n in doc["next"]), doc
        assert status({"CLOUDGPU_STATE_DIR": sd + "/nowhere"})["holds"] == []


if __name__ == "__main__":
    _t.main(globals())
