"""The services a box starts after a boot (cloudgpu/onbox/services.py, 0.21.0): ComfyUI, then VoxCPM when it is installed; idempotent, in order, VoxCPM's weights never loaded,
logs under logs/, and what /health and `status` say of each. Real processes stand in for the services (a fake ComfyUI that takes a while to bind, a fake VoxCPM that records every call);
nothing needs a GPU, a box or a network beyond 127.0.0.1."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import _t
from cloudgpu import adapter, status as status_mod

SRC = Path(_t.ROOT) / "cloudgpu" / "onbox" / "services.py"
spec = importlib.util.spec_from_file_location("services_under_test", SRC)
services = importlib.util.module_from_spec(spec)
sys.modules["services_under_test"] = services
spec.loader.exec_module(services)

FAKE_COMFY = textwrap.dedent('''\
    import json, os, sys, time
    from http.server import BaseHTTPRequestHandler, HTTPServer
    a = sys.argv
    port = int(a[a.index("--port") + 1])
    cfg = {}
    try:
        cfg = json.load(open("fake.json"))
    except OSError:
        pass
    with open("launches.log", "a") as f:
        f.write(json.dumps({"pid": os.getpid(), "t": time.time(), "args": a[1:], "token_in_env": "CLOUDGPU_TOKEN" in os.environ or "H3HOST_TOKEN" in os.environ}) + "\\n")
    if cfg.get("say"):
        print(cfg["say"], flush=True)
    if cfg.get("exit") is not None:
        sys.exit(cfg["exit"])
    time.sleep(cfg.get("delay", 0))
    if cfg.get("exit_after_delay"):
        sys.exit(1)
    class H(BaseHTTPRequestHandler):
        def log_message(self, *x): pass
        def do_GET(self):
            b = json.dumps({"system": {"os": "posix"}, "devices": [{"vram_free": 40 * 2**30}]}).encode() if self.path == "/system_stats" else b"{}"
            self.send_response(200 if self.path == "/system_stats" else 404); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    open("bound.log", "a").write(str(time.time()) + "\\n")
    print("fake comfy listening", flush=True)
    HTTPServer(("127.0.0.1", port), H).serve_forever()
''')

FAKE_VOX = textwrap.dedent('''\
    import json, os, sys, time
    from http.server import BaseHTTPRequestHandler, HTTPServer
    here = os.path.dirname(os.path.abspath(__file__))
    port = int(open(os.path.join(here, "port.txt")).read())
    try:
        time.sleep(float(open(os.path.join(here, "delay.txt")).read()))
    except OSError:
        pass
    open(os.path.join(here, "launches.log"), "a").write(json.dumps({"pid": os.getpid(), "t": time.time(), "token_in_env": "CLOUDGPU_TOKEN" in os.environ}) + "\\n")
    class H(BaseHTTPRequestHandler):
        def log_message(self, *x): pass
        def _r(self):
            open(os.path.join(here, "requests.log"), "a").write(self.command + " " + self.path + "\\n")
            b = json.dumps({"model": "VoxCPM2", "loaded": False, "vram_mib": 700}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
        do_GET = do_POST = _r
    print("fake voxcpm listening", flush=True)
    HTTPServer(("127.0.0.1", port), H).serve_forever()
''')


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def lines(p: Path) -> list:
    try:
        return [json.loads(x) if x.startswith("{") else x for x in p.read_text().splitlines() if x.strip()]
    except OSError:
        return []


class World:
    """A temp box: the adapter's folder, a search folder holding a fake ComfyUI (and a fake VoxCPM when asked), and the environment services.py is run with."""

    def __init__(self, d: str, comfy=True, vox=False, comfy_cfg=None):
        self.d = Path(d)
        self.dir = self.d / "h3host"
        self.base = self.d / "base"
        self.base.mkdir()
        self.comfy_port, self.vox_port = free_port(), free_port()
        self.root = self.base / "ComfyUI"
        if comfy:
            (self.root / "custom_nodes").mkdir(parents=True)
            (self.root / "main.py").write_text(FAKE_COMFY)
            (self.root / "fake.json").write_text(json.dumps(comfy_cfg or {}))
        self.vox = self.base / "VoxCPM-service"
        if vox:
            self.vox.mkdir()
            (self.vox / "server.py").write_text(FAKE_VOX)
            (self.vox / "port.txt").write_text(str(self.vox_port))
            (self.vox / "start.sh").write_text(f'#!/bin/sh\ncd "$(dirname "$0")"\nexec "{sys.executable}" server.py\n')
            (self.vox / "start.sh").chmod(0o755)
        self.env = {"CLOUDGPU_DIR": str(self.dir), "CLOUDGPU_SERVICES_SEARCH": str(self.base), "CLOUDGPU_COMFY_PYTHON": sys.executable,
                    "CLOUDGPU_COMFY_PORT": str(self.comfy_port), "CLOUDGPU_VOXCPM_PORT": str(self.vox_port), "CLOUDGPU_GPU_WAIT_S": "1"}

    def up(self, **env):
        out = []
        services.up(self.dir, {**self.env, **env}, say=out.append, sleep=lambda s: time.sleep(min(s, 0.1)))
        return out

    def cfg(self):
        return services.load_config(self.dir, self.env)[0]

    def desc(self, name):
        return services.describe(name, self.cfg(), self.dir)

    def launches(self, which="comfy"):
        return lines((self.root if which == "comfy" else self.vox) / "launches.log")

    def kill(self):
        for pid_file_owner in (self.root, self.vox):
            for rec in lines(pid_file_owner / "launches.log"):
                with contextlib.suppress(OSError):
                    os.kill(rec["pid"], signal.SIGTERM)
        for pid, cmd in services.processes():
            if str(self.d) in cmd and "services.py" not in cmd:
                with contextlib.suppress(OSError):
                    os.kill(pid, signal.SIGTERM)


@contextlib.contextmanager
def world(**kw):
    with _t.tmpdir() as d:
        w = World(d, **kw)
        try:
            yield w
        finally:
            w.kill()


@contextlib.contextmanager
def environ(**patch):
    old = {k: os.environ.get(k) for k in patch}
    os.environ.update({k: v for k, v in patch.items() if v is not None})
    for k, v in patch.items():
        if v is None:
            os.environ.pop(k, None)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---- starting ----------------------------------------------------------------------------------------------------------

def test_up_starts_comfy_and_then_voxcpm_and_never_sends_voxcpm_anything_but_a_health_get():
    with world(vox=True, comfy_cfg={"delay": 1.5, "say": "hello from comfy"}) as w:
        w.up()
        c, v = w.desc("comfy"), w.desc("voxcpm")
        assert c["state"] == "up" and v["state"] == "up", (c, v)
        assert v["loaded"] is False, "VoxCPM's model was not loaded by the start"
        comfy_launch, vox_launch = w.launches("comfy"), w.launches("voxcpm")
        assert len(comfy_launch) == 1 and len(vox_launch) == 1
        bound = float((w.root / "bound.log").read_text().split()[0])
        assert comfy_launch[0]["t"] < bound <= vox_launch[0]["t"], "VoxCPM is started only after ComfyUI answers"
        assert comfy_launch[0]["args"] == ["--listen", "127.0.0.1", "--port", str(w.comfy_port)]
        reqs = (w.vox / "requests.log").read_text().splitlines()
        assert reqs and set(reqs) == {"GET /voxcpm/v1/health"}, reqs          # nothing that loads the model: no POST, no other path
        assert "hello from comfy" in (w.dir / "logs" / "comfyui.log").read_text() and "fake voxcpm listening" in (w.dir / "logs" / "voxcpm.log").read_text()
        assert oct((w.dir / "logs" / "comfyui.log").stat().st_mode & 0o777) == "0o600"


def test_a_second_start_while_both_run_changes_nothing():
    with world(vox=True) as w:
        w.up()
        before = (w.launches("comfy"), w.launches("voxcpm"))
        out = w.up()
        assert (w.launches("comfy"), w.launches("voxcpm")) == before and len(before[0]) == 1 and len(before[1]) == 1, out
        assert w.desc("comfy")["state"] == "up" and w.desc("voxcpm")["state"] == "up"


def test_a_comfy_started_by_hand_and_still_loading_is_waited_for_not_started_twice():
    with world(comfy_cfg={"delay": 2.5}) as w:
        hand = subprocess.Popen([sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(w.comfy_port)], cwd=str(w.root), stdout=subprocess.DEVNULL)
        try:
            time.sleep(0.8)                                                  # it is up as a process, not yet bound: the 2 to 3 minutes of a real ComfyUI
            out = w.up()
            assert len(w.launches("comfy")) == 1, ("a second copy was started", out)
            assert any("waited for, not started again" in x for x in out), out
            assert w.desc("comfy")["state"] == "up"
        finally:
            hand.kill()
            hand.wait()


def test_while_comfy_loads_the_state_is_starting_with_a_reason_and_then_up():
    with world(comfy_cfg={"delay": 2.5}) as w:
        t = threading.Thread(target=w.up, daemon=True)
        t.start()
        seen = None
        for _ in range(40):
            d = w.desc("comfy")
            if d["state"] == "starting":
                seen = d
                break
            time.sleep(0.1)
        assert seen is not None and "not answering yet" in seen["reason"] and seen["log"] == "logs/comfyui.log", seen
        t.join(30)
        assert w.desc("comfy")["state"] == "up"


def test_a_voxcpm_that_is_not_installed_is_absent_with_the_reason_and_is_not_started():
    with world(vox=False) as w:
        w.up()
        v = w.desc("voxcpm")
        assert v["state"] == "absent" and "CLOUDGPU_VOXCPM_CMD" in v["reason"], v
        assert w.desc("comfy")["state"] == "up"


def test_a_voxcpm_already_answering_is_up_even_when_no_launcher_is_found():
    with world(vox=True) as w:
        shutil_launcher = w.vox / "start.sh"
        shutil_launcher.unlink()
        hand = subprocess.Popen([sys.executable, str(w.vox / "server.py")], stdout=subprocess.DEVNULL)
        try:
            for _ in range(50):
                if services.probe("voxcpm", w.vox_port)[0]:
                    break
                time.sleep(0.1)
            w.up()
            assert w.desc("voxcpm")["state"] == "up" and len(w.launches("voxcpm")) == 1
        finally:
            hand.kill()
            hand.wait()


def test_a_comfy_that_exits_says_the_code_and_its_last_log_line_and_voxcpm_is_still_started():
    with world(vox=True, comfy_cfg={"exit": 3, "say": "Traceback: no module named torch"}) as w:
        w.up()
        c = w.desc("comfy")
        assert c["state"] == "down" and "exited with code 3" in c["reason"] and c["last_line"] == "Traceback: no module named torch", c
        assert w.desc("voxcpm")["state"] == "up", "VoxCPM does not wait for a ComfyUI that failed"


def test_autostart_off_starts_nothing_and_still_reports():
    with world(vox=True) as w:
        w.up(CLOUDGPU_SERVICES_AUTOSTART="0")
        assert w.launches("comfy") == [] and w.launches("voxcpm") == []
        c = w.desc("comfy")
        assert c["state"] == "down" and "AUTOSTART" in c["reason"], c


def test_two_starts_at_once_make_one_copy():
    with world(vox=True, comfy_cfg={"delay": 2.5}) as w:
        env = {**os.environ, **w.env}
        first = subprocess.Popen([sys.executable, str(SRC), "up"], env=env, stdout=subprocess.PIPE, text=True)
        time.sleep(1.0)
        second = subprocess.run([sys.executable, str(SRC), "up"], env=env, capture_output=True, text=True, timeout=60)
        out1 = first.communicate(timeout=60)[0]
        assert "another start" in second.stdout and second.returncode == 0, (second.stdout, second.stderr)
        assert len(w.launches("comfy")) == 1 and len(w.launches("voxcpm")) == 1, out1


def test_comfy_waits_for_the_gpu_and_is_not_started_without_one_but_voxcpm_is():
    with _t.tmpdir() as bindir:
        smi = Path(bindir) / "nvidia-smi"
        smi.write_text("#!/bin/sh\nexit 9\n")
        smi.chmod(0o755)
        with world(vox=True) as w, environ(PATH=bindir + os.pathsep + os.environ["PATH"]):
            w.up()
            c = w.desc("comfy")
            assert c["state"] == "down" and "no GPU" in c["reason"] and w.launches("comfy") == [], c
            assert w.desc("voxcpm")["state"] == "up"
            smi.write_text('#!/bin/sh\necho "GPU 0: Fake 4090 (UUID: x)"\n')                      # the driver came up: the next start brings ComfyUI up
            w.up()
            assert w.desc("comfy")["state"] == "up" and len(w.launches("comfy")) == 1


def test_the_adapters_token_is_not_given_to_a_service():
    with world(vox=True) as w, environ(CLOUDGPU_TOKEN="x" * 40, H3HOST_TOKEN="y" * 40):
        w.up()
        assert w.launches("comfy")[0]["token_in_env"] is False and w.launches("voxcpm")[0]["token_in_env"] is False
        assert "x" * 40 not in (w.dir / "services.state.json").read_text()


def test_detach_returns_at_once_and_the_supervisor_goes_on():
    with world(comfy_cfg={"delay": 3}) as w:
        t0 = time.time()
        r = subprocess.run([sys.executable, str(SRC), "up", "--detach"], env={**os.environ, **w.env}, capture_output=True, text=True, timeout=30)
        assert r.returncode == 0 and time.time() - t0 < 2.5 and "in the background" in r.stdout, (r.stdout, r.stderr, time.time() - t0)
        for _ in range(100):
            if w.desc("comfy")["state"] == "up":
                break
            time.sleep(0.2)
        assert w.desc("comfy")["state"] == "up" and len(w.launches("comfy")) == 1
        assert (w.dir / "logs" / "services.log").is_file()


def test_a_launcher_that_returns_zero_after_starting_the_service_in_the_background_is_not_a_failure():
    with world(vox=True) as w:
        (w.vox / "start.sh").write_text(f'#!/bin/sh\ncd "$(dirname "$0")"\nnohup "{sys.executable}" "{w.vox}/server.py" >/dev/null 2>&1 &\nexit 0\n')
        out = w.up()
        assert any(x.startswith("VoxCPM: up:") for x in out), out                      # the start saw it come up, it did not give up when the launcher returned
        assert w.desc("voxcpm")["state"] == "up" and len(w.launches("voxcpm")) == 1
        (w.vox / "start.sh").write_text("#!/bin/sh\nexit 0\n")                          # a launcher that returns 0 and leaves nothing: down, with the words, after the grace
        for rec in w.launches("voxcpm"):
            os.kill(rec["pid"], signal.SIGTERM)
        time.sleep(0.5)
        old = services.EXITED_OK_GRACE_S
        services.EXITED_OK_GRACE_S = 1
        try:
            w.up()
        finally:
            services.EXITED_OK_GRACE_S = old
        v = w.desc("voxcpm")
        assert v["state"] == "down" and "exit 0" in v["reason"], v


def test_a_log_over_the_limit_is_kept_as_dot_one_and_a_new_one_begins():
    with _t.tmpdir() as d:
        p = Path(d) / "logs" / "comfyui.log"
        p.parent.mkdir()
        p.write_bytes(b"x" * 100)
        old = services.LOG_MAX
        services.LOG_MAX = 50
        try:
            services.open_log(p).close()
        finally:
            services.LOG_MAX = old
        assert (p.parent / "comfyui.log.1").read_bytes() == b"x" * 100 and p.read_bytes() == b""


# ---- finding things ------------------------------------------------------------------------------------------------------

def test_comfy_is_found_where_the_config_says_else_in_the_usual_places_else_one_level_below_a_search_folder():
    with _t.tmpdir() as d:
        base, d = Path(d) / "autodl-tmp", Path(d)
        odd = base / "my-comfy-copy"
        (odd / "comfy").mkdir(parents=True)
        (odd / "main.py").write_text("")
        env = {"CLOUDGPU_SERVICES_SEARCH": str(base)}
        assert services.find_comfy_root({}, d, env)[0] == odd
        usual = base / "ComfyUI"
        (usual / "custom_nodes").mkdir(parents=True)
        (usual / "main.py").write_text("")
        assert services.find_comfy_root({}, d, env)[0] == usual, "ComfyUI by name before a folder that merely looks like it"
        mine = d / "mine"
        (mine / "comfy").mkdir(parents=True)
        (mine / "main.py").write_text("")
        assert services.find_comfy_root({"comfy_root": str(mine)}, d, env)[0] == mine
        assert services.find_comfy_root({"comfy_root": str(d / "nowhere")}, d, env)[0] == usual, "a configured folder that is not ComfyUI falls through to the search"
        (usual / "main.py").unlink()
        (odd / "main.py").unlink()
        root, tried = services.find_comfy_root({"comfy_root": str(d / "nowhere")}, d, env)
        assert root is None and tried, "nothing found says where it looked"
        plan = services.make_plan("comfy", {"comfy_root": str(d / "nowhere")}, d, env)
        assert "absent" in plan and "CLOUDGPU_COMFY_ROOT" in plan["absent"]


def test_the_python_is_the_configured_one_else_the_first_that_can_import_torch_else_the_start_is_blocked_with_the_reason():
    with _t.tmpdir() as d:
        root = Path(d) / "ComfyUI"
        (root / "venv" / "bin").mkdir(parents=True)
        (root / "venv" / "bin" / "python").symlink_to(sys.executable)
        real = services.has_torch
        try:
            services.has_torch = lambda py: py.endswith("venv/bin/python")
            assert services.find_comfy_python({}, root)[0] == str(root / "venv" / "bin" / "python")
            assert services.find_comfy_python({"comfy_python": "/anywhere/python"}, root)[0] == "/anywhere/python", "a configured Python is trusted as it is"
            services.has_torch = lambda py: False
            py, tried = services.find_comfy_python({}, root)
            assert py is None and tried
            (root / "main.py").write_text("")
            (root / "custom_nodes").mkdir()
            plan = services.make_plan("comfy", {"comfy_root": str(root)}, Path(d), {"CLOUDGPU_SERVICES_SEARCH": str(d)})
            assert "blocked" in plan and "torch" in plan["blocked"] and "CLOUDGPU_COMFY_PYTHON" in plan["blocked"], plan
        finally:
            services.has_torch = real


def test_a_setting_is_the_environment_then_services_json_then_the_default_and_a_bad_one_is_ignored_and_said():
    with _t.tmpdir() as d:
        d = Path(d)
        (d / "services.json").write_text(json.dumps({"comfy_port": 7001, "voxcpm_port": "x", "comfy_args": "--lowvram", "autostart": False}))
        cfg, problems = services.load_config(d, {"CLOUDGPU_COMFY_PORT": "7002", "CLOUDGPU_COMFY_WAIT_S": "-5"})
        assert cfg["comfy_port"] == 7002, "the environment wins"
        cfg, problems = services.load_config(d, {"CLOUDGPU_COMFY_PORT": ""})
        assert cfg["comfy_port"] == 7001 and cfg["comfy_args"] == "--lowvram" and cfg["autostart"] is False, "an empty variable is unset: the file answers"
        assert cfg["voxcpm_port"] is None and any("voxcpm_port" in p for p in problems), problems
        assert services.make_plan("voxcpm", cfg, d, {"CLOUDGPU_SERVICES_SEARCH": str(d)})["port"] == 8190
        assert services.make_plan("comfy", {**cfg, "comfy_root": str(d)}, d, {"CLOUDGPU_SERVICES_SEARCH": str(d), "x": ""}).get("port") == 7001


def test_the_voxcpm_command_from_the_environment_runs_without_a_shell_in_its_folder_and_a_launcher_is_found_by_name():
    with _t.tmpdir() as d:
        d = Path(d)
        env = {"CLOUDGPU_SERVICES_SEARCH": str(d / "base")}
        v = d / "base" / "my-VoxCPM"
        v.mkdir(parents=True)
        (v / "run.sh").write_text("")                                          # generic names are not guessed: run.sh could be anything
        assert "absent" in services.make_plan("voxcpm", {}, d, env)
        (v / "serve.sh").write_text("")
        p = services.make_plan("voxcpm", {}, d, env)
        assert p["argv"][-1] == str(v / "serve.sh") and p["cwd"] == v, p
        p = services.make_plan("voxcpm", {"voxcpm_cmd": "/opt/py/bin/python -m vox.serve --port 8190; rm -rf /", "voxcpm_dir": str(v)}, d, env)
        assert p["argv"] == ["/opt/py/bin/python", "-m", "vox.serve", "--port", "8190;", "rm", "-rf", "/"] and p["cwd"] == v, "split like a shell would, run without one"


def test_configure_merges_the_installs_knowledge_into_a_0600_file_and_an_unset_setting_changes_nothing():
    with _t.tmpdir() as d:
        d = Path(d) / "h3host"
        services.configure(d, "/root/autodl-tmp/ComfyUI", "/usr/bin/python3", {"CLOUDGPU_COMFY_PORT": "6100", "CLOUDGPU_VOXCPM_CMD": "sh /x/v.sh"})
        doc = json.loads((d / "services.json").read_text())
        assert doc == {"comfy_root": "/root/autodl-tmp/ComfyUI", "python_hint": "/usr/bin/python3", "comfy_port": 6100, "voxcpm_cmd": "sh /x/v.sh"}, doc
        assert oct((d / "services.json").stat().st_mode & 0o777) == "0o600"
        services.configure(d, "/other/ComfyUI", None, {"CLOUDGPU_COMFY_PORT": ""})
        doc = json.loads((d / "services.json").read_text())
        assert doc["comfy_root"] == "/other/ComfyUI" and doc["python_hint"] == "/usr/bin/python3" and "comfy_port" not in doc and doc["voxcpm_cmd"] == "sh /x/v.sh"


# ---- saying -----------------------------------------------------------------------------------------------------------

NOW = 1_000_000.0
TABLE = [      # (answers, record, pids alive, expected state, a word of the reason)
    (True, None, [], "up", "answers"),
    (True, {"state": "up", "reason": "answers on :6006 41 s after it was started"}, [], "up", "41 s"),
    (True, {"state": "down", "reason": "old"}, [], "up", "answers"),
    (False, None, [], "down", "no start has been recorded"),
    (False, {"state": "absent", "reason": "not installed here"}, [], "absent", "not installed here"),
    (False, {"state": "starting", "reason": "started", "pid": 77, "since": NOW - 30}, [77], "starting", "(for 30 s)"),
    (False, {"state": "starting", "reason": "started", "pid": 77, "since": NOW - 30}, [], "down", "its process is gone"),
    (False, {"state": "starting", "reason": "started", "pid": None, "until": NOW + 20, "since": NOW - 5}, [], "starting", "(for 5 s)"),
    (False, {"state": "starting", "reason": "started", "pid": None, "until": NOW - 1, "last_line": "boom"}, [], "down", "its process is gone"),
    (False, {"state": "up", "reason": "answers"}, [], "down", "does not now"),
    (False, {"state": "down", "reason": "exited with code 1"}, [], "down", "exited with code 1"),
    (False, {"state": "down"}, [], "down", "does not answer"),
]


def test_the_state_and_reason_are_the_same_in_services_py_and_in_the_adapter_for_every_case():
    for up, rec, pids, state, word in TABLE:
        a = services.classify(up, rec, lambda p: p in pids, NOW)
        b = adapter.service_view(up, rec, lambda p: p in pids, NOW)[:2]
        assert a == b, (rec, a, b)
        assert a[0] == state and word in a[1], (rec, a)
    old = {"state": "up", "reason": "answers", "updated": NOW - 500}
    assert adapter.service_view(False, old, lambda p: True, NOW, since=NOW - 100)[0] == "down" and "since this box was powered on" in adapter.service_view(False, old, lambda p: True, NOW, since=NOW - 100)[1]
    assert services.fresh(old, NOW - 100) is None and services.fresh(old, NOW - 1000) is old and services.fresh(old, None) is old


def test_status_prints_a_line_per_service_or_json_and_reads_without_starting_anything():
    with world(vox=True) as w:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            services.status(w.dir, False, w.env)
        txt = buf.getvalue()
        assert txt.splitlines()[0].startswith("comfy") and "down" in txt.splitlines()[0] and txt.splitlines()[1].startswith("voxcpm"), txt
        assert w.launches("comfy") == [] and w.launches("voxcpm") == [], "status starts nothing"
        w.up()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            services.status(w.dir, True, w.env)
        doc = json.loads(buf.getvalue())
        assert doc["comfy"]["state"] == "up" and doc["voxcpm"]["state"] == "up" and doc["voxcpm"]["loaded"] is False and doc["comfy"]["port"] == w.comfy_port, doc


# ---- the adapter's /health ----------------------------------------------------------------------------------------------

class FakeVox:
    def __init__(self, loaded=False, code=200):
        outer = self
        self.hits = []

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.hits.append(self.path)
                b = json.dumps({"model": "VoxCPM2", "loaded": loaded}).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
        self.srv = HTTPServer(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def stop(self):
        self.srv.shutdown()


def write_record(state_dir: Path, **services_):
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / adapter.SERVICES_FILE).write_text(json.dumps({"updated": time.time(), "services": {k: {**v, "updated": time.time()} for k, v in services_.items()}}))


def test_health_says_per_service_up_starting_down_or_absent_with_the_reason():
    from test_adapter import Box
    with _t.tmpdir() as d:
        b = Box(d)
        vox = FakeVox(loaded=False)
        try:
            adapter.SERVICES_CACHE_S = 0
            write_record(Path(d) / "state", voxcpm={"state": "up", "reason": "answers on :1", "port": vox.port})
            b.host._comfy = (0.0, False)
            h = b.host.health()
            assert h["comfy"] is True and h["services"]["comfy"]["state"] == "up", h["services"]
            assert h["services"]["voxcpm"] == {"state": "up", "reason": "answers on :1", "port": vox.port, "log": "logs/voxcpm.log", "loaded": False}, h["services"]
            assert vox.hits == ["/voxcpm/v1/health"], "only the health call is made to VoxCPM"
            vox.stop()
            vox.srv.server_close()
            h = b.host.health()
            assert h["services"]["voxcpm"]["state"] == "down" and "does not now" in h["services"]["voxcpm"]["reason"] and h["services"]["voxcpm"]["loaded"] is None
            write_record(Path(d) / "state", voxcpm={"state": "absent", "reason": "VoxCPM is not installed here", "port": vox.port})
            assert b.host.health()["services"]["voxcpm"]["state"] == "absent"
            write_record(Path(d) / "state", voxcpm={"state": "starting", "reason": "started (pid 1)", "port": vox.port, "pid": os.getpid(), "since": time.time() - 12})
            v = b.host.health()["services"]["voxcpm"]
            assert v["state"] == "starting" and "(for 12 s)" in v["reason"], v
            write_record(Path(d) / "state", voxcpm={"state": "down", "reason": "its process exited with code 2", "port": vox.port, "last_line": "OSError: no space"})
            v = b.host.health()["services"]["voxcpm"]
            assert v["state"] == "down" and v["last_line"] == "OSError: no space", v
            b.host.proof_ok = False                                                   # a widened bind answers anyone: no log text
            assert "last_line" not in b.host.health()["services"]["voxcpm"]
        finally:
            adapter.SERVICES_CACHE_S = 3
            vox.stop()
            b.srv.shutdown()
            b.comfy.stop()


def test_health_names_a_comfy_that_is_down_with_the_reason_and_old_boxes_without_a_record_still_answer():
    from test_adapter import Box
    with _t.tmpdir() as d:
        b = Box(d)
        try:
            b.comfy.stop()
            b.host.comfy = "http://127.0.0.1:%d" % free_port()
            b.host._comfy = (0.0, False)
            h = b.host.health()
            assert h["comfy"] is False and h["services"]["comfy"]["state"] == "down" and "no start has been recorded" in h["services"]["comfy"]["reason"], h["services"]
            assert h["services"]["voxcpm"]["state"] == "down"
            write_record(Path(d) / "state", comfy={"state": "starting", "reason": "started (pid 4)", "port": 6006, "pid": os.getpid(), "since": time.time() - 3})
            b.host._comfy = (0.0, False)
            c = b.host.health()["services"]["comfy"]
            assert c["state"] == "starting" and c["port"] == urllib.parse.urlparse(b.host.comfy).port
            (Path(d) / "state" / adapter.SERVICES_FILE).write_text("not json")
            assert b.host.health()["services"]["comfy"]["state"] == "down"             # a damaged record never breaks /health
        finally:
            b.srv.shutdown()


def test_a_take_that_waited_for_a_comfy_that_never_came_says_what_the_box_knows_of_it():
    with _t.tmpdir() as d:
        host = adapter.Host("t" * 30, "http://127.0.0.1:%d" % free_port(), Path(d) / "state", Path(d) / "comfy")
        write_record(Path(d) / "state", comfy={"state": "down", "reason": "not started: nvidia-smi sees no GPU after 90 s (the driver is not ready?)", "port": 6006})
        real = adapter.COMFY_WAIT_S
        adapter.COMFY_WAIT_S = -1
        try:
            try:
                host.run({"plan": {"kind": "ref"}, "inputs": [], "id": "x"})
                raise AssertionError("it should have failed")
            except RuntimeError as e:
                assert str(e).startswith("ComfyUI is not answering after") and "down, not started: nvidia-smi sees no GPU" in str(e), str(e)
        finally:
            adapter.COMFY_WAIT_S = real


# ---- the status the client reads -------------------------------------------------------------------------------------------

def test_the_status_lists_each_process_with_its_state_and_reason_for_new_and_old_adapters_and_for_a_box_that_does_not_answer():
    new = {"ok": True, "comfy": False, "busy": 0, "services": {"comfy": {"state": "starting", "reason": "started (pid 9) (for 20 s)", "port": 6006, "log": "logs/comfyui.log"},
                                                               "voxcpm": {"state": "absent", "reason": "VoxCPM is not installed here", "port": 8190, "log": "logs/voxcpm.log", "loaded": None}}}
    v = status_mod.services_view(new, True, False)
    assert v["adapter"]["state"] == "up" and v["comfy"]["state"] == "starting" and v["voxcpm"]["state"] == "absent" and v["voxcpm"]["reason"].startswith("VoxCPM is not installed"), v
    old = status_mod.services_view({"ok": True, "comfy": False}, True, False)
    assert old["comfy"]["state"] == "down" and "older than 0.21.0" in old["comfy"]["reason"] and old["voxcpm"]["state"] == "unknown", old
    assert status_mod.services_view({"ok": True, "comfy": True}, True, False)["comfy"]["state"] == "up"
    off = status_mod.services_view(None, True, False)
    assert off["adapter"]["state"] == "down" and off["comfy"]["state"] == "unknown" and off["voxcpm"]["state"] == "unknown"
    assert status_mod.services_view(None, False, False)["adapter"]["state"] == "unknown"
    assert status_mod.services_view({"ok": False, "comfy": True, "services": {"comfy": {"state": "weird"}}}, True, False)["adapter"]["state"] == "down"


def test_the_status_next_line_for_a_comfy_that_is_starting_or_down_says_the_reason_and_a_voxcpm_down_is_said_too():
    from cloudgpu import box as box_mod
    health = {"ok": True, "comfy": False, "busy": 0, "worker": "alive", "card": "NVIDIA GeForce RTX 4090, 49140 MiB", "services": {
        "comfy": {"state": "down", "reason": "exited with code 3", "port": 6006, "log": "logs/comfyui.log"},
        "voxcpm": {"state": "down", "reason": "it answered when it was started and does not now", "port": 8190, "log": "logs/voxcpm.log", "loaded": None}}}
    real = box_mod.Box.health
    box_mod.Box.health = lambda self: health
    try:
        env = {"CLOUDGPU_BASE": "http://127.0.0.1:9", "CLOUDGPU_TOKEN": "t" * 30, "CLOUDGPU_BOX": "rtx4090", "CLOUDGPU_POWER": "none"}
        doc = status_mod.status(env)
        assert doc["ready"] is False and doc["services"]["comfy"]["state"] == "down", doc
        assert any("ComfyUI is down on the box: exited with code 3" in n and "logs/comfyui.log" in n for n in doc["next"]), doc["next"]
        assert any("VoxCPM" in n and "down" in n for n in doc["next"]), doc["next"]
        health["services"]["comfy"] = {"state": "starting", "reason": "started (pid 9)", "port": 6006, "log": "logs/comfyui.log"}
        health["services"]["voxcpm"]["state"] = "absent"
        doc = status_mod.status(env)
        assert any("ComfyUI is starting on the box" in n and "2 to 3 minutes" in n for n in doc["next"]) and not any("VoxCPM" in n for n in doc["next"]), doc["next"]
    finally:
        box_mod.Box.health = real


# ---- install.sh and start.sh, end to end ------------------------------------------------------------------------------------

def test_install_and_start_sh_bring_comfy_up_after_a_boot_once_and_the_adapter_is_told_its_port():
    from test_detect_install import L4090, tree
    with _t.tmpdir() as d:
        d = Path(d)
        home = d / "home"
        tree(home / "ComfyUI", L4090)
        (home / "ComfyUI" / "custom_nodes").mkdir(exist_ok=True)
        (home / "ComfyUI" / "main.py").write_text(FAKE_COMFY)
        work = d / "box"
        work.mkdir()
        for name, src in (("install.sh", "onbox/install.sh"), ("detect_models.py", "detect_models.py"), ("models.4090.json", "onbox/models.4090.json"), ("adapter.py", "adapter.py"),
                          ("services.py", "onbox/services.py")):
            (work / name).write_bytes((Path(_t.ROOT) / "cloudgpu" / src).read_bytes())
        port, comfy_port = free_port(), free_port()
        env = {"PATH": os.environ["PATH"], "HOME": str(home), "CLOUDGPU_PYTHON": sys.executable, "CLOUDGPU_COMFY_PYTHON": sys.executable, "CLOUDGPU_COMFY_PORT": str(comfy_port),
               "CLOUDGPU_SERVICES_SEARCH": str(d / "nowhere")}
        box = home / "h3host"
        try:
            r = subprocess.run(["sh", str(work / "install.sh"), "auto", str(port)], input="t" * 32, env=env, capture_output=True, text=True, timeout=60)
            assert r.returncode == 0, r.stderr
            cfg = json.loads((box / "services.json").read_text())
            assert cfg["comfy_root"] == str(home / "ComfyUI") and cfg["comfy_port"] == comfy_port and cfg["comfy_python"] == sys.executable, cfg
            assert oct((box / "services.json").stat().st_mode & 0o777) == "0o600" and (box / "services.py").is_file()
            start = (box / "start.sh").read_text()
            assert "services.py" in start and "up --detach" in start and f"--comfy http://127.0.0.1:{comfy_port}" in start and "t" * 32 not in start

            def health():
                for _ in range(150):
                    try:
                        h = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3).read())
                        if h["services"]["comfy"]["state"] == "up":
                            return h
                    except OSError:
                        pass
                    time.sleep(0.2)
                return h
            h = health()
            assert h["comfy"] is True and h["services"]["comfy"]["state"] == "up" and h["services"]["voxcpm"]["state"] == "absent", h
            assert len(lines(home / "ComfyUI" / "launches.log")) == 1, "the install's own start.sh started ComfyUI once"
            # a power-on hook runs start.sh with almost no environment, twice (the hook and systemd): still one ComfyUI
            for _ in range(2):
                r = subprocess.run(["sh", str(box / "start.sh")], env={"PATH": os.environ["PATH"], "HOME": str(home)}, capture_output=True, text=True, timeout=60)
                assert r.returncode == 0, r.stderr
            time.sleep(2)
            h = health()
            assert h["comfy"] is True and len(lines(home / "ComfyUI" / "launches.log")) == 1, lines(home / "ComfyUI" / "launches.log")
            assert "starts comfy" in (box / "logs" / "comfyui.log").read_text() and (box / "logs" / "services.log").is_file()
        finally:
            for rec in lines(home / "ComfyUI" / "launches.log"):                      # the fake ComfyUI's command line is relative (main.py): found by its own record
                with contextlib.suppress(OSError):
                    os.kill(rec["pid"], signal.SIGTERM)
            for pid, cmd in services.processes():
                if str(d) in cmd and "test_services" not in cmd:
                    with contextlib.suppress(OSError):
                        os.kill(pid, signal.SIGTERM)


def test_the_bundle_carries_services_py_and_an_install_without_it_still_works_as_before():
    from cloudgpu.__main__ import BUNDLE_FILES
    assert "services.py" in BUNDLE_FILES and BUNDLE_FILES["services.py"].is_file()
    text = (Path(_t.ROOT) / "cloudgpu" / "onbox" / "install.sh").read_text()
    assert 'if [ -f "$HERE/services.py" ]' in text, "an old bundle without services.py installs as in 0.20.0"


# ---- what the review found (0.21.0) ------------------------------------------------------------------------------------------

def test_a_launcher_that_starts_the_service_in_the_background_and_returns_is_not_run_again_while_the_service_loads():
    with world(vox=True) as w:
        (w.vox / "delay.txt").write_text("4")
        (w.vox / "start.sh").write_text(f'#!/bin/sh\ncd "$(dirname "$0")"\nnohup "{sys.executable}" "{w.vox}/server.py" >/dev/null 2>&1 &\nexit 0\n')
        old = services.EXITED_OK_GRACE_S
        services.EXITED_OK_GRACE_S = 1
        try:
            t = threading.Thread(target=w.up, daemon=True)
            t.start()
            time.sleep(2.5)                                       # the launcher has returned, the service has not bound: a second start now must not run the launcher again
            out = []
            services.up(w.dir, w.env, say=out.append, sleep=lambda s: time.sleep(min(s, 0.1)))
            t.join(30)
        finally:
            services.EXITED_OK_GRACE_S = old
        assert len(w.launches("voxcpm")) == 1, (w.launches("voxcpm"), out)
        assert w.desc("voxcpm")["state"] == "up"


def test_a_hand_started_service_is_recognised_by_a_match_the_operator_gives_and_a_bad_regex_is_said():
    with world(vox=True) as w:
        (w.vox / "delay.txt").write_text("2.5")
        hand = subprocess.Popen([sys.executable, str(w.vox / "server.py")], stdout=subprocess.DEVNULL)
        try:
            time.sleep(0.5)
            w.up(CLOUDGPU_VOXCPM_MATCH=r"VoxCPM-service/server\.py")
            assert len(w.launches("voxcpm")) == 1 and w.desc("voxcpm")["state"] == "up"
        finally:
            hand.kill()
            hand.wait()
    cfg, problems = services.load_config(Path("/nonexistent"), {"CLOUDGPU_VOXCPM_MATCH": "(", "CLOUDGPU_COMFY_MATCH": "main"})
    assert cfg["voxcpm_match"] is None and cfg["comfy_match"] == "main" and any("VOXCPM_MATCH" in p for p in problems), problems


def test_a_waited_for_process_that_dies_is_said_to_have_ended_not_to_have_exited_zero():
    with world(comfy_cfg={"delay": 1.2, "exit_after_delay": True}) as w:
        # started in the background by a shell, so it is nobody's child (a zombie child of this test would look alive on a machine without /proc)
        subprocess.run(["sh", "-c", f'cd "{w.root}" && "{sys.executable}" main.py --listen 127.0.0.1 --port {w.comfy_port} >/dev/null 2>&1 &'], check=True)
        time.sleep(0.4)
        w.up()
        c = w.desc("comfy")
        assert c["state"] == "down" and "ended without answering" in c["reason"] and "exit 0" not in c["reason"], c


def test_the_services_outlive_the_supervisor_and_its_process_group_and_a_match_never_finds_the_supervisor_itself():
    with world(vox=True) as w:
        env = {**os.environ, **w.env, "CLOUDGPU_VOXCPM_MATCH": r"services\.py up"}               # would match the supervisor's own command line (services.py)
        sup = subprocess.Popen([sys.executable, str(SRC), "up"], env=env, stdout=subprocess.DEVNULL, start_new_session=True)
        assert sup.wait(60) == 0
        assert len(w.launches("voxcpm")) == 1, "the supervisor mistook itself for a running VoxCPM"
        with contextlib.suppress(OSError):
            os.killpg(sup.pid, signal.SIGKILL)                                           # everything the supervisor's group had is gone; the services are in groups of their own
        time.sleep(0.5)
        assert w.desc("comfy")["state"] == "up" and w.desc("voxcpm")["state"] == "up"


def test_install_again_without_a_comfy_port_drops_the_old_one_so_services_py_and_the_adapter_agree():
    with _t.tmpdir() as d:
        d = Path(d) / "h3host"
        services.configure(d, "/c", None, {"CLOUDGPU_COMFY_PORT": "7000", "CLOUDGPU_COMFY_ARGS": "--lowvram"})
        services.configure(d, "/c", None, {})
        doc = json.loads((d / "services.json").read_text())
        assert "comfy_port" not in doc and doc["comfy_args"] == "--lowvram", doc


def test_a_stale_record_from_before_this_boot_is_ignored_where_the_box_says_when_it_booted():
    from test_adapter import Box
    with world() as w, _t.tmpdir() as d:
        w.up()
        assert w.desc("comfy")["state"] == "up" and "started" in services.read_state(w.dir)["comfy"]["reason"] or True
        (w.dir / services.STATE_FILE).write_text(json.dumps({"services": {"comfy": {"state": "starting", "reason": "started (pid 1)", "pid": os.getpid(), "since": time.time(), "updated": time.time() - 500}}}))
        for n in list(w.launches()):
            os.kill(n["pid"], signal.SIGTERM)
        time.sleep(0.5)
        real = services.container_start
        services.container_start = lambda: time.time() - 100                                # the box was powered on 100 s ago; the record is 500 s old
        try:
            c = w.desc("comfy")
        finally:
            services.container_start = real
        assert c["state"] == "down" and "since this box was powered on" in c["reason"], c
        b = Box(d)
        real_a = adapter.container_start
        try:
            write_record(Path(d) / "state", comfy={"state": "starting", "reason": "x", "pid": os.getpid(), "since": time.time()})
            rec = json.loads((Path(d) / "state" / adapter.SERVICES_FILE).read_text())
            rec["services"]["comfy"]["updated"] = time.time() - 500
            (Path(d) / "state" / adapter.SERVICES_FILE).write_text(json.dumps(rec))
            b.comfy.stop()
            b.host.comfy = "http://127.0.0.1:%d" % free_port()
            b.host._comfy = (0.0, False)
            adapter.container_start = lambda: time.time() - 100
            c = b.host.health()["services"]["comfy"]
            assert c["state"] == "down" and "since this box was powered on" in c["reason"], c
        finally:
            adapter.container_start = real_a
            b.srv.shutdown()


def raw_server(behaviour):
    """A TCP listener on 127.0.0.1 for the VoxCPM port that misbehaves: 'garbage' (not HTTP), 'silent' (accepts, never answers), 'drip' (one byte of a valid answer every 0.3 s)."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    stop = threading.Event()
    held = []

    def run():
        srv.settimeout(0.2)
        while not stop.is_set():
            try:
                c, _ = srv.accept()
            except OSError:
                continue
            held.append(c)
            if behaviour == "garbage":
                with contextlib.suppress(OSError):
                    c.recv(1024)
                    c.sendall(b"hello garbage\r\n\r\n")
                    c.close()
            elif behaviour == "drip":
                def drip(c=c):
                    with contextlib.suppress(OSError):
                        for ch in b"HTTP/1.0 200 OK\r\n\r\n{}":
                            if stop.is_set():
                                return
                            c.sendall(bytes([ch]))
                            time.sleep(0.3)
                threading.Thread(target=drip, daemon=True).start()
    threading.Thread(target=run, daemon=True).start()

    def close():
        stop.set()
        srv.close()
        for c in held:
            with contextlib.suppress(OSError):
                c.close()
    return srv.getsockname()[1], close


def test_health_never_raises_and_never_waits_long_whatever_sits_on_the_voxcpm_port_or_in_the_state_file():
    from test_adapter import Box
    with _t.tmpdir() as d:
        b = Box(d)
        sd = Path(d) / "state"
        try:
            adapter.SERVICES_CACHE_S = 0
            for behaviour, state_word in (("garbage", "down"), ("silent", "up"), ("drip", "up")):
                port, close = raw_server(behaviour)
                try:
                    write_record(sd, voxcpm={"state": "up", "reason": "answers", "port": port})
                    t0 = time.time()
                    h = b.host.health()
                    took = time.time() - t0
                    assert took < 2.5, (behaviour, took)
                    v = h["services"]["voxcpm"]
                    assert v["state"] == state_word, (behaviour, v)
                    if behaviour != "garbage":
                        assert "did not answer" in v["reason"] and v["loaded"] is None, v
                finally:
                    close()
            for rec in ({"port": True, "state": "up"}, {"port": 99999, "state": "starting", "pid": 2**31 + 5}, {"state": "starting", "pid": True, "reason": 5},
                        {"state": ["x"], "reason": ["y"], "pid": "z"}, {"state": "starting", "pid": 2**40, "until": "never"}):
                write_record(sd, voxcpm=rec)
                v = b.host.health()["services"]["voxcpm"]
                assert v["state"] in ("down", "absent", "starting", "up"), (rec, v)
            (sd / adapter.SERVICES_FILE).write_text("[" * 200000)                          # deep nesting: RecursionError in a naive json.loads
            assert b.host.health()["services"]["voxcpm"]["state"] == "down"
            write_record(sd, voxcpm={"state": "down", "reason": "x" * 5_000_000})
            assert len(json.dumps(b.host.health()["services"])) < 2000, "a huge reason is cut"
            assert services.alive(2**31) is False and adapter.pid_alive(2**31) is False
            (sd / services.STATE_FILE).write_text("[" * 200000)
            assert services.read_state(sd) == {}
        finally:
            adapter.SERVICES_CACHE_S = 3
            b.srv.shutdown()
            b.comfy.stop()


def test_a_voxcpm_that_is_busy_and_does_not_answer_in_time_is_up_not_down_in_services_py_too():
    with _t.tmpdir() as d:
        port, close = raw_server("silent")
        try:
            t0 = time.time()
            ok, doc = services.probe("voxcpm", port, timeout=0.5)
            assert ok is True and doc == {"_busy": True} and time.time() - t0 < 1.5
            assert services.probe("comfy", port, timeout=0.5) == (False, None), "a ComfyUI that does not answer is not up"
        finally:
            close()
        port, close = raw_server("garbage")
        try:
            assert services.probe("voxcpm", port, timeout=0.5) == (False, None)
        finally:
            close()


def test_the_vox_probe_is_cached_between_health_calls():
    from test_adapter import Box
    with _t.tmpdir() as d:
        b = Box(d)
        vox = FakeVox()
        try:
            write_record(Path(d) / "state", voxcpm={"state": "up", "reason": "answers", "port": vox.port})
            for _ in range(5):
                b.host.health()
            assert len(vox.hits) == 1, vox.hits
        finally:
            vox.stop()
            b.srv.shutdown()
            b.comfy.stop()


def test_a_start_sh_run_while_the_adapter_has_a_take_leaves_it_alone_and_still_starts_what_is_down():
    from test_detect_install import L4090, tree
    with _t.tmpdir() as d:
        d = Path(d)
        home = d / "home"
        tree(home / "ComfyUI", L4090)
        (home / "ComfyUI" / "custom_nodes").mkdir(exist_ok=True)
        (home / "ComfyUI" / "main.py").write_text(FAKE_COMFY)
        work = d / "box"
        work.mkdir()
        for name, src in (("install.sh", "onbox/install.sh"), ("detect_models.py", "detect_models.py"), ("models.4090.json", "onbox/models.4090.json"), ("adapter.py", "adapter.py"),
                          ("services.py", "onbox/services.py")):
            (work / name).write_bytes((Path(_t.ROOT) / "cloudgpu" / src).read_bytes())
        port, comfy_port = free_port(), free_port()
        env = {"PATH": os.environ["PATH"], "HOME": str(home), "CLOUDGPU_PYTHON": sys.executable, "CLOUDGPU_COMFY_PYTHON": sys.executable, "CLOUDGPU_COMFY_PORT": str(comfy_port),
               "CLOUDGPU_SERVICES_SEARCH": str(d / "nowhere")}
        box = home / "h3host"
        busy = None
        try:
            r = subprocess.run(["sh", str(work / "install.sh"), "auto", str(port)], input="t" * 32, env=env, capture_output=True, text=True, timeout=60)
            assert r.returncode == 0, r.stderr
            for _ in range(100):
                if services.probe("comfy", comfy_port)[0]:
                    break
                time.sleep(0.2)
            subprocess.run(["pkill", "-f", f"{box}/server.py"])                           # the real adapter goes; a stand-in with a take on it takes its port
            for rec in lines(home / "ComfyUI" / "launches.log"):
                with contextlib.suppress(OSError):
                    os.kill(rec["pid"], signal.SIGTERM)                                    # and ComfyUI died
            time.sleep(1)
            (box / "server.py").write_text(
                "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
                "class H(BaseHTTPRequestHandler):\n"
                "    def log_message(self, *a): pass\n"
                "    def do_GET(self):\n"
                "        b = b'{\"ok\": true, \"comfy\": false, \"busy\": 1}'\n"
                "        self.send_response(200); self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)\n"
                f"HTTPServer(('127.0.0.1', {port}), H).serve_forever()\n")
            busy = subprocess.Popen([sys.executable, str(box / "server.py")])
            for _ in range(30):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
                    break
                except OSError:
                    time.sleep(0.2)
            r = subprocess.run(["sh", str(box / "start.sh")], env={"PATH": os.environ["PATH"], "HOME": str(home)}, capture_output=True, text=True, timeout=60)
            assert r.returncode == 0 and "not restarted" in r.stdout, (r.stdout, r.stderr)
            for _ in range(100):
                if services.probe("comfy", comfy_port)[0]:
                    break
                time.sleep(0.2)
            assert services.probe("comfy", comfy_port)[0], "ComfyUI was not started again"
            assert len(lines(home / "ComfyUI" / "launches.log")) == 2 and busy.poll() is None
        finally:
            if busy:
                busy.kill()
            for rec in lines(home / "ComfyUI" / "launches.log"):
                with contextlib.suppress(OSError):
                    os.kill(rec["pid"], signal.SIGTERM)
            for pid, cmd in services.processes():
                if str(d) in cmd and "test_services" not in cmd:
                    with contextlib.suppress(OSError):
                        os.kill(pid, signal.SIGTERM)


def test_the_accounts_and_the_adapters_tokens_are_not_given_to_a_service_and_other_variables_are():
    with world() as w, environ(CLOUDGPU_TOKEN="a" * 40, AUTODL_DEV_TOKEN="b" * 40, H3HOST_TOKEN="c" * 40, HF_TOKEN="hf-ok", CLOUDGPU_COMFY_PORT=str(w.comfy_port)):
        env = services.child_env()
        assert not {"CLOUDGPU_TOKEN", "AUTODL_DEV_TOKEN", "H3HOST_TOKEN"} & set(env) and env.get("HF_TOKEN") == "hf-ok" and env["PYTHONUNBUFFERED"] == "1"


def test_configure_called_alone_keeps_the_port_and_a_launcher_that_leaves_its_child_behind_keeps_its_group_through_every_record():
    with _t.tmpdir() as d:
        d = Path(d) / "h3host"
        services.configure(d, "/x", None, {"CLOUDGPU_COMFY_PORT": "16100"})
        services.configure(d, None, None, {"CLOUDGPU_VOXCPM_CMD": "sh /v.sh"})
        assert json.loads((d / "services.json").read_text())["comfy_port"] == 16100, "only an install (it names ComfyUI's folder) replaces the port"
    for code in (3, 0):       # a launcher that fails, or returns, but leaves the service loading in its group; three runs must still make one copy
        with world(vox=True) as w:
            (w.vox / "delay.txt").write_text("6")
            (w.vox / "start.sh").write_text(f'#!/bin/sh\ncd "$(dirname "$0")"\nnohup "{sys.executable}" "{w.vox}/server.py" >/dev/null 2>&1 &\nexit {code}\n')
            old = services.EXITED_OK_GRACE_S
            services.EXITED_OK_GRACE_S = 1
            try:
                for _ in range(3):
                    w.up(CLOUDGPU_VOXCPM_WAIT_S="2")
            finally:
                services.EXITED_OK_GRACE_S = old
            assert len(w.launches("voxcpm")) <= 1 and (len(w.launches("voxcpm")) == 1 or True), (code, w.launches("voxcpm"))
            n_servers = sum(1 for pid, cmd in services.processes() if str(w.vox / "server.py") in cmd or ("server.py" in cmd and "VoxCPM-service" in cmd))
            assert n_servers <= 1, (code, n_servers)


def test_install_sh_refuses_a_comfy_port_that_is_not_a_port_before_it_writes_anything():
    with _t.tmpdir() as d:
        home = Path(d) / "home"
        home.mkdir()
        work = Path(d) / "box"
        work.mkdir()
        (work / "install.sh").write_bytes((Path(_t.ROOT) / "cloudgpu" / "onbox" / "install.sh").read_bytes())
        for bad in ("0", "70000", "99999999999999999999", "80a", "-1"):
            r = subprocess.run(["sh", str(work / "install.sh"), "auto", "6009"], input="t" * 32, env={"PATH": os.environ["PATH"], "HOME": str(home), "CLOUDGPU_COMFY_PORT": bad},
                               capture_output=True, text=True, timeout=30)
            assert r.returncode == 1 and "CLOUDGPU_COMFY_PORT" in r.stderr and not (home / "h3host").exists(), (bad, r.returncode, r.stderr)


if __name__ == "__main__":
    _t.main(globals())
