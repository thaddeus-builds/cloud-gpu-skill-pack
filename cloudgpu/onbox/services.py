#!/usr/bin/env python3
"""Run ON the box (Python 3.8+, standard library only): start the box's other services after a boot, in order, and say what state each is in.

    python3 services.py up [--detach] [--dir DIR]      start what is down: the adapter's start.sh runs this after the adapter (docs/cloud-gpu.md, "What a box runs")
    python3 services.py status [--json] [--dir DIR]    one line per service: up | starting | down | absent, and why (read only, starts nothing)
    python3 services.py configure [--dir DIR] [--comfy-root P] [--python P]   write the settings the install knew into DIR/services.json

The services, in the order they are started:

    comfy    ComfyUI on 127.0.0.1:6006 (`python main.py --listen 127.0.0.1 --port 6006` in its folder). Takes 2 to 3 minutes before it answers; up means GET /system_stats answers, the
             same test the adapter's /health uses for `comfy`.
    voxcpm   the VoxCPM voice service on 127.0.0.1:8190, started only when it is installed on this box (a launcher is found or CLOUDGPU_VOXCPM_CMD names one) and only AFTER ComfyUI
             answers (or the wait for it ended): ComfyUI gets the card first. This script never loads VoxCPM's weights: the service loads its model when the first voice is asked for
             (about 6 GB, which cannot sit next to H3's 46 GB on a 48 GB card), and the only call made to it is GET /voxcpm/v1/health, which loads nothing.

Idempotent: a service that answers is left alone, and so is one whose process is already running and still starting (a copy started by hand, by the vendor's panel or by an earlier
run: found by its command line), so a second start.sh never makes a second copy. A lock file keeps two starts (the vendor's power-on hook and systemd both run
start.sh at boot) from racing. Nothing is restarted, killed or stopped here.

Where things are (nothing is fixed to one image; each setting is the environment first, then DIR/services.json, then what is looked for):
    DIR                  CLOUDGPU_DIR, default ~/h3host (the adapter's folder; install.sh makes it)
    ComfyUI's folder     CLOUDGPU_COMFY_ROOT, else the folder install.sh was run with (services.json), else the first folder with a main.py and a comfy/ or custom_nodes/ folder among
                         ~/ComfyUI (/root/ComfyUI), /root/autodl-tmp/ComfyUI, ... and every folder one level under ~, /root, /root/autodl-tmp, /root/autodl-fs, /workspace, /opt
    ComfyUI's Python     CLOUDGPU_COMFY_PYTHON, else the first of ROOT/venv, ROOT/.venv, ~/miniconda3 (/root/miniconda3/bin/python), the Python install.sh ran with, this one, python3 on PATH, conda that can import torch
    other ComfyUI parts  CLOUDGPU_COMFY_PORT (6006), CLOUDGPU_COMFY_ARGS (extra arguments), CLOUDGPU_COMFY_CMD (the whole command, instead of the above), CLOUDGPU_COMFY_WAIT_S (420)
    VoxCPM               CLOUDGPU_VOXCPM_CMD (the whole command, run in CLOUDGPU_VOXCPM_DIR), else the launcher of a folder with "voxcpm" in its name (the operator's box:
                         `bash start.sh` in /root/voxcpm-svc) in CLOUDGPU_VOXCPM_DIR, DIR, or under ~, /root, /root/autodl-tmp, /root/autodl-fs, /workspace, /opt; launcher names:
                         start.sh, serve.sh, start_service.sh, run_service.sh, voxcpm_service.sh. A launcher may kill the copy it finds before it starts one (the operator's start.sh
                         kills the pid in svc.pid), so "is it running" is asked BEFORE the launcher is called, by the launcher's path and by the .py files its text names;
                         CLOUDGPU_VOXCPM_PORT (8190), CLOUDGPU_VOXCPM_WAIT_S (120), CLOUDGPU_VOXCPM_MATCH (a regex on the process command line, to recognise a copy that is already running)
    switch               CLOUDGPU_SERVICES_AUTOSTART=0 starts nothing (status still reports); CLOUDGPU_GPU_WAIT_S (90): how long to wait for nvidia-smi before ComfyUI is started
    CLOUDGPU_SERVICES_SEARCH  the folders searched (a path list) instead of the defaults above
Commands from the environment or services.json are split like a shell would and run WITHOUT a shell. The adapter's token is never given to a service.

Files, all under DIR: logs/comfyui.log, logs/voxcpm.log (what the service printed, appended; the old one is kept as .1 when it passes 20 MB), logs/services.log (this script),
services.state.json (what was seen last; the adapter's /health and `python3 -m cloudgpu status` say it), services.json (settings), services.lock. No secret is kept in any of them.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import fcntl
except ImportError:                                           # not a Linux box: no lock, the second start is on the person
    fcntl = None

ORDER = ("comfy", "voxcpm")
LABEL = {"comfy": "ComfyUI", "voxcpm": "VoxCPM"}
LOG_NAME = {"comfy": "comfyui.log", "voxcpm": "voxcpm.log"}
STATE_FILE, CONFIG_FILE, LOCK_FILE = "services.state.json", "services.json", "services.lock"
DEFAULT_PORT = {"comfy": 6006, "voxcpm": 8190}
DEFAULT_WAIT = {"comfy": 420, "voxcpm": 120}
HEALTH_PATH = {"comfy": "/system_stats", "voxcpm": "/voxcpm/v1/health"}
LOG_MAX = 20 * 1024 * 1024
LAUNCHERS = ("start.sh", "serve.sh", "start_service.sh", "run_service.sh", "voxcpm_service.sh")
EXITED_OK_GRACE_S = 60             # a launcher that returned 0 may have started the service in the background: wait this long more for it to answer
SELF_CMD = re.compile(r"(^|[\s/])services\.py\s+(up|status|configure)\b")      # this script's own command line is never "a copy of the service"
SETTINGS = {                       # key: (environment variable, kind)
    "autostart": ("CLOUDGPU_SERVICES_AUTOSTART", "flag"),
    "comfy_root": ("CLOUDGPU_COMFY_ROOT", "str"), "comfy_python": ("CLOUDGPU_COMFY_PYTHON", "str"), "comfy_args": ("CLOUDGPU_COMFY_ARGS", "str"),
    "comfy_cmd": ("CLOUDGPU_COMFY_CMD", "str"), "comfy_match": ("CLOUDGPU_COMFY_MATCH", "regex"), "comfy_port": ("CLOUDGPU_COMFY_PORT", "port"), "comfy_wait_s": ("CLOUDGPU_COMFY_WAIT_S", "secs"),
    "voxcpm_dir": ("CLOUDGPU_VOXCPM_DIR", "str"), "voxcpm_cmd": ("CLOUDGPU_VOXCPM_CMD", "str"), "voxcpm_port": ("CLOUDGPU_VOXCPM_PORT", "port"),
    "voxcpm_wait_s": ("CLOUDGPU_VOXCPM_WAIT_S", "secs"), "voxcpm_match": ("CLOUDGPU_VOXCPM_MATCH", "regex"),
    "gpu_wait_s": ("CLOUDGPU_GPU_WAIT_S", "secs"), "python_hint": ("CLOUDGPU_PYTHON", "str"),
}


# ---- settings ---------------------------------------------------------------------------------------------------------

def state_dir(arg: str | None = None, env=None) -> Path:
    env = os.environ if env is None else env
    return Path(os.path.expanduser(arg or env.get("CLOUDGPU_DIR") or "~/h3host"))


def coerce(kind: str, v):
    """The value in its kind, None when it is not one (a bad setting is ignored and said, never a crash)."""
    if v is None:
        return None
    s = str(v).strip()
    if kind == "flag":
        return None if s == "" else s.lower() not in ("0", "off", "false", "no")
    if kind == "str":
        return s or None
    if kind == "regex":
        try:
            re.compile(s)
        except re.error:
            return None
        return s or None
    try:
        n = int(s)
    except ValueError:
        return None
    if kind == "port":
        return n if 1 <= n <= 65535 else None
    return n if 0 <= n <= 86400 else None


def read_json(path: Path) -> dict:
    """The JSON object in a file, {} when it is missing, damaged, not an object or over 1 MB. Never raises."""
    try:
        with open(path, "rb") as f:
            doc = json.loads(f.read(1 << 20))
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def load_config(d: Path, env=None) -> tuple[dict, list]:
    """({key: value or None}, problems): the environment first (a variable that is set and not empty), then DIR/services.json."""
    env = os.environ if env is None else env
    file = read_json(d / CONFIG_FILE)
    out, problems = {}, []
    for key, (name, kind) in SETTINGS.items():
        raw = env.get(name)
        src = name
        if raw is None or str(raw).strip() == "":
            raw, src = file.get(key), f"{CONFIG_FILE} {key}"
        val = coerce(kind, raw)
        if raw not in (None, "") and val is None:
            problems.append(f"{src} is not a valid {kind}: ignored")
        out[key] = val
    return out, problems


def search_bases(d: Path, env=None) -> list[Path]:
    env = os.environ if env is None else env
    given = env.get("CLOUDGPU_SERVICES_SEARCH")
    if given:
        return [Path(os.path.expanduser(p)) for p in given.split(os.pathsep) if p]
    return [Path.home(), Path("/root"), Path("/root/autodl-tmp"), Path("/root/autodl-fs"), Path("/workspace"), Path("/opt")]


def is_comfy_root(p: Path) -> bool:
    return (p / "main.py").is_file() and ((p / "comfy").is_dir() or (p / "custom_nodes").is_dir())


def children(base: Path) -> list[Path]:
    try:
        return sorted(c for c in base.iterdir() if c.is_dir())
    except OSError:
        return []


def find_comfy_root(cfg: dict, d: Path, env=None) -> tuple[Path | None, list]:
    """(the folder, the places looked at): the configured one if it is ComfyUI's, else the first found."""
    tried, cands = [], []
    if cfg.get("comfy_root"):
        cands.append(Path(os.path.expanduser(cfg["comfy_root"])))
    bases = search_bases(d, env)
    for b in bases:
        cands += [b / "ComfyUI", b / "comfyui"]
    for b in bases:
        cands += children(b)
    seen = set()
    for c in cands:
        if c in seen:
            continue
        seen.add(c)
        if is_comfy_root(c):
            return c, tried
        if len(tried) < 4:
            tried.append(str(c))
    return None, tried


def has_torch(py: str) -> bool:
    try:
        r = subprocess.run([py, "-c", "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('torch') else 1)"], capture_output=True, timeout=60)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def find_comfy_python(cfg: dict, root: Path) -> tuple[str | None, list]:
    """(a Python that can import torch, the ones tried). The configured one is trusted as it is."""
    if cfg.get("comfy_python"):
        return cfg["comfy_python"], []
    cands = [str(root / "venv" / "bin" / "python"), str(root / ".venv" / "bin" / "python")]
    cands += [os.path.expanduser("~/miniconda3/bin/python"), "/root/miniconda3/bin/python"]
    if cfg.get("python_hint"):
        cands.append(cfg["python_hint"])
    cands += [sys.executable, shutil.which("python3") or "", shutil.which("python") or "", "/opt/conda/bin/python"]
    tried = []
    for c in cands:
        if not c or c in tried or not os.path.isfile(c):
            continue
        tried.append(c)
        if has_torch(c):
            return c, tried
    return None, tried


def find_voxcpm_launcher(cfg: dict, d: Path, env=None) -> tuple[Path | None, list]:
    dirs = []
    if cfg.get("voxcpm_dir"):
        dirs.append(Path(os.path.expanduser(cfg["voxcpm_dir"])))
    for b in [d] + search_bases(d, env):
        dirs += [c for c in children(b) if "voxcpm" in c.name.lower() and not re.search(r"bak|backup|old|tmp|orig|copy", c.name.lower())]
    tried = []
    for p in dirs:
        for name in LAUNCHERS:
            if (p / name).is_file():
                return p / name, tried
        if len(tried) < 3:
            tried.append(str(p))
    return None, tried


PY_RUN = r"^(\S*/)?(python[\d.]*|Python)\s+(\S+\s+)*?"      # `python [anything]` before the script: -X utf8, -u, -m ...


def launcher_texts(launcher: Path) -> list:
    """The launcher's text and, one level down, the texts of the sibling .sh scripts it names (a start.sh that calls an inner script)."""
    out = []
    try:
        text = launcher.read_text(errors="replace")[:50000]
    except OSError:
        return out
    out.append(text)
    for m in re.finditer(r"(?<![\w.-])([\w.-]+\.sh)\b", text):
        sib = launcher.parent / m.group(1)
        if sib != launcher and sib.is_file() and len(out) < 4:
            try:
                out.append(sib.read_text(errors="replace")[:50000])
            except OSError:
                pass
    return out


def launcher_match(launcher: Path) -> str:
    """A regex that recognises the service a launcher script starts, for the check that runs BEFORE the launcher is called (a launcher may kill the copy it finds and start another,
    as the operator's start.sh does with the pid in svc.pid): a process that runs the launcher (`bash PATH` or PATH), or runs one of the .py files its text (and its inner .sh scripts)
    mentions (voxcpm_service.py ...) as the script of a Python (with any flags, or `-m name`) or directly, or runs a Python/uvicorn/gunicorn with the launcher's folder in its command
    line. A command line that merely mentions a name (an editor, a tail, a grep) is not a copy. See also `pidfile_pids`, which the check adds."""
    lp = re.escape(str(launcher))
    parts = [r"^(\S*/)?(ba|da|z)?sh\s+(-\S+\s+)*%s(\s|$)" % lp, r"^%s(\s|$)" % lp]
    names = []
    for text in launcher_texts(launcher):
        for m in re.finditer(r"(?<![\w.-])([\w.-]+\.py)\b", text):
            if m.group(1) not in names and len(names) < 8:
                names.append(m.group(1))
    for n in names:
        e = re.escape(n)
        parts.append(PY_RUN + r"(\S*/)?%s(\s|$)" % e)
        parts.append(r"^(\S*/)?%s(\s|$)" % e)
        parts.append(PY_RUN + r"-m\s+%s(\s|$)" % re.escape(n[:-3]))
    mods = []
    for text in launcher_texts(launcher):
        for m in re.finditer(r"(?:-m\s+|(?:uvicorn|gunicorn)\s+(?:-\S+\s+)*)([A-Za-z_][\w.]*)", text):      # `python -m pkg.mod`, `uvicorn pkg.mod:app`
            if m.group(1) not in mods and m.group(1) not in ("pip", "venv") and len(mods) < 4:
                mods.append(m.group(1))
    for mod in mods:
        parts.append(r"^(\S*/)?(python[\d.]*|Python|uvicorn|gunicorn)\s+(\S+\s+)*?(-m\s+)?%s([\s:.]|$)" % re.escape(mod))
    parts.append(r"^(\S*/)?(python[\d.]*|Python|uvicorn|gunicorn)\s+.*%s/" % re.escape(str(launcher.parent)))
    return "|".join(parts)


def pidfile_pids(folder: Path) -> list:
    """[(pid file, pid)] of the *.pid files in a folder (a launcher such as the operator's start.sh keeps svc.pid and kills the pid in it before it starts)."""
    out = []
    try:
        files = sorted(folder.glob("*.pid"))
    except OSError:
        return out
    for f in files[:8]:
        try:
            out.append((f, int(f.read_text().strip())))
        except (OSError, ValueError):
            continue
    return out


def looks_like_the_service(cmd: str, folder: Path, plan_match: str) -> bool:
    """Is this command line the service of the launcher in `folder`? It matches the plan, or names the folder or "voxcpm"."""
    try:
        if plan_match and re.search(plan_match, cmd):
            return True
    except re.error:
        pass
    return str(folder) in cmd or "voxcpm" in cmd.lower()


def make_plan(name: str, cfg: dict, d: Path, env=None) -> dict:
    """What it takes to start one service: {argv, cwd, match, port, log, wait_s} - or {absent: why} (not installed here) or {blocked: why} (installed, cannot be started yet)."""
    port = cfg.get(name + "_port") or DEFAULT_PORT[name]
    base = {"name": name, "port": port, "log": d / "logs" / LOG_NAME[name], "wait_s": cfg.get(name + "_wait_s") or DEFAULT_WAIT[name]}
    if name == "comfy":
        root, tried = find_comfy_root(cfg, d, env)
        if cfg.get("comfy_cmd"):
            argv = shlex.split(cfg["comfy_cmd"])
            # the whole command as given, or ComfyUI's own `main.py ... --port N` line: a wrapper (`sh -c 'exec python main.py ...'`) shows a different line from the process it starts
            return {**base, "argv": argv, "cwd": root or d, "match": cfg.get("comfy_match") or r"(%s)|(main\.py\b.*--port[ =]%d(\s|$))" % (re.escape(" ".join(argv)), port)}
        if root is None:
            return {**base, "absent": "no ComfyUI folder found (a folder with main.py): looked at " + ", ".join(tried) + " and more; CLOUDGPU_COMFY_ROOT names it"}
        py, tried_py = find_comfy_python(cfg, root)
        if py is None:
            return {**base, "blocked": f"ComfyUI is at {root} but no Python that can import torch was found (tried {', '.join(tried_py) or 'none'}); CLOUDGPU_COMFY_PYTHON names it"}
        argv = [py, "main.py", "--listen", "127.0.0.1", "--port", str(port)] + (shlex.split(cfg["comfy_args"]) if cfg.get("comfy_args") else [])
        return {**base, "argv": argv, "cwd": root, "match": cfg.get("comfy_match") or r"main\.py\b.*--port[ =]%d(\s|$)" % port}
    if cfg.get("voxcpm_cmd"):
        argv = shlex.split(cfg["voxcpm_cmd"])
        cwd = Path(os.path.expanduser(cfg["voxcpm_dir"])) if cfg.get("voxcpm_dir") else d
        cwd = cwd if cwd.is_dir() else d
        script = next((cwd / a for a in argv[1:3] if a.endswith(".sh") and (cwd / a).is_file()), None)        # `bash start.sh`: the service is what that script starts
        match = cfg.get("voxcpm_match") or (launcher_match(script) if script else r"(^|\s)%s(\s|$)" % re.escape(" ".join(argv)))
        return {**base, "argv": argv, "cwd": cwd, "match": match, "guard": cwd if script else None}
    launcher, tried = find_voxcpm_launcher(cfg, d, env)
    if launcher is None:
        return {**base, "absent": "VoxCPM is not installed here: no launcher (" + ", ".join(LAUNCHERS) + ") in a voxcpm folder"
                + (f" (looked in {', '.join(tried)})" if tried else "") + "; CLOUDGPU_VOXCPM_CMD names the command"}
    shell = shutil.which("bash") or shutil.which("sh") or "sh"
    return {**base, "argv": [shell, str(launcher)], "cwd": launcher.parent, "match": cfg.get("voxcpm_match") or launcher_match(launcher), "guard": launcher.parent}


# ---- looking ------------------------------------------------------------------------------------------------------------

def http_get(port: int, path: str, timeout: float = 1.5) -> tuple:
    """(status, body, busy) of GET http://127.0.0.1:port/path. status None: nothing answered. busy: the connection was accepted but the answer did not arrive in full within
    `timeout` IN ALL (a single-threaded service that is working, or one that drips bytes): a timer shuts the socket at the deadline. At most 64 KB of the body are read. Never raises."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout * 2 + 1)      # the timer below is what ends the wait for an answer
    fired = []
    timer = None
    t0 = time.monotonic()
    try:
        try:
            conn.sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)         # the connect has the same budget
        except (socket.timeout, TimeoutError):
            return None, b"", True               # a connect that does not complete is a listener whose accept queue is full: working, not gone
        except Exception:
            return None, b"", False
        conn.sock.settimeout(timeout * 2 + 1)                # the timer is what ends the wait; the socket's own timeout only backs it up

        def stop():
            fired.append(1)
            try:
                conn.sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
        timer = threading.Timer(max(0.05, timeout - (time.monotonic() - t0)), stop)
        timer.daemon = True
        timer.start()
        try:
            conn.request("GET", path)
            r = conn.getresponse()
            body = b""
            while len(body) < 65536:
                chunk = r.read1(4096)
                if not chunk:
                    break
                body += chunk
            if fired:
                return None, b"", True
            return r.status, body, False
        except (socket.timeout, TimeoutError):
            return None, b"", True
        except Exception:
            return None, b"", bool(fired)
    finally:
        if timer is not None:
            timer.cancel()
        try:
            conn.close()
        except Exception:
            pass


def probe(name: str, port: int, timeout: float = 3.0) -> tuple[bool, dict | None]:
    """(answers, the JSON it said or None). ComfyUI: /system_stats must answer 200 with a JSON object. VoxCPM: any HTTP answer below 500 counts (a 404 is a server that is up), and
    so does a connection that is accepted but not answered in time (the service is working on a voice; the dict then is {"_busy": True}). Only GET /voxcpm/v1/health is ever sent
    to VoxCPM, and it loads nothing."""
    status, body, busy = http_get(port, HEALTH_PATH[name], timeout)
    if busy:
        return name == "voxcpm", ({"_busy": True} if name == "voxcpm" else None)
    if status is None:
        return False, None
    try:
        doc = json.loads(body)
    except Exception:
        doc = None
    doc = doc if isinstance(doc, dict) else None
    if name == "comfy":
        return status == 200 and doc is not None, doc
    return status < 500, doc


def alive(pid) -> bool:
    if type(pid) is not int or not 0 < pid < 2**31:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, OverflowError, ValueError):
        return False
    try:                                                      # a zombie is over
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat.rsplit(")", 1)[-1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def processes() -> list[tuple[int, str]]:
    """(pid, command line) of every process: /proc where there is one, else ps."""
    out = []
    if os.path.isdir("/proc/self"):
        for n in os.listdir("/proc"):
            if n.isdigit():
                try:
                    cmd = Path(f"/proc/{n}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
                except OSError:
                    continue
                if cmd:
                    out.append((int(n), cmd))
        if out:
            return out
    try:
        txt = subprocess.run(["ps", "-axo", "pid=,args="], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    for line in txt.splitlines():
        m = re.match(r"\s*(\d+)\s+(.*)", line)
        if m:
            out.append((int(m.group(1)), m.group(2)))
    return out


def process_groups() -> dict:
    """{pid: process group} of every process (/proc where there is one, else ps). A service started here is in a group of its own whose id is the pid of what was run, and a
    launcher that starts the service in the background and returns leaves it in that group."""
    out = {}
    if os.path.isdir("/proc/self"):
        for n in os.listdir("/proc"):
            if n.isdigit():
                try:
                    out[int(n)] = int(Path(f"/proc/{n}/stat").read_text().rsplit(")", 1)[1].split()[2])
                except (OSError, ValueError, IndexError):
                    continue
        if out:
            return out
    try:
        txt = subprocess.run(["ps", "-axo", "pid=,pgid="], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    for line in txt.splitlines():
        m = re.match(r"\s*(\d+)\s+(\d+)", line)
        if m:
            out[int(m.group(1))] = int(m.group(2))
    return out


def find_running(plan: dict, started_group=None) -> int | None:  # noqa: C901
    """The pid of a copy of the service that is running now: a process whose command line matches the plan, or one left in the group of the launcher an earlier run started
    (`started_group`, from the record of this boot); never this one. A pid file is not trusted: after a power-on the pids start over and an old number can belong to anything."""
    me = os.getpid()
    if plan.get("match"):
        try:
            rx = re.compile(plan["match"])
        except re.error:
            rx = None
        if rx is not None:
            for pid, cmd in processes():
                if pid != me and rx.search(cmd) and not SELF_CMD.search(cmd):
                    return pid
    if type(started_group) is int and started_group > 1:
        for pid, grp in process_groups().items():
            if grp == started_group and pid != me and alive(pid):
                return pid
    folder = plan.get("guard")
    if folder:                                               # the launcher's own pid file (svc.pid): its pid counts only when that process looks like the service
        table = dict(processes())
        for f, pid in pidfile_pids(folder):
            if pid != me and alive(pid) and pid in table and looks_like_the_service(table[pid], folder, plan.get("match") or ""):
                return pid
    return None


def retire_stale_pidfiles(plan: dict, say=print) -> None:
    """Before a launcher is called: a *.pid file in its folder whose process is alive but is NOT the service (after a power-on pids start over, and the number may be ComfyUI's or
    the adapter's) is renamed to .stale, so a launcher that kills the pid it finds does not kill an unrelated process."""
    folder = plan.get("guard")
    if not folder:
        return
    table = dict(processes())
    for f, pid in pidfile_pids(folder):
        if pid != os.getpid() and alive(pid) and not looks_like_the_service(table.get(pid, ""), folder, plan.get("match") or ""):
            try:
                os.replace(f, str(f) + ".stale")
                say(f"{f.name} named pid {pid}, which is not the service: set aside as {f.name}.stale so the launcher does not kill it")
            except OSError:
                pass


def container_start() -> float | None:
    """When this box (container or machine) was powered on, from its first process; None where /proc does not say. A record older than this is from an earlier run of the
    box: its pid may belong to another process now and it says nothing about this boot."""
    try:
        ticks = int(Path("/proc/1/stat").read_text().rsplit(")", 1)[1].split()[19])
        btime = next(int(l.split()[1]) for l in Path("/proc/stat").read_text().splitlines() if l.startswith("btime "))
        return btime + ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError, StopIteration, AttributeError):
        return None


def fresh(rec, since: float | None):
    """The record, or None when it is from before `since` (this boot)."""
    if not isinstance(rec, dict):
        return None
    if since is not None and isinstance(rec.get("updated"), (int, float)) and rec["updated"] < since - 2:
        return None
    return rec


def read_state(d: Path) -> dict:
    s = read_json(d / STATE_FILE).get("services")
    return s if isinstance(s, dict) else {}


def write_state(d: Path, name: str, **fields) -> None:
    cur = read_state(d)
    cur[name] = {**fields, "port": fields.get("port"), "updated": round(time.time(), 1)}
    tmp = d / (STATE_FILE + ".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"updated": round(time.time(), 1), "services": cur}, f, ensure_ascii=False)
    os.replace(tmp, d / STATE_FILE)


def classify(up: bool, rec: dict | None, is_alive, now: float) -> tuple[str, str]:
    """(state, reason) from what answers now and what the last start recorded. state: up | starting | down | absent. The adapter's /health carries a copy of this function
    (adapter.py `service_view`); tests/test_services.py holds the two equal on one table."""
    rec = rec if isinstance(rec, dict) else {}
    if up:
        return "up", (str(rec["reason"])[:300] if rec.get("state") == "up" and rec.get("reason") else "answers")
    st = rec.get("state")
    reason = str(rec.get("reason") or "")[:300]
    if not rec:
        return "down", "does not answer, and no start has been recorded since this box was powered on (start.sh has not started it yet, or does not start it)"
    if st == "absent":
        return "absent", reason or "not installed on this box"
    if st == "starting":
        since = rec.get("since")
        if is_alive(rec.get("pid")) or (isinstance(rec.get("until"), (int, float)) and now < rec["until"]):
            age = f" (for {int(now - since)} s)" if isinstance(since, (int, float)) else ""
            return "starting", (reason or "starting") + age
        return "down", "it was starting, but its process is gone and nothing answers"
    if st == "up":
        return "down", "it answered when it was started and does not now: its process ended or hangs; start.sh starts it again"
    return "down", reason or "does not answer"


def describe(name: str, cfg: dict, d: Path, now: float | None = None) -> dict:
    port = cfg.get(name + "_port") or DEFAULT_PORT[name]
    rec = fresh(read_state(d).get(name), container_start())
    ok, doc = probe(name, port)
    state, reason = classify(ok, rec, alive, time.time() if now is None else now)
    if ok and isinstance(doc, dict) and doc.get("_busy"):
        reason = "accepts connections but did not answer /voxcpm/v1/health in time (working on a voice?)"
    out = {"state": state, "reason": reason, "port": port, "log": f"logs/{LOG_NAME[name]}"}
    if name == "voxcpm":
        out["loaded"] = bool(doc.get("loaded")) if ok and isinstance(doc, dict) and "loaded" in doc and not doc.get("_busy") else None
    if isinstance(rec, dict) and rec.get("last_line") and state == "down":
        out["last_line"] = rec["last_line"]
    return out


# ---- starting -----------------------------------------------------------------------------------------------------------

def last_line(path: Path) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 4096))
            lines = [x.strip() for x in f.read().decode(errors="replace").splitlines() if x.strip()]
    except OSError:
        return ""
    return re.sub(r"[^\x20-\x7e -￿]", "", lines[-1])[:160] if lines else ""


def gpu_ready(wait_s: int, sleep=time.sleep) -> tuple[bool, str]:
    """ComfyUI on a card that is not there yet would fall back to the CPU or fail: wait for nvidia-smi to see a GPU. No nvidia-smi on this box: not checked."""
    exe = shutil.which("nvidia-smi") or next((p for p in ("/usr/bin/nvidia-smi", "/usr/local/nvidia/bin/nvidia-smi") if os.path.isfile(p)), None)
    if not exe:
        return True, "nvidia-smi not found: the GPU was not checked"
    end = time.time() + wait_s
    while True:
        try:
            r = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=30)
            if r.returncode == 0 and "GPU" in r.stdout:
                return True, "GPU seen"
        except (OSError, subprocess.SubprocessError):
            pass
        if time.time() >= end:
            return False, f"nvidia-smi sees no GPU after {wait_s} s (the driver is not ready?)"
        sleep(3)


def open_log(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if path.stat().st_size > LOG_MAX:
            os.replace(path, str(path) + ".1")
    except OSError:
        pass
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    return os.fdopen(fd, "ab")


def child_env() -> dict:
    env = {k: v for k, v in os.environ.items() if not (k.endswith("TOKEN") and k.split("_")[0] in ("CLOUDGPU", "H3HOST", "AUTODL"))}      # the adapter's and the account's tokens
    env["PYTHONUNBUFFERED"] = "1"
    return env


def launch(plan: dict, d: Path):
    """Start the service detached from this script, its output appended to its log. Returns the Popen."""
    logf = open_log(plan["log"])
    logf.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} cloudgpu services.py starts {plan['name']}: {' '.join(shlex.quote(a) for a in plan['argv'])} (in {plan['cwd']}) ===\n".encode())
    logf.flush()
    try:
        p = subprocess.Popen(plan["argv"], cwd=str(plan["cwd"]), stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT, env=child_env(), start_new_session=True)
    finally:
        logf.close()
    return p


class Starter:
    def __init__(self, d: Path, cfg: dict, say=print, sleep=time.sleep, env=None):
        self.d, self.cfg, self.say, self.sleep, self.env = d, cfg, say, sleep, env

    def rec(self, name, state, reason, plan=None, **extra):
        write_state(self.d, name, state=state, reason=reason, since=extra.pop("since", round(time.time(), 1)), log=f"logs/{LOG_NAME[name]}",
                    port=(plan or {}).get("port") or self.cfg.get(name + "_port") or DEFAULT_PORT[name], **extra)
        self.say(f"{LABEL[name]}: {state}: {reason}")

    def wait(self, plan, pid, proc, started, group=None):
        """Poll until the service answers, its process ends badly, or plan['wait_s'] passes. Returns True when it answers. `group`: the process group of what was started here
        (its pid): a launcher that returned 0 but left processes in it is still starting the service."""
        name, deadline, grace_end = plan["name"], started + plan["wait_s"], None
        while True:
            if probe(name, plan["port"])[0]:
                self.rec(name, "up", f"answers on :{plan['port']} {int(time.time() - started)} s after it was started", plan, pid=pid)
                return True
            code = proc.poll() if proc is not None else (None if alive(pid) else "gone")
            if code == "gone":
                self.rec(name, "down", f"the process (pid {pid}) that was starting ended without answering: see {plan['log'].relative_to(self.d)}", plan, pid=None, last_line=last_line(plan["log"]))
                return False
            if code is not None:
                ll = last_line(plan["log"])
                if code != 0:
                    self.rec(name, "down", f"its process exited with code {code} {int(time.time() - started)} s after it was started: see {plan['log'].relative_to(self.d)}",
                             plan, pid=None, group=group, last_line=ll)
                    return False
                grace_end = grace_end or min(deadline, time.time() + EXITED_OK_GRACE_S)        # a launcher that returned 0 may have left the service running in the background
                if time.time() >= grace_end and not (group and find_running(plan, group)):      # ... and while something it started still runs, the service is still loading
                    self.rec(name, "down", f"the process ended (exit 0) and nothing answers on :{plan['port']}: see {plan['log'].relative_to(self.d)}", plan, pid=pid, group=group, last_line=ll)
                    return False
            if time.time() >= deadline:
                live = (find_running(plan, group) if group else None) or pid
                self.rec(name, "starting", f"still not answering on :{plan['port']} {plan['wait_s']} s after it was started; the process is running (see {plan['log'].relative_to(self.d)})",
                         plan, pid=live, group=group, since=round(started, 1))
                return False
            self.sleep(1)

    def ensure(self, name: str) -> str:
        cfg, d = self.cfg, self.d
        plan = make_plan(name, cfg, d, self.env)
        port = plan["port"]
        if probe(name, port)[0]:
            self.rec(name, "up", f"answers on :{port}; it was already running", plan)
            return "up"
        if "absent" in plan:
            self.rec(name, "absent", plan["absent"], plan)
            return "absent"
        if cfg.get("autostart") is False:
            self.rec(name, "down", "not started: CLOUDGPU_SERVICES_AUTOSTART is off", plan)
            return "down"
        if "blocked" in plan:
            self.rec(name, "down", plan["blocked"], plan)
            return "down"
        rec = fresh(read_state(d).get(name), container_start())
        pid = find_running(plan, rec.get("group") if rec else None)
        if pid:                                              # a copy already runs (by hand, the panel, an earlier start): no second one
            started = time.time()
            grp = rec.get("group") if rec else None
            self.rec(name, "starting", f"already running (pid {pid}, not started by this run) and not answering yet on :{port}: waited for, not started again", plan, pid=pid,
                     group=grp, until=started + plan["wait_s"])
            return "up" if self.wait(plan, pid, None, started, group=grp) else "starting"
        if name == "comfy":
            ok, why = gpu_ready(cfg.get("gpu_wait_s") if cfg.get("gpu_wait_s") is not None else 90, self.sleep)
            if not ok:
                self.rec(name, "down", f"not started: {why}", plan)
                return "down"
        retire_stale_pidfiles(plan, self.say)
        try:
            proc = launch(plan, d)
        except OSError as e:
            self.rec(name, "down", f"could not start {' '.join(plan['argv'][:2])}: {e.strerror or e}", plan)
            return "down"
        started = time.time()
        self.rec(name, "starting", f"started (pid {proc.pid}); not answering yet on :{port}", plan, pid=proc.pid, group=proc.pid, until=started + plan["wait_s"])
        return "up" if self.wait(plan, proc.pid, proc, started, group=proc.pid) else "starting"


def up(d: Path, env=None, say=print, sleep=time.sleep) -> int:
    """Start what is down, in order. 0 when finished (whatever the states: they are in services.state.json), 0 also when another start holds the lock."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "logs").mkdir(parents=True, exist_ok=True)
    lockf = None
    if fcntl is not None:
        lockf = open(d / LOCK_FILE, "a+")
        try:
            fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            say("another start of the services is running: not starting a second one (python3 services.py status)")
            return 0
        except OSError as e:                                  # no locking on this filesystem: go on without it rather than never start anything
            say(f"no lock could be taken ({e.strerror or e}): starting without one")
    cfg, problems = load_config(d, env)
    for p in problems:
        say("setting: " + p)
    starter = Starter(d, cfg, say, sleep, env)
    for name in ORDER:
        try:
            starter.ensure(name)
        except Exception as e:                                # one service's trouble never stops the next from being tried
            try:
                starter.rec(name, "down", f"the start script failed: {type(e).__name__}: {e}"[:300])
            except OSError:
                say(f"{LABEL[name]}: the start script failed: {e}")
    return 0


def detach(d: Path) -> int:
    """Run `up` in a process of its own (the power-on hook must come back quickly) and say how to watch it."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "logs").mkdir(parents=True, exist_ok=True)
    logf = open_log(d / "logs" / "services.log")
    try:
        p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "up", "--dir", str(d)], stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT,
                             env=child_env(), start_new_session=True)
    finally:
        logf.close()
    print(f"services: starting ComfyUI, then VoxCPM if installed, in the background (pid {p.pid}); watch: {sys.executable} {os.path.abspath(__file__)} status --dir {d}; logs: {d}/logs/")
    return 0


def status(d: Path, as_json: bool, env=None) -> int:
    cfg, _ = load_config(d, env)
    rows = {n: describe(n, cfg, d) for n in ORDER}
    if as_json:
        print(json.dumps(rows, ensure_ascii=False))
    else:
        for n in ORDER:
            r = rows[n]
            print(f"{n:<7} {r['state']:<9} {r['reason']}")
    return 0


def configure(d: Path, comfy_root: str | None, python: str | None, env=None) -> int:
    """Merge what the install knew (and any CLOUDGPU_* setting in the environment) into DIR/services.json, 0600. A setting that is empty or unset changes nothing."""
    env = os.environ if env is None else env
    d.mkdir(parents=True, exist_ok=True)
    cur = read_json(d / CONFIG_FILE)
    if comfy_root:                       # an install (it names ComfyUI's folder): start.sh tells the adapter ComfyUI's port only from the install's environment, so the two must agree and a port from an earlier install is not kept
        cur.pop("comfy_port", None)
    for key, (name, kind) in SETTINGS.items():
        if key == "python_hint":
            continue
        v = coerce(kind, env.get(name))
        if v is not None:
            cur[key] = v
    if comfy_root:
        cur["comfy_root"] = comfy_root
    if python:
        cur["python_hint"] = python
    tmp = d / (CONFIG_FILE + ".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(cur, f, indent=1, sort_keys=True)
    os.replace(tmp, d / CONFIG_FILE)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="start and report the box's ComfyUI and VoxCPM services")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("up", "status", "configure"):
        s = sub.add_parser(c)
        s.add_argument("--dir", help="the adapter's folder (CLOUDGPU_DIR, default ~/h3host)")
        if c == "up":
            s.add_argument("--detach", action="store_true", help="run in the background and return at once")
        if c == "status":
            s.add_argument("--json", action="store_true")
        if c == "configure":
            s.add_argument("--comfy-root")
            s.add_argument("--python")
    a = ap.parse_args(argv)
    d = state_dir(a.dir)
    if a.cmd == "up":
        return detach(d) if a.detach else up(d)
    if a.cmd == "status":
        return status(d, a.json)
    return configure(d, a.comfy_root, a.python)


if __name__ == "__main__":
    sys.exit(main())
