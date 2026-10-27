"""cloudgpu.box, power and tunnel, offline: the box is powered on only when it is down and a backend can, never off while a take waits or on a
pass, the tunnel is opened by key with no prompt and only when needed, the SSH door comes from the account when it is not named and is never a shell
string, autodl is the built-in default backend, and the status never prints a setting's value. A fake adapter (the real one over a fake ComfyUI)
answers /health."""

import atexit
import hashlib
import hmac
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import _t

from cloudgpu import config, power, tunnel
from cloudgpu.vendors import autodl as vendor_autodl
from cloudgpu.box import Box
from cloudgpu.status import status


@contextmanager
def autodl_api(answer):
    """The autodl backend's one network call replaced: `answer(method, path, body, query)`; the calls are recorded."""
    calls = []
    real = vendor_autodl.AutoDL._call

    def call(self, method, path, body=None, query=None):
        calls.append((method, path, (body or {}).get("start_command")))
        return answer(method, path, body, query)
    vendor_autodl.AutoDL._call = call
    try:
        yield calls
    finally:
        vendor_autodl.AutoDL._call = real


def box(env, **kw):
    return Box({"CLOUDGPU_BASE": "https://b.example", "CLOUDGPU_TOKEN": "t" * 30, **env}, sleep=lambda s: None, **kw)


def test_settings_read_the_new_name_first_and_the_old_one_second_and_never_show_a_value():
    env = {"H3HOST_TOKEN": "old-secret", "CLOUDGPU_BASE": "https://x", "H3HOST_BASE": "https://old"}
    assert config.get("TOKEN", env) == "old-secret" and config.get("BASE", env) == "https://x" and config.get("BOX", env) is None
    shown = config.present({**env, "AUTODL_DEV_TOKEN": "dev-secret"})
    assert shown["TOKEN"] == "set" and shown["BOX"] == "unset" and shown["AUTODL_DEV_TOKEN"] == "set"
    assert "old-secret" not in json.dumps(shown) and "dev-secret" not in json.dumps(shown)
    with _t.tmpdir() as d:
        f = Path(d) / "e.env"
        f.write_text("# c\nCLOUDGPU_BASE='http://127.0.0.1:1'\nCLOUDGPU_TOKEN=abc\n\nnonsense\n")
        assert config.read_env_file(f) == {"CLOUDGPU_BASE": "http://127.0.0.1:1", "CLOUDGPU_TOKEN": "abc"}


def test_autodl_is_the_built_in_default_backend_and_none_and_an_unknown_name_are_handled():
    assert power.backend({}).name == "autodl" and not power.backend({}).configured() and power.BACKENDS == ("none", "autodl")                  # default, but it says what to set when used
    assert power.backend({"CLOUDGPU_INSTANCE": "pro-1", "AUTODL_DEV_TOKEN": "d"}).configured()
    assert power.backend({"H3HOST_INSTANCE": "pro-1", "AUTODL_DEV_TOKEN": "d"}).configured()          # the older variable name
    assert power.backend({"CLOUDGPU_POWER": "none"}).name == "none"
    with _t.raises("unknown"):
        power.backend({"CLOUDGPU_POWER": "vastai"})
    base = power.Power()                       # the contract every backend inherits: it can say nothing and turn nothing off
    assert base.status() is None and base.power_off() is False and base.pass_info() == {} and base.ssh() is None and not base.configured()
    with _t.raises("key_missing") as c:                                       # the default backend used with nothing set: what to set, or to say none
        power.backend({}).status()
    assert "CLOUDGPU_POWER=none" in " ".join(c.err.next)


def test_the_box_is_powered_on_only_when_it_is_down_and_a_backend_can():
    with autodl_api(lambda m, path, b, q: "shutdown" if path == "status" else None) as calls:
        b = box({"CLOUDGPU_INSTANCE": "pro-1", "AUTODL_DEV_TOKEN": "dev"})
        answers = iter([None, None, {"ok": True, "comfy": False}, {"ok": True, "comfy": True, "busy": 0}])
        b.health = lambda: next(answers)
        b.ensure_up()
        assert calls == [("GET", "status", None), ("POST", "power_on", "/root/h3host/start.sh")], calls
        b.ensure_up()                                          # up once: no second call
        assert len(calls) == 2
        calls.clear()
        b = box({"CLOUDGPU_INSTANCE": "pro-1", "AUTODL_DEV_TOKEN": "dev"})
        b.health = lambda: {"ok": True, "comfy": True, "busy": 0}          # already on: no power call
        b.ensure_up()
        assert calls == []
    with autodl_api(lambda *a: None) as calls:                              # down, backend `none`: refused, nothing powered
        b = box({"CLOUDGPU_POWER": "none"})
        b.health = lambda: None
        with _t.raises("down"):
            b.ensure_up()
        assert calls == []
    b = box({})                                                             # down, the default backend with no variables: told what to set
    b.health = lambda: None
    with _t.raises("key_missing"):
        b.ensure_up()


def test_it_gives_up_when_the_box_never_comes_up():
    with autodl_api(lambda m, path, b, q: "running"):
        b = box({"CLOUDGPU_INSTANCE": "pro-1", "AUTODL_DEV_TOKEN": "dev"})
        b.health = lambda: None
        import cloudgpu.box as bx
        real = bx.UP_TIMEOUT_S
        bx.UP_TIMEOUT_S = -1
        try:
            with _t.raises("timeout"):
                b.ensure_up()
        finally:
            bx.UP_TIMEOUT_S = real


def test_the_box_is_powered_off_only_when_nothing_waits_on_it_and_it_is_not_a_pass():
    billing = ["payg"]

    def api(m, path, b, q):
        return {"list": [{"uuid": "pro-1", "charge_type": billing[0]}]} if path == "list" else None
    with autodl_api(api) as calls:
        b = box({"CLOUDGPU_INSTANCE": "pro-1", "AUTODL_DEV_TOKEN": "dev"})
        b.health = lambda: {"ok": True, "comfy": True, "busy": 2}          # another process has takes on it
        assert b.power_off_if_idle() is False and calls == []
        b.health = lambda: {"ok": True, "comfy": True, "busy": 0}
        assert b.power_off_if_idle() is True and calls[-1][:2] == ("POST", "power_off")
        calls.clear()
        billing[0] = "daily"                                                # a day pass is paid anyway: left on
        assert b.power_off_if_idle() is False and ("POST", "power_off", None) not in calls
    for env in ({"CLOUDGPU_POWER": "none"}, {}):                             # no backend, or an unconfigured default: nothing to do
        b = box(env)
        b.health = lambda: {"ok": True, "comfy": True, "busy": 0}
        assert b.power_off_if_idle() is False


def test_the_tunnel_is_opened_by_key_with_no_prompt_and_only_when_configured_and_needed():
    with _t.tmpdir() as d:
        key = Path(d) / "k"
        key.write_text("private")
        key.chmod(0o600)
        spawned = []
        spawn = lambda argv: spawned.append(argv) or True       # noqa: E731
        b = box({"CLOUDGPU_BASE": "http://127.0.0.1:16009", "CLOUDGPU_SSH_KEY": str(key), "CLOUDGPU_SSH": "ops@gpu.example:2222"}, spawn=spawn)
        b.health = lambda: None                                          # nothing answers: open it
        assert b.tunnel() is True
        argv = spawned[0]
        assert argv[:3] == ["ssh", "-f", "-N"] and "127.0.0.1:16009:127.0.0.1:6009" in argv and argv[argv.index("-p") + 1] == "2222"
        assert argv[argv.index("-i") + 1] == str(key) and "BatchMode=yes" in argv and "IdentitiesOnly=yes" in argv and argv[-1] == "ops@gpu.example"
        assert not any("sshpass" in a or "ASKPASS" in a for a in argv)
        b.health = lambda: {"ok": True, "comfy": True, "busy": 0}        # already answering (an open tunnel): nothing is spawned
        spawned.clear()
        assert b.tunnel() is False and spawned == []
        b = box({"CLOUDGPU_BASE": "https://public.example:8443", "CLOUDGPU_SSH_KEY": str(key), "CLOUDGPU_SSH": "gpu.example:22"}, spawn=spawn)
        b.health = lambda: None                                          # a public address is not tunnelled to
        assert b.tunnel() is False and spawned == []
        b = box({"CLOUDGPU_BASE": "http://127.0.0.1:16009"}, spawn=spawn)    # no key: not configured
        b.health = lambda: None
        assert b.tunnel() is False and spawned == []
        key.chmod(0o644)                                                 # a key anyone can read is refused, as ssh itself would
        b = box({"CLOUDGPU_BASE": "http://127.0.0.1:16009", "CLOUDGPU_SSH_KEY": str(key), "CLOUDGPU_SSH": "gpu.example:22"}, spawn=spawn)
        b.health = lambda: None
        with _t.raises("key_missing"):
            b.tunnel()
        key.chmod(0o600)
        b = Box({"CLOUDGPU_TOKEN": "t" * 30, "CLOUDGPU_SSH_KEY": str(key)})                     # no base: the local default
        assert b.root() == "http://127.0.0.1:16009"


def test_the_ssh_door_comes_from_the_account_when_it_is_not_named_and_is_never_a_shell_string():
    assert tunnel.parse_ssh("root@gpu.example:2222") == ("root", "gpu.example", 2222) and tunnel.parse_ssh("gpu.example:22") == ("root", "gpu.example", 22)
    for bad in ("gpu.example", "gpu.example:0", "gpu.example:99999", "gpu.example;rm -rf /:22", "$(id)@gpu.example:22", "gpu example:22", ":22", ""):
        with _t.raises("key_missing"):
            tunnel.parse_ssh(bad)
    with _t.tmpdir() as d:
        key = Path(d) / "k"
        key.write_text("private")
        key.chmod(0o600)
        spawned = []
        with autodl_api(lambda m, path, b, q: {"proxy_host": "gpu7.example", "ssh_port": 31337, "root_password": "never-read"}) as calls:
            b = box({"CLOUDGPU_BASE": "http://127.0.0.1:16009", "CLOUDGPU_SSH_KEY": str(key), "CLOUDGPU_INSTANCE": "pro-1", "AUTODL_DEV_TOKEN": "dev"},
                    spawn=lambda argv: spawned.append(argv) or True)
            b.health = lambda: None
            assert b.tunnel() is True and calls[0][:2] == ("GET", "snapshot")
            assert spawned[0][spawned[0].index("-p") + 1] == "31337" and spawned[0][-1] == "root@gpu7.example"
        b = box({"CLOUDGPU_BASE": "http://127.0.0.1:16009", "CLOUDGPU_SSH_KEY": str(key), "CLOUDGPU_POWER": "none"})      # no name and no backend to say it
        b.health = lambda: None
        with _t.raises("key_missing"):
            b.tunnel()


class FakeAdapter:
    """What the real adapter answers on /health. `token`: it proves it knows it (HMAC of the nonce), as the real one does; None: a process that does not know it.
    `card`/`worker` are what it reports; `seen` is every request's Authorization header."""

    def __init__(self, comfy=True, busy=0, token="t" * 30, card="NVIDIA GeForce RTX 4090, 49140 MiB", worker="alive"):
        self.seen = seen = []

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                seen.append(self.headers.get("Authorization"))
                d = {"ok": worker == "alive", "comfy": comfy, "busy": busy, "worker": worker, "card": card}
                nonce = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get("nonce", [""])[0]
                if token and nonce:
                    d["proof"] = hmac.new(token.encode(), nonce.encode(), hashlib.sha256).hexdigest()
                doc = json.dumps(d).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(doc)))
                self.end_headers()
                self.wfile.write(doc)
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()


def test_the_status_reads_the_box_and_the_pass_and_never_prints_a_secret():
    a = FakeAdapter()
    listing = {"list": [{"uuid": "pro-1", "status": "running", "charge_type": "daily", "gpu_spec_uuid": "v-48g", "region_name": "x",
                         "expired_at": {"Time": "2999-01-01T00:00:00+08:00", "Valid": True}}]}
    env = {"CLOUDGPU_BASE": a.base, "CLOUDGPU_TOKEN": "t" * 30, "CLOUDGPU_BOX": "rtx4090", "CLOUDGPU_INSTANCE": "pro-1", "AUTODL_DEV_TOKEN": "dev-secret-value",
           "CLOUDGPU_AUTOOFF": "1"}
    try:
        with autodl_api(lambda m, path, b, q: listing):
            doc = status(env)
        raw = json.dumps(doc)
        assert doc["ready"] is True and doc["box"] == {"answers": True, "comfy": True, "busy": 0, "worker": "alive", "card": "NVIDIA GeForce RTX 4090, 49140 MiB"}, doc
        assert doc["warnings"] == [] and not any("card" in n for n in doc["next"]), doc
        assert doc["account"]["backend"] == "autodl" and doc["account"]["billing"] == "daily" and doc["account"]["hours_left"] > 1000
        assert "t" * 30 not in raw and "dev-secret-value" not in raw and a.base not in raw
        assert any("CLOUDGPU_AUTOOFF" in n for n in doc["next"]), doc["next"]
    finally:
        a.srv.shutdown()


def test_the_status_says_what_is_missing_when_nothing_is_set_and_when_the_box_is_off():
    doc = status({})
    assert doc["ready"] is False and doc["box"]["answers"] is False and any("CLOUDGPU_TOKEN" in n for n in doc["next"]) and any("CLOUDGPU_BOX" in n for n in doc["next"])
    doc = status({"CLOUDGPU_BASE": "http://127.0.0.1:9", "CLOUDGPU_TOKEN": "t" * 30, "CLOUDGPU_BOX": "b"})
    assert doc["ready"] is False and doc["box"]["answers"] is False
    assert any("not configured" in n for n in doc["next"]) and any("CLOUDGPU_SSH_KEY" in n for n in doc["next"]), doc["next"]


def test_a_present_key_wins_even_empty_off_means_off_everywhere_and_a_token_loses_its_newline():
    assert config.get("AUTOOFF", {"CLOUDGPU_AUTOOFF": "", "H3HOST_AUTOOFF": "1"}) is None                 # present and empty: off, not the older name's 1
    assert config.get("AUTOOFF", {"H3HOST_AUTOOFF": "1"}) == "1" and config.get("TOKEN", {"CLOUDGPU_TOKEN": " abc\n"}) == "abc"
    assert config.get("TOKEN", {"CLOUDGPU_TOKEN": "  \n"}) is None
    for v, on in (("1", True), ("true", True), ("0", False), ("off", False), ("", False), ("no", False)):
        assert config.flag("AUTOOFF", {"CLOUDGPU_AUTOOFF": v}) is on, v
    assert config.flag("AUTOOFF", {"CLOUDGPU_AUTOOFF": "", "H3HOST_AUTOOFF": "1"}) is False
    pass_api = {"list": [{"uuid": "pro-1", "charge_type": "daily", "expired_at": {"Time": "2999-01-01T00:00:00+08:00", "Valid": True}}]}
    a = FakeAdapter()
    try:
        with autodl_api(lambda m, path, b, q: pass_api):
            for v, expect in (("0", False), ("1", True)):                                                       # AUTOOFF=0 on a pass: no "unset it" advice
                doc = status({"CLOUDGPU_BASE": a.base, "CLOUDGPU_TOKEN": "t" * 30, "CLOUDGPU_BOX": "rtx4090", "CLOUDGPU_INSTANCE": "pro-1",
                              "AUTODL_DEV_TOKEN": "d", "CLOUDGPU_AUTOOFF": v})
                assert any("CLOUDGPU_AUTOOFF" in n for n in doc["next"]) is expect, (v, doc["next"])
    finally:
        a.srv.shutdown()


def test_nothing_unknown_powers_a_box_off():
    billing = ["payg"]
    listing = [{"uuid": "pro-1", "charge_type": billing[0]}]
    with autodl_api(lambda m, path, b, q: {"list": listing} if path == "list" else None) as calls:
        b = box({"CLOUDGPU_INSTANCE": "pro-1", "AUTODL_DEV_TOKEN": "dev"})
        for h in (None, {}, {"ok": True, "comfy": True}, {"ok": True, "comfy": True, "busy": None}, {"ok": False, "comfy": True, "busy": 0, "worker": "dead"},
                  {"ok": True, "comfy": True, "busy": 1}):
            b.health = lambda h=h: h
            assert b.power_off_if_idle() is False, h
        assert ("POST", "power_off", None) not in calls
        b.health = lambda: (_ for _ in ()).throw(_t_error())                          # a listener that is not the adapter
        assert b.power_off_if_idle() is False
        b.health = lambda: {"ok": True, "comfy": True, "busy": 0}
        listing.clear()                                                                # the account lists no such instance: pass_info says error
        assert b.power_off_if_idle() is False and ("POST", "power_off", None) not in calls
        listing.append({"uuid": "pro-1", "charge_type": "payg"})
        assert b.power_off_if_idle() is True
    with autodl_api(lambda m, path, b, q: (_ for _ in ()).throw(_t_error())) as calls:      # the account call itself fails
        b = box({"CLOUDGPU_INSTANCE": "pro-1", "AUTODL_DEV_TOKEN": "dev"})
        b.health = lambda: {"ok": True, "comfy": True, "busy": 0}
        assert b.power_off_if_idle() is False


def _t_error():
    from cloudgpu.errors import CloudGpuError
    return CloudGpuError("http", "x")


def test_the_power_off_at_exit_is_registered_only_by_the_process_that_powered_the_box_on():
    registered = []
    real = atexit.register
    atexit.register = lambda f, *a: registered.append(f)
    try:
        with autodl_api(lambda m, path, b, q: "shutdown" if path == "status" else None):
            env = {"CLOUDGPU_INSTANCE": "pro-1", "AUTODL_DEV_TOKEN": "dev", "CLOUDGPU_AUTOOFF": "1"}
            b = box(env)
            b.health = lambda: {"ok": True, "comfy": True, "busy": 0}                 # already on when this process came: it did not start it
            b.ensure_up()
            assert registered == []
            b = box(env)
            answers = iter([None, None, {"ok": True, "comfy": True, "busy": 0}])
            b.health = lambda: next(answers)
            b.ensure_up()                                                              # this process powered it on
            assert len(registered) == 1 and registered[0] == b._power_off_at_exit
            b = box({**env, "CLOUDGPU_AUTOOFF": "0"})
            answers = iter([None, None, {"ok": True, "comfy": True, "busy": 0}])
            b.health = lambda: next(answers)
            b.ensure_up()
            assert len(registered) == 1                                                # AUTOOFF=0: off
    finally:
        atexit.register = real


def test_the_exit_hook_is_registered_as_soon_as_this_process_powers_the_box_on_so_a_timeout_leaves_nothing_on():
    import cloudgpu.box as bx
    registered, real, real_to = [], atexit.register, bx.UP_TIMEOUT_S
    atexit.register = lambda f, *a: registered.append(f)
    bx.UP_TIMEOUT_S = -1                                                                   # the wait gives up at once
    try:
        status = ["shutdown"]
        listing = [{"uuid": "pro-1", "charge_type": "payg"}]

        def api(m, path, b, q):
            return status[0] if path == "status" else {"list": listing} if path == "list" else None
        with autodl_api(api) as calls:
            env = {"CLOUDGPU_INSTANCE": "pro-1", "AUTODL_DEV_TOKEN": "dev", "CLOUDGPU_AUTOOFF": "1"}
            b = box(env)
            b.health = lambda: None
            with _t.raises("timeout"):
                b.ensure_up()
            assert len(registered) == 1 and ("POST", "power_on", "/root/h3host/start.sh") in calls
            with _t.raises("timeout"):                                                     # a second try: it is starting, no second power-on, one hook
                status[0] = "starting"
                b.ensure_up()
            assert len(registered) == 1 and [c for c in calls if c[1] == "power_on"] == [("POST", "power_on", "/root/h3host/start.sh")]
            calls.clear()
            for h in (None, {"ok": True, "comfy": True, "busy": 1}, {"ok": True, "comfy": True}):          # the hook does nothing when the box is unknown or busy
                b.health = lambda h=h: h
                registered[0]()
            assert ("POST", "power_off", None) not in calls
            b.health = lambda: {"ok": True, "comfy": True, "busy": 0}
            registered[0]()                                                                # idle, hourly, ours: off
            assert calls[-1][:2] == ("POST", "power_off")
            calls.clear()
            b._powered = False                                                             # a box this process did not power on is never powered off by the hook
            registered[0]()
            assert calls == []
            b = box(env)                                                                   # powered on by the call, power_on itself failing: nothing to hook
            b.health = lambda: None
            with autodl_api(lambda m, path, bb, q: (_ for _ in ()).throw(_t_error()) if path == "power_on" else "shutdown"):
                with _t.raises("http"):
                    b.ensure_up()
            assert len(registered) == 1
    finally:
        atexit.register, bx.UP_TIMEOUT_S = real, real_to


def test_only_a_clearly_off_box_is_powered_on_and_only_an_hourly_one_is_powered_off():
    for word, on in (("shutdown", True), ("off", True), ("running", False), ("starting", False), ("booting", False), ("", False), (None, False), ("???", False)):
        with autodl_api(lambda m, path, b, q, w=word: w if path == "status" else None) as calls:
            b = box({"CLOUDGPU_INSTANCE": "pro-1", "AUTODL_DEV_TOKEN": "dev"})
            answers = iter([None, None, {"ok": True, "comfy": True, "busy": 0}])
            b.health = lambda: next(answers, {"ok": True, "comfy": True, "busy": 0})
            b.ensure_up()
            assert (("POST", "power_on", "/root/h3host/start.sh") in calls) is on, word
    for billing, off in (("payg", True), (None, False), ("", False), ("hourly", False), ("PAYG", False), ("daily", False), ("unknown", False)):
        row = {"uuid": "pro-1"} if billing is None else {"uuid": "pro-1", "charge_type": billing}
        with autodl_api(lambda m, path, b, q, r=row: {"list": [r]} if path == "list" else None) as calls:
            b = box({"CLOUDGPU_INSTANCE": "pro-1", "AUTODL_DEV_TOKEN": "dev"})
            b.health = lambda: {"ok": True, "comfy": True, "busy": 0}
            assert b.power_off_if_idle() is off, billing
            assert (("POST", "power_off", None) in calls) is off, billing


def test_the_proof_is_checked_again_every_minute_before_the_token_is_handed_out_and_a_failure_forgets_the_box_is_ready():
    good, squatter = FakeAdapter(), FakeAdapter(token=None)
    gone = FakeAdapter()
    gone.srv.shutdown()
    gone.srv.server_close()
    try:
        env = {"CLOUDGPU_BASE": good.base}
        b = box(env)
        b.ensure_up()
        assert b._ready and len(good.seen) == 1
        b.env["CLOUDGPU_BASE"] = squatter.base                         # the tunnel died and another process holds the port
        assert b.token() == "t" * 30 and squatter.seen == []           # proved a moment ago: no new request inside the minute
        b.ensure_up()
        assert squatter.seen == []
        b._proved -= 61                                                # older than a minute
        with _t.raises("http"):
            b.token()
        assert b._ready is False and squatter.seen == [None], squatter.seen        # the challenge went out without the token, and nothing was handed out
        b = box(env)
        b.ensure_up()
        b.env["CLOUDGPU_BASE"] = squatter.base
        b._proved -= 61
        with _t.raises("http"):
            b.ensure_up()                                              # also before a submit that goes by ensure_up
        assert b._ready is False
        b = box(env)
        b.ensure_up()
        b.env["CLOUDGPU_BASE"] = gone.base                             # nothing listens any more
        b._proved -= 61
        with _t.raises("down") as c:
            b.token()
        assert "t" * 30 not in c.err.message and b._ready is False
        b.env["CLOUDGPU_BASE"] = good.base                             # and the way back: the proof passes again
        assert b.token() == "t" * 30
        b = box({"CLOUDGPU_BASE": "https://b.example"})                # a public https address is named by its certificate: no proof, no request
        b.health = lambda: (_ for _ in ()).throw(AssertionError("no health call"))
        assert b.token() == "t" * 30
    finally:
        good.srv.shutdown()
        squatter.srv.shutdown()


def test_a_local_box_is_seen_as_up_whatever_the_proxy_environment_says():
    good = FakeAdapter()
    keys = ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "no_proxy", "NO_PROXY")
    saved = {k: os.environ.pop(k, None) for k in keys}
    dead = FakeAdapter()
    dead.srv.shutdown()
    dead.srv.server_close()
    try:
        os.environ.update(http_proxy=dead.base, HTTP_PROXY=dead.base, https_proxy=dead.base, HTTPS_PROXY=dead.base)
        with autodl_api(lambda m, path, b, q: "shutdown" if path == "status" else None) as calls:
            b = box({"CLOUDGPU_BASE": good.base, "CLOUDGPU_INSTANCE": "pro-1", "AUTODL_DEV_TOKEN": "dev"})
            b.ensure_up()
            assert calls == [] and b._ready, calls                     # it was up: nothing powered on, no 900 s wait
            assert good.seen == [None]                                 # the challenge reached the adapter itself, with no token
    finally:
        for k in keys:
            os.environ.pop(k, None)
            if saved[k] is not None:
                os.environ[k] = saved[k]
        good.srv.shutdown()


def test_the_base_url_may_be_http_only_on_this_machine():
    for ok in ("https://b.example", "https://b.example:8443/x", "http://127.0.0.1:16009", "http://localhost:1", "http://[::1]:16009"):
        assert Box({"CLOUDGPU_BASE": ok}).root() == ok
    for bad in ("http://b.example", "http://10.0.0.5:6009", "http://127.0.0.1.evil.example", "http://", "ftp://b.example", "file:///etc"):
        with _t.raises("key_missing") as c:
            Box({"CLOUDGPU_BASE": bad}).root()
        assert "b.example" not in c.err.message and "10.0.0.5" not in c.err.message          # the address is not echoed
    with _t.raises("key_missing") as c:
        Box({"CLOUDGPU_TOKEN": "abc\ndef"}).token()
    assert "abc" not in c.err.message
    assert Box({"CLOUDGPU_TOKEN": "abc\n"}).token() == "abc"


def test_ports_are_ascii_digits_in_range_and_a_known_hosts_file_makes_the_host_key_strict():
    for bad in ("gpu.example:\u00b2", "gpu.example:\u0661\u0662", "gpu.example:22\n", "gpu.example:0", "gpu.example:65536", "gpu.example\n:22"):
        with _t.raises("key_missing"):
            tunnel.parse_ssh(bad)
    for bad in ("0", "70000", "\u00b2", "6 009", "", "x", None):
        with _t.raises("key_missing"):
            tunnel.valid_port(bad)
    assert tunnel.valid_port("6009") == 6009 and tunnel.valid_port(" 16009 ") == 16009
    a = tunnel.argv("k", "root", "h", 22, 16009, 6009)
    assert "StrictHostKeyChecking=accept-new" in a and not any(x.startswith("UserKnownHostsFile") for x in a)
    a = tunnel.argv("k", "root", "h", 22, 16009, 6009, "/x/known")
    assert "StrictHostKeyChecking=yes" in a and "UserKnownHostsFile=/x/known" in a and "StrictHostKeyChecking=accept-new" not in a
    with _t.tmpdir() as d:
        key = Path(d) / "k"
        key.write_text("private")
        key.chmod(0o600)
        spawned = []
        env = {"CLOUDGPU_BASE": "http://127.0.0.1:16009", "CLOUDGPU_SSH_KEY": str(key), "CLOUDGPU_SSH": "gpu.example:22"}
        for extra in ({"CLOUDGPU_REMOTE_PORT": "99999"}, {"CLOUDGPU_REMOTE_PORT": "\u00b2"}, {"CLOUDGPU_KNOWN_HOSTS": str(Path(d) / "nofile")}, {"CLOUDGPU_BASE": "http://127.0.0.1:99999"}):
            b = box({**env, **extra}, spawn=lambda argv: spawned.append(argv) or True)
            b.health = lambda: None
            with _t.raises("key_missing"):
                b.tunnel()
        assert spawned == []
        kh = Path(d) / "known"
        kh.write_text("[gpu.example]:22 ssh-ed25519 AAAA\n")
        b = box({**env, "CLOUDGPU_KNOWN_HOSTS": str(kh), "CLOUDGPU_REMOTE_PORT": "7000"}, spawn=lambda argv: spawned.append(argv) or True)
        b.health = lambda: None
        assert b.tunnel() is True and "UserKnownHostsFile=" + str(kh) in spawned[0] and "127.0.0.1:16009:127.0.0.1:7000" in spawned[0]


def test_a_listener_on_the_tunnels_port_that_does_not_know_the_token_is_never_trusted_or_sent_it():
    good, squatter, wrong = FakeAdapter(), FakeAdapter(token=None), FakeAdapter(token="u" * 30)
    try:
        b = box({"CLOUDGPU_BASE": good.base})
        assert b.health()["comfy"] is True and good.seen == [None]                    # the proof checked out, and no token went with the challenge
        for fake in (squatter, wrong):
            b = box({"CLOUDGPU_BASE": fake.base})
            with _t.raises("http") as c:
                b.health()
            assert "t" * 30 not in c.err.message
            assert fake.seen == [None]                                                 # the challenge carried no Authorization header
        spawned = []
        with _t.tmpdir() as d:
            key = Path(d) / "k"
            key.write_text("private")
            key.chmod(0o600)
            b = box({"CLOUDGPU_BASE": squatter.base, "CLOUDGPU_SSH_KEY": str(key), "CLOUDGPU_SSH": "gpu.example:22"}, spawn=lambda argv: spawned.append(argv) or True)
            with _t.raises("http"):
                b.ensure_up()                                                          # refused loudly, no tunnel opened over it, not an endless wait
            assert spawned == []
        doc = status({"CLOUDGPU_BASE": squatter.base, "CLOUDGPU_TOKEN": "t" * 30, "CLOUDGPU_BOX": "rtx4090", "CLOUDGPU_POWER": "none"})
        assert doc["ready"] is False and doc["box"]["answers"] is False and any("does not prove" in n for n in doc["next"]) and not any("does not answer" in n for n in doc["next"])
        # no token configured: nothing to prove with and nothing to send; a public https base is not challenged (TLS names it), here simulated by a non-loopback name
        assert Box({"CLOUDGPU_BASE": squatter.base}).health()["comfy"] is True
    finally:
        for f in (good, squatter, wrong):
            f.srv.shutdown()


def test_the_status_says_which_card_it_is_when_it_is_not_the_declared_one_and_when_the_worker_is_dead_and_warns_about_a_readable_env_file():
    a, dead = FakeAdapter(card="NVIDIA GeForce RTX 4090, 49140 MiB"), FakeAdapter(worker="dead", card="unknown")
    try:
        base = {"CLOUDGPU_TOKEN": "t" * 30, "CLOUDGPU_POWER": "none"}
        doc = status({**base, "CLOUDGPU_BASE": a.base, "CLOUDGPU_BOX": "rtx-pro6000"})
        assert any("RTX 4090" in n and "rtx-pro6000" in n for n in doc["next"]), doc["next"]
        assert not any("card" in n for n in status({**base, "CLOUDGPU_BASE": a.base, "CLOUDGPU_BOX": "rtx4090"})["next"])
        doc = status({**base, "CLOUDGPU_BASE": dead.base, "CLOUDGPU_BOX": "rtx4090"})
        assert doc["ready"] is True and doc["box"]["worker"] == "dead" and any("worker is dead" in n for n in doc["next"]) and not any("card" in n for n in doc["next"])
        with _t.tmpdir() as d:
            f = Path(d) / "e.env"
            f.write_text("CLOUDGPU_TOKEN=abc\n")
            f.chmod(0o644)
            w = status({}, str(f))["warnings"]
            assert len(w) == 1 and "chmod 600" in w[0] and "abc" not in w[0]
            f.chmod(0o600)
            assert status({}, str(f))["warnings"] == [] and config.env_file_warning(Path(d) / "missing") is None
    finally:
        a.srv.shutdown()
        dead.srv.shutdown()


if __name__ == "__main__":
    _t.main(globals())
