"""The guards on every call that carries a secret, offline against servers on 127.0.0.1: a token never reaches another host through a redirect and never appears in an exception
(a trailing newline is stripped, an unusable character is refused by name), `http://` goes only to this machine, and `bench` opens only the URLs it may and names its file safely."""

import contextlib
import hashlib
import hmac
import io
import json
import os
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import _t

from cloudgpu import bench, config, net
from cloudgpu.vendors import autodl as va


class Server:
    """A server on a free local port; `handle(path, headers)` returns (code, headers, body) and every request is appended to `seen` as (method, path, Authorization)."""

    def __init__(self, handle):
        self.seen = seen = []

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _go(self):
                n = int(self.headers.get("Content-Length") or 0)
                if n:
                    self.rfile.read(n)
                seen.append((self.command, self.path, self.headers.get("Authorization")))
                code, hdrs, body = handle(self.path)
                self.send_response(code)
                for k, v in hdrs.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            do_GET = do_POST = _go
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def stop(self):
        self.srv.shutdown()
        self.srv.server_close()


OK = (200, {}, b'{"code":"Success","data":"fine"}')


def health(path, token="t" * 30):
    """What the adapter answers on /health at a local address: ok, and the proof of the nonce under the token (token None: a process that cannot prove it)."""
    d = {"ok": True, "comfy": True, "busy": 0}
    nonce = urllib.parse.parse_qs(urllib.parse.urlparse(path).query).get("nonce", [""])[0]
    if token and nonce:
        d["proof"] = hmac.new(token.encode(), nonce.encode(), hashlib.sha256).hexdigest()
    return 200, {}, json.dumps(d).encode()


def test_check_url_allows_https_anywhere_and_http_only_on_this_machine_and_never_echoes_the_url():
    for ok in ("https://a.example/x", "http://127.0.0.1:1/x", "http://localhost/x", "http://[::1]:5/x", "HTTP://LOCALHOST/x"):
        assert net.check_url(ok) == ok, ok
    for bad in ("http://a.example/files/SECRETKEY.mp4", "file:///etc/passwd", "ftp://a.example/x", "/relative", "https://", "http://127.0.0.1@evil.example/x", "gopher://127.0.0.1/"):
        with _t.raises("key_missing") as c:
            net.check_url(bad, "the URL")
        assert "SECRETKEY" not in c.err.message and "evil.example" not in c.err.message and "etc/passwd" not in c.err.message
    assert net.header_safe("abc DEF-1_2") and not net.header_safe("a\nb") and not net.header_safe("a\r") and not net.header_safe("tök") and not net.header_safe("a\x00b")


def test_a_redirect_to_another_host_or_port_is_refused_and_the_token_does_not_follow():
    other = Server(lambda path: OK)
    here = Server(lambda path: (302, {"Location": f"{other.base}/steal"}, b"") if path.endswith("/leave") else (302, {"Location": "/api/stay"}, b"") if path.endswith("/hop") else OK)
    real = va.API
    va.API = here.base + "/api"
    try:
        v = va.AutoDL({"AUTODL_DEV_TOKEN": "tok-abc\n"})                          # a trailing newline (a file, a pipe) is dropped, not sent and not fatal
        assert v._call("GET", "stay") == "fine" and here.seen[-1][2] == "tok-abc"
        assert v._call("GET", "hop") == "fine"                                    # a redirect on the same host and port is followed
        with _t.raises("http") as c:
            v._call("GET", "leave")
        assert c.err.status == 302 and "tok-abc" not in c.err.message + c.err.detail
        assert other.seen == [], other.seen                                       # nothing reached the other host
    finally:
        va.API = real
        here.stop()
        other.stop()


def test_a_token_never_appears_in_an_exception():
    secret = "SECRET-TOKEN-VALUE-123"
    for bad in (secret + "\nX-Evil: 1", secret[:8] + "\r\n" + secret[8:], secret + "\x00", secret + "é"):
        with _t.raises("key_missing") as c:
            va.AutoDL({"AUTODL_DEV_TOKEN": bad})._call("GET", "status")
        assert secret not in c.err.message and secret[:8] not in c.err.message and "AUTODL_DEV_TOKEN" in c.err.message
    real = va.net.urlopen                                                         # whatever http.client raises with the header in its text
    va.net.urlopen = lambda *a, **k: (_ for _ in ()).throw(ValueError(f"Invalid header value b'{secret}'"))
    try:
        with _t.raises("http") as c:
            va.AutoDL({"AUTODL_DEV_TOKEN": secret})._call("GET", "status")
        assert secret not in c.err.message + c.err.detail and secret not in str(c.err) and c.err.__cause__ is None
    finally:
        va.net.urlopen = real
    assert config.present({"AUTODL_DEV_TOKEN": " \n"})["AUTODL_DEV_TOKEN"] == "unset" and not va.AutoDL({"AUTODL_DEV_TOKEN": " \n"}).configured()


class Api:
    """A box for bench: the autodl.art contract, with the result URL it will name."""

    def __init__(self, result_url):
        self.result_url = result_url
        self.s = Server(self.handle)

    def handle(self, path):
        if path.startswith("/health"):
            return health(path)
        if path.endswith("/minimax_h3_lightx2v"):
            return 200, {}, b'{"code":"Success","data":{"task_id":"h3-1"}}'
        if "/result/" in path:
            return 200, {}, ('{"code":"Success","data":{"status":"SUCCESS","results":[{"url":"%s"}]}}' % self.result_url).encode()
        if path.startswith("/files/"):
            return 200, {}, b"MP4DATA"
        return 404, {}, b"{}"


def run_bench(d, base, name="take", mode=0o600, token="t" * 30):
    env = Path(d) / "e.env"
    env.write_text(f"CLOUDGPU_BASE={base}\nCLOUDGPU_TOKEN={token}\n")
    env.chmod(mode)
    img = Path(d) / "i.png"
    img.write_bytes(b"png")
    out, err, cwd = io.StringIO(), io.StringIO(), os.getcwd()
    os.chdir(d)
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = bench.main([str(env), name, "minimax_h3_lightx2v", "480p竖", "4", str(img)])
    finally:
        os.chdir(cwd)
    return rc, out.getvalue(), err.getvalue()


def test_bench_opens_only_urls_it_may_names_its_file_safely_and_warns_about_a_readable_env_file():
    with _t.tmpdir() as d:
        api = Api("/files/k.mp4")
        try:
            rc, out, err = run_bench(d, "http://box.example", "x")                 # clear-text http to another machine: the token would travel in the open
            assert rc == 2 and "https" in out and "t" * 30 not in out + err and api.s.seen == []
            rc, out, err = run_bench(d, api.s.base, "../../evil name\x00", mode=0o644)
            assert rc == 0 and "chmod 600" in err and "t" * 30 not in out + err, (rc, out, err)
            assert (Path(d) / "_.._evil_name_.mp4").read_bytes() == b"MP4DATA" and sorted(p.name for p in Path(d).glob("*.mp4")) == ["_.._evil_name_.mp4"]
            assert not (Path(d).parent / "evil.mp4").exists()
            assert [a for m, p, a in api.s.seen if p.startswith("/files/")] == [None]    # the download carries no Authorization header
            for url in ("file:///etc/hosts", "http://box.example/files/SECRET.mp4", "ftp://box.example/x.mp4"):
                api.result_url = url
                before = set(os.listdir(d))
                rc, out, err = run_bench(d, api.s.base, "take2")
                assert rc == 1 and "SECRET" not in out and set(os.listdir(d)) == before, (url, rc, out)
                assert not (Path(d) / "take2.mp4").exists()
        finally:
            api.s.stop()


def test_bench_does_not_follow_a_redirect_to_another_host_with_the_token():
    other = Server(lambda path: OK)
    here = Server(lambda path: health(path) if path.startswith("/health") else (302, {"Location": f"{other.base}/steal"}, b""))
    try:
        with _t.tmpdir() as d:
            rc, out, err = run_bench(d, here.base)
            assert rc == 2 and other.seen == [] and "t" * 30 not in out + err, (rc, out)
            assert [a for m, p, a in here.seen if not p.startswith("/health")] == ["t" * 30]       # the submit did carry the token, to the host that proved it
    finally:
        here.stop()
        other.stop()


def test_bench_sends_the_token_only_to_a_local_listener_that_proves_it_knows_it():
    squatter = Server(lambda path: health(path, token=None))
    gone = Server(lambda path: OK)
    gone.stop()
    try:
        with _t.tmpdir() as d:
            rc, out, err = run_bench(d, squatter.base)
            assert rc == 2 and "does not prove" in out and "t" * 30 not in out + err, (rc, out)
            assert all(a is None for m, p, a in squatter.seen) and all(p.startswith("/health") for m, p, a in squatter.seen), squatter.seen      # no submit, no token
            rc, out, err = run_bench(d, gone.base)                                      # nothing listens: the token is not sent anywhere
            assert rc == 2 and "t" * 30 not in out + err, (rc, out)
    finally:
        squatter.stop()


def test_a_request_to_this_machine_does_not_go_through_a_proxy_from_the_environment():
    from cloudgpu.box import Box
    dead = Server(lambda path: OK)
    dead.stop()                                                                          # a proxy address nobody answers at
    here = Server(lambda path: health(path) if path.startswith("/health") else OK)
    keys = ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "no_proxy", "NO_PROXY", "all_proxy", "ALL_PROXY")
    saved = {k: os.environ.pop(k, None) for k in keys}
    try:
        os.environ.update(http_proxy=dead.base, HTTP_PROXY=dead.base, https_proxy=dead.base, HTTPS_PROXY=dead.base)
        with net.urlopen(net.urllib.request.Request(here.base + "/x"), timeout=5) as r:
            assert r.status == 200 and here.seen[-1][1] == "/x"                          # straight to the local server
        with net.urlopen(here.base + "/y", timeout=5) as r:                              # a plain string works too
            assert r.status == 200
        b = Box({"CLOUDGPU_BASE": here.base, "CLOUDGPU_TOKEN": "t" * 30})
        assert b.health()["comfy"] is True and b.token() == "t" * 30                     # the box is seen as up (no power-on, no 900 s wait) and the token is handed out
    finally:
        for k in keys:
            os.environ.pop(k, None)
            if saved[k] is not None:
                os.environ[k] = saved[k]
        here.stop()


if __name__ == "__main__":
    _t.main(globals())
