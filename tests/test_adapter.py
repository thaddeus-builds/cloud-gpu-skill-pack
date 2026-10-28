"""cloudgpu/adapter.py, offline: a box answers the autodl.art ComfyUI workflow contract. A fake ComfyUI on 127.0.0.1 stands
in for the card; nothing leaves the machine. The token is checked on every call but /health, a request the box would not run is a
400 before anything is queued, both graphs are the open templates' (ref2va for the audio-driven take, fl2va for Mode A), a failed
take says why, and a restart never silently re-runs a take."""

import base64
import errno
import hashlib
import hmac
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import types
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import _t

from cloudgpu import adapter as adapter  # noqa: E402

TOKEN = "t" * 32
API = adapter.API
PNG = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode()
WAV = "data:audio/wav;base64," + base64.b64encode(b"RIFFfakeWAVE").decode()


class FakeComfy:
    """ComfyUI's four calls the box makes. `mode`: ok | refuse | fail. Makes a real small mp4 when ffmpeg is there."""

    def __init__(self, root: Path, mode="ok"):
        self.root, self.mode, self.graphs, self.done, self.calls = root, mode, [], {}, []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _j(self, code, doc):
                b = json.dumps(doc).encode()
                self.send_response(code)
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                if self.path == "/system_stats":
                    return self._j(200, {"system": {}})
                if self.path.startswith("/history/"):
                    pid = self.path.rsplit("/", 1)[-1]
                    return self._j(200, {pid: outer.done[pid]} if pid in outer.done else {})
                self._j(404, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.calls.append(self.path)
                if self.path == "/interrupt":
                    return self._j(200, {})
                if self.path in ("/queue", "/free"):
                    return self._j(200, {})
                if outer.mode == "refuse":
                    return self._j(400, {"error": {"message": "Prompt outputs failed validation"}})
                g = body["prompt"]
                outer.graphs.append(g)
                pid = f"p{len(outer.graphs)}"
                if outer.mode == "fail":
                    outer.done[pid] = {"status": {"completed": False, "status_str": "error", "messages": [["execution_error", {"exception_message": "CUDA out of memory"}]]}}
                else:
                    prefix = g["16"]["inputs"]["filename_prefix"]
                    out = outer.root / "output" / Path(prefix).parent
                    out.mkdir(parents=True, exist_ok=True)
                    name = Path(prefix).name + "_00001_.mp4"
                    if shutil.which("ffmpeg"):
                        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=24", "-f", "lavfi", "-i", "sine=frequency=440",
                                        "-t", "10", "-shortest", "-pix_fmt", "yuv420p", str(out / name)], check=True)
                    else:
                        (out / name).write_bytes(b"fakemp4")
                    outer.done[pid] = {"status": {"completed": True, "status_str": "success"},
                                       "outputs": {"16": {"images": [{"filename": name, "subfolder": Path(prefix).parent.as_posix(), "type": "output"}], "animated": [True]}}}
                self._j(200, {"prompt_id": pid})

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def stop(self):
        self.srv.shutdown()


class Box:
    """A running adapter over a temp folder and a FakeComfy."""

    def __init__(self, d: str, mode="ok", tts=None):
        self.root = Path(d) / "comfy"
        (self.root / "input").mkdir(parents=True)
        self.comfy = FakeComfy(self.root, mode)
        self.host = adapter.Host(TOKEN, self.comfy.url, Path(d) / "state", self.root, tts=tts)
        self.host.start_worker()
        self.srv = adapter.ThreadingHTTPServer(("127.0.0.1", 0), adapter.handler(self.host))
        self.base = f"http://127.0.0.1:{self.srv.server_address[1]}"
        self.host.public_base = self.base
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def call(self, method, path, body=None, token=TOKEN):
        r = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None, method=method,
                                   headers={"Authorization": token, "Content-Type": "application/json"} if token else {})
        try:
            with urllib.request.urlopen(r, timeout=20) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def wait(self, tid, timeout=20):
        end = time.time() + _t.patient(timeout)
        while time.time() < end:
            code, d = self.call("GET", f"{API}/result/{tid}")
            assert code == 200, d
            if d["data"]["status"] in ("SUCCESS", "FAILED"):
                return d["data"]
            time.sleep(0.05)
        raise AssertionError("the take did not finish")

    def close(self):
        self.srv.shutdown()
        self.comfy.stop()


REF = {"resolution": "768p竖", "ref_image_0": PNG, "ref_audio_0": WAV, "audio_duration": 8}
FL = {"resolution": "480p竖", "prompt": "She says: hello.", "first_frame": PNG, "last_frame": PNG, "duration": 4, "seed": 7}


def test_frames_follow_the_templates_rule():
    assert adapter.frames_for(8) == 192 and adapter.frames_for(15) == 362 and adapter.frames_for(1) == 39
    assert all(adapter.frames_for(s / 2) % 17 == 5 for s in range(2, 31))


def test_the_token_guards_every_call_but_health():
    with _t.tmpdir() as d:
        b = Box(d)
        try:
            h = b.call("GET", "/health", token=None)[1]
            assert h["ok"] is True and h["comfy"] is True and h["busy"] == 0 and h["worker"] == "alive" and "card" in h and "proof" not in h, h
            for tok in (None, "wrong", TOKEN[:-1], TOKEN + "x"):
                assert b.call("POST", f"{API}/minimax_h3_lightx2v", FL, token=tok)[0] == 401
                assert b.call("GET", f"{API}/result/h3-abc", token=tok)[0] == 401
            assert not list((b.root / "input").iterdir())          # a refused caller wrote nothing
        finally:
            b.close()


def test_a_request_the_box_would_not_run_is_refused_before_anything_is_queued():
    with _t.tmpdir() as d:
        b = Box(d)
        try:
            bad = [("nope", FL), ("minimax_h3_lightx2v", {**FL, "resolution": "1080p竖"}), ("minimax_h3_lightx2v", {**FL, "duration": 16}),
                   ("minimax_h3_lightx2v", {**FL, "duration": 0}), ("minimax_h3_lightx2v", {**FL, "prompt": ""}),
                   ("minimax_h3_lightx2v", {**FL, "first_frame": "http://127.0.0.1:9/x.png"}),      # this machine is not fetched from
                   ("minimax_h3_lightx2v", {**FL, "first_frame": "https://10.0.0.5/x.png"}),
                   ("minimax_h3_lightx2v", {**FL, "first_frame": "file:///etc/passwd"}),
                   ("minimax_h3_lightx2v", {**FL, "first_frame": WAV}),                              # audio where an image goes
                   ("minimax_h3_image_audio_to_video", {**REF, "ref_audio_0": PNG}),
                   ("minimax_h3_image_audio_to_video", {**REF, "audio_duration": "x"}),
                   ("minimax_h3_image_audio_to_video", {k: v for k, v in REF.items() if k != "ref_image_0"})]
            for wf, body in bad:
                code, doc = b.call("POST", f"{API}/{wf}", body)
                assert code == 400 and doc["code"] == "RequestParameterIsWrong", (wf, body.get("first_frame", "")[:30], code, doc)
            assert not list((b.root / "input").iterdir()) and not list((b.host.tasks_dir).iterdir())
            assert not b.comfy.graphs
        finally:
            b.close()


def test_the_audio_driven_take_builds_the_ref2va_graph_and_returns_a_video():
    with _t.tmpdir() as d:
        b = Box(d)
        try:
            code, doc = b.call("POST", f"{API}/minimax_h3_image_audio_to_video", REF)
            assert code == 200 and doc["code"] == "Success", doc
            r = b.wait(doc["data"]["task_id"])
            assert r["status"] == "SUCCESS", r
            g = b.comfy.graphs[0]
            assert g["7"]["class_type"] == "MiniMaxH3ReferenceToVideo" and g["1"]["inputs"]["unet_name"] == adapter.MODELS["unet_ref"]
            assert (g["7"]["inputs"]["width"], g["7"]["inputs"]["height"], g["7"]["inputs"]["length"]) == (768, 1344, 192)
            assert g["6"]["class_type"] == "LoadAudio" and g["6"]["inputs"]["audio"].endswith(".wav")
            assert g["5"]["inputs"]["image"].endswith(".png")
            assert g["10"]["inputs"]["steps"] == adapter.MODELS["steps_ref"]
            assert not list((b.root / "input").iterdir())          # the inputs were removed once the take ended
            url = r["results"][0]["url"]
            with urllib.request.urlopen(url, timeout=20) as resp:
                blob = resp.read()
            assert resp.headers["Content-Type"] == "video/mp4" and len(blob) > 100
            if shutil.which("ffprobe"):          # the pad to 5 mod 17 frames is cut back to the 8 s that were asked for
                p = Path(d) / "got.mp4"
                p.write_bytes(blob)
                dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)]))
                assert 7.8 < dur < 8.2, dur
        finally:
            b.close()


def test_mode_a_builds_the_fl2va_graph_with_the_still_as_first_and_last_frame():
    with _t.tmpdir() as d:
        b = Box(d)
        try:
            tid = b.call("POST", f"{API}/minimax_h3_lightx2v", FL)[1]["data"]["task_id"]
            assert b.wait(tid)["status"] == "SUCCESS"
            g = b.comfy.graphs[0]
            assert g["7"]["class_type"] == "MiniMaxH3ImageToVideo" and g["1"]["inputs"]["unet_name"] == adapter.MODELS["unet_fl"]
            assert (g["7"]["inputs"]["width"], g["7"]["inputs"]["height"]) == (480, 864)
            assert g["7"]["inputs"]["first_frame"] == ["5", 0] and g["7"]["inputs"]["last_frame"] == ["5b", 0]
            assert g["5"]["inputs"]["image"] == g["5b"]["inputs"]["image"]
            assert g["9"]["inputs"]["noise_seed"] == 7 and g["7"]["inputs"]["prompt"] == "She says: hello."
            assert "6" not in g                                    # H3 speaks: no audio input
        finally:
            b.close()


def test_a_failed_take_says_why_and_the_worker_goes_on():
    with _t.tmpdir() as d:
        b = Box(d, mode="fail")
        try:
            r = b.wait(b.call("POST", f"{API}/minimax_h3_lightx2v", FL)[1]["data"]["task_id"])
            assert r["status"] == "FAILED" and "CUDA out of memory" in r["error"], r
            b.comfy.mode = "refuse"
            r = b.wait(b.call("POST", f"{API}/minimax_h3_lightx2v", FL)[1]["data"]["task_id"])
            assert r["status"] == "FAILED" and "refused the graph" in r["error"], r
            b.comfy.mode = "ok"
            assert b.wait(b.call("POST", f"{API}/minimax_h3_lightx2v", FL)[1]["data"]["task_id"])["status"] == "SUCCESS"
            assert not list((b.root / "input").iterdir())
        finally:
            b.close()


def test_a_restart_fails_waiting_takes_and_never_re_runs_them():
    with _t.tmpdir() as d:
        state = Path(d) / "state"
        (state / "tasks").mkdir(parents=True)
        for tid, st in (("h3-aaa", "QUEUED"), ("h3-bbb", "RUNNING"), ("h3-ccc", "SUCCESS")):
            (state / "tasks" / f"{tid}.json").write_text(json.dumps({"id": tid, "status": st, "file_key": "k" * 32}))
        (state / "out").mkdir()
        (state / "out" / f"{'k' * 32}.mp4").write_bytes(b"MP4")                    # the finished take's video is still there
        h = adapter.Host(TOKEN, "http://127.0.0.1:9", state, Path(d))
        assert [h.result(t)["status"] for t in ("h3-aaa", "h3-bbb", "h3-ccc")] == ["FAILED", "FAILED", "SUCCESS"]
        assert "restarted" in h.result("h3-aaa")["error"] and h.q.empty()


def test_unknown_tasks_and_odd_ids_are_404_and_files_need_their_secret_name():
    with _t.tmpdir() as d:
        b = Box(d)
        try:
            for tid in ("h3-nope", "..%2f..%2fetc%2fpasswd", "x", "h3-"):
                assert b.call("GET", f"{API}/result/{tid}")[0] == 404, tid
            for p in ("/files/" + "a" * 32 + ".mp4", "/files/../tasks.mp4", "/files/short.mp4"):
                assert b.call("GET", p, token=None)[0] == 404, p
        finally:
            b.close()


def test_models_json_overrides_the_file_names_and_refuses_unknown_keys():
    with _t.tmpdir() as d:
        f = Path(d) / "m.json"
        f.write_text(json.dumps({"clip": "x/clip_int8.safetensors", "steps_ref": 8, "lora_ref": "x/ref_turbo.safetensors"}))
        old = dict(adapter.MODELS)
        try:
            adapter.MODELS.update(json.loads(f.read_text()))
            g = adapter.graph("ref", w=480, h=864, seconds=8, image="a.png", audio="a.wav", prompt="p", seed=1, prefix="x")
            assert g["2"]["inputs"]["clip_name"] == "x/clip_int8.safetensors" and g["10"]["inputs"]["steps"] == 8
            assert g["20"]["inputs"]["lora_name"] == "x/ref_turbo.safetensors"
        finally:
            adapter.MODELS.clear()
            adapter.MODELS.update(old)
        f.write_text(json.dumps({"nope": 1}))
        r = subprocess.run([sys.executable, str(Path(_t.ROOT) / "cloudgpu" / "adapter.py"), "--models", str(f), "--state", str(Path(d) / "s")],
                           env={"PATH": "/usr/bin:/bin", "CLOUDGPU_TOKEN": "t" * 32}, capture_output=True, text=True, timeout=30)
        assert r.returncode != 0 and "unknown keys" in r.stderr, r.stderr


def test_the_box_presets_name_exactly_the_keys_the_adapter_knows():
    for f in sorted((Path(_t.ROOT) / "cloudgpu" / "onbox").glob("models.*.json")):
        assert set(json.loads(f.read_text())) == set(adapter.MODELS), f.name


def test_the_tunnel_script_never_falls_back_to_a_password():
    sh = Path(_t.ROOT) / "cloudgpu" / "onbox" / "tunnel.sh"
    r = subprocess.run(["sh", str(sh), "h.example", "22", "/nonexistent-key"], capture_output=True, text=True, timeout=30)
    assert r.returncode == 2 and "not readable" in r.stderr, (r.returncode, r.stderr)
    text = sh.read_text()
    assert "BatchMode=yes" in text and "IdentitiesOnly=yes" in text and "sshpass" not in text and "SSH_ASKPASS" not in text


def test_a_token_that_is_too_short_does_not_start_the_server():
    r = subprocess.run([sys.executable, str(Path(_t.ROOT) / "cloudgpu" / "adapter.py"), "--state", "/nonexistent-x"],
                       env={"PATH": "/usr/bin:/bin", "CLOUDGPU_TOKEN": "short"}, capture_output=True, text=True, timeout=30)
    assert r.returncode != 0 and "at least 24" in r.stderr, (r.returncode, r.stderr)


def bare_host(d, **kw):
    """A Host with no worker thread and no ComfyUI: for tests that step the worker by hand."""
    root = Path(d) / "comfy"
    (root / "input").mkdir(parents=True, exist_ok=True)
    return adapter.Host(TOKEN, "http://127.0.0.1:9", Path(d) / "state", root, **kw)


def test_health_says_the_card_the_worker_and_proves_the_token_to_a_client_that_holds_it():
    with _t.tmpdir() as d:
        b = Box(d)
        real = os.environ["PATH"]
        try:
            fake = Path(d) / "bin"
            fake.mkdir()
            smi = fake / "nvidia-smi"
            smi.write_text("#!/bin/sh\necho 'NVIDIA GeForce RTX 4090, 49140 MiB'\n")
            smi.chmod(0o755)
            os.environ["PATH"] = f"{fake}:{real}"
            b.host._card = ("unknown", 0.0)
            nonce = "ab12" * 8
            h = b.call("GET", "/health?nonce=" + nonce, token=None)[1]
            assert h["card"] == "NVIDIA GeForce RTX 4090, 49140 MiB" and h["worker"] == "alive" and h["ok"] is True, h
            assert h["memory_mib"] == 49140 and type(h["memory_mib"]) is int                       # the size as a number, next to the text
            assert h["proof"] == hmac.new(TOKEN.encode(), nonce.encode(), hashlib.sha256).hexdigest()
            assert "proof" not in b.call("GET", "/health", token=None)[1] and "proof" not in b.call("GET", "/health?nonce=" + "a" * 200, token=None)[1]
            os.environ["PATH"] = str(fake)
            smi.write_text("#!/bin/sh\nexit 1\n")
            b.host._card = ("unknown", 0.0)
            h = b.call("GET", "/health", token=None)[1]
            assert h["card"] == "unknown" and h["memory_mib"] is None                    # nvidia-smi fails: "unknown", never an error
            smi.write_text("#!/bin/sh\necho 'NVIDIA Odd Card, no size'\n")
            b.host._card = ("unknown", 0.0)
            h = b.call("GET", "/health", token=None)[1]
            assert h["card"] == "NVIDIA Odd Card, no size" and h["memory_mib"] is None   # a text with no MiB: no number is invented
        finally:
            os.environ["PATH"] = real
            b.close()


def test_the_proof_is_answered_only_on_a_loopback_bind_to_a_hex_nonce_and_at_a_limited_rate():
    for bind, ok in (("127.0.0.1", True), ("127.1.2.3", True), ("::1", True), ("[::1]", True), ("localhost", True), ("::ffff:127.0.0.1", True),
                     ("0.0.0.0", False), ("::", False), ("", False), ("10.0.0.5", False), ("192.0.2.1", False), ("box.example", False), ("::ffff:10.0.0.1", False)):
        assert adapter.loopback_bind(bind) is ok, bind
    with _t.tmpdir() as d:
        nonce = "0123456789abcdef"
        wide = bare_host(d, bind="0.0.0.0")                                       # CLOUDGPU_BIND widened: no proof, whatever is asked
        assert "proof" not in wide.health(nonce) and "proof" not in wide.health("a" * 32)
        h = bare_host(Path(d) / "two")
        want = hmac.new(TOKEN.encode(), nonce.encode(), hashlib.sha256).hexdigest()
        assert h.health(nonce)["proof"] == want
        assert h.health("a" * 16)["proof"] and h.health("f" * 64)["proof"]       # 16 and 64 characters are the limits
        for bad in ("", "abc123", "a" * 15, "a" * 65, "A" * 32, "g" * 32, "ab12 " * 7, "0123456789abcdef\n", "é" * 16):
            assert "proof" not in h.health(bad), repr(bad)
        busy = bare_host(Path(d) / "three")
        got = [("proof" in busy.health(secrets_hex(i))) for i in range(adapter.PROOF_PER_MIN + 5)]
        assert got == [True] * adapter.PROOF_PER_MIN + [False] * 5, got           # an offline guessing oracle is rate limited
        assert busy.health(nonce)["ok"] is False and "card" in busy.health(nonce)  # everything but the proof is still answered
        busy._proofs = [time.time() - 61] * adapter.PROOF_PER_MIN                 # a minute later it answers again
        assert "proof" in busy.health(nonce)


def secrets_hex(i: int) -> str:
    return f"{i:032x}"


def test_a_take_whose_video_was_swept_reads_failed_result_expired_not_success_and_a_404():
    with _t.tmpdir() as d:
        b = Box(d)
        try:
            tid = b.call("POST", f"{API}/minimax_h3_lightx2v", FL)[1]["data"]["task_id"]
            r = b.wait(tid)
            assert r["status"] == "SUCCESS"
            video = next(b.host.out_dir.glob("*.mp4"))
            assert b.call("GET", f"{API}/result/{tid}")[1]["data"]["status"] == "SUCCESS" and b.host.file(video.stem) is not None
            old = time.time() - adapter.FILE_TTL_S - 5                             # the file is older than FILE_TTL_S, the task record (TASK_TTL_S) is not
            os.utime(video, (old, old))
            code, doc = b.call("GET", f"{API}/result/{tid}")
            assert code == 200 and doc["data"]["status"] == "FAILED" and "result expired" in doc["data"]["error"] and "results" not in doc["data"], doc
            assert b.host.busy() == 0                                              # it no longer holds the box busy
            video.unlink()                                                         # and the same once the sweep has deleted it
            assert b.call("GET", f"{API}/result/{tid}")[1]["data"]["status"] == "FAILED"
            assert b.call("GET", f"{API}/result/h3-nope")[0] == 404
        finally:
            b.close()


def test_health_is_dead_when_the_worker_thread_is_dead_and_never_reads_the_task_files():
    with _t.tmpdir() as d:
        h = bare_host(d)
        assert h.health()["worker"] == "dead" and h.health()["ok"] is False             # no worker was started
        h.start_worker()
        assert h.health()["worker"] == "alive" and h.health()["ok"] is True
        real_load = h.load
        h.load = lambda tid: (_ for _ in ()).throw(AssertionError("/health read a task file"))
        for _ in range(3):
            h.health()
        h.load = real_load
        h.work = lambda tid: (_ for _ in ()).throw(RuntimeError("boom"))                # whatever goes wrong in a take, the thread lives on
        h.n_active = 1
        h.q.put("h3-x")
        end = time.time() + _t.patient(5)
        while h.n_active and time.time() < end:
            time.sleep(0.02)
        assert h.n_active == 0 and h.worker_thread.is_alive() and h.health()["worker"] == "alive"


def test_a_submit_is_busy_from_acceptance_a_finished_take_stays_busy_until_collected_or_the_grace_ends():
    with _t.tmpdir() as d:
        b = Box(d)
        real_fetch, gate, seen = adapter.fetch, threading.Event(), []
        try:
            def slow(*a, **k):
                seen.append(b.host.busy())                  # inside the fetch of the inputs: the take is already counted
                gate.wait(10)
                return real_fetch(*a, **k)
            adapter.fetch = slow
            res = []
            t = threading.Thread(target=lambda: res.append(b.call("POST", f"{API}/minimax_h3_lightx2v", FL)), daemon=True)
            t.start()
            end = time.time() + _t.patient(5)
            while not seen and time.time() < end:
                time.sleep(0.01)
            assert seen == [1] and b.call("GET", "/health", token=None)[1]["busy"] == 1
            gate.set()
            t.join(20)
            tid = res[0][1]["data"]["task_id"]
            assert b.wait(tid)["status"] == "SUCCESS"
            end = time.time() + _t.patient(5)
            while b.host.n_active and time.time() < end:
                time.sleep(0.01)
            assert b.host.busy() == 1, "a finished take that nobody fetched is still busy"
            url = b.call("GET", f"{API}/result/{tid}")[1]["data"]["results"][0]["url"]
            assert b.host.busy() == 1                       # reading the result is not collecting the video
            urllib.request.urlopen(url, timeout=20).read()
            assert b.host.busy() == 0
            # a failed take is collected by reading its result; an uncollected one stops counting after the grace
            b.comfy.mode = "fail"
            tid = b.call("POST", f"{API}/minimax_h3_lightx2v", FL)[1]["data"]["task_id"]
            end = time.time() + _t.patient(10)
            while b.host.busy() != 1 and time.time() < end:
                time.sleep(0.01)
            assert b.wait(tid)["status"] == "FAILED" and b.host.busy() == 0
            b.comfy.mode = "ok"
            tid = b.call("POST", f"{API}/minimax_h3_lightx2v", FL)[1]["data"]["task_id"]
            b.wait(tid)
            assert b.host.busy() == 1
            b.host.holds[tid] = (time.time() - adapter.GRACE_S - 1, "k")
            assert b.host.busy() == 0
        finally:
            adapter.fetch = real_fetch
            gate.set()
            b.close()


def test_a_refused_or_failed_submit_stops_counting_and_a_full_disk_is_a_json_answer_with_nothing_left_behind():
    with _t.tmpdir() as d:
        b = Box(d)
        real_fetch = adapter.fetch
        try:
            shutil.rmtree(b.root / "input")                     # ComfyUI/input missing: created
            first = b.call("POST", f"{API}/minimax_h3_lightx2v", FL)
            assert first[0] == 200
            n = []

            def full(ref, into, kinds, what):
                n.append(1)
                if len(n) == 1:
                    return real_fetch(ref, into, kinds, what)          # the first file is written, the second hits the full disk
                raise OSError(errno.ENOSPC, "No space left on device")
            adapter.fetch = full
            assert b.wait(first[1]["data"]["task_id"])["status"] == "SUCCESS"
            time.sleep(0.2)
            b.host.holds.clear()                                     # that first take was never collected: not what this test is about
            before = set(p.name for p in b.host.tasks_dir.iterdir())
            code, doc = b.call("POST", f"{API}/minimax_h3_lightx2v", {**FL, "last_frame": WAV.replace("audio/wav", "image/jpeg")})
            assert code == 507 and doc["code"] == "InternalError" and "disk full" in doc["msg"], (code, doc)
            assert set(p.name for p in b.host.tasks_dir.iterdir()) == before and not list((b.root / "input").iterdir())
            assert b.host.busy() == 0
            adapter.fetch = real_fetch
            assert b.call("POST", f"{API}/minimax_h3_lightx2v", {**FL, "first_frame": "https://10.0.0.5/x.png"})[0] == 400
            assert b.host.busy() == 0
        finally:
            adapter.fetch = real_fetch
            b.close()


def test_a_save_that_fails_in_the_worker_fails_the_take_and_keeps_the_thread():
    with _t.tmpdir() as d:
        h = bare_host(d)
        tid = h.submit("minimax_h3_lightx2v", FL)
        assert h.load(tid)["inputs"] and list((h.root / "input").iterdir())
        real = h.save
        h.save = lambda t: (_ for _ in ()).throw(OSError(errno.ENOSPC, "full"))
        h.q.get_nowait()
        h.work(tid)
        h.finish(tid)
        r = h.result(tid)
        assert r["status"] == "FAILED" and "disk" in r["error"] and h.busy() == 0 and not list((h.root / "input").iterdir()), r
        h.save = real


def test_a_restart_deletes_the_inputs_it_failed_and_clears_comfyuis_queue_and_old_task_files_are_pruned():
    with _t.tmpdir() as d:
        root = Path(d) / "comfy"
        (root / "input").mkdir(parents=True)
        state = Path(d) / "state"
        (state / "tasks").mkdir(parents=True)
        (root / "input" / "cloudgpu_a.png").write_bytes(b"x")
        (root / "input" / "cloudgpu_old.png").write_bytes(b"x")
        os.utime(root / "input" / "cloudgpu_old.png", (1, 1))
        (state / "tasks" / "h3-run.json").write_text(json.dumps({"id": "h3-run", "status": "RUNNING", "inputs": ["cloudgpu_a.png", "../evil"]}))
        (state / "tasks" / "h3-old.json").write_text(json.dumps({"id": "h3-old", "status": "SUCCESS", "file_key": "k" * 32}))
        (state / "tasks" / "h3-new.json").write_text(json.dumps({"id": "h3-new", "status": "SUCCESS", "file_key": "k" * 32}))
        os.utime(state / "tasks" / "h3-old.json", (1, 1))
        comfy = FakeComfy(root)
        try:
            h = adapter.Host(TOKEN, comfy.url, state, root)
            assert not (root / "input" / "cloudgpu_a.png").exists() and not (root / "input" / "cloudgpu_old.png").exists()
            assert not (state / "tasks" / "h3-old.json").exists() and (state / "tasks" / "h3-new.json").exists() and h.result("h3-run")["status"] == "FAILED"
            assert comfy.calls == []                                # the constructor does not talk to ComfyUI
            h.start_worker()
            end = time.time() + _t.patient(5)
            while len(comfy.calls) < 2 and time.time() < end:
                time.sleep(0.02)
            assert comfy.calls == ["/interrupt", "/queue"], comfy.calls
        finally:
            comfy.stop()


def test_ffmpeg_that_hangs_fails_the_take_instead_of_the_worker():
    if not shutil.which("ffmpeg"):
        return _t.skip("no ffmpeg")
    with _t.tmpdir() as d:
        b = Box(d)
        real, old = adapter.subprocess, adapter.FFMPEG_TIMEOUT_S

        def run(args, **k):
            if "libx264" in args:
                raise subprocess.TimeoutExpired(args, k["timeout"])
            return subprocess.run(args, **k)
        try:
            adapter.subprocess = types.SimpleNamespace(run=run, TimeoutExpired=subprocess.TimeoutExpired, SubprocessError=subprocess.SubprocessError)
            r = b.wait(b.call("POST", f"{API}/minimax_h3_lightx2v", FL)[1]["data"]["task_id"])
            assert r["status"] == "FAILED" and "ffmpeg timed out" in r["error"], r
            assert adapter.FFMPEG_TIMEOUT_S == old and not list(b.host.out_dir.glob("*.mp4"))
        finally:
            adapter.subprocess = real
            b.close()


def test_an_empty_token_is_refused_by_the_host_and_by_the_handler():
    with _t.tmpdir() as d:
        try:
            adapter.Host("", "http://127.0.0.1:9", Path(d) / "s2", Path(d))
        except ValueError:
            pass
        else:
            raise AssertionError("Host accepted an empty token")
        b = Box(d)
        try:
            b.host.token = ""                                       # even if one got through, a request with no Authorization is not accepted
            assert b.call("POST", f"{API}/minimax_h3_lightx2v", FL, token=None)[0] == 401
            assert b.call("GET", f"{API}/result/h3-x", token="")[0] == 401
        finally:
            b.close()


def test_the_ssrf_filter_refuses_every_non_global_address_and_a_url_is_fetched_from_the_address_that_was_checked():
    for ip in ("127.0.0.1", "10.1.2.3", "192.168.0.1", "172.16.0.1", "169.254.169.254", "100.100.100.200", "100.64.0.1", "0.0.0.0", "::1", "fe80::1", "fc00::1",
               "::ffff:127.0.0.1", "::ffff:10.0.0.1", "224.0.0.1", "192.0.2.1"):
        assert adapter.resolve_public(ip, 80) is None and not adapter.public_host(ip), ip
    assert adapter.resolve_public("8.8.8.8", 80) == "8.8.8.8" and adapter.public_host("2606:4700:4700::1111")
    seen = []

    class H(adapter.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            seen.append((self.headers["Host"], self.path))
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "http://127.0.0.1:9/x")
                self.end_headers()
                return
            self.send_response(200)
            if self.path == "/png":
                self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", "4")
            self.end_headers()
            self.wfile.write(b"\x89PNG")
    srv = adapter.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    real = adapter.resolve_public
    calls = []
    adapter.resolve_public = lambda host, p: calls.append((host, p)) or "127.0.0.1"          # the one resolution, and where it says to connect
    try:
        with _t.tmpdir() as d:
            into = Path(d)
            p = adapter.fetch(f"http://files.example:{port}/png", into, ("image",), "x")      # "files.example" resolves nowhere: only the pinned address can answer
            assert p.read_bytes() == b"\x89PNG" and calls == [("files.example", port)] and seen == [(f"files.example:{port}", "/png")], (calls, seen)
            for path, why in (("/nothing", "no content type"), ("/redirect", "")):
                try:
                    adapter.fetch(f"http://files.example:{port}{path}", into, ("image",), "x")
                except adapter.Refused as e:
                    assert why in str(e) and "127.0.0.1" not in str(e) and str(port) not in str(e), str(e)
                else:
                    raise AssertionError(path)
            srv.shutdown()
            srv.server_close()
            msgs = set()
            for u in (f"http://files.example:{port}/png", "http://files.example:1/png"):         # nothing listens: refused connection, always the same words
                try:
                    adapter.fetch(u, into, ("image",), "x")
                except adapter.Refused as e:
                    msgs.add(str(e))
            assert msgs == {"x: could not be fetched"}, msgs
            assert not list(into.iterdir()) or len(list(into.iterdir())) == 1
    finally:
        adapter.resolve_public = real


def test_a_stalled_client_loses_its_connection_and_the_adapter_listens_on_loopback_unless_told_otherwise():
    with _t.tmpdir() as d:
        old = adapter.SOCKET_TIMEOUT_S
        adapter.SOCKET_TIMEOUT_S = 0.5
        try:
            h = bare_host(d)
            assert adapter.handler(h).timeout == 0.5
            srv = adapter.ThreadingHTTPServer(("127.0.0.1", 0), adapter.handler(h))
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            with socket.create_connection(srv.server_address, timeout=10) as c:
                c.sendall(b"POST /x HTTP/1.1\r\nContent-Le")                 # the rest never comes
                assert c.recv(100) == b""                                 # the server closed it
            srv.shutdown()
        finally:
            adapter.SOCKET_TIMEOUT_S = old
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        env = {"PATH": "/usr/bin:/bin", "CLOUDGPU_TOKEN": "t" * 32}
        p = subprocess.Popen([sys.executable, str(Path(_t.ROOT) / "cloudgpu" / "adapter.py"), "--port", str(port), "--state", str(Path(d) / "s"),
                              "--comfy-root", str(Path(d) / "c")], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            end = time.time() + _t.patient(15)
            up = False
            while time.time() < end and not up:
                try:
                    up = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3).read())["ok"]
                except OSError:
                    time.sleep(0.2)
            assert up
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as u:          # the address this machine would use to leave it (nothing is sent)
                    u.connect(("10.255.255.255", 1))
                    other = u.getsockname()[0]
            except OSError:
                other = "127.0.0.1"
            if other.startswith("127."):
                _t.skip("this machine's name resolves to loopback: the bind cannot be told from outside")
            else:
                try:
                    socket.create_connection((other, port), timeout=3).close()
                except OSError:
                    pass
                else:
                    raise AssertionError("the adapter answers on a non-loopback address by default")
        finally:
            p.kill()
            p.wait()


def test_the_tunnel_script_reuses_a_listener_only_when_it_proves_it_knows_the_token():
    sh = Path(_t.ROOT) / "cloudgpu" / "onbox" / "tunnel.sh"
    with _t.tmpdir() as d:
        key = Path(d) / "k"
        key.write_text("private")
        b = Box(d)

        class Squat(adapter.BaseHTTPRequestHandler):             # answers like a box, knows no token
            def log_message(self, *a):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", "16")
                self.end_headers()
                self.wfile.write(b'{"comfy": false}')
        squat = adapter.ThreadingHTTPServer(("127.0.0.1", 0), Squat)
        threading.Thread(target=squat.serve_forever, daemon=True).start()

        def run(port, token, *extra):
            env = {"PATH": os.environ["PATH"], **({"CLOUDGPU_TOKEN": token} if token else {})}
            return subprocess.run(["sh", str(sh), "h.example", "22", str(key), str(port), *extra], env=env, capture_output=True, text=True, timeout=60)
        try:
            port = b.srv.server_address[1]
            r = run(port, TOKEN)
            assert r.returncode == 0 and json.loads(r.stdout)["comfy"] is True and "proof" not in r.stdout and TOKEN not in r.stdout + r.stderr, (r.stdout, r.stderr)
            r = run(port, "w" * 32)                                              # a listener that cannot prove the token we hold
            assert r.returncode == 3 and "does not prove" in r.stderr and "w" * 32 not in r.stdout + r.stderr, (r.returncode, r.stderr)
            r = run(squat.server_address[1], TOKEN)                              # a process that merely holds the port
            assert r.returncode == 3 and "another process" in r.stderr, (r.returncode, r.stdout, r.stderr)
            r = run(port, None)                                                  # no token to check with: used, but it says so
            assert r.returncode == 0 and "not verified" in r.stderr
            for bad in (["h.example", "22", str(key), "70000"], ["h.example", "2\u00b2", str(key)], ["h;x", "22", str(key)], ["h.example", "22", str(key), "16009", "0"]):
                r = subprocess.run(["sh", str(sh), *bad], capture_output=True, text=True, timeout=30)
                assert r.returncode == 2 and ("port" in r.stderr or "host" in r.stderr), (bad, r.returncode, r.stderr)
        finally:
            squat.shutdown()
            b.close()


VOICE = {"prompt_text": "Hello there.", "prompt_simple": WAV}
FAKE_WORKER = """import json, sys
job = json.load(open(sys.argv[1]))
open(job["out"] + ".seen", "w").write(json.dumps(job))
if job["text"] == "boom":
    print("CUDA out of memory", file=sys.stderr); sys.exit(1)
if job["text"] == "nothing":
    sys.exit(0)
open(job["out"], "wb").write(b"RIFF" + b"\\0" * 60 if job["text"] != "junk" else b"junk" * 20)
"""


def voice_box(d, **kw):
    """A box with a voice: a fake worker script that stands in for IndexTTS2 (it writes the job it was given next to its output)."""
    worker = Path(d) / "fake_worker.py"
    worker.write_text(FAKE_WORKER)
    return Box(d, tts={"python": sys.executable, "dir": str(Path(d) / "ckpt"), "worker": str(worker)}, **kw)


def test_a_voice_take_returns_a_wav_with_the_hosted_call_shape_and_is_served_as_audio():
    with _t.tmpdir() as d:
        b = voice_box(d)
        try:
            assert b.call("GET", "/health", token=None)[1]["voice"] is True
            body = {**VOICE, "emo_control_method": "使用情感向量控制", "emo_happy": 0.6, "emo_calm": 0.2, "emo_ref_audio": ""}
            code, r = b.call("POST", f"{API}/indextts2-v1", body)
            assert code == 200, r
            res = b.wait(r["data"]["task_id"])
            assert res["status"] == "SUCCESS", res
            item = res["results"][0]
            assert item["type"] == "audio" and item["file_type"] == "wav" and item["url"].endswith(".wav"), item
            raw = urllib.request.urlopen(item["url"], timeout=10)
            assert raw.headers["Content-Type"] == "audio/wav" and raw.read(4) == b"RIFF"
            seen = json.loads(next((b.host.out_dir).glob("*.wav.seen")).read_text())
            assert seen["emo"] == "vector" and seen["vector"] == [0.6, 0, 0, 0, 0, 0, 0, 0.2] and seen["text"] == "Hello there." and seen["random"] is False
            assert Path(seen["spk"]).parent == b.root / "input"
            b.host.maintain(force=True)
            assert b.call("GET", "/health", token=None)[1]["busy"] >= 0
            assert not list((b.root / "input").iterdir()) and not list(Path(d, "state").glob("voice-*"))          # input and job file are gone
            assert "/free" in b.comfy.calls                                                          # ComfyUI was asked to make room
        finally:
            b.close()


def test_a_voice_that_is_the_samples_own_by_default_and_the_emotion_audio_is_fetched_only_when_asked_for():
    with _t.tmpdir() as d:
        b = voice_box(d)
        try:
            tid = b.call("POST", f"{API}/indextts2-v1", {**VOICE, "emo_ref_audio": WAV, "emo_alpha": 0.5})[1]["data"]["task_id"]
            assert b.wait(tid)["status"] == "SUCCESS"
            seen = json.loads(next(b.host.out_dir.glob("*.wav.seen")).read_text())
            assert seen["emo"] == "same" and seen["emo_audio"] is None and seen["vector"] is None and seen["alpha"] == 0.5       # an unused sample is not fetched
            tid = b.call("POST", f"{API}/indextts2-v1", {**VOICE, "emo_control_method": "使用情感参考音频", "emo_ref_audio": WAV})[1]["data"]["task_id"]
            assert b.wait(tid)["status"] == "SUCCESS"
            assert any(json.loads(p.read_text())["emo"] == "audio" and json.loads(p.read_text())["emo_audio"] for p in b.host.out_dir.glob("*.wav.seen"))
        finally:
            b.close()


def test_a_voice_request_the_box_would_not_run_is_a_400_with_nothing_queued():
    with _t.tmpdir() as d:
        b = voice_box(d)
        try:
            bad = [({"prompt_simple": WAV}, "prompt_text"), ({"prompt_text": " ", "prompt_simple": WAV}, "prompt_text"), ({"prompt_text": "x" * 2001, "prompt_simple": WAV}, "split"),
                   ({"prompt_text": "hi"}, "prompt_simple"), ({**VOICE, "emo_control_method": "使用情感描述文本控制"}, "text"),
                   ({**VOICE, "emo_control_method": "nope"}, "not served"), ({**VOICE, "emo_control_method": "使用情感向量控制"}, "every emo_"),
                   ({**VOICE, "emo_control_method": "使用情感向量控制", "emo_sad": 1.5}, "0 to 1"), ({**VOICE, "emo_control_method": "使用情感向量控制", "emo_sad": "x"}, "0 to 1"),
                   ({**VOICE, "emo_control_method": "使用情感参考音频"}, "emo_ref_audio"), ({**VOICE, "emo_alpha": 2}, "0 to 1"),
                   ({**VOICE, "prompt_simple": PNG}, "prompt_simple")]
            for body, word in bad:
                code, r = b.call("POST", f"{API}/indextts2-v1", body)
                assert code == 400 and word in r["msg"], (body, code, r)
            assert not list((b.root / "input").iterdir()) and b.call("GET", "/health", token=None)[1]["busy"] == 0
        finally:
            b.close()


def test_a_box_without_a_voice_says_so_in_health_and_refuses_the_voice_workflow_before_fetching_anything():
    with _t.tmpdir() as d:
        b = Box(d)
        try:
            assert b.call("GET", "/health", token=None)[1]["voice"] is False
            code, r = b.call("POST", f"{API}/indextts2-v1", VOICE)
            assert code == 400 and "no voice installed" in r["msg"] and not list((b.root / "input").iterdir()), (code, r)
        finally:
            b.close()


def test_a_voice_that_fails_says_why_leaves_no_file_and_the_worker_goes_on():
    with _t.tmpdir() as d:
        b = voice_box(d)
        try:
            for text, word in (("boom", "CUDA out of memory"), ("nothing", "without a wav"), ("junk", "without a wav")):
                tid = b.call("POST", f"{API}/indextts2-v1", {**VOICE, "prompt_text": text})[1]["data"]["task_id"]
                res = b.wait(tid)
                assert res["status"] == "FAILED" and word in res["error"], (text, res)
            assert not list(b.host.out_dir.glob("*.wav"))
            tid = b.call("POST", f"{API}/indextts2-v1", VOICE)[1]["data"]["task_id"]
            assert b.wait(tid)["status"] == "SUCCESS" and b.host.worker_thread.is_alive()
        finally:
            b.close()


def test_the_voice_process_does_not_inherit_the_token_and_an_expired_voice_reads_failed():
    with _t.tmpdir() as d:
        worker = Path(d) / "env_worker.py"
        worker.write_text("import json, os, sys\njob = json.load(open(sys.argv[1]))\n"
                          "open(job['out'], 'wb').write(b'RIFF' + b'0' * 60)\n"
                          "open(job['out'] + '.env', 'w').write(json.dumps(sorted(k for k in os.environ if k in ('CLOUDGPU_TOKEN', 'H3HOST_TOKEN'))))\n")
        os.environ["CLOUDGPU_TOKEN"] = TOKEN
        b = Box(d, tts={"python": sys.executable, "dir": d, "worker": str(worker)})
        try:
            tid = b.call("POST", f"{API}/indextts2-v1", VOICE)[1]["data"]["task_id"]
            res = b.wait(tid)
            assert res["status"] == "SUCCESS"
            assert json.loads(next(b.host.out_dir.glob("*.wav.env")).read_text()) == []
            for p in b.host.out_dir.glob("*.wav"):
                os.utime(p, (time.time() - adapter.FILE_TTL_S - 5,) * 2)
            res = b.call("GET", f"{API}/result/{tid}")[1]["data"]
            assert res["status"] == "FAILED" and "expired" in res["error"], res
        finally:
            os.environ.pop("CLOUDGPU_TOKEN", None)
            b.close()


LINE = adapter.LINE


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_the_line_lets_one_caller_go_at_a_time_in_the_order_they_joined():
    with _t.tmpdir() as d:
        b = Box(d)
        b.host.line.clock = clock = Clock()
        try:
            a = b.call("POST", LINE, {"label": "host1/take-a"})[1]["data"]
            c = b.call("POST", LINE, {"label": "host2/take-b"})[1]["data"]
            e = b.call("POST", LINE, {"label": "host3/take-c"})[1]["data"]
            assert (a["state"], a["ahead"], a["running"]) == ("go", 0, None) and a["lease_s"] == adapter.LEASE_S, a
            assert (c["state"], c["ahead"], c["running"]) == ("waiting", 1, "host1/take-a") and (e["state"], e["ahead"]) == ("waiting", 2), (c, e)
            assert b.call("GET", f"{LINE}/{c['ticket']}")[1]["data"]["state"] == "waiting"
            assert b.call("DELETE", f"{LINE}/{a['ticket']}")[0] == 200
            assert b.call("DELETE", f"{LINE}/{a['ticket']}")[0] == 200                        # idempotent
            got = b.call("GET", f"{LINE}/{c['ticket']}")[1]["data"]
            assert (got["state"], got["ahead"]) == ("go", 0), got
            assert b.call("GET", f"{LINE}/{e['ticket']}")[1]["data"]["ahead"] == 1
            lst = b.call("GET", LINE)[1]["data"]
            assert [(x["state"], x["label"]) for x in lst["line"]] == [("go", "host2/take-b"), ("waiting", "host3/take-c")] and "ticket" not in json.dumps(lst)
            clock.t += 5
            assert b.call("GET", LINE)[1]["data"]["line"][0]["age_s"] == 5
        finally:
            b.close()


def test_a_ticket_nobody_asks_about_is_dropped_so_a_dead_caller_never_blocks_the_line_and_asking_renews_it():
    with _t.tmpdir() as d:
        b = Box(d)
        b.host.line.clock = clock = Clock()
        try:
            a = b.call("POST", LINE, {"label": "dead"})[1]["data"]
            c = b.call("POST", LINE, {"label": "alive"})[1]["data"]
            for _ in range(5):                                                                 # the second keeps asking, the first never does
                clock.t += adapter.LEASE_S / 3
                assert b.call("GET", f"{LINE}/{c['ticket']}")[0] == 200
            got = b.call("GET", f"{LINE}/{c['ticket']}")[1]["data"]
            assert got["state"] == "go" and got["ahead"] == 0, got                              # the dead one's lease ran out and the line moved up
            assert b.call("GET", f"{LINE}/{a['ticket']}")[0] == 404
            clock.t += adapter.LEASE_S + 1
            assert b.call("GET", f"{LINE}/{c['ticket']}")[0] == 404 and b.call("GET", LINE)[1]["data"]["line"] == []
            assert b.call("GET", f"{LINE}/zzzz")[0] == 404 and b.call("GET", f"{LINE}/../health")[0] == 404
        finally:
            b.close()


def test_the_line_is_token_checked_bounded_and_its_labels_are_plain_text():
    with _t.tmpdir() as d:
        b = Box(d)
        try:
            for tok in (None, "wrong"):
                assert b.call("POST", LINE, {"label": "x"}, token=tok)[0] == 401 and b.call("GET", LINE, token=tok)[0] == 401
                assert b.call("GET", f"{LINE}/abc", token=tok)[0] == 401 and b.call("DELETE", f"{LINE}/abc", token=tok)[0] == 401
            assert b.host.line.count() == 0
            t = b.call("POST", LINE, {"label": "a\x00b\n" + "z" * 200})[1]["data"]
            lab = b.call("GET", LINE)[1]["data"]["line"][0]["label"]
            assert "\x00" not in lab and "\n" not in lab and len(lab) <= adapter.LABEL_MAX, lab
            assert b.call("POST", LINE, [1])[0] == 400 and b.call("POST", LINE, {"label": 5})[0] == 200          # not an object: refused; a label that is not text is read as text
            while b.host.line.count() < adapter.MAX_LINE:
                b.host.line.join("x")
            code, r = b.call("POST", LINE, {"label": "one too many"})
            assert code == 429 and r["code"] == "LineFull", (code, r)
            b.call("DELETE", f"{LINE}/{t['ticket']}")
        finally:
            b.close()


def test_a_caller_in_the_line_keeps_the_box_busy_and_a_take_without_a_ticket_still_runs():
    with _t.tmpdir() as d:
        b = Box(d)
        b.host.line.clock = clock = Clock()
        try:
            assert b.call("GET", "/health", token=None)[1]["busy"] == 0
            t = b.call("POST", LINE, {"label": "waits"})[1]["data"]
            assert b.call("GET", "/health", token=None)[1]["busy"] == 1                         # no power-off, no restart under a caller that waits
            tid = b.call("POST", f"{API}/minimax_h3_lightx2v", FL)[1]["data"]["task_id"]        # an older client knows nothing of the line
            assert b.wait(tid)["status"] == "SUCCESS"
            b.call("DELETE", f"{LINE}/{t['ticket']}")
            b.call("GET", f"{API}/result/{tid}")
            assert b.call("GET", "/health", token=None)[1]["busy"] <= 1
            u = b.call("POST", LINE, {})[1]["data"]
            clock.t += adapter.LEASE_S + 1                                                      # the lease ends: busy goes back down by itself
            assert b.host.line.count() == 0 and u["state"] == "go"
        finally:
            b.close()


if __name__ == "__main__":
    _t.main(globals())
