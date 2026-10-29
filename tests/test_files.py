"""cloudgpu.files: a picture sent once and named by its sha256 afterwards. A real adapter over HTTP (the file store of 0.10.0) and the Box client; an adapter without the store is
not asked and the consumer sends the data inline as before."""

import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import _t
import test_adapter as ta

from cloudgpu import files
from cloudgpu.box import Box as RealBox
from cloudgpu.errors import CloudGpuError


def client(base):
    return RealBox({"CLOUDGPU_BASE": base, "CLOUDGPU_TOKEN": ta.TOKEN, "CLOUDGPU_POWER": "none"})


def test_a_file_is_sent_once_and_the_box_is_asked_before_every_later_take():
    with _t.tmpdir() as d:
        a = ta.Box(d)
        try:
            box = client(a.base)
            pic = Path(d) / "still.png"
            pic.write_bytes(ta.BLOB)
            assert box.supports_files() is True
            sha = files.sha256_of(pic)
            assert files.has(box, sha) is False
            ref = box.ensure_file(pic)
            assert ref == f"sha256:{sha}" and files.has(box, sha) is True
            stored = list(a.host.files_dir.glob(f"{sha}.*"))
            assert len(stored) == 1
            first = stored[0].stat().st_mtime_ns
            assert box.ensure_file(pic) == ref, "the same answer"
            assert stored[0].stat().st_mtime_ns == first, "and nothing was sent again"
            code, doc = a.call("POST", f"{ta.API}/minimax_h3_image_audio_to_video", {**ta.REF, "ref_image_0": ref})
            assert code == 200 and a.wait(doc["data"]["task_id"])["status"] == "SUCCESS", doc
        finally:
            a.close()


def test_an_unsupported_type_and_a_wrong_token_are_errors_and_a_changed_file_gets_its_own_sha():
    with _t.tmpdir() as d:
        a = ta.Box(d)
        try:
            box = client(a.base)
            txt = Path(d) / "note.txt"
            txt.write_text("not a picture")
            with _t.raises("http"):
                files.put(box, txt)
            pic = Path(d) / "still.png"
            pic.write_bytes(ta.BLOB)
            one = files.sha256_of(pic)
            pic.write_bytes(ta.BLOB + b"changed")
            assert files.sha256_of(pic) != one, "the sum follows the file's size and time, not its name"
            bad = RealBox({"CLOUDGPU_BASE": a.base, "CLOUDGPU_TOKEN": "wrong-token-0123456789abcdef", "CLOUDGPU_POWER": "none"})
            with _t.raises("http"):
                files.ensure(bad, pic)
        finally:
            a.close()


class OldAdapter(BaseHTTPRequestHandler):
    """An adapter older than 0.10.0: /health without `files`, and 404 on every other path."""

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = b'{"ok": true, "comfy": true, "busy": 0, "worker": "alive", "card": "unknown", "voice": false}' if self.path.startswith("/health") else b"{}"
        self.send_response(200 if self.path.startswith("/health") else 404)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_an_adapter_without_the_store_is_not_asked_and_a_box_that_does_not_answer_is_not_supported():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), OldAdapter)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        box = client(f"http://127.0.0.1:{srv.server_address[1]}")
        assert box.supports_files() is False
    finally:
        srv.shutdown()
    assert client("http://127.0.0.1:9").supports_files() is False


if __name__ == "__main__":
    _t.main(globals())
