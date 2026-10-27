"""cloudgpu/detect_models.py and box/install.sh: detect_models.py reads which H3 files an image carries (both real layouts the cards were run on, a Blackwell and an Ada
card, links followed, no guess by substring), and install.sh installs and starts the adapter in a folder of its own from a bare box (no vendor path)."""

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import _t

BOX = Path(_t.ROOT) / "cloudgpu" / "onbox"
from cloudgpu import detect_models as dm  # noqa: E402

L4090 = ["diffusion_models/minimax/minimax_h3_ref2va_pruned_int8_convrot.safetensors", "diffusion_models/minimax/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
         "diffusion_models/minimax/minimax_h3_ref2va_XUELUO_int8_convrot.safetensors", "diffusion_models/minimax/minimax_h3_ref2va_int8_convrot.safetensors",
         "diffusion_models/minimax/Minimax-h3_Singularity_ref2va_v2_int8.safetensors",
         "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
         "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot_uncensored_FeiHou.safetensors",
         "vae/minimax_h3_video_vae_fp16.safetensors", "vae/minimax_h3_audio_vae_fp32.safetensors",
         "loras/minimax/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", "loras/minimax/minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
         "loras/minimax/minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors", "loras/minimax/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"]
L6000 = ["diffusion_models/MiniMax/minimax_h3_ref2va_pruned_int8_convrot.safetensors", "diffusion_models/MiniMax/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
         "text_encoders/MiniMax/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "text_encoders/MiniMax/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
         "vae/MiniMax/minimax_h3_video_vae_fp16.safetensors", "vae/MiniMax/minimax_h3_audio_vae_fp32.safetensors",
         "loras/MiniMax/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors", "loras/MiniMax/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"]


def tree(root: Path, files):
    for f in files:
        p = root / "models" / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    return root


def test_the_4090_image_layout_gives_the_preset_that_was_run_on_the_card():
    with _t.tmpdir() as d:
        r = dm.detect(tree(Path(d), L4090), 8.9)
        assert r["missing"] == [] and r["models"] == json.loads((BOX / "models.4090.json").read_text()), r


def test_the_pro6000_image_layout_takes_the_nvfp4_encoder_on_a_blackwell_card_and_the_int8_one_otherwise():
    with _t.tmpdir() as d:
        r = dm.detect(tree(Path(d), L6000), 12.0)
        assert r["missing"] == [] and r["models"]["clip"].endswith("nvfp4_awq.safetensors") and r["models"]["unet_ref"].startswith("MiniMax/"), r
        assert r["models"]["lora_ref"].endswith("fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors")          # no ref2v turbo lora here: the fl2v one
        assert dm.detect(Path(d), 8.9)["models"]["clip"].endswith("int8_convrot.safetensors")
        assert dm.detect(Path(d), 0.0)["models"]["clip"].endswith("int8_convrot.safetensors")                    # capability unknown: the safe one
    preset = json.loads((BOX / "models.pro6000.json").read_text())
    assert set(preset) == set(dm.wanted(True)) | set(dm.STEPS)


def test_an_authors_merge_or_an_uncensored_variant_is_never_picked_and_a_missing_file_is_named():
    drop = {"diffusion_models/minimax/minimax_h3_ref2va_pruned_int8_convrot.safetensors", "diffusion_models/minimax/minimax_h3_ref2va_int8_convrot.safetensors",
            "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"}
    with _t.tmpdir() as d:
        tree(Path(d), [f for f in L4090 if f not in drop])        # left: two authors' merges, an uncensored encoder, an nvfp4 encoder (not for this card)
        r = dm.detect(Path(d), 8.9)
        assert "unet_ref" in r["missing"] and "clip" in r["missing"] and "unet_ref" not in r["models"] and "clip" not in r["models"], r
        assert dm.detect(Path(d), 12.0)["models"]["clip"] == "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors".split("/", 1)[1]


def test_links_are_followed_because_the_shared_pools_are_links():
    with _t.tmpdir() as d:
        root = Path(d)
        pool = root / "pool"
        for f in L4090:
            p = pool / Path(f).name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")
        for f in L4090:
            link = root / "comfy" / "models" / f
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(pool / Path(f).name)
        assert dm.detect(root / "comfy", 8.9)["missing"] == []


def test_the_cli_writes_the_file_only_when_every_key_was_found():
    with _t.tmpdir() as d:
        root, out = Path(d) / "comfy", Path(d) / "models.json"
        tree(root, L4090[:3])
        r = subprocess.run([sys.executable, str(Path(_t.ROOT) / "cloudgpu" / "detect_models.py"), "--comfy-root", str(root), "--compute-cap", "8.9", "--out", str(out)],
                           capture_output=True, text=True)
        assert r.returncode == 1 and not out.exists() and "missing" in r.stdout
        tree(root, L4090)
        r = subprocess.run([sys.executable, str(Path(_t.ROOT) / "cloudgpu" / "detect_models.py"), "--comfy-root", str(root), "--compute-cap", "8.9", "--out", str(out)],
                           capture_output=True, text=True)
        assert r.returncode == 0 and json.loads(out.read_text()) == json.loads((BOX / "models.4090.json").read_text())


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_install_sh_installs_in_a_folder_of_its_own_detects_the_models_and_starts_the_adapter():
    if not (Path("/bin/sh").exists()):
        return _t.skip("no sh")
    with _t.tmpdir() as d:
        home = Path(d) / "home"
        tree(home / "ComfyUI", L4090)
        work = Path(d) / "box"
        work.mkdir()
        for f in ("onbox/install.sh", "detect_models.py", "onbox/models.4090.json", "adapter.py"):
            src = Path(_t.ROOT) / "cloudgpu" / f
            (work / Path(f).name).write_bytes(src.read_bytes())
        port = free_port()
        env = {"PATH": os.environ["PATH"], "HOME": str(home), "CLOUDGPU_PYTHON": sys.executable}
        r = subprocess.run(["sh", str(work / "install.sh"), "auto", str(port)], input="x" * 8, env=env, capture_output=True, text=True, timeout=60)
        assert r.returncode != 0 and "shorter than 24" in r.stderr, (r.returncode, r.stderr)
        r = subprocess.run(["sh", str(work / "install.sh"), "auto", str(port)], input="x" * 23 + "\n", env=env, capture_output=True, text=True, timeout=60)
        assert r.returncode != 0 and "shorter than 24" in r.stderr, "a 23 character token with its newline is 24 bytes but not 24 characters"
        token = "t" * 32
        try:
            r = subprocess.run(["sh", str(work / "install.sh"), "auto", str(port)], input=token, env=env, capture_output=True, text=True, timeout=60)
            assert r.returncode == 0, r.stderr
            box = home / "h3host"
            assert json.loads((box / "models.json").read_text())["clip"] == "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"      # no GPU here: the safe encoder
            assert (box / "server.py").is_file() and (box / "start.sh").is_file() and oct((box / "token").stat().st_mode & 0o777) == "0o600"
            assert token not in (box / "start.sh").read_text()                                   # the token is read from its file, never written into a script
            h = None
            for _ in range(30):
                try:
                    h = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3).read())
                    break
                except OSError:
                    time.sleep(0.3)
            assert h is not None and h["ok"] is True and h["comfy"] is False, h              # up; no ComfyUI in this test
        finally:
            out = subprocess.run(["pgrep", "-f", f"{home}/h3host/server.py"], capture_output=True, text=True).stdout.split()
            for pid in out:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except OSError:
                    pass


def test_start_sh_leaves_an_adapter_with_a_take_on_it_alone_and_the_adapter_binds_loopback():
    with _t.tmpdir() as d:
        home = Path(d) / "home"
        tree(home / "ComfyUI", L4090)
        work = Path(d) / "box"
        work.mkdir()
        for f in ("onbox/install.sh", "detect_models.py", "onbox/models.4090.json", "adapter.py"):
            (work / Path(f).name).write_bytes((Path(_t.ROOT) / "cloudgpu" / f).read_bytes())
        port = free_port()
        env = {"PATH": os.environ["PATH"], "HOME": str(home), "CLOUDGPU_PYTHON": sys.executable}
        box = home / "h3host"
        busy = None
        try:
            r = subprocess.run(["sh", str(work / "install.sh"), "auto", str(port)], input="t" * 32 + "\n", env=env, capture_output=True, text=True, timeout=60)
            assert r.returncode == 0, r.stderr                                          # a 32 character token with a newline is fine
            start = (box / "start.sh").read_text()
            assert "CLOUDGPU_BIND" in start and "--bind" not in start                   # the default (127.0.0.1) is the adapter's; the script only passes a wider one on
            subprocess.run(["pkill", "-f", f"{box}/server.py"])
            time.sleep(0.5)
            (box / "server.py").write_text(                                             # stands in for an adapter with a take running
                "import sys\nfrom http.server import BaseHTTPRequestHandler, HTTPServer\n"
                "class H(BaseHTTPRequestHandler):\n"
                "    def log_message(self, *a): pass\n"
                "    def do_GET(self):\n"
                "        b = b'{\"ok\": true, \"comfy\": true, \"busy\": 1}'\n"
                "        self.send_response(200); self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)\n"
                f"HTTPServer(('127.0.0.1', {port}), H).serve_forever()\n")
            busy = subprocess.Popen([sys.executable, str(box / "server.py")])
            for _ in range(30):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).read()
                    break
                except OSError:
                    time.sleep(0.2)
            r = subprocess.run(["sh", str(box / "start.sh")], env=env, capture_output=True, text=True, timeout=60)
            assert r.returncode == 0 and "not restarted" in r.stdout, (r.stdout, r.stderr)
            time.sleep(0.5)
            assert busy.poll() is None, "start.sh killed an adapter that had a take on it"
            # busy 0 (or nothing answering) is restarted as before
            busy.kill()
            busy.wait()
            busy = None
            (box / "server.py").write_bytes((work / "adapter.py").read_bytes())
            r = subprocess.run(["sh", str(box / "start.sh")], env=env, capture_output=True, text=True, timeout=60)
            assert r.returncode == 0 and '"busy": 0' in r.stdout, (r.stdout, r.stderr)
        finally:
            if busy:
                busy.kill()
            for pid in subprocess.run(["pgrep", "-f", f"{home}/h3host/server.py"], capture_output=True, text=True).stdout.split():
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except OSError:
                    pass


if __name__ == "__main__":
    _t.main(globals())
