"""cloudgpu/adapter.py, the audio-locked workflow (0.20.0), offline: a fake ComfyUI stands in for the card and answers /object_info like ComfyUI does.

minimax_h3_image_audio_locked is the audio-driven take with one more node that anchors the take's own audio: the official MiniMaxH3AddGuide (default) or the third-party
VRGDG_MiniMaxH3AudioDrive (off unless the operator sets CLOUDGPU_ALLOW_NODE_VRGDG=1). What these tests hold: the audiodrive graph is the one measured on a 4090 (modulo model names); the
addguide graph changes the unlocked one in exactly the two places its definition asks; no model file name is written in the lock code; the unlocked workflows build exactly the
graphs of 0.19.0; a lock that cannot be served is a coded error and never another graph; /health says what is served and why; the licence switch is read from the environment."""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import _t
from test_adapter import API, PNG, REF, TOKEN, Box

from cloudgpu import adapter  # noqa: E402

LOCKED = f"{API}/{adapter.LOCKED_WORKFLOW}"
UNLOCKED = f"{API}/minimax_h3_image_audio_to_video"
SHA_A, SHA_B = "a" * 40, "b" * 40
PREFIX = "comfyui-vrgamedevgirl"

# What /object_info/<node> answers for the two nodes (the input names are the ones the graphs wire; python_module is how ComfyUI names where a node comes from).
DEFS = {
    "addguide": {"input": {"required": {"positive": ["CONDITIONING"], "latent": ["LATENT"], "frame_idx": ["INT", {"default": 0}]},
                           "optional": {"vae": ["VAE"], "audio_vae": ["VAE"], "image": ["IMAGE"], "audio": ["AUDIO"]}},
                 "output": ["CONDITIONING"], "python_module": "comfy_extras.nodes_minimax_h3"},
    "audiodrive": {"input": {"required": {"av_latent": ["LATENT"], "source_audio": ["AUDIO"], "audio_vae": ["VAE"]}},
                   "output": ["LATENT", "AUDIO"], "python_module": f"custom_nodes.{PREFIX}"},
}

# The graph the operator measured on a real 4090 (2026-11-03: audio correlation 0.945, same still and voice), API format, with its placeholders kept as they were.
TESTED = {
    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax/minimax_h3_ref2va_pruned_int8_convrot.safetensors", "weight_dtype": "default"}},
    "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "type": "minimax", "device": "default"}},
    "3": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
    "4": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
    "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
    "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": 0}},
    "10": {"class_type": "BasicScheduler", "inputs": {"model": ["20", 0], "scheduler": "simple", "steps": 4, "denoise": 1.0}},
    "11": {"class_type": "BasicGuider", "inputs": {"model": ["20", 0], "conditioning": ["7", 0]}},
    "12": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["9", 0], "guider": ["11", 0], "sampler": ["8", 0], "sigmas": ["10", 0], "latent_image": ["21", 0]}},
    "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
    "15": {"class_type": "CreateVideo", "inputs": {"images": ["13", 0], "audio": ["21", 1], "fps": 24.0}},
    "16": {"class_type": "SaveVideo", "inputs": {"video": ["15", 0], "filename_prefix": "cloudgpu/h3-{TAKE}", "format": "auto", "codec": "auto"}},
    "20": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "minimax/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", "strength_model": 1.0}},
    "5": {"class_type": "LoadImage", "inputs": {"image": "{STILL}.png"}},
    "6": {"class_type": "LoadAudio", "inputs": {"audio": "{VOICE}.wav"}},
    "7": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
        "clip": ["2", 0], "vae": ["3", 0], "audio_vae": ["4", 0],
        "prompt": "[Shot 1] <Subject 1> (S1) whose appearance comes from <Picture 1> looks into the lens and says, <d>[Chinese] {LINE}</d>\n"
                  "<Audio 1>: fully_copy - the original audio of <Audio 1> is the complete final track; <Subject 1> (S1) lip-syncs exactly to it.\n"
                  "No captions, subtitles, logos or on-screen text appear.",
        "width": 768, "height": 1344, "length": 124, "ref_image_size": "match", "ref_images.ref_image_0": ["5", 0], "ref_audios.ref_audio_0": ["6", 0]}},
    "21": {"class_type": "VRGDG_MiniMaxH3AudioDrive", "inputs": {"av_latent": ["7", 1], "source_audio": ["6", 0], "audio_vae": ["4", 0]}},
}
PRESET_4090 = json.loads((Path(_t.ROOT) / "cloudgpu" / "onbox" / "models.4090.json").read_text())


class Models:
    """`adapter.MODELS` set to `names` for a block, then put back."""

    def __init__(self, names):
        self.names, self.old = names, dict(adapter.MODELS)

    def __enter__(self):
        adapter.MODELS.update(self.names)

    def __exit__(self, *a):
        adapter.MODELS.clear()
        adapter.MODELS.update(self.old)


def build(lock=None, **kw):
    args = dict(w=768, h=1344, seconds=5, image="{STILL}.png", audio="{VOICE}.wav", prompt=TESTED["7"]["inputs"]["prompt"], seed=0, prefix="cloudgpu/h3-{TAKE}")
    return adapter.graph("ref", **{**args, **kw}, lock=lock)


def box(d, nodes=("addguide",), allow=False):
    b = Box(d, allow_vrgdg=allow)
    b.comfy.nodes = {adapter.LOCK_NODES[k]: DEFS[k] for k in nodes}
    return b


BODY = {**REF, "audio_duration": 5, "seed": 3, "prompt": "[Shot 1] <Subject 1> (S1) looks into the lens and says, <d>[Chinese] 你好</d>\n<Audio 1>: fully_copy"}


def test_the_audiodrive_graph_is_the_one_measured_on_the_4090():
    with Models(PRESET_4090):
        assert build("audiodrive") == TESTED
    names = {"unet_ref": "S/unet.safetensors", "lora_ref": "S/lora.safetensors", "steps_ref": 3, "clip": "S/clip.safetensors", "vae_video": "S/vv.safetensors", "vae_audio": "S/va.safetensors"}
    swap = {PRESET_4090["unet_ref"]: names["unet_ref"], PRESET_4090["lora_ref"]: names["lora_ref"], PRESET_4090["clip"]: names["clip"],
            PRESET_4090["vae_video"]: names["vae_video"], PRESET_4090["vae_audio"]: names["vae_audio"]}
    want = json.loads(json.dumps(TESTED))
    for n in want.values():
        for k, v in n["inputs"].items():
            if isinstance(v, str) and v in swap:
                n["inputs"][k] = swap[v]
    want["10"]["inputs"]["steps"] = 3
    with Models(names):
        assert build("audiodrive") == want, "the model names and the steps come from the box's MODELS, whatever they are"


def test_the_addguide_graph_is_the_unlocked_graph_with_one_node_and_the_guider_on_its_conditioning():
    with Models(PRESET_4090):
        plain, locked = build(None), build("addguide")
    assert plain["7"]["class_type"] == "MiniMaxH3ReferenceToVideo" and "21" not in plain
    assert locked["21"] == {"class_type": "MiniMaxH3AddGuide", "inputs": {"positive": ["7", 0], "latent": ["7", 1], "frame_idx": 0, "vae": ["3", 0], "audio_vae": ["4", 0], "audio": ["6", 0]}}
    assert locked["11"]["inputs"]["conditioning"] == ["21", 0]
    rest = {k: v for k, v in locked.items() if k != "21"}
    rest["11"] = json.loads(json.dumps(rest["11"]))
    rest["11"]["inputs"]["conditioning"] = ["7", 0]
    assert rest == plain, "nothing else differs: the latent, the audio decode, the video and the reference node are the unlocked take's"
    assert locked["12"]["inputs"]["latent_image"] == ["7", 1] and locked["15"]["inputs"]["audio"] == ["14", 0] and locked["14"]["class_type"] == "VAEDecodeAudio"


def test_audiodrive_leaves_the_reference_node_the_prompt_and_every_picture_as_in_the_unlocked_graph():
    with Models(PRESET_4090):
        plain = build(None, extra_images=["b.png", "c.png"])
        locked = build("audiodrive", extra_images=["b.png", "c.png"])
    for k in plain:
        if k not in ("12", "14", "15"):
            assert locked[k] == plain[k], k
    assert set(locked) == (set(plain) - {"14"}) | {"21"}
    assert locked["7"]["inputs"]["ref_audios.ref_audio_0"] == ["6", 0], "the node's own description: the same audio is also the reference <Audio 1>"


def test_the_unlocked_workflows_build_exactly_the_graphs_of_0_19_0():
    golden = json.loads((Path(__file__).parent / "graphs_0_19_0.json").read_text())
    assert set(golden) == {"ref", "ref_3_pictures", "fl", "t2v"}
    with Models(PRESET_4090):
        for name, case in golden.items():
            args = dict(case["args"])
            kind = args.pop("kind")
            assert adapter.graph(kind, **args) == case["graph"], name
            assert adapter.graph(kind, **args, lock=None) == case["graph"], name


def test_no_model_file_name_is_written_in_the_lock_code():
    src = Path(adapter.__file__).read_text().splitlines()
    start = next(i for i, ln in enumerate(src) if ln.startswith("MODELS = {"))
    end = next(i for i in range(start, len(src)) if src[i] == "}")
    outside = [ln for i, ln in enumerate(src) if ".safetensors" in ln and not start <= i <= end]
    assert not outside, "a model file name outside MODELS: " + outside[0][:120]
    lock_code = "\n".join(src[next(i for i, ln in enumerate(src) if ln.startswith("def lock_addguide")):next(i for i, ln in enumerate(src) if ln.startswith("LOCK_BUILDERS"))])
    assert "safetensors" not in lock_code and "MODELS[" not in lock_code, "the lock builders change the finished graph and never name a model"


def test_every_lock_value_builds_a_graph_that_is_not_the_unlocked_one_or_stops():
    with Models(PRESET_4090):
        plain = build(None)
        for lock in adapter.LOCKS:
            if lock == "custom":
                continue
            g = build(lock)
            assert g != plain and "21" in g, lock
        for lock in ("custom", "nope", ""):
            try:
                g = build(lock)
            except adapter.LockUnavailable as e:
                assert str(e).startswith(("lock_not_implemented:", "lock_unknown:")), (lock, str(e))
            else:
                raise AssertionError(f"lock {lock!r} built a graph ({'unlocked' if g == plain else 'other'}) instead of stopping")
        try:
            adapter.graph("fl", w=480, h=864, seconds=4, image="a.png", audio=None, prompt="p", seed=1, prefix="x", lock="addguide")
        except adapter.LockUnavailable as e:
            assert str(e).startswith("lock_wrong_workflow:")
        else:
            raise AssertionError("a lock on a graph that has no audio to anchor")
    assert set(adapter.LOCK_BUILDERS) == set(adapter.LOCKS) and set(adapter.LOCK_NODES) == set(adapter.LOCKS) - {"custom"}


def test_a_locked_take_runs_with_the_official_node_by_default_and_the_adapter_adds_no_words_of_its_own():
    with _t.tmpdir() as d:
        b = box(d)
        try:
            for body in (BODY, {**BODY, "lock": "addguide"}):
                code, doc = b.call("POST", LOCKED, body)
                assert code == 200 and doc["code"] == "Success", doc
                r = b.wait(doc["data"]["task_id"])
                assert r["status"] == "SUCCESS", r
                g = b.comfy.graphs[-1]
                assert g["21"]["class_type"] == "MiniMaxH3AddGuide" and g["11"]["inputs"]["conditioning"] == ["21", 0]
                assert g["7"]["inputs"]["prompt"] == BODY["prompt"], "the prompt is the request's, word for word"
                assert (g["7"]["inputs"]["width"], g["7"]["inputs"]["height"], g["7"]["inputs"]["length"]) == (768, 1344, 124)
                assert g["9"]["inputs"]["noise_seed"] == 3 and g["1"]["inputs"]["unet_name"] == adapter.MODELS["unet_ref"]
                assert g["20"]["inputs"]["lora_name"] == adapter.MODELS["lora_ref"] and g["10"]["inputs"]["steps"] == adapter.MODELS["steps_ref"]
                assert not any(n["class_type"].startswith("VRGDG") for n in g.values())
            assert b.host.load(doc["data"]["task_id"])["plan"]["lock"] == "addguide"
            # no prompt: the unlocked workflow's default, which has no speech in it
            code, doc = b.call("POST", LOCKED, {k: v for k, v in BODY.items() if k != "prompt"})
            assert code == 200 and b.wait(doc["data"]["task_id"])["status"] == "SUCCESS"
            p = b.comfy.graphs[-1]["7"]["inputs"]["prompt"]
            assert p == adapter.DEFAULT_REF_PROMPT and "<d>" not in p and "says" not in p
        finally:
            b.close()


def test_the_audiodrive_lock_is_refused_with_the_licence_words_unless_the_operator_switched_it_on_even_when_the_node_is_there():
    with _t.tmpdir() as d:
        b = box(d, nodes=("addguide", "audiodrive"), allow=False)
        try:
            code, doc = b.call("POST", LOCKED, {**BODY, "lock": "audiodrive"})
            assert code == 400 and doc["code"] == "RequestParameterIsWrong" and doc["msg"].startswith("lock_not_allowed:"), doc
            for word in ("VRGDG_MiniMaxH3AudioDrive", "comfyui-vrgamedevgirl", "VRGameDevGirl Community and Commercial License v1.0", "not an open licence", "commercial licence",
                         "CLOUDGPU_ALLOW_NODE_VRGDG=1", "owner", "addguide"):
                assert word in doc["msg"], (word, doc["msg"])
            assert not b.comfy.graphs and not list(b.host.tasks_dir.iterdir()) and not list((b.root / "input").iterdir()), "nothing was fetched, saved or sent"
            # the same refusal when ComfyUI knows nothing yet: the licence is not waiting on a node list
            b.comfy.object_info_fail = True
            b.host._lock_caps = (0.0, None)
            assert b.call("POST", LOCKED, {**BODY, "lock": "audiodrive"})[1]["msg"].startswith("lock_not_allowed:")
            # and the other lock is untouched by it
            b.comfy.object_info_fail = False
            b.host._lock_caps = (0.0, None)
            code, doc = b.call("POST", LOCKED, BODY)
            assert code == 200 and b.wait(doc["data"]["task_id"])["status"] == "SUCCESS"
        finally:
            b.close()


def test_the_audiodrive_lock_runs_when_the_operator_switched_it_on_and_health_says_so_with_the_pack_and_its_commit():
    with _t.tmpdir() as d:
        b = box(d, nodes=("addguide", "audiodrive"), allow=True)
        try:
            pack = b.root / "custom_nodes" / PREFIX
            (pack / ".git").mkdir(parents=True)
            (pack / ".git" / "HEAD").write_text(SHA_A + "\n")
            (pack / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "1.2.3"\n')
            code, doc = b.call("POST", LOCKED, {**BODY, "lock": "audiodrive"})
            assert code == 200, doc
            assert b.wait(doc["data"]["task_id"])["status"] == "SUCCESS"
            g = b.comfy.graphs[-1]
            assert g["21"]["class_type"] == "VRGDG_MiniMaxH3AudioDrive" and "14" not in g and g["15"]["inputs"]["audio"] == ["21", 1]
            h = b.call("GET", "/health", token=None)[1]
            assert h["audio_lock"] == ["addguide", "audiodrive"], h
            det = h["audio_lock_detail"]
            assert det["checked"] is True
            assert det["addguide"]["node"] == "MiniMaxH3AddGuide" and det["addguide"]["present"] is True and det["addguide"]["usable"] is True
            assert det["addguide"]["module"] == "comfy_extras.nodes_minimax_h3" and det["addguide"]["problem"] is None
            a = det["audiodrive"]
            assert a["node"] == "VRGDG_MiniMaxH3AudioDrive" and a["present"] is True and a["allowed"] is True and a["usable"] is True and a["problem"] is None
            assert a["status"] == "audiodrive (third-party, licence-restricted, enabled)"
            assert a["pack"] == PREFIX and a["commit"] == SHA_A and a["version"] == "1.2.3" and "VRGameDevGirl Community and Commercial License v1.0" in a["licence"]
        finally:
            b.close()


def test_health_says_which_locks_are_served_and_why_not_when_the_switch_is_off_or_a_node_is_missing_or_comfyui_is_silent():
    with _t.tmpdir() as d:
        b = box(d, nodes=("addguide", "audiodrive"), allow=False)
        try:
            h = b.call("GET", "/health", token=None)[1]
            assert h["audio_lock"] == ["addguide"] and h["audio_lock_detail"]["checked"] is True
            a = h["audio_lock_detail"]["audiodrive"]
            assert a["present"] is True and a["allowed"] is False and a["usable"] is False and a["status"] == "audiodrive (third-party, licence-restricted, off: CLOUDGPU_ALLOW_NODE_VRGDG is not set)"
            assert a["problem"].startswith("lock_not_allowed:")
            # no node at all
            b.comfy.nodes = {}
            b.host._lock_caps = (0.0, None)
            h = b.call("GET", "/health", token=None)[1]
            assert h["audio_lock"] == [] and h["audio_lock_detail"]["addguide"]["present"] is False
            assert h["audio_lock_detail"]["addguide"]["problem"].startswith("lock_node_missing:") and "MiniMaxH3AddGuide" in h["audio_lock_detail"]["addguide"]["problem"]
            # ComfyUI up but its node list unreadable: nothing is known, nothing is offered
            b.comfy.nodes = {adapter.LOCK_NODES["addguide"]: DEFS["addguide"]}
            b.comfy.object_info_fail = True
            b.host._lock_caps = (0.0, None)
            h = b.call("GET", "/health", token=None)[1]
            assert h["comfy"] is True and h["audio_lock"] == [] and h["audio_lock_detail"]["checked"] is False
            assert h["audio_lock_detail"]["addguide"]["present"] is None and h["audio_lock_detail"]["addguide"]["problem"].startswith("lock_state_unknown:")
            # and it is read again, not remembered: the cache for a silent ComfyUI is the short one
            b.comfy.object_info_fail = False
            time.sleep(adapter.COMFY_CACHE_S + 0.2)
            assert b.call("GET", "/health", token=None)[1]["audio_lock"] == ["addguide"]
        finally:
            b.close()


def test_health_of_a_box_whose_comfyui_is_down_offers_no_lock():
    with _t.tmpdir() as d:
        h = adapter.Host(TOKEN, "http://127.0.0.1:9", Path(d) / "state", Path(d) / "comfy")
        out = h.health()
        assert out["comfy"] is False and out["audio_lock"] == [] and out["audio_lock_detail"]["checked"] is False
        assert out["audio_lock_detail"]["audiodrive"]["allowed"] is False and out["audio_lock_detail"]["addguide"]["usable"] is False
        assert out["files"] == 1 and out["t2v"] is True and "voice" in out, "the fields that were there are still there"


def test_a_lock_whose_node_is_missing_is_a_coded_error_that_names_the_node_and_no_other_graph_is_sent():
    with _t.tmpdir() as d:
        b = box(d, nodes=(), allow=True)
        try:
            code, doc = b.call("POST", LOCKED, BODY)
            assert code == 400 and doc["code"] == "RequestParameterIsWrong" and doc["msg"].startswith("lock_node_missing:") and "MiniMaxH3AddGuide" in doc["msg"], doc
            assert "no other graph was used" in doc["msg"]
            code, doc = b.call("POST", LOCKED, {**BODY, "lock": "audiodrive"})
            assert code == 400 and doc["msg"].startswith("lock_node_missing:") and "VRGDG_MiniMaxH3AudioDrive" in doc["msg"] and "comfyui-vrgamedevgirl" in doc["msg"], doc
            assert not b.comfy.graphs, "no fallback to the unlocked graph, to the other lock, or to anything"
            assert not list(b.host.tasks_dir.iterdir()) and not list((b.root / "input").iterdir()) and b.host.busy() == 0
            # the unlocked workflow is not affected by the lock nodes at all
            code, doc = b.call("POST", UNLOCKED, BODY)
            assert code == 200 and b.wait(doc["data"]["task_id"])["status"] == "SUCCESS"
            assert "21" not in b.comfy.graphs[-1] and "14" in b.comfy.graphs[-1]
        finally:
            b.close()


def test_a_node_whose_definition_lacks_an_input_the_graph_wires_is_incompatible_and_refused():
    with _t.tmpdir() as d:
        b = box(d, nodes=("addguide",))
        try:
            thin = json.loads(json.dumps(DEFS["addguide"]))
            del thin["input"]["optional"]["audio"]
            b.comfy.nodes = {"MiniMaxH3AddGuide": thin}
            code, doc = b.call("POST", LOCKED, BODY)
            assert code == 400 and doc["msg"].startswith("lock_node_incompatible:") and "audio" in doc["msg"] and "MiniMaxH3AddGuide" in doc["msg"], doc
            assert not b.comfy.graphs
            h = b.call("GET", "/health", token=None)[1]
            assert h["audio_lock"] == [] and h["audio_lock_detail"]["addguide"]["present"] is True and h["audio_lock_detail"]["addguide"]["usable"] is False
        finally:
            b.close()


def test_a_refusal_is_not_stale_a_node_that_appears_after_the_first_look_is_used_without_waiting_for_the_cache():
    with _t.tmpdir() as d:
        b = box(d, nodes=())
        try:
            assert b.call("GET", "/health", token=None)[1]["audio_lock"] == []          # cached as missing
            b.comfy.nodes = {"MiniMaxH3AddGuide": DEFS["addguide"]}                      # ComfyUI restarted with it
            code, doc = b.call("POST", LOCKED, BODY)
            assert code == 200 and b.wait(doc["data"]["task_id"])["status"] == "SUCCESS", doc
        finally:
            b.close()


def test_a_take_accepted_while_comfyui_could_not_say_is_failed_in_words_and_is_not_run_as_another_graph():
    with _t.tmpdir() as d:
        b = box(d, nodes=("addguide",))
        try:
            b.comfy.object_info_fail = True
            code, doc = b.call("POST", LOCKED, BODY)
            assert code == 200, "ComfyUI is still starting: the take waits for it as every take does"
            r = b.wait(doc["data"]["task_id"])
            assert r["status"] == "FAILED" and "lock_state_unknown:" in r["error"] and "MiniMaxH3AddGuide" in r["error"], r
            assert not b.comfy.graphs, "failed in words, not run as another graph"
            b.comfy.object_info_fail = False                                              # ComfyUI can say now, and has the node
            code, doc = b.call("POST", LOCKED, BODY)
            assert code == 200 and b.wait(doc["data"]["task_id"])["status"] == "SUCCESS"
            assert len(b.comfy.graphs) == 1 and b.comfy.graphs[0]["21"]["class_type"] == "MiniMaxH3AddGuide"
        finally:
            b.close()


def test_a_run_time_check_stops_a_take_whose_node_vanished_after_it_was_accepted():
    with _t.tmpdir() as d:
        b = box(d, nodes=("addguide",))
        try:
            h = b.host
            p = adapter.plan(adapter.LOCKED_WORKFLOW, BODY)
            assert p["lock"] == "addguide" and p["kind"] == "ref"
            t = {"id": "h3-" + "0" * 20, "workflow": adapter.LOCKED_WORKFLOW, "status": "RUNNING", "created": time.time(), "plan": p, "inputs": ["s.png", "v.wav"]}
            b.comfy.nodes = {}
            try:
                h.run(t)
            except adapter.LockUnavailable as e:
                assert str(e).startswith("lock_node_missing:")
            else:
                raise AssertionError("the take ran without its node")
            assert not b.comfy.graphs
        finally:
            b.close()


def test_a_lock_the_box_does_not_serve_or_sent_to_a_workflow_that_does_not_lock_is_refused_before_anything_is_queued():
    with _t.tmpdir() as d:
        b = box(d, nodes=("addguide", "audiodrive"), allow=True)
        try:
            cases = [(LOCKED, {**BODY, "lock": "custom"}, "lock_not_implemented:"), (LOCKED, {**BODY, "lock": "nope"}, "lock_unknown:"), (LOCKED, {**BODY, "lock": ""}, "lock_unknown:"),
                     (LOCKED, {**BODY, "lock": ["addguide"]}, "lock_unknown:"), (LOCKED, {**BODY, "lock": 1}, "lock_unknown:"),
                     (UNLOCKED, {**BODY, "lock": "addguide"}, "lock_wrong_workflow:"), (f"{API}/minimax_h3_lightx2v", {"lock": "addguide", "prompt": "x", "first_frame": PNG, "last_frame": PNG, "duration": 4}, "lock_wrong_workflow:"),
                     (f"{API}/minimax_h3_t2v", {"lock": "audiodrive", "prompt": "x", "duration": 4}, "lock_wrong_workflow:")]
            for url, body, code_word in cases:
                code, doc = b.call("POST", url, body)
                assert code == 400 and doc["msg"].startswith(code_word), (url, body.get("lock"), doc)
            assert "reserved for a future lock" in b.call("POST", LOCKED, {**BODY, "lock": "custom"})[1]["msg"]
            assert not b.comfy.graphs and not list(b.host.tasks_dir.iterdir())
            # the locked workflow keeps the unlocked one's validation
            for bad in ({**BODY, "ref_image_0": ""}, {**BODY, "ref_audio_0": PNG}, {**BODY, "audio_duration": 16}, {**BODY, "resolution": "1080p竖"}, {**BODY, "ref_image_2": PNG}):
                assert b.call("POST", LOCKED, bad)[0] == 400, bad
            assert b.call("POST", LOCKED, {**BODY, "ref_image_1": PNG, "resolution": "480p横"})[0] == 200
        finally:
            b.close()


def test_the_unlocked_workflow_never_asks_comfyui_for_its_nodes_and_its_take_is_what_it_was():
    with _t.tmpdir() as d:
        b = box(d, nodes=("addguide",))
        try:
            code, doc = b.call("POST", UNLOCKED, REF)
            assert code == 200 and b.wait(doc["data"]["task_id"])["status"] == "SUCCESS"
            assert not [g for g in b.comfy.gets if g.startswith("/object_info")], "a take of the unlocked workflow does not look for lock nodes"
            assert "lock" not in b.host.load(doc["data"]["task_id"])["plan"]
            g = b.comfy.graphs[-1]
            assert "21" not in g and g["11"]["inputs"]["conditioning"] == ["7", 0] and g["14"]["class_type"] == "VAEDecodeAudio" and g["12"]["inputs"]["latent_image"] == ["7", 1]
            assert adapter.WORKFLOWS["minimax_h3_image_audio_to_video"] == "ref" and adapter.WORKFLOWS[adapter.LOCKED_WORKFLOW] == "ref"
        finally:
            b.close()


def test_the_pack_helpers_read_the_folder_and_the_commit_without_running_git():
    with _t.tmpdir() as d:
        root = Path(d)
        assert adapter.pack_of("custom_nodes.comfyui-vrgamedevgirl") == PREFIX and adapter.pack_of("custom_nodes.Some.Pack_1") == "Some.Pack_1"
        for bad in (None, 3, "comfy_extras.nodes_minimax_h3", "custom_nodes.", "custom_nodes.../etc", "custom_nodes.a/b", "custom_nodes..x", "custom_nodes.-x"):
            assert adapter.pack_of(bad) is None, bad
        f = root / "pack"
        (f / ".git" / "refs" / "heads").mkdir(parents=True)
        assert adapter.git_head(f) is None
        (f / ".git" / "HEAD").write_text(SHA_A)
        assert adapter.git_head(f) == SHA_A                                                  # detached
        (f / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
        assert adapter.git_head(f) is None                                                   # the ref is not there yet
        (f / ".git" / "refs" / "heads" / "main").write_text(SHA_B + "\n")
        assert adapter.git_head(f) == SHA_B                                                  # a loose ref
        (f / ".git" / "refs" / "heads" / "main").unlink()
        (f / ".git" / "packed-refs").write_text(f"# pack-refs\n{SHA_A} refs/heads/other\n{SHA_B} refs/heads/main\n")
        assert adapter.git_head(f) == SHA_B                                                  # packed
        (f / ".git" / "HEAD").write_text("ref: refs/../../etc/passwd")
        assert adapter.git_head(f) is None                                                   # a ref that leaves .git is never followed
        (f / ".git" / "HEAD").write_text("garbage")
        assert adapter.git_head(f) is None
        (root / "file").write_text("x")
        assert adapter.git_head(root / "file") is None and adapter.git_head(root / "absent") is None
        assert adapter.pack_version(f) is None
        (f / "pyproject.toml").write_text('[tool.x]\nversion_info = 1\n[project]\nversion = "0.9.1"\n')
        assert adapter.pack_version(f) == "0.9.1"
        # the same answer as detect_models, which the installer uses (two copies of ten lines, kept equal by this test)
        from cloudgpu import detect_models as dm
        assert dm.git_head(f) == adapter.git_head(f)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_adapter(d, switch):
    """The real adapter process (main()) against a fake ComfyUI that has both lock nodes; `switch` is the value of CLOUDGPU_ALLOW_NODE_VRGDG, None for unset. Returns (/health, what it wrote to stderr)."""
    from test_adapter import FakeComfy
    comfy_root = Path(d) / "comfy"
    (comfy_root / "input").mkdir(parents=True, exist_ok=True)
    fc = FakeComfy(comfy_root)
    fc.nodes = {adapter.LOCK_NODES[k]: DEFS[k] for k in adapter.LOCK_NODES}
    port = free_port()
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "CLOUDGPU_TOKEN": TOKEN}
    if switch is not None:
        env[adapter.ENV_ALLOW_VRGDG] = switch
    proc = subprocess.Popen([sys.executable, str(Path(adapter.__file__)), "--port", str(port), "--comfy", fc.url, "--comfy-root", str(comfy_root), "--state", str(Path(d) / "state")],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        end = time.time() + _t.patient(20)
        while time.time() < end:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                    health = json.loads(r.read())
                    break
            except (urllib.error.URLError, OSError):
                time.sleep(0.1)
        else:
            raise AssertionError("the adapter did not start")
    finally:
        proc.terminate()
        err = proc.communicate(timeout=10)[1]
        fc.stop()
    return health, err


def test_only_an_explicit_one_in_the_environment_switches_the_third_party_lock_on():
    cases = ((None, False), ("", False), ("0", False), ("no", False), ("off", False), ("false", False), ("2", False), ("enabled", False),
             ("1", True), ("true", True), (" Yes ", True), ("ON", True))
    with _t.tmpdir() as d:
        for i, (value, on) in enumerate(cases):
            h, err = run_adapter(f"{d}/{i}", value)
            assert h["audio_lock_detail"]["audiodrive"]["allowed"] is on, (value, h["audio_lock_detail"]["audiodrive"])
            assert (h["audio_lock"] == ["addguide", "audiodrive"]) is on and (h["audio_lock"] == ["addguide"]) is (not on), (value, h["audio_lock"])
            assert ("ENABLED" in err) is on, (value, err)


if __name__ == "__main__":
    _t.main(globals())
