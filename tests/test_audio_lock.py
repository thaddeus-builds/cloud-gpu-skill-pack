"""cloudgpu/adapter.py, the audio-locked workflow (0.20.0), offline: a fake ComfyUI stands in for the card and answers /object_info like ComfyUI does.

minimax_h3_image_audio_locked anchors the take's own audio: the official lock (default, 0.21.0, "E3": core nodes only, the line text in, the three-line prompt built here), the core node
MiniMaxH3AddGuide, or the third-party VRGDG_MiniMaxH3AudioDrive (off unless the operator sets CLOUDGPU_ALLOW_NODE_VRGDG=1). What these tests hold: the official graph is the operator's tested E3
graph (modulo the box's file names and the take's fields); the audiodrive graph is the one measured on a 4090 (modulo model names); the
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
    "official": {"input": {"required": {"video_latent": ["LATENT"], "audio_latent": ["LATENT"]}}, "output": ["LATENT"], "python_module": "comfy_extras.nodes_lt"},
}
MORE_DEFS = {n: {"input": {"required": {i: ["X"] for i in ins}}, "python_module": "nodes"} for n, ins in adapter.LOCK_MORE["official"].items()}


def node_defs(keys):
    """What a fake ComfyUI's /object_info knows for these locks (the official lock needs five core nodes)."""
    out = {adapter.LOCK_NODES[k]: DEFS[k] for k in keys}
    if "official" in keys:
        out.update(MORE_DEFS)
    return out


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
E3_FILE = json.loads((Path(__file__).parent / "e3_lockaudio_api.json").read_text())        # the operator's tested graph, a data file (never run here)
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
    b.comfy.nodes = node_defs(nodes)
    return b


BODY = {**REF, "audio_duration": 5, "seed": 3, "lock": "addguide", "prompt": "[Shot 1] <Subject 1> (S1) looks into the lens and says, <d>[Chinese] 你好</d>\n<Audio 1>: fully_copy"}
LINE = "今天先说三件小事，第一件，把水杯放在手边。"
OBODY = {**REF, "audio_duration": 4.8, "seed": 3, "line": LINE}           # the official lock (the default): the line, no prompt, no lock key


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


def test_an_addguide_take_runs_with_the_core_node_and_the_adapter_adds_no_words_of_its_own():
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
                         "CLOUDGPU_ALLOW_NODE_VRGDG=1", "owner", "lock=official"):
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
        b = box(d, nodes=("official", "addguide", "audiodrive"), allow=True)
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
            assert h["audio_lock"] == ["official", "addguide", "audiodrive"], h
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
            code, doc = b.call("POST", LOCKED, OBODY)                     # the default lock, too
            assert code == 400 and doc["msg"].startswith("lock_node_missing:") and "LTXVConcatAVLatent" in doc["msg"] and "update ComfyUI" in doc["msg"], doc
            code, doc = b.call("POST", LOCKED, {**BODY, "lock": "audiodrive"})
            assert code == 400 and doc["msg"].startswith("lock_node_missing:") and "VRGDG_MiniMaxH3AudioDrive" in doc["msg"] and "comfyui-vrgamedevgirl" in doc["msg"], doc
            assert not b.comfy.graphs, "no fallback to the unlocked graph, to the other lock, or to anything"
            assert not list(b.host.tasks_dir.iterdir()) and not list((b.root / "input").iterdir()) and b.host.busy() == 0
            # the unlocked workflow is not affected by the lock nodes at all
            code, doc = b.call("POST", UNLOCKED, {k: v for k, v in BODY.items() if k != "lock"})
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
            assert h.lock_caps()["locks"]["addguide"]["usable"], "the cache says the node is there"
            b.comfy.nodes = {}                                          # ComfyUI restarted without it, inside the cache's 30 s
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
            cases = [(LOCKED, {**BODY, "lock": "custom"}, "lock_unknown:"), (LOCKED, {**BODY, "lock": "nope"}, "lock_unknown:"), (LOCKED, {**BODY, "lock": ""}, "lock_unknown:"),
                     (LOCKED, {**BODY, "lock": ["addguide"]}, "lock_unknown:"), (LOCKED, {**BODY, "lock": 1}, "lock_unknown:"),
                     (UNLOCKED, BODY, "lock_wrong_workflow:"), (UNLOCKED, {**OBODY, "lock": None}, "lock_wrong_workflow:"), (UNLOCKED, {**BODY, "lock": None, "line_language": "Chinese"}, "lock_wrong_workflow:"), (f"{API}/minimax_h3_lightx2v", {"lock": "addguide", "prompt": "x", "first_frame": PNG, "last_frame": PNG, "duration": 4}, "lock_wrong_workflow:"),
                     (f"{API}/minimax_h3_t2v", {"lock": "audiodrive", "prompt": "x", "duration": 4}, "lock_wrong_workflow:")]
            for url, body, code_word in cases:
                code, doc = b.call("POST", url, body)
                assert code == 400 and doc["msg"].startswith(code_word), (url, body.get("lock"), doc)
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
    fc.nodes = node_defs(adapter.LOCK_NODES)
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
            assert h["audio_lock"] == (["official", "addguide", "audiodrive"] if on else ["official", "addguide"]), (value, h["audio_lock"])
            assert ("ENABLED" in err) is on, (value, err)


def test_a_node_that_vanished_inside_the_cache_window_is_refused_at_the_door_and_stops_a_take_that_was_already_accepted():
    with _t.tmpdir() as d:
        b = box(d, nodes=("official", "addguide", "audiodrive"), allow=True)
        try:
            h = b.call("GET", "/health", token=None)[1]
            assert h["audio_lock"] == ["official", "addguide", "audiodrive"], "the cache is primed with every node"
            b.comfy.nodes = {}                                          # a ComfyUI restart without them, inside LOCK_CACHE_S
            for lock in ("official", "addguide", "audiodrive"):
                code, doc = b.call("POST", LOCKED, {**(OBODY if lock == "official" else BODY), "lock": lock})
                assert code == 400 and doc["msg"].startswith("lock_node_missing:") and adapter.LOCK_NODES[lock] in doc["msg"], (lock, doc)
            assert not b.comfy.graphs
            b.comfy.nodes = {"VRGDG_MiniMaxH3AudioDrive": DEFS["audiodrive"]}        # and one that came back is used at once
            code, doc = b.call("POST", LOCKED, {**BODY, "lock": "audiodrive"})
            assert code == 200 and b.wait(doc["data"]["task_id"])["status"] == "SUCCESS"
        finally:
            b.close()


def test_a_widened_bind_does_not_tell_anyone_the_commit_or_version_of_the_third_party_pack():
    with _t.tmpdir() as d:
        for bind, shown in (("127.0.0.1", True), ("0.0.0.0", False)):
            b = Box(d + "/" + bind, bind=bind, allow_vrgdg=True)
            try:
                b.comfy.nodes = node_defs(adapter.LOCK_NODES)
                pack = b.root / "custom_nodes" / PREFIX
                (pack / ".git").mkdir(parents=True)
                (pack / ".git" / "HEAD").write_text(SHA_A + "\n")
                (pack / "pyproject.toml").write_text('version = "1.2.3"\n')
                a = b.call("GET", "/health", token=None)[1]["audio_lock_detail"]["audiodrive"]
                assert (a["commit"] == SHA_A and a["version"] == "1.2.3") is shown and a["pack"] == PREFIX and a["usable"] is True, (bind, a)
                if not shown:
                    assert a["commit"] is None and a["version"] is None
                    assert b.host.lock_caps()["locks"]["audiodrive"]["commit"] == SHA_A, "only the answer is blanked, not what the adapter knows"
            finally:
                b.close()


def test_the_new_lock_code_names_no_model_and_no_step_count():
    import ast
    tree = ast.parse(Path(adapter.__file__).read_text())
    new = {"lock_official", "official_frames", "official_length", "clean_line", "official_prompt", "lock_addguide", "lock_audiodrive", "plan_lock", "lock_words", "lock_entry", "node_inputs", "pack_of", "git_head", "pack_version",
           "unknown_locks", "lock_caps", "lock_problem"}

    def clean(node, what):
        for n in ast.walk(node):
            assert not (isinstance(n, ast.Name) and n.id == "MODELS"), f"{what} reads MODELS"
            assert not (isinstance(n, ast.Constant) and isinstance(n.value, str) and ("safetensors" in n.value or "steps" in n.value)), f"{what} names a model file or the steps"

    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in new:
            seen.add(node.name)
            clean(node, node.name)
    assert seen == new, new - seen
    graph_fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "graph")
    block = [n for n in graph_fn.body if isinstance(n, ast.If) and "lock" in ast.dump(n.test)]
    assert len(block) == 1, "graph() has one lock block"
    clean(block[0], "graph()'s lock block")


# ---- the official lock, "E3" (0.21.0) ----------------------------------------------------------------------------------

E3_LINE = E3_FILE["8"]["inputs"]["prompt"].split("[Chinese] ")[1].split("</d>")[0]


def e3_build(**kw):
    args = dict(w=768, h=1344, seconds=4.5, image="h3f_ref.png", audio="h3f_drive.wav", prompt=adapter.official_prompt(E3_LINE), seed=973438621, prefix="h3lock/E3-1")
    return adapter.graph("ref", **{**args, **kw}, lock="official")


def test_the_official_graph_is_the_operators_tested_e3_graph_node_for_node():
    with Models(PRESET_4090):
        assert e3_build() == E3_FILE, "the file is the graph that ran on a 4090 (captions 1 of 9, audio correlation 0.999 to 1.000)"
    assert E3_FILE["8"]["inputs"]["length"] == 124 and E3_FILE["19"]["inputs"]["length"] == 108 and E3_FILE["14"]["inputs"]["noise_seed"] == 973438621
    # on another image the file names, the steps and the seed come from the box, and nothing else changes
    names = {"unet_ref": "S/unet.safetensors", "lora_ref": "S/lora.safetensors", "steps_ref": 6, "clip": "S/clip.safetensors", "vae_video": "S/vv.safetensors", "vae_audio": "S/va.safetensors"}
    with Models(names):
        g = e3_build(seed=5)
    assert (g["1"]["inputs"]["unet_name"], g["2"]["inputs"]["lora_name"], g["3"]["inputs"]["clip_name"], g["4"]["inputs"]["vae_name"], g["5"]["inputs"]["vae_name"]) == \
        ("S/unet.safetensors", "S/lora.safetensors", "S/clip.safetensors", "S/vv.safetensors", "S/va.safetensors")
    assert g["15"]["inputs"]["steps"] == 6 and g["14"]["inputs"]["noise_seed"] == 5
    want = json.loads(json.dumps(E3_FILE))
    want["1"]["inputs"]["unet_name"], want["2"]["inputs"]["lora_name"], want["3"]["inputs"]["clip_name"] = "S/unet.safetensors", "S/lora.safetensors", "S/clip.safetensors"
    want["4"]["inputs"]["vae_name"], want["5"]["inputs"]["vae_name"] = "S/vv.safetensors", "S/va.safetensors"
    want["15"]["inputs"]["steps"], want["14"]["inputs"]["noise_seed"] = 6, 5
    assert g == want


def test_the_official_graph_wires_the_audio_the_way_e3_does():
    with Models(PRESET_4090):
        g = e3_build()
    classes = [n["class_type"] for n in g.values()]
    assert "VAEDecodeAudio" not in classes and "MiniMaxH3AddGuide" not in classes and not any(c.startswith("VRGDG") for c in classes), "core nodes only, the audio is never decoded"
    ref = g["8"]["inputs"]
    assert ref["ref_audios.ref_audio_0"] == ["7", 0] and ref["audio_vae"] == ["5", 0] and g["9"]["inputs"] == {"audio": ["7", 0], "vae": ["5", 0]}, "one audio file feeds the latent and <Audio 1>"
    assert g["20"]["inputs"]["audio"] == ["7", 0], "the original audio goes into the video"
    assert g["10"]["inputs"]["value"] == 0.0 and g["11"]["inputs"] == {"samples": ["9", 0], "mask": ["10", 0]}, "the mask is on the audio latent only"
    assert g["12"]["inputs"] == {"video_latent": ["8", 1], "audio_latent": ["11", 0]} and g["17"]["inputs"]["latent_image"] == ["12", 0]
    assert g["16"]["inputs"]["conditioning"] == ["8", 0] and g["18"]["inputs"]["samples"] == ["17", 0] and g["19"]["inputs"]["image"] == ["18", 0]
    assert g["13"]["inputs"]["sampler_name"] == "res_multistep" and g["15"]["inputs"]["scheduler"] == "simple" and g["15"]["inputs"]["denoise"] == 1.0
    assert (ref["width"], ref["height"]) == (768, 1344) and g["21"]["inputs"]["filename_prefix"] == "h3lock/E3-1"


def test_the_frame_count_and_the_length_follow_the_e3_rule_with_a_floor_of_124():
    assert [adapter.official_frames(x) for x in (1, 4.5, 4.8, 5, 7, 15)] == [24, 108, 115, 120, 168, 360]
    assert [adapter.official_length(x) for x in (1, 3, 4.5, 4.8, 5, 5.17, 5.2, 7, 15)] == [124, 124, 124, 124, 124, 124, 141, 175, 362]
    for i in range(100, 1501):
        sec = i / 100
        frames, length = adapter.official_frames(sec), adapter.official_length(sec)
        assert length >= max(frames, adapter.OFFICIAL_MIN_LENGTH) and length % 17 == 5, sec
        assert length - frames < 17 or length == adapter.OFFICIAL_MIN_LENGTH, "no longer than the next 17k+5 above the frames (or the floor)"
        if adapter.frames_for(sec) >= adapter.OFFICIAL_MIN_LENGTH:
            assert length == adapter.frames_for(sec), "above the floor it is the rule every other graph uses"
    with Models(PRESET_4090):
        for sec, frames, length in ((4.8, 115, 124), (7, 168, 175), (15, 360, 362)):
            g = e3_build(seconds=sec)
            assert g["8"]["inputs"]["length"] == length and g["19"]["inputs"]["length"] == frames, sec


def test_the_official_prompt_is_the_three_e3_lines_and_the_line_is_the_only_thing_that_changes():
    p = adapter.official_prompt(E3_LINE)
    assert p == E3_FILE["8"]["inputs"]["prompt"]
    l1, l2, l3 = p.split("\n")
    assert l1 == f"[Shot 1] <Subject 1> (S1) whose appearance comes from <Picture 1> looks into the lens and says, <d>[Chinese] {E3_LINE}</d>"
    assert l2 == "<Audio 1>: fully_copy - the original audio of <Audio 1> is the complete final track; <Subject 1> (S1) lip-syncs exactly to it."
    assert l3 == "No captions, subtitles, logos or on-screen text appear."
    q = adapter.official_prompt("  Hello there.  ", "English")
    assert q.split("\n")[0].endswith("<d>[English] Hello there.</d>") and q.split("\n")[1:] == [l2, l3]
    for bad in (None, "", "   ", 5, "two\nlines", "tab\there", "a <b> c", "x</d>y", 'says "hi"', "\u201chi\u201d", "x" * (adapter.LINE_MAX_CHARS + 1)):
        try:
            adapter.official_prompt(bad)
        except adapter.Refused as e:
            assert str(e).startswith(("lock_line_required:", "lock_line_invalid:")), (bad, str(e))
        else:
            raise AssertionError(f"line {bad!r} was accepted")
    assert adapter.official_prompt("x" * adapter.LINE_MAX_CHARS)
    for bad in ("", "Chi\nnese", "<d>", "x", "a" * 30, 3, "Chinese]"):
        try:
            adapter.official_prompt("hi", bad)
        except adapter.Refused as e:
            assert str(e).startswith("lock_line_invalid:"), (bad, str(e))
        else:
            raise AssertionError(f"language {bad!r} was accepted")


def test_the_default_lock_is_official_and_its_request_carries_the_line_not_a_prompt():
    assert adapter.DEFAULT_LOCK == "official" and adapter.LOCKS == ("official", "addguide", "audiodrive")
    p = adapter.plan(adapter.LOCKED_WORKFLOW, OBODY)
    assert p["lock"] == "official" and p["seconds"] == 4.8 and p["prompt"] == adapter.official_prompt(LINE)
    assert adapter.plan(adapter.LOCKED_WORKFLOW, {**OBODY, "line_language": "English"})["prompt"] == adapter.official_prompt(LINE, "English")
    cases = [({**OBODY, "line": None}, "lock_line_required:"), ({k: v for k, v in OBODY.items() if k != "line"}, "lock_line_required:"), ({**OBODY, "line": "a\nb"}, "lock_line_invalid:"),
             ({**OBODY, "prompt": "my own words"}, "lock_prompt_unused:"), ({**OBODY, "prompt": ""}, "lock_prompt_unused:"), ({**OBODY, "ref_image_1": PNG}, "lock_one_picture:"),
             ({**BODY, "line": LINE}, "lock_line_unused:"), ({**BODY, "line_language": "Chinese"}, "lock_line_unused:"),
             ({**OBODY, "lock": "audiodrive", "prompt": "p"}, "lock_line_unused:")]
    for body, word in cases:
        try:
            adapter.plan(adapter.LOCKED_WORKFLOW, body)
        except adapter.Refused as e:
            assert str(e).startswith(word), (body, str(e))
        else:
            raise AssertionError(f"{body} was accepted")
    assert adapter.plan(adapter.LOCKED_WORKFLOW, {**OBODY, "lock": "official", "resolution": "480p横"})["w"] == 864
    assert adapter.plan(adapter.LOCKED_WORKFLOW, BODY)["prompt"] == BODY["prompt"], "the other locks take the request's own prompt as they did"


def test_an_official_take_runs_on_core_nodes_with_the_line_in_the_prompt_and_queues_nothing_else():
    with _t.tmpdir() as d:
        b = box(d, nodes=("official",))
        try:
            code, doc = b.call("POST", LOCKED, OBODY)
            assert code == 200 and doc["code"] == "Success", doc
            r = b.wait(doc["data"]["task_id"])
            assert r["status"] == "SUCCESS", r
            g = b.comfy.graphs[-1]
            assert g["8"]["inputs"]["prompt"] == adapter.official_prompt(LINE)
            assert (g["8"]["inputs"]["width"], g["8"]["inputs"]["height"], g["8"]["inputs"]["length"], g["19"]["inputs"]["length"]) == (768, 1344, 124, 115)
            assert g["14"]["inputs"]["noise_seed"] == 3 and g["1"]["inputs"]["unet_name"] == adapter.MODELS["unet_ref"] and g["15"]["inputs"]["steps"] == adapter.MODELS["steps_ref"]
            assert g["6"]["inputs"]["image"].endswith(".png") and g["7"]["inputs"]["audio"].endswith(".wav")
            assert [n["class_type"] for n in g.values()].count("SaveVideo") == 1 and len(b.comfy.graphs) == 1
            assert b.host.load(doc["data"]["task_id"])["plan"]["lock"] == "official"
            h = b.call("GET", "/health", token=None)[1]
            assert h["audio_lock"] == ["official"], h
            o = h["audio_lock_detail"]["official"]
            assert o["node"] == "LTXVConcatAVLatent" and o["usable"] is True and o["problem"] is None and o["module"] == "comfy_extras.nodes_lt"
            assert o["needs"] == ["LTXVConcatAVLatent", "VAEEncodeAudio", "SetLatentNoiseMask", "SolidMask", "ImageFromBatch"]
        finally:
            b.close()


def test_the_official_lock_is_unusable_and_says_which_node_when_any_one_of_its_nodes_is_missing_or_changed():
    with _t.tmpdir() as d:
        b = box(d, nodes=("official",))
        try:
            for node in adapter.LOCK_MORE["official"]:
                nodes = node_defs(("official",))
                del nodes[node]
                b.comfy.nodes = nodes
                code, doc = b.call("POST", LOCKED, OBODY)
                assert code == 400 and doc["msg"].startswith("lock_node_missing:") and f"needs the node {node}," in doc["msg"], (node, doc)
                h = b.call("GET", "/health", token=None)[1]
                assert h["audio_lock"] == [] and h["audio_lock_detail"]["official"]["present"] is True and h["audio_lock_detail"]["official"]["usable"] is False
            thin = node_defs(("official",))
            thin["SetLatentNoiseMask"] = {"input": {"required": {"samples": ["LATENT"]}}}
            b.comfy.nodes = thin
            code, doc = b.call("POST", LOCKED, OBODY)
            assert code == 400 and doc["msg"].startswith("lock_node_incompatible:") and "SetLatentNoiseMask" in doc["msg"] and "mask" in doc["msg"], doc
            thin = node_defs(("official",))
            thin["LTXVConcatAVLatent"] = {"input": {"required": {"video_latent": ["LATENT"]}}}
            b.comfy.nodes = thin
            code, doc = b.call("POST", LOCKED, OBODY)
            assert code == 400 and doc["msg"].startswith("lock_node_incompatible:") and "audio_latent" in doc["msg"], doc
            assert not b.comfy.graphs and not list(b.host.tasks_dir.iterdir()), "nothing was fetched, saved or sent; no other graph"
        finally:
            b.close()


def test_the_official_lock_needs_no_operator_switch_and_the_other_two_are_asked_for_by_name():
    with _t.tmpdir() as d:
        b = box(d, nodes=("official", "addguide", "audiodrive"), allow=False)
        try:
            h = b.call("GET", "/health", token=None)[1]
            assert h["audio_lock"] == ["official", "addguide"]
            assert b.call("POST", LOCKED, {**BODY, "lock": "audiodrive"})[1]["msg"].startswith("lock_not_allowed:")
            code, doc = b.call("POST", LOCKED, OBODY)
            assert code == 200 and b.wait(doc["data"]["task_id"])["status"] == "SUCCESS"
            code, doc = b.call("POST", LOCKED, {k: v for k, v in BODY.items() if k != "lock"})
            assert code == 400 and doc["msg"].startswith("lock_prompt_unused:"), "no lock key means the official lock: a prompt is not silently dropped"
            code, doc = b.call("POST", LOCKED, BODY)
            assert code == 200 and b.wait(doc["data"]["task_id"])["status"] == "SUCCESS" and b.comfy.graphs[-1]["21"]["class_type"] == "MiniMaxH3AddGuide"
        finally:
            b.close()


if __name__ == "__main__":
    _t.main(globals())
