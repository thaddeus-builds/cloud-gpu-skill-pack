"""The adapter that runs ON the GPU box: MiniMax-H3 in the box's ComfyUI, behind the HTTP contract of autodl.art's ComfyUI workflow API.

Runs ON the GPU box (Python 3.8+, standard library only), next to a ComfyUI that has the H3 weights. It takes the same two
calls a client of that API already makes, so a client needs a base URL and a token and nothing else:

    POST /api/v1/comfyui/comfyui_workflow/{workflow}        -> {"code":"Success","data":{"task_id":...}}
    GET  /api/v1/comfyui/comfyui_workflow/result/{task_id}  -> {"code":"Success","data":{"status":QUEUED|RUNNING|SUCCESS|FAILED,
                                                                "results":[{"url":...,"type":"video","file_type":"mp4"}]}}
    GET  /health[?nonce=X]                                  -> {"ok":bool,"comfy":bool,"busy":N,"worker":"alive"|"dead","card":"NVIDIA ..., 49140 MiB"|"unknown",
                                                                "memory_mib":49140|null,"vram_free_mib":N|null,"voice":bool,"t2v":bool,"audio_lock":[...],"audio_lock_detail":{...} [,"proof":HMAC-SHA256(token, X) hex]}   (no token; `ok` is false when the worker thread is
                                                                dead; N = takes accepted and not yet over, plus finished ones not yet collected until
                                                                GRACE_S after they finished: a caller shuts the box down only at 0. `card` is the GPU the
                                                                box runs on, so a caller can check the box it declared; `memory_mib` is the first GPU's
                                                                memory as a number, null when unknown; `vram_free_mib` is the free memory of ComfyUI's first device as ComfyUI's /system_stats
                                                                says it (at most COMFY_CACHE_S old), null when ComfyUI is down or does not say; `voice` is true when indextts2-v1 is installed; `t2v` is true when this adapter serves minimax_h3_t2v (an adapter without it does not say, and refuses the workflow). `audio_lock` lists the `lock`
                                                                values of minimax_h3_image_audio_locked this box serves RIGHT NOW (["addguide"], ["addguide","audiodrive"], or [] when ComfyUI has no usable lock node or
                                                                has not answered yet; an adapter older than 0.20.0 has no such key and refuses the workflow); `audio_lock_detail` says per lock whether its node
                                                                was found in ComfyUI's /object_info, where it comes from (module, pack, commit) and, for audiodrive, whether the operator switched it on. `proof` lets a client that holds the token tell
                                                                this adapter from another process holding the tunnel's port: it is answered only when the
                                                                bind address is loopback, for a nonce of 16 to 64 lowercase hex characters, at most
                                                                PROOF_PER_MIN a minute; on a widened bind there is no proof (it would be an offline
                                                                guessing oracle for the token) and a client then refuses an http base: reach such a box
                                                                by its https address)
    GET  /files/<random>.mp4|.wav                           -> the finished video or voice (the URL is the secret; valid FILE_TTL_S, after which the task
                                                                reads FAILED "result expired")

    POST /cloudgpu/line {"label"}   -> {"data":{"ticket":ID,"state":"go"|"waiting","ahead":N,"running":LABEL|null,"lease_s":60}}   the line of callers that use this box
    GET  /cloudgpu/line/ID          -> the same; asking renews the ticket's lease (404: its lease ran out, or the adapter restarted: join again)
    DELETE /cloudgpu/line/ID        -> leave (idempotent)
    POST /cloudgpu/free             -> {"data":{"freed":bool}}   ask ComfyUI to unload its models and free its GPU memory (token-checked, best effort; `freed` is false when a take is
                                                                accepted or running, or ComfyUI did not answer: nothing is done then, and a caller ignores the answer)
    GET  /cloudgpu/line             -> {"data":{"line":[{"state","label","age_s"}],"lease_s":60}}: who runs and who waits (no ticket ids)
                                                                (token-checked. A ticket is a place in an ordered line: "go" for the front one, so callers on DIFFERENT machines take the box one at a
                                                                time in the order they joined. A ticket not asked about for LEASE_S is dropped, so a dead caller never blocks the line. The line
                                                                orders the callers that use it; a caller that submits without a ticket still queues in the worker as before. While a ticket is live
                                                                the box counts as busy in /health, so it is not powered off or restarted under a caller that waits.)

    GET  /cloudgpu/tasks[?since=EPOCH]  -> {"data":{"tasks":[{"workflow","status","created","started","done"}],"now":EPOCH,"ttl_s":86400}}   what `python3 -m cloudgpu util` reads (0.12.0)
                                                                (token-checked, read only. The task records of the last TASK_TTL_S, newest last; `done` is when the take finished (the file's
                                                                modification time for one the adapter failed itself), null while it waits or runs; no task id and nothing a take carried.)

    Authorization: <CLOUDGPU_TOKEN>    the raw token, as autodl's own header (no "Bearer"); compared in constant time.

Workflows (the names autodl publishes, so a project's avatar.workflow keeps working):
    minimax_h3_image_audio_to_video   ref_image_0 (+ ref_image_1 .. ref_image_8) + ref_audio_0 + audio_duration + resolution
                                      more pictures are <Picture 2>, <Picture 3> .. in the prompt (no gaps in the numbering); with no prompt given the default is used
                                      FL2VA-free path: the reference-to-video model (ref2va) with the take's audio as <Audio 1>
    minimax_h3_image_audio_locked     the same fields as minimax_h3_image_audio_to_video + lock: "addguide" (default) | "audiodrive"; the audio on the lips is the take's audio (0.20.0, below)
    minimax_h3_lightx2v               prompt + first_frame + last_frame + duration + resolution (+ seed)
                                      Mode A: the FL2VA model speaks the prompt's line itself
    minimax_h3_t2v                    prompt + duration + resolution (+ seed): text to video, no picture and no audio in; the FL2VA model (turbo lora) makes the picture and the sound
    resolution: "480p竖" 480x864 | "768p竖" 768x1344 | "480p横" 864x480 | "768p横" 1344x768 (landscape, 0.15.0). 1080p is refused: it was never run on this card.
    indextts2-v1                      prompt_text + prompt_simple (the voice sample) + emo_control_method (+ emo_happy .. emo_calm, emo_ref_audio, emo_alpha, emo_random)
                                      the voice: IndexTTS2 run by tts_worker.py in its own Python environment (--tts-python, --tts-dir); the result is a wav
                                      (type "audio", file_type "wav"). Without a voice installed the box answers 400 and /health says "voice": false.
Inputs are http(s) URLs or data: URIs. A URL that points at this machine or a private address is refused (this box is
not a proxy into its own network). The graphs are ComfyUI's own open templates (video_minimax_h3_r2v, video_minimax_h3_i2v),
rebuilt as API graphs here (text to video is the i2v one with both frame inputs left out); the box's other H3 workflows carry their author's lock and are not used.

The audio-locked workflow (0.20.0). The unlocked audio-driven graph lets the model REGENERATE speech from the audio it is given (measured on a 4090: sample correlation with the given audio
0.23 to 0.39). minimax_h3_image_audio_locked builds the same graph and adds one node that anchors the take's own audio, chosen by the request's `lock`:
    addguide    (default) the OFFICIAL ComfyUI node MiniMaxH3AddGuide (comfy_extras.nodes_minimax_h3): the audio is encoded as a guide on the conditioning; the audio of the video is decoded from the sampled latent as before
    audiodrive  the THIRD-PARTY node VRGDG_MiniMaxH3AudioDrive (a custom-node pack under a licence that is NOT open: see docs/cloud-gpu.md). OFF unless the box's operator sets CLOUDGPU_ALLOW_NODE_VRGDG=1
    custom      reserved for a future graph of official nodes only; not implemented, always refused
The adapter never falls back from a lock to the unlocked graph or to another lock: a lock whose node ComfyUI does not have, or which the operator has not allowed, is a coded error (400 on the
take; a take accepted while ComfyUI was still starting and then found without the node is FAILED with the same words), the words starting `lock_unknown`, `lock_not_implemented`,
`lock_wrong_workflow`, `lock_not_allowed`, `lock_node_missing`, `lock_node_incompatible` or `lock_state_unknown`. The nodes are found by name in ComfyUI's /object_info at start and again when asked
(at most LOCK_CACHE_S old, and fresh for each take of a lock); no model file name is written in the lock code: the models are the box's MODELS as for every graph.

The H3 models live in the ComfyUI process, not in this adapter (standard library only, no torch), and ComfyUI keeps them on the card after a take. So after a take that leaves
nothing accepted or queued, the adapter asks ComfyUI to free them (POST /free, best effort: a failure never fails a take). The next take loads its models again; FREE_AFTER_TAKE
(--free-after-take 0|1, CLOUDGPU_FREE_AFTER_TAKE, default 1) turns this off for a box that only ever makes video.

One GPU runs one take at a time (ComfyUI queues), so tasks, a voice included, are served in order by one worker. A task found QUEUED or RUNNING
after a restart is FAILED ("restarted"), never silently re-run, its input files are deleted, and ComfyUI is told to interrupt and clear its queue so the
orphan job does not run on (and cost GPU time) for nobody. Task files older than TASK_TTL_S are deleted at start and every MAINTAIN_EVERY_S.

Binds 127.0.0.1 (the SSH tunnel needs only that); CLOUDGPU_BIND or --bind widens it (0.0.0.0 for a vendor's public port: then ComfyUI must not be on one).
A URL input is resolved once, every address must be global (not private, loopback, link-local, 100.64.0.0/10, reserved), and the connection goes to that address.

    CLOUDGPU_TOKEN=... python3 adapter.py [--port 6009] [--bind 127.0.0.1] [--comfy http://127.0.0.1:6006] [--comfy-root ~/ComfyUI] [--state ~/h3host] [--models FILE] [--tts-python PY --tts-dir DIR]
"""

from __future__ import annotations

import argparse
import base64
import errno
import hashlib
import hmac
import http.client
import ipaddress
import json
import os
import queue
import random
import re
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

API = "/api/v1/comfyui/comfyui_workflow"
RESOLUTIONS = {"480p竖": (480, 864), "768p竖": (768, 1344), "480p横": (864, 480), "768p横": (1344, 768)}
MAX_SECONDS = 15
MAX_BODY = 96 * 1024 * 1024
MAX_INPUT = 64 * 1024 * 1024
TASK_TIMEOUT_S = 1800
COMFY_WAIT_S = 600                 # a cold box: ComfyUI is still starting when the first task arrives
FILE_TTL_S = 3600                  # a task whose video is older than this is FAILED "result expired" (its record lives TASK_TTL_S)
FILES = "/cloudgpu/files"           # the files a consumer sends once and then names by their sha256 (0.10.0)
FILES_TTL_S = 7 * 24 * 3600        # a stored file nobody used for this long is deleted
FILES_MAX = 512 * 1024 * 1024      # and the oldest go first when the store is over this
SHA_REF = re.compile(r"sha256:([0-9a-f]{64})")
FILE_KINDS = {".png": "image", ".jpg": "image", ".webp": "image", ".wav": "audio", ".mp3": "audio"}
TASK_TTL_S = 24 * 3600             # a task file (and an input left behind) older than this is deleted
MAINTAIN_EVERY_S = 600
GRACE_S = 300                      # a finished take counts as busy this long unless it was collected
FFMPEG_TIMEOUT_S = 600
SOCKET_TIMEOUT_S = 60              # a client that stalls this long loses its connection and its handler thread
COMFY_CACHE_S, CARD_RETRY_S = 2, 60
FREE_AFTER_TAKE = True             # after a take that leaves nothing waiting, ComfyUI is asked to unload its models (the next take loads them again)
PROOF_PER_MIN = 30                 # answers with a `proof` per minute: a client asks about once a minute
NONCE = re.compile(r"[0-9a-f]{16,64}")
FPS = 24
# File names are relative to ComfyUI's models folders and differ between the images the box can come from (the PRO6000 image keeps them in
# MiniMax/, the 4090 one in minimax/ and the vae/clip at the top): --models FILE.json overrides any key. The default is the PRO6000 image's.
MODELS = {
    "unet_ref": "MiniMax/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "unet_fl": "MiniMax/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "lora_ref": "MiniMax/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",   # the fl2v lora: that image has no ref2v turbo one
    "steps_ref": 4,
    "lora_fl": "MiniMax/minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors",                # the i2v template's own pair: 8-step lora run at 6
    "steps_fl": 6,
    "clip": "MiniMax/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",                        # nvfp4: Blackwell cards only; others use the int8 file
    "vae_video": "MiniMax/minimax_h3_video_vae_fp16.safetensors",
    "vae_audio": "MiniMax/minimax_h3_audio_vae_fp32.safetensors",
}
LINE = "/cloudgpu/line"
TASKS = "/cloudgpu/tasks"
LEASE_S = 60                       # a ticket nobody asked about for this long is dropped; a caller asks about every LEASE_S / 3
MAX_LINE = 64
LABEL_MAX = 80
TTS_WORKFLOW = "indextts2-v1"
LOCKED_WORKFLOW = "minimax_h3_image_audio_locked"           # the audio-driven take whose audio is anchored to the given one (0.20.0); the unlocked one stays as it was
WORKFLOWS = {"minimax_h3_image_audio_to_video": "ref", LOCKED_WORKFLOW: "ref", "minimax_h3_lightx2v": "fl", "minimax_h3_t2v": "t2v", TTS_WORKFLOW: "tts"}
LOCKS = ("addguide", "audiodrive", "custom")                # the values of `lock`; "addguide" is the default
DEFAULT_LOCK = "addguide"
CUSTOM_WORDS = "lock_not_implemented: lock=custom is reserved for a future lock made of official ComfyUI nodes only; it is not served yet (use addguide)"
LOCK_NODES = {"addguide": "MiniMaxH3AddGuide",                      # the official node: ComfyUI's comfy_extras.nodes_minimax_h3
              "audiodrive": "VRGDG_MiniMaxH3AudioDrive"}            # third party; "custom" has no node (an extension point: LOCK_BUILDERS, below)
# The inputs each graph below wires, by the names of the node definitions: a node whose /object_info lacks one is "present but incompatible", never used.
LOCK_INPUTS = {"addguide": ("positive", "latent", "frame_idx", "vae", "audio_vae", "audio"),
               "audiodrive": ("av_latent", "source_audio", "audio_vae")}
ENV_ALLOW_VRGDG = "CLOUDGPU_ALLOW_NODE_VRGDG"               # the operator's switch for the third-party lock: off unless set to 1/true/yes/on
VRGDG_LICENCE = "VRGameDevGirl Community and Commercial License v1.0"
VRGDG_TAG = "third-party, licence-restricted"
LOCK_CACHE_S = 30                                           # how long a read of ComfyUI's node list is trusted (a node appears only when ComfyUI restarts)
MAX_EXTRA_IMAGES = 8                # ref_image_1 .. ref_image_8 beside ref_image_0 (the hosted workflow takes up to 9 pictures)
TTS_MAX_CHARS = 2000
TTS_TIMEOUT_S = 900                # the model is loaded for each voice (its VRAM is given back after), so this includes the load
TTS_MAGIC = b"RIFF"
EMOTIONS = ("happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm")      # IndexTTS2's vector order
EMO_METHODS = {"与音色参考音频相同": "same", "使用情感参考音频": "audio", "使用情感向量控制": "vector"}     # the hosted workflow's words; the text-described one is not served
EMO_TEXT_METHOD = "使用情感描述文本控制"
CONTENT_TYPES = {"mp4": "video/mp4", "wav": "audio/wav"}
DEFAULT_REF_PROMPT = ("The person from <Picture 1> faces the camera and talks naturally to the viewer, soft light, vertical phone "
                      "video. Use <Picture 1> as reference frame and <Audio 1> exactly as it is. No captions, subtitles, titles, "
                      "logos, watermarks or any text appears anywhere in the frame; the picture stays as clean as <Picture 1>.")


class Refused(Exception):
    """A request the box will not run: answered 400 with this text, nothing queued."""


class LockUnavailable(RuntimeError):
    """A lock the box cannot serve (reserved, not allowed, its node missing): a coded message (`lock_...: words`). Raised where the take is already accepted (at run time);
    at the door the same words are a Refused."""


def frames_for(seconds: float) -> int:
    """The templates' own rule: at least 5 frames and 5 mod 17."""
    n = max(5, round(seconds * FPS))
    return n + (5 - n % 17) % 17


# ---- ComfyUI graph -------------------------------------------------------------------------------------------------

def graph(kind: str, *, w: int, h: int, seconds: float, image: str | None, audio: str | None, prompt: str, seed: int, prefix: str,
          extra_images: tuple | list = (), lock: str | None = None) -> dict:
    mk = "fl" if kind == "t2v" else kind            # text to video runs the FL2VA model and its lora, with no frames given
    lora, steps = MODELS["lora_" + mk], MODELS["steps_" + mk]
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": MODELS["unet_" + mk], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": MODELS["clip"], "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": MODELS["vae_video"]}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": MODELS["vae_audio"]}},
        "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "9": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "10": {"class_type": "BasicScheduler", "inputs": {"model": ["20", 0], "scheduler": "simple", "steps": steps, "denoise": 1.0}},
        "11": {"class_type": "BasicGuider", "inputs": {"model": ["20", 0], "conditioning": ["7", 0]}},
        "12": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["9", 0], "guider": ["11", 0], "sampler": ["8", 0],
                                                                  "sigmas": ["10", 0], "latent_image": ["7", 1]}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
        "14": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["12", 0], "vae": ["4", 0]}},
        "15": {"class_type": "CreateVideo", "inputs": {"images": ["13", 0], "audio": ["14", 0], "fps": FPS}},
        "16": {"class_type": "SaveVideo", "inputs": {"video": ["15", 0], "filename_prefix": prefix, "format": "auto", "codec": "auto"}},
        "20": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": lora, "strength_model": 1.0}},
    }
    n = frames_for(seconds)
    if kind != "t2v":
        g["5"] = {"class_type": "LoadImage", "inputs": {"image": image}}
    if kind == "ref":
        g["6"] = {"class_type": "LoadAudio", "inputs": {"audio": audio}}
        g["7"] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["2", 0], "vae": ["3", 0], "audio_vae": ["4", 0], "prompt": prompt, "width": w, "height": h, "length": n,
            "ref_image_size": "match", "ref_images.ref_image_0": ["5", 0], "ref_audios.ref_audio_0": ["6", 0]}}
        for i, name in enumerate(extra_images, 1):          # <Picture 2> is ref_image_1, and so on
            g[f"5r{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
            g["7"]["inputs"][f"ref_images.ref_image_{i}"] = [f"5r{i}", 0]
    elif kind == "t2v":
        g["7"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["2", 0], "vae": ["3", 0], "prompt": prompt, "width": w, "height": h, "length": n}}
    else:
        g["7"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["2", 0], "vae": ["3", 0], "prompt": prompt, "width": w, "height": h, "length": n,
            "first_frame": ["5", 0], "last_frame": ["5b", 0]}}
        g["5b"] = {"class_type": "LoadImage", "inputs": {"image": image}}
    if lock is not None:                    # only the audio-locked workflow passes a lock; every other graph is exactly what it was
        if kind != "ref":
            raise LockUnavailable(f"lock_wrong_workflow: a lock anchors the audio of an audio-driven take; kind {kind!r} has none")
        builder = LOCK_BUILDERS.get(lock)
        if builder is None:
            raise LockUnavailable(f"lock_unknown: lock {lock!r} is not one of {', '.join(LOCKS)}")
        builder(g)                          # no else, no fallback: a lock that cannot be built stops the take
    return g


# ---- the locks of minimax_h3_image_audio_locked (0.20.0) ----------------------------------------------------------------
# Each builder changes the finished audio-driven graph `g` in place and nothing else. The graph has the nodes 7 (MiniMaxH3ReferenceToVideo: conditioning 0, audio-video latent 1),
# 6 (LoadAudio, also wired into the reference node as <Audio 1>), 4 (the audio VAE), 3 (the video VAE), 11 (the guider), 12 (the sampler), 14 (VAEDecodeAudio) and 15 (CreateVideo).

def lock_addguide(g: dict) -> None:
    """The official node MiniMaxH3AddGuide (ComfyUI core, comfy_extras.nodes_minimax_h3): its inputs are positive and latent (required), frame_idx (0: from the first frame) and,
    optional, vae, audio_vae, image, audio; its one output is the conditioning with the guide added. The given audio is the guide (no image: the picture is the reference node's). The
    sampler keeps the reference node's latent and the guider takes the new conditioning, so the audio of the video is still decoded from the sampled latent (node 14, untouched)."""
    g["21"] = {"class_type": LOCK_NODES["addguide"], "inputs": {"positive": ["7", 0], "latent": ["7", 1], "frame_idx": 0,
                                                              "vae": ["3", 0], "audio_vae": ["4", 0], "audio": ["6", 0]}}
    g["11"]["inputs"]["conditioning"] = ["21", 0]


def lock_audiodrive(g: dict) -> None:
    """The third-party node VRGDG_MiniMaxH3AudioDrive, referred to by name in a graph (behind the capability check and the operator's switch; nothing of its code is here). Wiring as the operator measured it on a 4090: the node takes the reference node's audio-video latent, the same audio and the audio VAE; its first output is the latent the sampler
    starts from and its second is the audio of the video, so the audio is not decoded from the sampled latent (node 14 is gone)."""
    g["21"] = {"class_type": LOCK_NODES["audiodrive"], "inputs": {"av_latent": ["7", 1], "source_audio": ["6", 0], "audio_vae": ["4", 0]}}
    g["12"]["inputs"]["latent_image"] = ["21", 0]
    del g["14"]
    g["15"]["inputs"]["audio"] = ["21", 1]


def lock_custom(g: dict) -> None:
    """EXTENSION POINT, not implemented: a future lock built only of official nodes (for example VAEEncodeAudio and SetLatentNoiseMask) replaces this body and is then also given an entry
    in LOCK_NODES / LOCK_INPUTS (the nodes it needs, so /health and the missing-node error name them) and cleared in plan(). Until then `lock: custom` is refused at the door and, should a graph
    ever be asked for, here: it must not build the unlocked graph."""
    raise LockUnavailable(CUSTOM_WORDS)


LOCK_BUILDERS = {"addguide": lock_addguide, "audiodrive": lock_audiodrive, "custom": lock_custom}


# ---- inputs ----------------------------------------------------------------------------------------------------------

def resolve_public(host: str, port: int) -> str | None:
    """The address to connect to when EVERY address `host` resolves to is a global one (ip.is_global: not private, loopback, link-local, shared 100.64.0.0/10 which
    holds some clouds' metadata address, documentation or reserved), else None. The caller connects to the returned address, so a name cannot answer differently later."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError):
        return None
    pick = None
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if ip.version == 6 and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if not ip.is_global or ip.is_multicast:
            return None
        pick = pick or str(ip)
    return pick


def public_host(host: str) -> bool:
    """False when the name resolves to this machine or a private network: the box must not fetch for a caller from its own side."""
    return resolve_public(host, 443) is not None


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def _pinned(conn_class, ip: str):
    """`conn_class` that connects to `ip` whatever name it was given (TLS still checks the name)."""
    class Pinned(conn_class):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._create_connection = lambda address, timeout=None, source_address=None: socket.create_connection((ip, address[1]), timeout, source_address)
    return Pinned


class PinnedHTTP(urllib.request.HTTPHandler):
    def __init__(self, ip: str):
        super().__init__()
        self.ip = ip

    def http_open(self, req):
        return self.do_open(_pinned(http.client.HTTPConnection, self.ip), req)


class PinnedHTTPS(urllib.request.HTTPSHandler):
    def __init__(self, ip: str):
        super().__init__()
        self.ip = ip

    def https_open(self, req):
        return self.do_open(_pinned(http.client.HTTPSConnection, self.ip), req, context=self._context)


def fetch(ref: str, into: Path, kinds: tuple, what: str, files: Path | None = None) -> Path:
    """Write one input (a data: URI, an http(s) URL or `sha256:<hex>` of a file stored with PUT /cloudgpu/files) into `into` and return the path; `kinds` are the accepted MIME prefixes."""
    if not isinstance(ref, str) or not ref:
        raise Refused(f"{what} is missing")
    m = SHA_REF.fullmatch(ref)
    if m:
        stored = stored_file(files, m.group(1)) if files is not None else None
        if stored is None:
            raise Refused(f"file_unknown: {what}: sha256:{m.group(1)} is not stored on this box (PUT {FILES}/{m.group(1)} first)")
        if FILE_KINDS.get(stored.suffix) not in tuple(k.rstrip("/") for k in kinds):
            raise Refused(f"{what}: the stored file is a {FILE_KINDS.get(stored.suffix, '?')}, expected {'/'.join(kinds)}")
        path = into / f"cloudgpu_{uuid.uuid4().hex[:16]}{stored.suffix}"
        try:
            shutil.copyfile(stored, path)                  # a copy: the take's inputs are deleted when it is over, the stored file stays
        except OSError:
            path.unlink(missing_ok=True)
            raise
        os.utime(stored, None)                             # used now: it is not old
        return path
    if ref.startswith("data:"):
        head, _, b64 = ref.partition(",")
        mime = head[5:].split(";")[0]
        if ";base64" not in head or not mime.startswith(kinds):
            raise Refused(f"{what}: a data: URI of type {mime or '?'} is not accepted here ({'/'.join(kinds)})")
        try:
            blob = base64.b64decode(b64, validate=False)
        except ValueError:
            raise Refused(f"{what}: the data: URI is not base64") from None
    else:
        try:
            u = urllib.parse.urlparse(ref)
            port = u.port or (443 if u.scheme == "https" else 80)
        except ValueError:
            raise Refused(f"{what} must be an http(s) URL or a data: URI") from None
        if u.scheme not in ("http", "https") or not u.hostname:
            raise Refused(f"{what} must be an http(s) URL or a data: URI")
        ip = resolve_public(u.hostname, port)
        if ip is None:
            raise Refused(f"{what}: this host is not fetched from here")
        # no proxy (the pinned address must be the one used), no redirect, and every failure is the same words: this box is not a port scanner for its caller
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect, PinnedHTTP(ip), PinnedHTTPS(ip))
        try:
            with opener.open(urllib.request.Request(ref, headers={"User-Agent": "cloudgpu"}), timeout=60) as r:
                mime = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
                blob = r.read(MAX_INPUT + 1)
        except (urllib.error.URLError, OSError, http.client.HTTPException, ValueError):
            raise Refused(f"{what}: could not be fetched") from None
        if not mime.startswith(kinds + ("application/octet-stream", "binary/octet-stream")):
            raise Refused(f"{what}: the URL says {mime or 'no content type'}, expected {'/'.join(kinds)}")
    if not blob or len(blob) > MAX_INPUT:
        raise Refused(f"{what} is empty or over {MAX_INPUT // 2**20} MB")
    ext = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "audio/wav": ".wav", "audio/mpeg": ".mp3",
           "audio/x-wav": ".wav", "audio/mp3": ".mp3"}.get(mime, ".png" if kinds[0] == "image" else ".wav")
    path = into / f"cloudgpu_{uuid.uuid4().hex[:16]}{ext}"
    try:
        path.write_bytes(blob)
    except OSError:
        path.unlink(missing_ok=True)                 # a half-written input on a full disk
        raise
    return path


def stored_file(files: Path, sha: str):
    """The stored file of this sha256, or None."""
    for ext in FILE_KINDS:
        p = files / f"{sha}{ext}"
        if p.is_file():
            return p
    return None


def plan(workflow: str, body: dict) -> dict:
    """Validate a request body (no side effects): the take's parameters, or Refused."""
    kind = WORKFLOWS.get(workflow)
    if kind is None:
        raise Refused(f"unknown workflow {workflow!r}; this box serves {', '.join(sorted(WORKFLOWS))}")
    lock = plan_lock(workflow, body)          # first: a lock sent to a workflow that does not anchor the audio is refused whatever else is in the body
    if kind == "tts":
        return plan_voice(body)
    res = body.get("resolution", "768p竖")
    if res not in RESOLUTIONS:
        raise Refused(f"resolution {res!r} is not served here ({', '.join(RESOLUTIONS)})")
    w, h = RESOLUTIONS[res]
    try:
        seed = int(body["seed"]) if body.get("seed") is not None else random.randrange(2**31)
        if kind == "ref":
            secs = float(body.get("audio_duration", 0))
        else:
            secs = float(body.get("duration", 0))
    except (TypeError, ValueError):
        raise Refused("seed and the duration must be numbers") from None
    if not 1 <= secs <= MAX_SECONDS:
        raise Refused(f"the duration must be 1 to {MAX_SECONDS} seconds")
    if kind == "ref":
        for k in ("ref_image_0", "ref_audio_0"):
            if not body.get(k):
                raise Refused(f"{k} is required")
        prompt = body.get("prompt") or (DEFAULT_REF_PROMPT if h > w else DEFAULT_REF_PROMPT.replace("vertical phone video", "wide screen video"))
        extra = [k for k in body if isinstance(k, str) and k.startswith("ref_image_") and k != "ref_image_0"]
        want = [f"ref_image_{i}" for i in range(1, len(extra) + 1)]
        if sorted(extra) != want or len(extra) > MAX_EXTRA_IMAGES:
            raise Refused(f"the extra pictures are ref_image_1 .. ref_image_{MAX_EXTRA_IMAGES}, in order with no gap")
        if not all(body[k] for k in extra):
            raise Refused("an extra reference picture is empty")
    else:
        for k in ("prompt",) if kind == "t2v" else ("prompt", "first_frame", "last_frame"):
            if not body.get(k):
                raise Refused(f"{k} is required")
        prompt = body["prompt"]
    p = {"kind": kind, "w": w, "h": h, "seconds": secs, "seed": seed, "prompt": str(prompt)}
    if kind == "ref" and extra:
        p["extra_images"] = len(extra)
    if lock is not None:
        p["lock"] = lock
    return p


def plan_lock(workflow: str, body: dict) -> str | None:
    """The `lock` of a take (no side effects): None for every workflow but the audio-locked one, which defaults to "addguide". A coded Refused (`lock_...: words`) for a lock that is not
    one of LOCKS, for "custom" (reserved, not implemented), and for a `lock` sent to a workflow that does not lock: the caller asked for an anchored audio and would otherwise get an unanchored one."""
    lock = body.get("lock")
    if workflow != LOCKED_WORKFLOW:
        if lock is not None:
            raise Refused(f"lock_wrong_workflow: lock belongs to {LOCKED_WORKFLOW}; {workflow} does not anchor the audio, so the take was not queued")
        return None
    lock = DEFAULT_LOCK if lock is None else lock
    if not isinstance(lock, str) or lock not in LOCKS:
        raise Refused(f"lock_unknown: lock {lock!r} is not served; use one of {', '.join(LOCKS)}")
    if lock == "custom":
        raise Refused(CUSTOM_WORDS)
    return lock


def _unit(v, what: str) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise Refused(f"{what} must be a number from 0 to 1") from None
    if not 0.0 <= f <= 1.0:            # also False for nan
        raise Refused(f"{what} must be a number from 0 to 1")
    return f


def plan_voice(body: dict) -> dict:
    """The voice's parameters (no side effects): the line, the emotion mode and its numbers, or Refused. The call shape is the hosted workflow's."""
    text = body.get("prompt_text")
    if not isinstance(text, str) or not text.strip():
        raise Refused("prompt_text is required (the line to speak)")
    if len(text) > TTS_MAX_CHARS:
        raise Refused(f"prompt_text is over {TTS_MAX_CHARS} characters: split the line")
    if not body.get("prompt_simple"):
        raise Refused("prompt_simple is required (the voice sample to clone)")
    method = body.get("emo_control_method") or "与音色参考音频相同"
    if method == EMO_TEXT_METHOD:
        raise Refused("an emotion described by text is not served on this box; use the emotion vector (emo_happy ... emo_calm) or emo_ref_audio")
    mode = EMO_METHODS.get(method)
    if mode is None:
        raise Refused(f"emo_control_method {method!r} is not served here ({', '.join(EMO_METHODS)})")
    p = {"kind": "tts", "text": text, "emo": mode, "vector": None, "alpha": None, "random": bool(body.get("emo_random", False))}
    if body.get("emo_alpha") is not None:
        p["alpha"] = _unit(body["emo_alpha"], "emo_alpha")
    if mode == "vector":
        p["vector"] = [_unit(body.get("emo_" + e, 0), "emo_" + e) for e in EMOTIONS]
        if not any(p["vector"]):
            raise Refused("emo_control_method asks for the emotion vector but every emo_<emotion> is 0")
    if mode == "audio" and not body.get("emo_ref_audio"):
        raise Refused("emo_ref_audio is required for the emotion-from-audio method")
    return p


class Line:
    """The ordered line of callers for this box (in memory: an adapter restart already fails every take, and a caller whose ticket is forgotten joins again at the back).
    The ticket at the front is "go"; the others wait. Every call drops the tickets whose lease ran out, so the next one in line moves up even when its caller has not asked yet."""

    def __init__(self, clock=time.monotonic):
        self.clock, self.lock, self.tickets = clock, threading.Lock(), []

    def _advance(self, now: float) -> None:
        self.tickets = [t for t in self.tickets if t["expires"] > now]
        if self.tickets:
            self.tickets[0]["state"] = "go"            # the front one is always the one that goes: a ticket behind it was never granted

    def _view(self, t: dict) -> dict:
        i = self.tickets.index(t)
        return {"ticket": t["id"], "state": t["state"], "ahead": i, "running": self.tickets[0]["label"] if i else None, "lease_s": LEASE_S}

    def join(self, label) -> dict:
        label = "".join(c for c in str(label or "") if c.isprintable())[:LABEL_MAX]
        now = self.clock()
        with self.lock:
            self._advance(now)
            if len(self.tickets) >= MAX_LINE:
                raise Refused("the line is full")
            t = {"id": secrets.token_hex(8), "label": label, "state": "waiting", "since": now, "expires": now + LEASE_S}
            self.tickets.append(t)
            self._advance(now)
            return self._view(t)

    def poll(self, tid: str):
        """The ticket's place, with its lease renewed; None when there is no such ticket (lease over, left, or an adapter restarted)."""
        now = self.clock()
        with self.lock:
            self._advance(now)
            for t in self.tickets:
                if t["id"] == tid:
                    t["expires"] = now + LEASE_S
                    return self._view(t)
            return None

    def leave(self, tid: str) -> None:
        with self.lock:
            self.tickets = [t for t in self.tickets if t["id"] != tid]
            self._advance(self.clock())

    def count(self) -> int:
        with self.lock:
            self._advance(self.clock())
            return len(self.tickets)

    def listing(self) -> list:
        now = self.clock()
        with self.lock:
            self._advance(now)
            return [{"state": t["state"], "label": t["label"], "age_s": int(now - t["since"])} for t in self.tickets]


# ---- the lock nodes: what ComfyUI has, where it comes from, whether a lock may be used ----------------------------------

def node_inputs(info) -> set:
    """The input names of a node as ComfyUI's /object_info/<node> describes it (required and optional); empty for anything that is not such a description."""
    out: set = set()
    inp = info.get("input") if isinstance(info, dict) else None
    if isinstance(inp, dict):
        for section in ("required", "optional"):
            names = inp.get(section)
            if isinstance(names, dict):
                out |= {k for k in names if isinstance(k, str)}
    return out


def pack_of(module) -> str | None:
    """The custom-node folder a node comes from, from the `python_module` ComfyUI reports ("custom_nodes.<folder>"); None for a core node or anything odd (the name is used as a folder name)."""
    if not isinstance(module, str) or not module.startswith("custom_nodes."):
        return None
    name = module[len("custom_nodes."):]
    return name if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.\-]{0,99}", name) and ".." not in name else None


def git_head(folder: Path) -> str | None:
    """The commit a folder's git checkout is on (40 hex), read from .git/HEAD and its ref or packed-refs without running git; None when it is not a checkout or cannot be read."""
    git = folder / ".git"
    try:
        head = (git / "HEAD").read_text().strip()
        if re.fullmatch(r"[0-9a-f]{40}", head):
            return head
        if not head.startswith("ref: "):
            return None
        ref = head[5:].strip()
        if ".." in ref or ref.startswith("/"):
            return None
        try:
            val = (git / ref).read_text().strip()
        except OSError:
            val = next((ln.split()[0] for ln in (git / "packed-refs").read_text().splitlines() if ln.endswith(" " + ref)), "")
        return val if re.fullmatch(r"[0-9a-f]{40}", val) else None
    except (OSError, UnicodeDecodeError, IndexError):
        return None


def pack_version(folder: Path) -> str | None:
    """The version line of a pack's pyproject.toml (`version = "x"`), None when there is none; for a pack installed without git."""
    try:
        with open(folder / "pyproject.toml", encoding="utf-8", errors="replace") as f:
            m = re.search(r'^version\s*=\s*"([^"\n]{1,40})"', f.read(8192), re.M)
    except OSError:
        return None
    return m.group(1) if m else None


def lock_words(lock: str, why: str, **kw) -> str:
    """The coded words of a lock that cannot be served. `why`: allowed, missing, incompatible (missing=[inputs]) or unknown."""
    node = LOCK_NODES[lock]
    if why == "allowed":
        return (f"lock_not_allowed: lock=audiodrive uses the third-party node {node} (pack comfyui-vrgamedevgirl) under the {VRGDG_LICENCE}, which is not an open licence: "
                f"serving it to others through an API or a hosted workflow needs the author's paid commercial licence. It is off on this box until the owner decides and the operator "
                f"sets {ENV_ALLOW_VRGDG}=1 for the adapter. Use lock=addguide (the official node). Nothing was queued.")
    if why == "missing":
        where = ("it is a ComfyUI core node (comfy_extras.nodes_minimax_h3): update ComfyUI" if lock == "addguide"
                 else "it comes from the third-party custom-node pack comfyui-vrgamedevgirl: install it on the box image, pinned to a commit (docs/cloud-gpu.md)")
        return f"lock_node_missing: lock={lock} needs the node {node}, which this box's ComfyUI does not have ({where}). The take was not run and no other graph was used instead."
    if why == "incompatible":
        return (f"lock_node_incompatible: the node {node} in this box's ComfyUI has no input {', '.join(kw['missing'])} that the adapter wires (its definition changed?). "
                f"The take was not run and no other graph was used instead.")
    return (f"lock_state_unknown: lock={lock} needs the node {node}, but ComfyUI's node list could not be read (ComfyUI down or still starting). "
            f"The take was not run and no other graph was used instead.")


def lock_entry(lock: str, info, checked: bool, allowed: bool, root: Path) -> dict:
    """What /health says of one lock and whether it may be used: `info` is the node's /object_info description (None: ComfyUI does not have it), `checked` False when ComfyUI did not answer
    (then nothing is known), `allowed` the operator's switch (it matters for audiodrive only). `problem` is the coded words a take of this lock gets, None when it can run: the licence
    switch is looked at first, then the node, then its inputs."""
    node = LOCK_NODES[lock]
    module = info.get("python_module") if isinstance(info, dict) and isinstance(info.get("python_module"), str) else None
    e: dict = {"node": node, "present": (info is not None) if checked else None}
    problem = None
    if lock == "audiodrive" and not allowed:
        problem = lock_words(lock, "allowed")
    elif not checked:
        problem = lock_words(lock, "unknown")
    elif info is None:
        problem = lock_words(lock, "missing")
    else:
        lacking = [n for n in LOCK_INPUTS[lock] if n not in node_inputs(info)]
        if lacking:
            problem = lock_words(lock, "incompatible", missing=lacking)
    if lock == "addguide":
        e["module"] = module[:100] if module else None
    else:
        pack = pack_of(module)
        folder = root / "custom_nodes" / pack if pack else None
        e.update(allowed=allowed, licence=VRGDG_LICENCE, pack=pack, commit=git_head(folder) if folder else None, version=pack_version(folder) if folder else None)
        on = "enabled" if allowed else f"off: {ENV_ALLOW_VRGDG} is not set"
        e["status"] = f"audiodrive ({VRGDG_TAG}, {on})"
    e["usable"] = problem is None
    e["problem"] = problem
    return e


# ---- the server ------------------------------------------------------------------------------------------------------

class Host:
    def __init__(self, token: str, comfy: str, state: Path, comfy_root: Path, public_base: str = "", bind: str = "127.0.0.1", tts: dict | None = None,
                 free_after_take: bool = FREE_AFTER_TAKE, allow_vrgdg: bool = False):
        if not token:
            raise ValueError("the token is empty: the adapter would answer anyone")
        self.token, self.comfy, self.state, self.root = token, comfy.rstrip("/"), state, comfy_root
        self.tasks_dir, self.out_dir, self.files_dir = state / "tasks", state / "out", state / "files"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.public_base = public_base.rstrip("/")
        self.line = Line()
        self.free_after_take = free_after_take
        self.tts = tts                      # {"python", "dir", "worker"} when a voice is installed, else None
        self.q: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        self.n_active = 0                   # takes accepted (even while their inputs are still being fetched) and not over: /health reads this, not the files
        self.holds: dict = {}               # finished takes not yet collected: task id -> (finished at, file key), busy for GRACE_S
        self.mem: dict = {}                 # a task whose file could not be written (disk full) lives here
        self.worker_thread = None
        self._comfy = (0.0, False)
        self._card, self._card_lock, self._last_maintain = ("unknown", 0.0), threading.Lock(), 0.0
        self._mem_mib = None
        self._vram_free = None              # ComfyUI's own count of the first device's free memory (MiB), from the /system_stats that comfy_up asks for
        self.allow_vrgdg = bool(allow_vrgdg)   # the operator's switch for the third-party lock (CLOUDGPU_ALLOW_NODE_VRGDG=1); off unless a person set it
        self._locks_lock = threading.Lock()
        self._lock_caps: tuple = (0.0, None)   # (when ComfyUI's lock nodes were last read, what they said)
        self.proof_ok = loopback_bind(bind)     # on any other bind /health never answers `proof`
        self._proofs: list = []                 # when the last answers with a proof were made (the last minute)
        self.orphans = self.restarted()
        self.maintain(force=True)

    # tasks are small json files: one source of truth that survives a restart
    def _path(self, tid: str) -> Path:
        return self.tasks_dir / f"{tid}.json"

    def load(self, tid: str):
        if tid in self.mem:
            return dict(self.mem[tid])
        try:
            return json.loads(self._path(tid).read_text())
        except (OSError, ValueError):
            return None

    def save(self, t: dict):
        tmp = self._path(t["id"]).with_suffix(".tmp")
        tmp.write_text(json.dumps(t, ensure_ascii=False))
        os.replace(tmp, self._path(t["id"]))

    def keep(self, t: dict) -> bool:
        """save() for the worker: an OSError (disk full) keeps the task in memory instead of killing the thread; False then."""
        try:
            self.save(t)
        except OSError:
            self.mem[t["id"]] = dict(t)
            return False
        self.mem.pop(t["id"], None)
        return True

    def drop_inputs(self, t: dict):
        names = t.get("inputs")
        for n in set(names if isinstance(names, list) else []):
            if isinstance(n, str) and Path(n).name == n:
                try:
                    (self.root / "input" / n).unlink(missing_ok=True)
                except OSError:
                    pass

    def busy(self) -> int:
        """Takes accepted and not over, and finished ones not yet collected (until GRACE_S after they finished): counters, no file is read."""
        now = time.time()
        with self.lock:
            for tid, (done, _) in list(self.holds.items()):
                if now - done > GRACE_S:
                    del self.holds[tid]
            n = self.n_active + len(self.holds)
        return n + self.line.count()                    # a caller in the line holds the box busy: it is not powered off or restarted under them

    def collected(self, tid: str = "", key: str = ""):
        """A finished take was fetched (its video served, or its failure read): it no longer holds the box busy."""
        with self.lock:
            for t, (_, k) in list(self.holds.items()):
                if t == tid or (key and k == key):
                    del self.holds[t]

    def records(self, since: float = 0.0) -> list:
        """What `util` reads: one small dict per task record created at or after `since` (no id, no plan, no input), oldest first. `done` is `finished`, else the record file's
        modification time for a take that is over, else None. Read only."""
        out = []
        for p in self.tasks_dir.glob("*.json"):
            t = self.load(p.stem)
            if not isinstance(t, dict) or not isinstance(t.get("created"), (int, float)) or t["created"] < since:
                continue
            done = t.get("finished")
            if done is None and t.get("status") in ("SUCCESS", "FAILED"):
                try:
                    done = p.stat().st_mtime
                except OSError:
                    pass
            out.append({"workflow": t.get("workflow"), "status": t.get("status"), "created": t["created"], "started": t.get("started"), "done": done})
        return sorted(out, key=lambda r: r["created"])

    def restarted(self) -> int:
        n = 0
        for p in self.tasks_dir.glob("*.json"):
            t = self.load(p.stem)
            if t and t["status"] in ("QUEUED", "RUNNING"):
                t.update(status="FAILED", error="the adapter restarted while this take was waiting or running; it was not re-run")
                self.save(t)
                self.drop_inputs(t)
                n += 1
        return n

    def maintain(self, force: bool = False):
        """Delete task files, leftover temp files and input files older than TASK_TTL_S, and expired videos; at start and then every MAINTAIN_EVERY_S."""
        now = time.time()
        if not force and now - self._last_maintain < MAINTAIN_EVERY_S:
            return
        self._last_maintain = now
        for pat, folder in (("*.json", self.tasks_dir), ("*.tmp", self.tasks_dir), ("cloudgpu_*", self.root / "input")):
            for p in list(folder.glob(pat)):
                try:
                    if now - p.stat().st_mtime > TASK_TTL_S:
                        t = self.load(p.stem) if p.suffix == ".json" else None
                        if not (t and t.get("status") in ("QUEUED", "RUNNING")):
                            p.unlink()
                except OSError:
                    pass
        stored = []
        for p in self.files_dir.glob("*"):
            try:
                st = p.stat()
                if p.suffix == ".tmp" and now - st.st_mtime > TASK_TTL_S or p.suffix != ".tmp" and now - st.st_mtime > FILES_TTL_S:
                    p.unlink()
                elif p.suffix != ".tmp":
                    stored.append((st.st_mtime, st.st_size, p))
            except OSError:
                pass
        total = sum(sz for _, sz, _ in stored)
        for _, sz, p in sorted(stored):                    # over the cap: the longest unused go first
            if total <= FILES_MAX:
                break
            try:
                p.unlink()
                total -= sz
            except OSError:
                pass
        for tid, t in list(self.mem.items()):
            if now - t.get("finished", t.get("created", now)) > TASK_TTL_S:
                self.mem.pop(tid, None)
        self.sweep()

    def put_file(self, sha: str, blob: bytes, mime: str) -> int:
        """Store a file under its sha256 (the caller's claim is checked against the bytes). Returns its size; Refused when it is not what it says."""
        ext = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/mpeg": ".mp3", "audio/mp3": ".mp3"}.get(mime)
        if ext is None:
            raise Refused(f"a file of type {mime or '?'} is not stored here (png, jpeg, webp, wav, mp3)")
        if not blob or len(blob) > MAX_INPUT:
            raise Refused(f"the file is empty or over {MAX_INPUT // 2**20} MB")
        if hashlib.sha256(blob).hexdigest() != sha:
            raise Refused("the bytes are not the sha256 in the address")
        have = stored_file(self.files_dir, sha)
        if have is not None:
            os.utime(have, None)
            return have.stat().st_size
        tmp = self.files_dir / f"{sha}.{uuid.uuid4().hex[:8]}.tmp"
        try:
            tmp.write_bytes(blob)
            os.replace(tmp, self.files_dir / f"{sha}{ext}")
        except OSError:
            tmp.unlink(missing_ok=True)
            raise
        return len(blob)

    def submit(self, workflow: str, body: dict) -> str:
        p = plan(workflow, body)
        if p["kind"] == "tts" and not self.tts:
            raise Refused("this box has no voice installed (IndexTTS2: start the adapter with --tts-python and --tts-dir, see docs/cloud-gpu.md)")
        if p.get("lock"):
            why = self.lock_problem(p["lock"], door=True)        # before anything is fetched or queued
            if why:
                raise Refused(why)
        with self.lock:
            self.n_active += 1              # busy from here: fetching the inputs below can take a minute, and a power-off must not fall into it
        tid = "h3-" + uuid.uuid4().hex[:20]
        t = {"id": tid, "workflow": workflow, "status": "QUEUED", "created": time.time(), "plan": p, "inputs": []}
        files = []
        try:
            # fetch before queueing: a bad input is the caller's 400 now, not a failed take later
            inp = self.root / "input"
            inp.mkdir(parents=True, exist_ok=True)
            if p["kind"] == "tts":
                files.append(fetch(body["prompt_simple"], inp, ("audio",), "prompt_simple", self.files_dir))
                if p["emo"] == "audio":
                    files.append(fetch(body["emo_ref_audio"], inp, ("audio",), "emo_ref_audio", self.files_dir))
            elif p["kind"] == "ref":
                files.append(fetch(body["ref_image_0"], inp, ("image",), "ref_image_0", self.files_dir))
                files.append(fetch(body["ref_audio_0"], inp, ("audio",), "ref_audio_0", self.files_dir))
                for i in range(1, p.get("extra_images", 0) + 1):          # after the audio, so files[0] and files[1] mean what they always did
                    files.append(fetch(body[f"ref_image_{i}"], inp, ("image",), f"ref_image_{i}", self.files_dir))
            elif p["kind"] == "t2v":
                pass                        # nothing to fetch: the prompt is all there is
            else:
                files.append(fetch(body["first_frame"], inp, ("image",), "first_frame", self.files_dir))
                files.append(files[0] if body["last_frame"] == body["first_frame"]
                             else fetch(body["last_frame"], inp, ("image",), "last_frame", self.files_dir))
            t["inputs"] = [f.name for f in files]
            self.save(t)
        except BaseException:               # a refused input, a full disk: leave nothing behind and stop counting
            for f in set(files):
                try:
                    f.unlink(missing_ok=True)
                except OSError:
                    pass
            for leftover in (self._path(tid), self._path(tid).with_suffix(".tmp")):
                try:
                    leftover.unlink(missing_ok=True)
                except OSError:
                    pass
            with self.lock:
                self.n_active -= 1
            raise
        self.q.put(tid)
        return tid

    def result(self, tid: str):
        t = self.load(tid) if tid.startswith("h3-") and tid[3:].isalnum() else None
        if t is None:
            return None
        d = {"status": t["status"]}
        if t["status"] == "SUCCESS":
            if self.file(t.get("file_key") or "", t.get("file_ext", "mp4")) is None:       # the video was swept (FILE_TTL_S) while the record lives on (TASK_TTL_S)
                self.collected(tid=tid)
                return {"status": "FAILED", "error": "result expired: the video was deleted from the box; run the take again"}
            ext = t.get("file_ext", "mp4")
            d["results"] = [{"url": f"{self.public_base}/files/{t['file_key']}.{ext}", "type": "audio" if ext == "wav" else "video", "file_type": ext}]
        if t["status"] == "FAILED":
            d["error"] = t.get("error", "")
            self.collected(tid=tid)
        return d

    def file(self, key: str, ext: str = "mp4"):
        if len(key) != 32 or not key.isalnum() or ext not in CONTENT_TYPES:
            return None
        p = self.out_dir / f"{key}.{ext}"
        return p if p.is_file() and time.time() - p.stat().st_mtime < FILE_TTL_S else None

    def card(self) -> str:
        """The GPU this box runs on, as `nvidia-smi` names it with its memory ("NVIDIA GeForce RTX 4090, 49140 MiB"), "unknown" when it cannot say. Cached; an unknown is asked again after CARD_RETRY_S."""
        with self._card_lock:
            card, asked = self._card
            if card != "unknown" or time.time() - asked < CARD_RETRY_S:
                return card
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
                lines = [x.strip() for x in r.stdout.splitlines() if x.strip()] if r.returncode == 0 else []
                card = "; ".join(lines)[:200] or "unknown"
                m = re.search(r",\s*(\d+)\s*MiB\s*$", lines[0]) if lines else None
                self._mem_mib = int(m.group(1)) if m else None
            except (OSError, subprocess.SubprocessError):
                card, self._mem_mib = "unknown", None
            self._card = (card, time.time())
            return card

    def health(self, nonce: str = "") -> dict:
        wt = self.worker_thread
        alive = wt is not None and wt.is_alive()
        card = self.card()
        up = self.comfy_up()
        caps = self.lock_caps() if up else self.unknown_locks()          # no ComfyUI, no knowledge: no lock is offered
        d = {"ok": alive, "comfy": up, "busy": self.busy(), "worker": "alive" if alive else "dead", "card": card,
             "memory_mib": self._mem_mib if card != "unknown" else None, "vram_free_mib": self._vram_free if up else None, "voice": bool(self.tts), "files": 1, "t2v": True,
             "audio_lock": [k for k, e in caps["locks"].items() if e["usable"]], "audio_lock_detail": {"checked": caps["checked"], **caps["locks"]}}
        if nonce and self.proof_ok and NONCE.fullmatch(nonce) and self._proof_allowed():
            d["proof"] = hmac.new(self.token.encode(), nonce.encode(), "sha256").hexdigest()
        return d

    def _proof_allowed(self) -> bool:
        now = time.time()
        with self.lock:
            self._proofs = [x for x in self._proofs if now - x < 60]
            if len(self._proofs) >= PROOF_PER_MIN:
                return False
            self._proofs.append(now)
            return True

    # ---- the locks (0.20.0) ----
    def unknown_locks(self) -> dict:
        """What is known of the lock nodes when ComfyUI has not answered: nothing (`checked` false); audiodrive is still off without the operator's switch."""
        return {"checked": False, "locks": {k: lock_entry(k, None, False, self.allow_vrgdg, self.root) for k in LOCK_NODES}}

    def lock_caps(self, fresh: bool = False) -> dict:
        """ComfyUI's lock nodes as {"checked": bool, "locks": {lock: lock_entry}}, read from its /object_info (one node at a time: the whole list is megabytes). Kept LOCK_CACHE_S, or COMFY_CACHE_S
        when ComfyUI did not answer; `fresh` asks again. Never raises."""
        with self._locks_lock:
            when, caps = self._lock_caps
            if not fresh and caps is not None and time.time() - when < (LOCK_CACHE_S if caps["checked"] else COMFY_CACHE_S):
                return caps
            infos = {}
            try:
                for lock, node in LOCK_NODES.items():
                    doc = self.comfy_json("/object_info/" + urllib.parse.quote(node), timeout=5)
                    info = doc.get(node) if isinstance(doc, dict) else None
                    infos[lock] = info if isinstance(info, dict) else None
                caps = {"checked": True, "locks": {k: lock_entry(k, infos[k], True, self.allow_vrgdg, self.root) for k in LOCK_NODES}}
            except Exception:                         # ComfyUI silent or an answer that is not a node description: nothing is known, nothing is offered (never an error to /health)
                caps = self.unknown_locks()
            self._lock_caps = (time.time(), caps)
            return caps

    def lock_problem(self, lock: str, door: bool) -> str | None:
        """The coded words saying why a take of `lock` cannot be served now, None when it can. `door`: the take is only being accepted. A lock that is not usable on the cached read is asked about again
        before it is refused (a node appears when ComfyUI restarts, a refusal must not be stale). At the door a ComfyUI that did not answer is not a refusal (a cold box is still starting: the take
        waits for it, and run() asks again); once the take runs, the same silence fails it. Never a fallback to another graph."""
        if lock not in LOCK_NODES:
            return CUSTOM_WORDS if lock == "custom" else f"lock_unknown: lock {lock!r} is not one of {', '.join(LOCKS)}"
        if lock == "audiodrive" and not self.allow_vrgdg:
            return lock_words(lock, "allowed")        # the licence switch is looked at before ComfyUI is asked anything, and it does not change while the adapter runs
        e = self.lock_caps()["locks"][lock]
        if e["usable"]:
            return None
        caps = self.lock_caps(fresh=True)
        e = caps["locks"][lock]
        if e["usable"]:
            return None
        if door and not caps["checked"]:
            return None
        return e["problem"]

    # ---- worker ----
    def comfy_json(self, path: str, body=None, timeout=60):
        r = urllib.request.Request(self.comfy + path, data=json.dumps(body).encode() if body is not None else None,
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return json.loads(resp.read())

    def comfy_up(self) -> bool:
        checked, up = self._comfy
        if time.time() - checked < COMFY_CACHE_S:       # /health is asked often and must not wait on ComfyUI each time
            return up
        try:
            stats = self.comfy_json("/system_stats", timeout=5)
            up = True
            try:
                self._vram_free = int(stats["devices"][0]["vram_free"]) // 2**20
            except (KeyError, IndexError, TypeError, ValueError):
                self._vram_free = None
        except (urllib.error.URLError, OSError, ValueError):
            up = False
        self._comfy = (time.time(), up)
        return up

    def interrupt(self):
        """Ask ComfyUI to stop what it is running (best effort: it answers an empty body, and may be down)."""
        try:
            self.comfy_json("/interrupt", {}, timeout=5)
        except (urllib.error.URLError, OSError, ValueError):
            pass

    def clear_comfy(self):
        """After a restart that failed takes: interrupt ComfyUI and empty its queue, or the orphaned job runs on for nobody and a new take waits behind it."""
        self.interrupt()
        try:
            self.comfy_json("/queue", {"clear": True}, timeout=5)
        except (urllib.error.URLError, OSError, ValueError):
            pass

    def start_worker(self) -> threading.Thread:
        th = threading.Thread(target=self.worker, daemon=True)
        self.worker_thread = th
        th.start()
        return th

    def worker(self):
        self.worker_thread = threading.current_thread()
        if self.orphans:
            self.clear_comfy()
        while True:
            try:
                tid = self.q.get(timeout=MAINTAIN_EVERY_S)
            except queue.Empty:
                tid = None
            try:
                if tid is not None:
                    self.work(tid)
                self.maintain()
            except Exception:                                # the thread must not die: /health says `dead` and nothing would run again
                pass
            finally:
                if tid is not None:
                    self.finish(tid)

    def finish(self, tid: str):
        """The take left the queue: it no longer counts as accepted, and until it is collected (or GRACE_S) it still holds the box busy."""
        with self.lock:
            self.n_active = max(0, self.n_active - 1)
        t = self.load(tid)
        if t and t["status"] in ("SUCCESS", "FAILED"):
            with self.lock:
                self.holds[tid] = (time.time(), t.get("file_key"))
        if self.free_after_take and t and t["plan"]["kind"] != "tts":       # a voice already freed ComfyUI before it ran
            self.free_if_idle()

    def work(self, tid: str):
        t = self.load(tid)
        if not t or t["status"] != "QUEUED":
            return
        t["status"] = "RUNNING"
        t["started"] = time.time()
        if not self.keep(t):                                 # the disk is full: the take is failed, not the worker
            t.update(status="FAILED", error="the box could not write its state (disk full?); the take was not run", finished=time.time())
            self.keep(t)
            self.drop_inputs(t)
            return
        try:
            self.run(t)
            t["status"] = "SUCCESS"
        except Exception as e:                               # a failed take says why; the worker lives on
            t["status"], t["error"] = "FAILED", f"{type(e).__name__}: {e}"[:600]
        finally:
            self.drop_inputs(t)
        t["finished"] = time.time()
        self.keep(t)

    def run(self, t: dict):
        if t["plan"]["kind"] == "tts":
            return self.run_tts(t)
        deadline = time.time() + COMFY_WAIT_S
        while not self.comfy_up():
            if time.time() > deadline:
                raise RuntimeError("ComfyUI is not answering")
            time.sleep(3)
        p, files = t["plan"], t["inputs"]
        lock = p.get("lock")
        if lock:                                             # ComfyUI is up now: ask for its nodes again, and stop the take in words rather than run another graph
            why = self.lock_problem(lock, door=False)
            if why:
                raise LockUnavailable(why)
        prefix = f"cloudgpu/{t['id']}"
        g = graph(p["kind"], w=p["w"], h=p["h"], seconds=p["seconds"], image=files[0] if files else None, audio=files[1] if p["kind"] == "ref" else None,
                  prompt=p["prompt"], seed=p["seed"], prefix=prefix, extra_images=files[2:] if p["kind"] == "ref" else (), lock=lock)
        if p["kind"] == "fl" and files[1] != files[0]:
            g["5b"]["inputs"]["image"] = files[1]
        try:
            pid = self.comfy_json("/prompt", {"prompt": g, "client_id": t["id"]})["prompt_id"]
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"ComfyUI refused the graph: {e.read().decode(errors='replace')[:400]}") from None
        end = time.time() + TASK_TIMEOUT_S
        while True:
            h = self.comfy_json("/history/" + pid).get(pid)
            if h and h.get("status", {}).get("completed") is not None:
                break
            if time.time() > end:
                self.interrupt()
                raise RuntimeError(f"timed out after {TASK_TIMEOUT_S}s")
            time.sleep(2)
        if h["status"].get("status_str") != "success":
            raise RuntimeError("ComfyUI: " + json.dumps(h["status"].get("messages"), ensure_ascii=False)[-400:])
        src = None
        for out in h.get("outputs", {}).values():
            for item in out.get("images", []) + out.get("videos", []) + out.get("gifs", []):
                if str(item.get("filename", "")).endswith(".mp4"):
                    src = self.root / "output" / item.get("subfolder", "") / item["filename"]
        if src is None or not src.is_file():
            raise RuntimeError("ComfyUI finished without a video file")
        key = secrets.token_hex(16)
        dst = self.out_dir / f"{key}.mp4"
        want = p["seconds"]
        if shutil.which("ffmpeg"):          # the rule above pads to 5 mod 17 frames: cut back to what was asked for
            try:
                r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-t", f"{want:.3f}", "-c:v", "libx264", "-crf", "17",
                                    "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(dst)],
                                   capture_output=True, text=True, timeout=FFMPEG_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                dst.unlink(missing_ok=True)
                raise RuntimeError(f"ffmpeg timed out after {FFMPEG_TIMEOUT_S}s") from None
            if r.returncode != 0:
                dst.unlink(missing_ok=True)
                raise RuntimeError("ffmpeg: " + r.stderr[-300:])
        else:
            shutil.copy(src, dst)
        t["file_key"] = key
        self.sweep()

    def run_tts(self, t: dict):
        """The voice: tts_worker.py in the IndexTTS2 environment writes one wav. ComfyUI is first asked to give its GPU memory back (the H3 models stay resident
        otherwise and the voice model may not fit next to them); the next video take loads them again."""
        p, files = t["plan"], t["inputs"]
        self.free_comfy()
        key = secrets.token_hex(16)
        dst = self.out_dir / f"{key}.wav"
        job = {"dir": str(self.tts["dir"]), "text": p["text"], "spk": str(self.root / "input" / files[0]), "out": str(dst), "emo": p["emo"], "vector": p["vector"],
               "alpha": p["alpha"], "random": p["random"], "emo_audio": str(self.root / "input" / files[1]) if p["emo"] == "audio" else None}
        jobfile = self.state / f"voice-{t['id']}.job"
        env = {k: v for k, v in os.environ.items() if k not in ("CLOUDGPU_TOKEN", "H3HOST_TOKEN")}       # the voice process has no use for the token
        try:
            fd = os.open(jobfile, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(job, f, ensure_ascii=False)
            try:
                r = subprocess.run([str(self.tts["python"]), str(self.tts["worker"]), str(jobfile)], capture_output=True, text=True, timeout=TTS_TIMEOUT_S,
                                   env=env, cwd=str(self.state))
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"the voice timed out after {TTS_TIMEOUT_S}s") from None
            if r.returncode != 0:
                raise RuntimeError("the voice failed: " + (r.stderr.strip().splitlines() or ["no message"])[-1][-300:])
            try:
                with open(dst, "rb") as f:
                    ok = f.read(4) == TTS_MAGIC and dst.stat().st_size > 44
            except OSError:
                ok = False
            if not ok:
                raise RuntimeError("the voice finished without a wav file")
        except BaseException:
            dst.unlink(missing_ok=True)
            raise
        finally:
            jobfile.unlink(missing_ok=True)
        t["file_key"], t["file_ext"] = key, "wav"
        self.sweep()

    def free_comfy(self) -> bool:
        """Best effort: ask ComfyUI to unload its models and free its memory (it may be down, or an older one without the call). True when it answered."""
        try:
            self.comfy_json("/free", {"unload_models": True, "free_memory": True}, timeout=10)
            return True
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def free_if_idle(self) -> bool:
        """Free ComfyUI's memory when no take is accepted or running and none is queued (a take that arrives meanwhile only waits for ComfyUI, which applies /free between its own jobs). Never raises."""
        with self.lock:
            if self.n_active > 0 or not self.q.empty():
                return False
        try:
            return self.free_comfy()
        except Exception:
            return False

    def sweep(self):
        now = time.time()
        for p in list(self.out_dir.glob("*.mp4")) + list(self.out_dir.glob("*.wav")):
            try:
                if now - p.stat().st_mtime > FILE_TTL_S:
                    p.unlink(missing_ok=True)
            except OSError:
                pass


def loopback_bind(bind: str) -> bool:
    """True when the adapter listens on this machine only: 127.0.0.0/8, ::1, or `localhost`. Anything else (0.0.0.0, ::, a public or private address, a name) is not."""
    if bind.lower() == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(bind.strip("[]"))
    except ValueError:
        return False
    return (getattr(ip, "ipv4_mapped", None) or ip).is_loopback


def handler(host: Host):
    class H(BaseHTTPRequestHandler):
        server_version = "cloudgpu"
        timeout = SOCKET_TIMEOUT_S          # StreamRequestHandler: a stalled client frees its thread

        def log_message(self, *a):          # the access log would carry a task id; the token is never in a URL
            pass

        def _send(self, code: int, doc: dict):
            b = json.dumps(doc, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _authed(self) -> bool:
            got = self.headers.get("Authorization", "")
            if host.token and hmac.compare_digest(got.encode(), host.token.encode()):       # an empty token is never matched, not even by no header
                return True
            self._send(401, {"code": "Unauthorized", "data": None, "msg": "token refused"})
            return False

        def _join_line(self):
            if not self._authed():
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                n = -1
            if not 0 <= n <= 4096:
                return self._send(413, {"code": "RequestParameterIsWrong", "data": None, "msg": "the body is at most 4 KB"})
            try:
                body = json.loads(self.rfile.read(n)) if n else {}
                if not isinstance(body, dict):
                    raise ValueError
            except ValueError:
                return self._send(400, {"code": "RequestParameterIsWrong", "data": None, "msg": "the body is not a JSON object"})
            try:
                d = host.line.join(body.get("label"))
            except Refused as e:
                return self._send(429, {"code": "LineFull", "data": None, "msg": str(e)})
            self._send(200, {"code": "Success", "data": d, "msg": ""})

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/health":
                nonce = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get("nonce", [""])[0]
                return self._send(200, host.health(nonce))
            if path.startswith("/files/") and path.rsplit(".", 1)[-1] in CONTENT_TYPES:
                ext = path.rsplit(".", 1)[-1]
                p = host.file(path[len("/files/"):-(len(ext) + 1)], ext)
                if p is None:
                    return self._send(404, {"code": "NotFound", "data": None, "msg": ""})
                self.send_response(200)
                self.send_header("Content-Type", CONTENT_TYPES[ext])
                self.send_header("Content-Length", str(p.stat().st_size))
                self.end_headers()
                with open(p, "rb") as f:
                    shutil.copyfileobj(f, self.wfile)
                host.collected(key=p.stem)           # sent to the end: the take is collected
                return
            if path.startswith(FILES + "/"):
                if not self._authed():
                    return
                sha = path[len(FILES) + 1:]
                p = stored_file(host.files_dir, sha) if re.fullmatch(r"[0-9a-f]{64}", sha) else None
                if p is None:
                    return self._send(404, {"code": "NotFound", "data": None, "msg": "no such file"})
                return self._send(200, {"code": "Success", "data": {"sha256": sha, "size": p.stat().st_size}, "msg": ""})
            if path == LINE or path.startswith(LINE + "/"):
                if not self._authed():
                    return
                if path == LINE:
                    return self._send(200, {"code": "Success", "data": {"line": host.line.listing(), "lease_s": LEASE_S}, "msg": ""})
                d = host.line.poll(path[len(LINE) + 1:])
                if d is None:
                    return self._send(404, {"code": "NotFound", "data": None, "msg": "no such ticket: its lease ran out or the adapter restarted; join again"})
                return self._send(200, {"code": "Success", "data": d, "msg": ""})
            if path == TASKS:
                if not self._authed():
                    return
                try:
                    since = float(urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get("since", ["0"])[0])
                except ValueError:
                    return self._send(400, {"code": "RequestParameterIsWrong", "data": None, "msg": "since is a number of seconds since 1970"})
                return self._send(200, {"code": "Success", "data": {"tasks": host.records(since), "now": time.time(), "ttl_s": TASK_TTL_S}, "msg": ""})
            if path.startswith(API + "/result/"):
                if not self._authed():
                    return
                d = host.result(path.rsplit("/", 1)[-1])
                if d is None:
                    return self._send(404, {"code": "NotFound", "data": None, "msg": "no such task"})
                return self._send(200, {"code": "Success", "data": d, "msg": ""})
            self._send(404, {"code": "NotFound", "data": None, "msg": ""})

        def do_PUT(self):
            path = self.path.split("?")[0]
            if not path.startswith(FILES + "/"):
                return self._send(404, {"code": "NotFound", "data": None, "msg": ""})
            if not self._authed():
                return
            sha = path[len(FILES) + 1:]
            if not re.fullmatch(r"[0-9a-f]{64}", sha):
                return self._send(400, {"code": "RequestParameterIsWrong", "data": None, "msg": "the address is /cloudgpu/files/<sha256 in lower-case hex>"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                n = -1
            if not 0 < n <= MAX_INPUT:
                return self._send(413, {"code": "RequestParameterIsWrong", "data": None, "msg": f"the file must be 1 byte to {MAX_INPUT // 2**20} MB"})
            blob = self.rfile.read(n)
            try:
                size = host.put_file(sha, blob, (self.headers.get("Content-Type") or "").split(";")[0].strip().lower())
            except Refused as e:
                return self._send(400, {"code": "RequestParameterIsWrong", "data": None, "msg": str(e)})
            except OSError as e:
                return self._send(507 if e.errno == errno.ENOSPC else 500, {"code": "InternalError", "data": None, "msg": "the box could not store the file"})
            self._send(200, {"code": "Success", "data": {"sha256": sha, "size": size}, "msg": ""})

        def do_DELETE(self):
            path = self.path.split("?")[0]
            if not path.startswith(LINE + "/"):
                return self._send(404, {"code": "NotFound", "data": None, "msg": ""})
            if not self._authed():
                return
            host.line.leave(path[len(LINE) + 1:])
            self._send(200, {"code": "Success", "data": None, "msg": ""})

        def do_POST(self):
            path = self.path.split("?")[0]
            if path == LINE:
                return self._join_line()
            if path == "/cloudgpu/free":
                if not self._authed():
                    return
                return self._send(200, {"code": "Success", "data": {"freed": host.free_if_idle()}, "msg": ""})
            if not path.startswith(API + "/") or path.startswith(API + "/result/"):
                return self._send(404, {"code": "NotFound", "data": None, "msg": ""})
            if not self._authed():
                return
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                n = -1
            if not 0 < n <= MAX_BODY:
                return self._send(413, {"code": "RequestParameterIsWrong", "data": None, "msg": f"body must be 1 byte to {MAX_BODY // 2**20} MB"})
            try:
                body = json.loads(self.rfile.read(n))
                if not isinstance(body, dict):
                    raise ValueError
            except ValueError:
                return self._send(400, {"code": "RequestParameterIsWrong", "data": None, "msg": "the body is not a JSON object"})
            try:
                tid = host.submit(path[len(API) + 1:], body)
            except Refused as e:
                return self._send(400, {"code": "RequestParameterIsWrong", "data": None, "msg": str(e)})
            except OSError as e:                     # a full disk: an answer the caller can read, not a dropped connection
                return self._send(507 if e.errno == errno.ENOSPC else 500,
                                  {"code": "InternalError", "data": None, "msg": "the box could not store the inputs (disk full?); nothing was queued"})
            except Exception:
                return self._send(500, {"code": "InternalError", "data": None, "msg": "the box could not accept the take; nothing was queued"})
            self._send(200, {"code": "Success", "data": {"task_id": tid}, "msg": ""})
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=6009)
    ap.add_argument("--bind", default=os.environ.get("CLOUDGPU_BIND") or "127.0.0.1",
                    help="the address to listen on (CLOUDGPU_BIND); 127.0.0.1 is all the SSH tunnel needs, 0.0.0.0 only for a vendor's public port")
    ap.add_argument("--comfy", default="http://127.0.0.1:6006", help="ComfyUI's address on this box")
    ap.add_argument("--comfy-root", default=os.path.expanduser("~/ComfyUI"))
    ap.add_argument("--state", default=os.path.expanduser("~/h3host"))
    ap.add_argument("--models", help="a JSON file of MODELS overrides (the keys at the top of this module)")
    ap.add_argument("--tts-python", default=os.environ.get("CLOUDGPU_TTS_PYTHON", ""), help="the Python of the IndexTTS2 environment (CLOUDGPU_TTS_PYTHON); the voice is served only with --tts-dir too")
    ap.add_argument("--tts-dir", default=os.environ.get("CLOUDGPU_TTS_DIR", ""), help="IndexTTS2's checkpoints folder, the one with config.yaml (CLOUDGPU_TTS_DIR)")
    ap.add_argument("--free-after-take", type=int, choices=(0, 1), default=0 if os.environ.get("CLOUDGPU_FREE_AFTER_TAKE", "1").strip().lower() in ("0", "false", "no", "off") else 1,
                    help="1 (default): after a take that leaves nothing waiting, ask ComfyUI to unload its models and free GPU memory (CLOUDGPU_FREE_AFTER_TAKE=0 turns it off)")
    ap.add_argument("--public-base", default=os.environ.get("CLOUDGPU_PUBLIC_BASE") or os.environ.get("H3HOST_PUBLIC_BASE", ""),
                    help="https://host:port the callers reach this server at; result URLs are built from it")
    a = ap.parse_args()
    if a.models:
        over = json.loads(Path(a.models).read_text())
        unknown = set(over) - set(MODELS)
        if unknown:
            raise SystemExit(f"--models: unknown keys {sorted(unknown)}")
        MODELS.update(over)
    token = (os.environ.get("CLOUDGPU_TOKEN") or os.environ.get("H3HOST_TOKEN") or "").strip()
    if len(token) < 24:
        raise SystemExit("set CLOUDGPU_TOKEN to a secret of at least 24 characters")
    worker = Path(__file__).resolve().with_name("tts_worker.py")
    tts = {"python": a.tts_python, "dir": a.tts_dir, "worker": worker} if a.tts_python and a.tts_dir and worker.is_file() and (Path(a.tts_dir) / "config.yaml").is_file() else None
    allow_vrgdg = os.environ.get(ENV_ALLOW_VRGDG, "").strip().lower() in ("1", "true", "yes", "on")
    host = Host(token, a.comfy, Path(a.state), Path(a.comfy_root), a.public_base, a.bind, tts, bool(a.free_after_take), allow_vrgdg)
    if allow_vrgdg:
        print(f"audio lock: audiodrive is ENABLED by {ENV_ALLOW_VRGDG} ({VRGDG_TAG}: {VRGDG_LICENCE}); the operator has taken the decision", file=sys.stderr, flush=True)
    host.lock_caps(fresh=True)                      # the lock nodes are read at start; ComfyUI may still be starting, then /health and the first take read them again
    host.start_worker()
    ThreadingHTTPServer((a.bind, a.port), handler(host)).serve_forever()


if __name__ == "__main__":
    main()
