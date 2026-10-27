"""Submit one take to a box and time it: `python3 -m cloudgpu bench ENVFILE NAME WORKFLOW RES SECONDS IMAGE [AUDIO]`.

ENVFILE holds the settings (CLOUDGPU_BASE and CLOUDGPU_TOKEN, KEY=value lines). WORKFLOW is minimax_h3_image_audio_to_video (needs AUDIO) or
minimax_h3_lightx2v (Mode A: the still is the first and last frame). RES is 480p竖 or 768p竖. Writes NAME.mp4 in the current folder and prints one JSON
line: the wall time from submit to done, which includes any wait in the box's queue; run a take twice, the first one loads the models. This is a paid
take (the box's hours): a person runs it (a convention: nothing here asks for a person, there is no --yes and no tty check, so an agent must not run it on its own). NAME becomes the
file name with only letters, digits, dot, dash and underscore kept.
"""

from __future__ import annotations

import base64
import json
import re
import sys
import time
import urllib.error
import urllib.request

from . import config, net
from .box import Box
from .errors import CloudGpuError

LINE = "大家好，今天给大家推荐一款特别好用的产品。"


def main(argv: list[str]) -> int:
    try:
        return _run(argv)
    except CloudGpuError as e:                       # the box did not prove it knows the token, or stopped answering: the token was not sent
        print(e.message)
        return 2


def _run(argv: list[str]) -> int:
    if len(argv) < 6:
        print(__doc__)
        return 2
    envf, name, wf, res, secs, image, *rest = argv
    env = config.read_env_file(envf)
    warn = config.env_file_warning(envf)
    if warn:
        print(warn, file=sys.stderr)
    base, tok = (config.get("BASE", env) or "").rstrip("/"), config.get("TOKEN", env)
    if not base or not tok:
        print("ENVFILE needs CLOUDGPU_BASE and CLOUDGPU_TOKEN")
        return 2
    try:
        net.check_url(base, "CLOUDGPU_BASE")
    except CloudGpuError as e:
        print(e.message)
        return 2
    if not net.header_safe(tok):
        print("CLOUDGPU_TOKEN has a character that cannot be sent in an HTTP header")
        return 2
    box = Box(env)                                   # the token is handed out only through it: at a local address it is proved first, and again every minute
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name).lstrip(".")[:80] or "take"
    api = base + "/api/v1/comfyui/comfyui_workflow"

    def uri(path, mime):
        return f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()

    def call(method, url, body=None):
        r = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None, method=method,
                                   headers={"Authorization": box.token(), "Content-Type": "application/json"})
        try:
            with net.urlopen(r, timeout=120) as x:
                return json.loads(x.read())
        except urllib.error.HTTPError as e:          # a redirect to another host lands here too: its body is not the API's
            try:
                return json.loads(e.read())
            except ValueError:
                return {"code": f"HTTP{e.code}", "data": None}

    img = uri(image, "image/png")
    if wf == "minimax_h3_image_audio_to_video":
        if not rest:
            print("AUDIO is required for", wf)
            return 2
        body = {"resolution": res, "ref_image_0": img, "ref_audio_0": uri(rest[0], "audio/wav"), "audio_duration": int(secs)}
    else:
        body = {"resolution": res, "first_frame": img, "last_frame": img, "duration": int(secs), "seed": 7,
                "prompt": f'The person from the first frame looks into the camera and says in Mandarin: "{LINE}" '
                          "She speaks naturally with small head and hand movements, soft light, vertical phone video."}
    t0 = time.time()
    d = call("POST", f"{api}/{wf}", body)
    if d.get("code") != "Success":
        print(json.dumps({"name": name, "submit": d}, ensure_ascii=False))
        return 2
    tid = d["data"]["task_id"]
    while True:
        r = call("GET", f"{api}/result/{tid}").get("data")
        if not r:
            print(json.dumps({"name": name, "error": "the box did not answer the result call as its API does"}))
            return 2
        if r["status"] in ("SUCCESS", "FAILED"):
            break
        time.sleep(3)
    out = {"name": name, "workflow": wf, "res": res, "seconds": int(secs), "status": r["status"], "wall_s": round(time.time() - t0, 1),
           "wall_s_per_video_s": round((time.time() - t0) / int(secs), 2)}
    if r["status"] == "SUCCESS":
        url = r["results"][0]["url"]
        url = base + url if url.startswith("/") else url
        try:
            net.check_url(url, "the result URL the box returned")      # a box can name any URL, file:// included
        except CloudGpuError as e:
            out["error"] = e.message
            print(json.dumps(out, ensure_ascii=False))
            return 1
        with net.urlopen(urllib.request.Request(url), timeout=120) as x, open(f"{name}.mp4", "wb") as f:
            f.write(x.read())
    else:
        out["error"] = r.get("error")
    print(json.dumps(out, ensure_ascii=False))
    return 0 if r["status"] == "SUCCESS" else 1
