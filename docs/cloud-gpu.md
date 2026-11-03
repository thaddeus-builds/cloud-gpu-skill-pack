# cloudgpu: run ComfyUI workflows (MiniMax-H3 first) on a GPU you rent from any vendor that gives SSH

A GPU rented by the hour or the day, with ComfyUI and the H3 weights on it, behind a small token-checked HTTP adapter. A project sends a take, the box
makes it, nothing is billed per second of video: the machine's hours are the cost. Four things differ per vendor and each has one seam, so a new vendor
costs nothing or about thirty lines; everything else (the adapter, its queue, the ComfyUI graphs) is the same on every box.

| what differs | seam | for a new vendor |
|---|---|---|
| how the box is reached | `CLOUDGPU_BASE` (a public address) or the SSH tunnel the package opens by key (`CLOUDGPU_SSH_KEY`, `CLOUDGPU_SSH`) | nothing: any vendor with SSH works; a vendor that exposes a port can use it |
| how it is powered, and the cloud account | a **cloud vendor** (`cloudgpu/vendors/`: list, status, snapshot, power, create, release) and the power backend built on it, chosen by `CLOUDGPU_POWER`: **`autodl` is built in and the default**, `none` | `none` fits an always-on box, a pass, or a vendor with no API; with an API, write one `Vendor` (below) |
| which model files the image carries | `cloudgpu/detect_models.py` reads them off the box into `models.json` | nothing: it matches exact file names, in any folder layout |
| what a second of video costs | price rows per box, `<workflow>@<resolution>#<box>`, from recorded runs (`python3 -m cloudgpu price`) | run the bench on the box, record the runs |

## What it costs and what it does (measured 2026-10-23, one take at a time, models warm)

Seconds of machine per second of video, and what that is at the box's hourly price (`cloudgpu/data/runs.jsonl` has every run). The hosted price is
autodl.art's own workflow catalog, peak / off-peak, ¥ per second of video.

| card (¥/h, pay as you go) | take | machine s per video s | ¥ per video s | hosted (peak / off-peak) |
|---|---|---|---|---|
| RTX 4090 48G (2.88) | audio-driven, 480p, 8 s | 6.6 | 0.0053 | 0.04 / 0.03 |
| | audio-driven, 480p, 15 s | 9.0 | 0.0072 | 0.04 / 0.03 |
| | audio-driven, 768p, 8 s | 22.9 | 0.0183 | 0.05 / 0.04 |
| | audio-driven, 768p, 15 s | 36.7 | 0.0294 | 0.05 / 0.04 |
| | Mode A (first = last frame), 480p, 8 s | 11.1 | 0.0089 | 0.04 / 0.03 |
| | Mode A, 768p, 8 s | 34.9 | 0.0279 | 0.05 / 0.04 |
| RTX PRO 6000 96G (6.98) | audio-driven, 480p, 8 s / 15 s | 7.3 / 8.1 | 0.014 / 0.016 | 0.04 / 0.03 |
| | audio-driven, 768p, 8 s / 15 s | 20.8 / 26.7 | 0.040 / 0.052 | 0.05 / 0.04 |
| | Mode A, 768p, 8 s | 32.3 | 0.063 | 0.05 / 0.04 |

- Full-use numbers: an idle hour is paid on top. A day or month pass is paid whether the card works or not; by the hour with the power-off below pays only the
  hours that were on. Check what the vendor charges for a stopped box (autodl: the system disk, ¥0.10/day; an instance off for 60 days is released with its data).
- The first take after a start is slow (about 160 s on the 4090 for 8 s of 480p: the 26 GB text encoder and the 20 GB model load).
- A 4090 has 48 GB and ran every take above, 768p of 15 s included, but at 768p the memory is full and ComfyUI swaps to system RAM: 10% slower than the
  PRO 6000 at 8 s and 40% slower at 15 s. At 480p it is as fast as the PRO 6000 for 41% of the price. A 32 GB card was not tried.
- Quality was judged from contact sheets by eye (mouth moves, the still is kept); lip-sync to the exact audio was not measured.

## Set up a box

1. Rent an instance from an image that already carries H3 in ComfyUI (a shared model pool on the vendor's side means no download); note its SSH line.
2. `python3 -m cloudgpu bundle` makes `cloudgpu-box.tgz` (the adapter, the detector, `install.sh`, `tunnel.sh`, the two verified model presets). `scp` it to the box,
   `tar xzf` it, and run `sh install.sh auto 6009 < token-file` (a 24+ character secret). `auto` runs `detect_models.py`, which reads the image's files and picks, by
   exact name, the models for this card (the nvfp4 text encoder needs a Blackwell GPU, every other card takes the int8 one); it names what is missing and writes
   nothing then. install.sh needs only a python and curl, works in `$CLOUDGPU_DIR` (default `~/h3host`, the folder name the first boxes got), and installs a systemd unit
   when the box runs systemd (a container has none: the vendor's power-on hook runs `start.sh`). The token is read from stdin, written 0600, never an argument.
3. Reach it. Best by key: put the public half of a key in the box's `authorized_keys`, restricted:
   `restrict,port-forwarding,permitopen="127.0.0.1:6009",permitlisten="127.0.0.1:1",command="/bin/false"`. That is a key that can forward (`-L`) the adapter's port and
   nothing else: `restrict` turns everything off, `port-forwarding` gives back forwarding only, `permitopen` limits `-L` to the adapter, `permitlisten` (OpenSSH 7.8 or newer)
   limits `-R` to a loopback listener on port 1, which nothing uses, so the key cannot publish anything from the box. Set `CLOUDGPU_SSH_KEY` (and `CLOUDGPU_SSH=[user@]host:port`):
   the package opens the tunnel by itself (BatchMode, no prompt) when nothing answers, and again after a power-on. `onbox/tunnel.sh HOST PORT KEYFILE` does the same by hand.
   Or, when the vendor exposes a port, start the adapter on it (`CLOUDGPU_PUBLIC_BASE` and `CLOUDGPU_BIND=0.0.0.0` for `start.sh`: the adapter listens on 127.0.0.1 only unless
   told otherwise) and set `CLOUDGPU_BASE` (a vendor's proxy often exposes only a few ports and an image may already use them: check what answers there first).
   **The first connection trusts the host key** (`StrictHostKeyChecking=accept-new`: remembered the first time, a changed key refused afterwards), so someone who answers as the box on that
   first connection is trusted. To close that, get the box's host key another way (the vendor's console, a person on the box) into a file and set `CLOUDGPU_KNOWN_HOSTS=file`: the key
   must then be in it (`StrictHostKeyChecking=yes`).
   **The local end of the tunnel is checked before it is used.** Any process on your machine can hold `127.0.0.1:16009` and read what is sent to it, so before the token goes anywhere the
   client sends a random nonce to `/health?nonce=...` (no token) and the adapter answers `proof`, the HMAC-SHA256 of the nonce under the token; the client checks it with the token it
   holds. No proof: `CloudGpuError(kind="http")` and nothing was sent (`tunnel.sh` does the same with `CLOUDGPU_TOKEN` in the environment, exit 3). This applies to local addresses only
   (a public https address is named by its certificate), so an adapter older than 0.2.1 must be reinstalled on the box (`bundle`, `install.sh`) to be reached through a tunnel. The proof is
   checked again when the last one is older than 60 s: `Box.token()` and `Box.ensure_up()` do it, so a tunnel that died and a port another process took are caught before the token goes
   (`bench` takes the token the same way). A request to this machine never goes through `http_proxy`/`HTTPS_PROXY`. **A widened bind (`CLOUDGPU_BIND`) has no proof**: the adapter answers
   `proof` only when it listens on loopback (otherwise anyone could ask it for HMAC(token, nonce) and guess the token offline), only for a nonce of 16 to 64 lowercase hex characters, and
   at most 30 times a minute; a client refuses an `http://` base to anything but this machine, so such a box is reached by its https address.

## Settings (environment; names only, a value is never printed)

```
CLOUDGPU_TOKEN=...          the secret install.sh read
CLOUDGPU_BOX=rtx4090        this box's name: prices end in #rtx4090
CLOUDGPU_BASE=...           where the box answers (http://127.0.0.1:16009 through the tunnel, or its public https address); unset with a key: that default
CLOUDGPU_SSH_KEY=...        a private key file (chmod 600): the package opens the tunnel by itself
CLOUDGPU_SSH=...            [user@]host:port of the box's SSH door; unset: the power backend says it (autodl's API does, and the port can change)
CLOUDGPU_REMOTE_PORT=...    the adapter's port on the box (default 6009)
CLOUDGPU_POWER=...          none | a cloud vendor's name (autodl, the default)
CLOUDGPU_VENDOR=...         the vendor `instances` talks to when CLOUDGPU_POWER is none (default autodl)
CLOUDGPU_KNOWN_HOSTS=...    a file with the box's SSH host key: the tunnel then trusts nothing on first use (default: accept-new)
CLOUDGPU_AUTOOFF=1          power the box off when the process ends, if THIS process powered it on, no take waits on it and the box is billed by the hour (1/true/yes/on; 0, off or empty is off)
CLOUDGPU_INSTANCE=pro-…     the instance the power backend powers (autodl ids look like pro-…)
CLOUDGPU_PASS_WARN_DAYS=3   warn this many days before a pass (daily, weekly, monthly) ends (default 3; 'Passes', below)
AUTODL_DEV_TOKEN=…          the autodl backend: the account's developer token (console -> account -> settings), not the workflow token
```

The older `H3HOST_<NAME>` names are read second, and only when the new name is absent: a key that is present wins even when empty (`CLOUDGPU_AUTOOFF=` is off, it does not fall through
to `H3HOST_AUTOOFF`). Values are stripped of surrounding whitespace (a token read from a file keeps no newline). A consumer loads its own `.env` into the environment, or passes a
mapping: `Box(env)`; `python3 -m cloudgpu status --env-file FILE` and `bench` read a file and warn when other users can read it (chmod 600).
On the box, `start.sh` also takes `CLOUDGPU_BIND` (default 127.0.0.1), `CLOUDGPU_PUBLIC_BASE`, `CLOUDGPU_ALLOW_NODE_VRGDG` (empty or unset is off, the default; `1`, `true`, `yes` or `on` is on: the owner's switch for the third-party audiodrive lock, below), and leaves an adapter that has a take on it running (it restarts only at `busy` 0). `install.sh` writes the value it was run with (letters and digits only; anything else stops the install) into `start.sh` as the default, so the decision is made at each install and **lasts across reboots and power-ons**, which run `start.sh` with no environment; a reinstall without the variable is off again. A value set when `start.sh` runs wins over the default, `0` or empty included, and the adapter writes `audio lock: audiodrive is ENABLED ...` to its log at each start it is on.

## Use it from code

```python
from cloudgpu import Box, CloudGpuError
box = Box()                      # or Box({"CLOUDGPU_BASE": ..., ...})
box.ensure_up()                  # tunnel by key if configured, power on if it is off and a backend can; CloudGpuError(kind=...) if not
api, token = box.api_base(), box.token()
# POST {api}/{workflow}   header Authorization: <token> (raw, no Bearer)   -> {"code":"Success","data":{"task_id":...}}
# GET  {api}/result/{task_id}                                              -> {"data":{"status":QUEUED|RUNNING|SUCCESS|FAILED,"results":[{"url":...}]}}
box.power_off_if_idle()          # only when GET /health answered and says busy is 0, and the account says the box is billed by the hour
```

Workflows (the names autodl publishes): `minimax_h3_image_audio_to_video` (`ref_image_0`, optional `ref_image_1`..`ref_image_8`, `ref_audio_0`, `audio_duration` 1..15, `resolution`) and `minimax_h3_lightx2v` (Mode A:
`prompt`, `first_frame`, `last_frame`, `duration` 1..15, `resolution`, `seed`). `resolution` is `480p竖` (480x864), `768p竖` (768x1344), `480p横` (864x480) or `768p横` (1344x768; landscape, 0.15.0). 1080p was never run on these cards. Inputs are http(s) URLs or
`data:` URIs. The result URL is `/files/<random>.mp4` (relative when the box has no public address: join it to the base you reached it at) and lives one hour: download at once.
A body the box will not run is HTTP 400 `RequestParameterIsWrong` before anything is queued. One take runs at a time; a take found waiting after a restart is failed, never re-run.
More than one reference picture (0.8.0): `ref_image_1`..`ref_image_8` take the same URL or `data:` rules as `ref_image_0`, in order with no gap (a gap or a ninth is a 400). In the prompt they are
`<Picture 2>`, `<Picture 3>` and so on (`ref_image_0` is `<Picture 1>`), e.g. "holding the box shown in <Picture 2>; the box looks exactly like <Picture 2>". With no `prompt` the default is used as before and
reads only `<Picture 1>`, so a caller that sends extra pictures should send its own prompt. Measured by the issue's author on the 4090 (48 GB) with one extra picture: 480p, 4.4 s take, 66 s (queued behind another take); 768p, 135 s;
the large print on a held pack is legible, small print is not. **Not measured:** the 32 GB card (4080 SUPER) with extra pictures: it held 31 GB with one picture, so it may run out of memory.
A third workflow, `indextts2-v1`, is the voice (0.6.0, below): the same call as the hosted one, the result a wav.
The audio-locked take (0.20.0): `minimax_h3_image_audio_locked` takes the same fields as `minimax_h3_image_audio_to_video` and one more, `lock` (`official`, the default since 0.21.0, `addguide` or `audiodrive`; section "The audio-locked take", below; `official` takes `line`, the spoken words, instead of `prompt`), and the voice on the lips is the audio you sent, not a speech the model made up from it. `minimax_h3_image_audio_to_video` is unchanged and stays for a box whose ComfyUI has no lock node.
Text to video (0.12.0): `minimax_h3_t2v` takes `prompt`, `duration` 1..15 and optionally `resolution` (default `768p竖`) and `seed`, and no picture or audio: it is the Mode A graph (same FL2VA model,
same turbo lora and steps, same `--models` names) with both frame inputs left out, so the model invents the picture and speaks what the prompt says. A consumer found the graph by hand on a 4090 box (three
clips, 768x1344, 90 frames, 6 steps); the adapter's own run of it has not been measured. An adapter older than 0.12.0 answers 400 `unknown workflow`: ask `/health` for `"t2v": true` first.

`GET /health` (no token) answers `{"ok", "comfy", "busy", "worker": "alive"|"dead", "card", "memory_mib", "vram_free_mib", "voice", "files", "t2v"}` and, when asked with `?nonce=`, a `proof`. `ok` is false when the worker thread is dead. `busy` counts
a take from the moment it is accepted (while its inputs are still being fetched) until it is over **and collected** (the video downloaded, or the failure read), or 300 s after it finished.
`card` is the GPU the box runs on, as `nvidia-smi` names it with its memory (`NVIDIA GeForce RTX 4090, 49140 MiB`; `unknown` when it cannot say); `memory_mib` is the first card's memory as a number (`null` when unknown), so a consumer can check the size without parsing text: compare it with the box you declared
(`CLOUDGPU_BOX`); `python3 -m cloudgpu status` does and says so in `next` when they differ. `vram_free_mib` is the free memory of ComfyUI's first device as ComfyUI itself reports it (`null` when ComfyUI is down or does not say), so a consumer can see whether a second model fits
before it loads it. `voice` is true when the box has the voice installed; `t2v` is true when it serves `minimax_h3_t2v`. From 0.7.0 a caller waiting in the box's line also counts in `busy`.
From 0.20.0 `/health` also has `audio_lock` and `audio_lock_detail` (section "The audio-locked take", below): `audio_lock` is the list of `lock` values the box serves right now (`["official", "addguide"]`, `["official", "addguide", "audiodrive"]`, or `[]`); an adapter older than 0.20.0 has no such key and answers the locked workflow 400 `unknown workflow`.

Errors are `CloudGpuError` with `kind`: `key_missing` (a setting or a key file is not right), `http` (a vendor API answered no), `unknown` (not a backend), `down` (the box
does not answer and nothing can start it), `timeout` (it did not come up); `next` lists what a person can do.

## A voice on the box (0.6.0)

A Mode B take (a cloned voice drives the lips) or a narrator line needs speech before the picture. The adapter serves it as `indextts2-v1`, with the call shape of autodl's hosted workflow, so one
rented box makes the voice and the picture and no hosted per-second API is called:

```
POST {api}/indextts2-v1   {"prompt_text": the line (at most 2000 characters),
                           "prompt_simple": the voice sample to clone (an http(s) URL or a data: URI, 5 to 15 s of speech),
                           "emo_control_method": "与音色参考音频相同" (default: the sample's own manner) | "使用情感向量控制" | "使用情感参考音频",
                           "emo_happy" "emo_angry" "emo_sad" "emo_afraid" "emo_disgusted" "emo_melancholic" "emo_surprised" "emo_calm": 0 to 1 (the vector method),
                           "emo_ref_audio": a sample of the manner to copy (the audio method), "emo_alpha": 0 to 1, "emo_random": bool}
GET  {api}/result/{task_id}   -> results: [{"url": ".../files/<random>.wav", "type": "audio", "file_type": "wav"}]
```

The emotion described by a text (`使用情感描述文本控制`) is not served: it answers 400 and says to use the vector. A voice runs in the same single worker as the video takes, so it waits its turn behind them.

**How it runs.** IndexTTS2 needs its own Python environment (torch and its dependencies), so the adapter, which stays standard library only, starts `tts_worker.py` in that environment once per voice, passing the job in a
0600 file and without the token in its environment. The worker calls IndexTTS2's own `indextts.infer_v2.IndexTTS2.infer`. The model is loaded for each voice and its GPU memory is given back when the process ends; before
each voice the adapter asks ComfyUI to unload its models (`POST /free`) so the voice fits next to them, and the next video take loads them again. That load is paid on every voice-then-video pair: measure it
before a price is quoted for Mode B on a box (`bench`).

**Set up.** `install.sh` looks for IndexTTS2's checkpoints (a folder with `config.yaml` and `s2mel.pth`, under ComfyUI's `models/` or a checkout in the home folder: `~/index-tts`, `~/IndexTTS-2`) and for a Python that can
`import indextts.infer_v2` (`.venv` next to the checkout, else the box's own), and wires them into `start.sh` as `--tts-python` and `--tts-dir`. When it finds nothing the voice is simply not installed
(the install does not fail): name them with `CLOUDGPU_TTS_DIR` and `CLOUDGPU_TTS_PYTHON` and run `install.sh` again. `GET /health` says `"voice": true|false`, and a box without a voice answers the workflow 400
("this box has no voice installed") before anything is fetched. Install IndexTTS2 itself from its own project (`hf download IndexTeam/IndexTTS-2`), not from this package.

**Not yet verified on a real card.** The adapter side is tested offline with a stand-in worker; `tts_worker.py` follows IndexTTS2's README (read 2026-10-28) but has not been run on a rented box, and the
folder layout of the vendor images is not known (hence the two settings). The first voice on a box must be a person's paid test: check the wav, the time (model load included), and that ComfyUI and the voice
fit the card together.

## A pool of boxes (0.3.0)

More than one box, in order of preference: `CLOUDGPU_POOL=a,b`, and each box's settings as `CLOUDGPU_<A>_<NAME>` (`CLOUDGPU_A_TOKEN`, `CLOUDGPU_A_BASE`, `CLOUDGPU_A_SSH`, `CLOUDGPU_A_SSH_KEY`, `CLOUDGPU_A_BOX`,
`CLOUDGPU_A_POWER`, and the same with `B`); give each box **a local tunnel port of its own** in its `BASE` (`http://127.0.0.1:16011`, `http://127.0.0.1:16009`). Settings that are not a box's
(`AUTODL_DEV_TOKEN`) are shared. No `CLOUDGPU_POOL`: one box on the plain settings, a pool of one.

```python
from cloudgpu import Pool
pool = Pool.from_env(os.environ, pin_file="my_box")      # the file written in a take's folder
with pool.take(out_dir, sent=lambda d: (d / "task_id").exists()) as box:
    ...                                                  # send the take to `box`, wait, fetch the result from `box`
```

A take waits for the first box in the list that is not making another take (a box makes one at a time) and comes up (`Box.ensure_up`). A box that fails to come up is tried again after a
minute; every box down with none busy raises the last box's `CloudGpuError` instead of waiting for ever. The choice is written to the take's folder (also by a pool of one named box, `CLOUDGPU_POOL=b`; a plain unnamed box writes none), and a take the caller says was sent
earlier runs on the box that holds it and on no other: only that box can give its result back. Threads of one process share one pool, so with two boxes two takes run at once.

### A person's hold on a box (0.4.0)

A person who needs a box for an hour (a test, a manual run) says so, and takes through the `Pool` stay off it:

```
python3 -m cloudgpu hold BOX --minutes 60 --who NAME --why "manual test"     # BOX: the name in CLOUDGPU_POOL, or default for a pool of one
python3 -m cloudgpu release BOX
```

A hold is one small file per box in `CLOUDGPU_STATE_DIR` (default `~/.cloudgpu/holds/`), so it works across processes and restarts and **ends by itself at its time** (at most 480 minutes; someone
else's active hold is not replaced, your own is extended). `Pool.take` counts a held box as busy and uses another; when every box it may use is held (a pool of one, or a take sent earlier that must
return to its box) it waits, saying `waiting for the hold on BOX by NAME until HH:MM` once on stderr (or through `Pool(on_wait=fn)`), and goes when the hold ends. The default `on_wait` (stderr) says a wait at once and then one line per `wait_say_every_s` (default 180 s) with the wait so far (`... (still waiting, 200 s)`); a function passed as `on_wait` gets every change of the text, as before. `Pool.power_off_if_idle` leaves a held
box on. `python3 -m cloudgpu status` lists the active holds. **Limit:** a hold governs callers that go through `Pool`; someone who talks to ComfyUI directly is not stopped, and a hold on a box
does not stop a take already running on it. 

### A line for the takes of one machine (0.5.0)

Two processes (two jobs, or a job and a person's run) that both see a free box no longer both send. Every take through `Pool.take` takes a **ticket** in `<CLOUDGPU_STATE_DIR>/turns/` and goes
when it is the oldest waiting take that may use a box that is free (nothing running on it), not held, and comes up; a take pinned to one box (a result to fetch) waits in the line for that box only.
A ticket is alive while its owner holds an `flock` on it, so a take that dies, however it dies, leaves no stale ticket and never blocks the line. A take that waits says so once
(`waiting for its turn: take-3 (pid 4121) is running on a`), and `python3 -m cloudgpu status` lists the line (`queue`: state, box, pid, take folder, since) and adds a line to `next` for each.
**Limits:** this line is per machine and per state directory (callers on different machines use the box's own line, next section), and two pools of different physical boxes that share a
state directory and box names would wait on each other (give each set its own `CLOUDGPU_STATE_DIR`). Without `fcntl` (Windows) or a state directory that cannot be written there is no line and a take
goes as in 0.4.

### Two takes on a box at once: `CLOUDGPU_QUEUE_DEPTH` (0.9.0)

By default one take of this machine is on a box from the moment it is sent until its result is fetched. The box sees only the sampling of that take; while the next take's pictures and audio travel to it
and while the last result comes back it is idle. Measured on 2026-10-29 on a consumer far from the boxes (RTX 4090 and 4080 SUPER, one pack making talking-head takes at 480x864, the character still sent
inline with every take): created to done 40 s and 54 to 63 s on the box, a new take arriving every 134 s and 164 s, the boxes working 29 and 33 percent of the time, at most one connection to the adapter at
any moment on either box (issue 12). `CLOUDGPU_QUEUE_DEPTH=2` lets two takes of this machine be on a box: the second is sent while the first is sampled, the adapter queues it behind the first (ComfyUI
is first in, first out), and the box goes from one take to the next with no transfer in between.

- **Choice of box.** A take goes to the box with fewest takes of this machine, the first in the list on a tie, so two takes go to two boxes before either box gets a second.
- **The line (line.py).** Only the first take on a box takes a place in its line, the one after it rides on that place; a take of another machine that gets "go" meanwhile is queued by the adapter behind ours.
- **The queue of this machine (turns.py).** Counts running takes per box against the depth, oldest first as before.
- **Values.** 1 is today's behaviour and the default; 2 is the one to try; 4 is the most. Raise it only while the box has memory for a second queued request (the adapter holds the body, a few MB) and the consumer
  has the scenes in flight to fill it (a pack's `parallel`).
- A take sent earlier and fetched later stays on its box as before. The result of a take is fetched inside its place, so a depth of 2 also overlaps one take's download with the next take's sampling.

### A picture sent once: the box's file store (0.10.0)

A take's body carries its pictures inline as `data:` URIs, so the same character still travels with every take, twice for a first-and-last-frame take (measured on 2026-10-29: 2.84 MB a body, 18 KB/s up,
so a minute and more of every take, with the box idle). The adapter now keeps files by their sha256: `GET /cloudgpu/files/<sha256>` (200 with the size, or 404), `PUT /cloudgpu/files/<sha256>` (the raw bytes,
`Content-Type` png, jpeg, webp, wav or mp3, up to 64 MB; the bytes must hash to the address; sending it again is no error and stores nothing twice), and a take's body may carry `sha256:<hex>` wherever it
carries a picture or an audio. A sha the box does not have is a 400 whose message starts `file_unknown:`; the box says it has the store with `"files": 1` in `/health`. Files the box has not been asked for in a week go,
and the oldest go first when the store passes 512 MB. A consumer asks `box.supports_files()` and then `box.ensure_file(path)` (cloudgpu/files.py), which sends the file once and returns the `sha256:<hex>`
to put in the body; an adapter older than 0.10.0 does not say `files`, so it is never asked and the consumer sends the data inline as before. The adapter has to be installed on the box again for this (the
consumer's `CLOUDGPU_QUEUE_DEPTH` does not need it).

## Cloud vendors and power

A **vendor** (`cloudgpu/vendors/`, one module each) is the cloud account's own API behind one small interface (`class Vendor`): `list`/`all`, `get`, `status`, `snapshot`,
`power_on`, `power_off`, `create`, `release`, `gpu_specs`. Its answers are normalised (`Instance`: id, status, name, billing, ends, gpu_spec, gpus, region, created/started/stopped;
a snapshot: price, SSH door, services, the card's name, never a password or a token). A **power backend** (`class Power`: `status`, `power_on`, `power_off`, `pass_info`, `ssh`)
is made from a vendor and the instance id `CLOUDGPU_INSTANCE` (`VendorPower`), chosen by `CLOUDGPU_POWER`: `none` (an always-on box, a pass, or a vendor with no API: nothing is
powered, the package only waits for `/health`) or a vendor's name. **autodl is built in and is the default**: with `CLOUDGPU_INSTANCE` and `AUTODL_DEV_TOKEN` it powers the
box on and off; without them it is "not configured" and the first call says what to set, or to set `CLOUDGPU_POWER=none`. `power_off_if_idle` powers off only when the adapter says
`busy` is 0 (a second process using the box is never cut off) and only when the account says the box is billed `payg` (hourly); a pass, or a word it does not know, is left on.
`ensure_up` powers on only a box the account says is `shutdown` or `off` (a box that is `starting`, or whose state is unknown, is waited for), and with `CLOUDGPU_AUTOOFF` the exit hook is
registered the moment this process has sent the power-on, so a timeout leaves nothing running; the hook never powers off a box this process did not power on. Anything unknown is "do not power off": a `/health` that does not answer (tunnel down, adapter restarting), a worker that
is not alive, an account call that fails or does not list the instance.

### Who may power a box

`Box.ensure_up()` powers a box on **without asking a person**. That is the design, not an oversight: who may power boxes on is decided by whoever holds the account's token
(`AUTODL_DEV_TOKEN`) and `CLOUDGPU_INSTANCE`, the operator team, and not by an agent or a customer. A process that must not be able to start billing is simply not given those two.
`CLOUDGPU_AUTOOFF=1` powers the box off again at exit, but only when this process was the one that powered it on (a box that was already on is left to whoever started it).
The `--yes` gate is on the CLI (`instances power-on`, `power-off`, `create`, `release`; an agent inside a person's allowance: "An allowance for an agent", below): it is **a person's convention, not something enforced** (there is no tty check, any process can
pass `--yes`), so the rule "a person gives `--yes`" is kept by whoever runs the command, and by not giving a process the account token. `bench` is a paid take and has no gate at all.

**autodl** covers every call of the account API ([docs](https://www.autodl.art/docs/app_api/): create, snapshot, status, list, power_on, power_off, release; the card spec ids of
its appendix, plus `6000d`, seen on a real instance). Checked against the live API on 2026-10-23: a snapshot's `payg_price` 6980 is ¥6.98 an hour (the console's price: the unit is
0.001 ¥/h); a day pass's snapshot says `payg_price` 0, so a pass's price is the console's, not the API's; `status`, `snapshot` and `list` take the query string (the body form the page
shows answers `RequestParameterIsWrong`). A create request is checked offline against the documented limits (1 to 4 cards, 0 to 500 GB of extra system disk, a known spec id, the
required fields, no stray field) before anything is sent.

```
python3 -m cloudgpu instances list | status ID | snapshot ID | specs      # reading: free, no secret printed
python3 -m cloudgpu instances power-on ID | power-off ID | release ID      # a dry run that says what it would do, until --yes
python3 -m cloudgpu instances create --spec request.json                   # the same: a dry run until --yes
```

Everything that changes the account is a dry run until `--yes`, which only a person gives (by convention; see "Who may power a box"): **`create` starts billing, `release` destroys the instance's data** (and it refuses an
instance that is not shut down), power-on starts the hourly billing. The provider's own power-on before an approved take calls the library, not the CLI.

**To take in a new cloud** (the plan: one at a time as they are bought): write `cloudgpu/vendors/<name>.py` with a `Vendor` subclass: its `token_env`, the calls it documents (a call it does not have
stays "no such call": said, never guessed), the normalising of its answers into `Instance` and the snapshot dict, and `create_problems` from its documented limits; add it to `registry()`;
test it offline against the documented answers (copy `tests/test_vendors.py`). The power backend, the CLI, the status check and the skill then work for it with no other change.
A vendor with no API at all needs nothing: `CLOUDGPU_POWER=none`. A call never runs a command taken from a file or the environment.

### Free cards now: `instances stock` (0.17.0)

autodl's developer API has no call for free cards, so a power-on is the only way to find out, and an afternoon of refusals is common (2026-11-03: the same instance, every try, `InternalError 当前算力规格暂无库存`
and `ServerBusy 服务正忙` alternating). The public market page (https://www.autodl.art/app/market) shows `(free/total)` on each card; `instances stock` reads the call behind it:

```
python3 -m cloudgpu instances stock                          # every card in every data center: free and total
python3 -m cloudgpu instances stock --data-center westDC3 --spec v-48g --spec pro6000-p
```

It prints one JSON document (`any_idle`, `rows` of `data_center`, `region`, `spec` (the id `instances specs` and a create request use; `null` for a card this package does not know), `card`, `idle`, `total`, and a note) and
exits **0 when a card is free, 3 when none is, 1 on an error**, so it goes before a power-on: `instances stock --spec v-48g --data-center westDC3 && instances power-on ID --yes`.
Read-only, and it needs no token and sends none (the developer token is refused on that call). Limits, said plainly: the call is the market page's own and not documented, so it can change or be limited (keep to one read a
half minute); a free card now does not make a power-on succeed; whether an existing instance is tied to its own machine (so a free card elsewhere does not help it) is **not verified**, but there is one sample for it (2026-11-03: the page showed 1 free PRO6000 in the northwest, and a power-on of an existing PRO6000 instance in that data center was refused at once with 暂无库存); and every number seen
so far was 0 (2026-11-03), so that the anonymous count equals what a logged-in page shows is not verified either. `ServerBusy 服务正忙` now counts as no stock (`ensure` retries it, as it does `暂无库存`).

### A hard cap on how long a box stays on: `--max-minutes` (0.19.0)

A box left on bills by the hour, and "shut it down when done" is the step that gets forgotten (it was written by hand by the remote agent on 2026-11-03, after it had to). `instances power-on ID --yes --max-minutes N` and
`instances ensure ... --yes --max-minutes N` start a **guard** in the background for each box they switch on or create: after N minutes by the wall clock (a laptop that slept still acts when it wakes) it asks the vendor to shut the box down
and reads the status until the vendor itself says `shutdown` (a refused request, `ServerBusy` included, is asked again a few times). The call prints the guard's `pid`, `off_at` and `log`; the log's last line is one JSON document, `off: true` only when the vendor's status says so.
`instances guard ID --minutes N [--agent]` is the same in the foreground. Without `--yes` the dry run says `would_guard`. A cap is a number above 0 and at most a day. The guard never releases anything, and an `--agent` guard works only on a box the ledger shows this agent created.

Limits, said plainly: the guard is a process on this machine, so a machine that is switched off or a guard that is killed shuts nothing (the vendor's balance is the other stop); it shuts the box down even if a take is running (a hard cap is that); and it was
tested offline against a fake vendor, **not yet against a real box**: the first real use should be watched.

### Passes: the base, and the warning before one ends (0.16.0)

The base capacity is **passes** (daily, weekly or monthly boxes) that **a person buys and renews in the vendor's console**; this package cannot do either (autodl's API only creates pay-as-you-go
instances). Boxes an agent makes through the API are hourly, a temporary addition. The whole operating model, and who does what, is in `docs/how-to-run-boxes.md`.

What the package does for a pass is notice and name it (`cloudgpu/expiry.py`). For every instance billed `daily`, `weekly` or `monthly` it works out the time left from the account's `expired_at`
(read with its own offset, not the machine's) and gives a state:

| state | level | meaning | who |
|---|---|---|---|
| `expired` | error | the pass has ended | a person renews it in the console; powering on will not help |
| `expiring` | warn | ends in less than `CLOUDGPU_PASS_WARN_DAYS` (default **3**; fractions allowed; a bad value is 3) | a person renews it before then |
| `no_end` | warn | the account gave no end time, so it cannot be watched | a person looks in the console |
| `off` | info | switched off, the pass runs on | power it on (`instances power-on ID --yes`, or `ensure` does it); nothing to renew |
| `active` | ok | nothing to say | |

A pay-as-you-go instance never has a state: it has no end. Each message names the instance (id and name), the end time, and who acts. Where it shows:

- `python3 -m cloudgpu instances list` still prints the same JSON list, with a `pass` object on each pass; the warnings are lines on stderr (`WARN: ...`, `ERROR: ...`). `--warn-days N` overrides the setting.
- `python3 -m cloudgpu instances status ID` adds the same `pass` for that instance.
- `python3 -m cloudgpu instances expiry [ID ...] [--warn-days N]` is the check for a script or a cron: every pass of the account (or the ones named), `level`, `alerts`; **exit 0** well (or only `off`),
  **1** one ends soon, **2** one has ended.
- `python3 -m cloudgpu status` (the box wired in the settings) puts the message in `next` and the whole row in `account.expiry`.
- `instances ensure` puts a message in `next` for every wired pass that ends soon or has ended, and the rows in `passes`. A pass that is off and has not ended is powered on before any hourly box
  (it costs nothing more); one that has ended or gives no end time is left to a person.
- A consumer's health check calls `expiry.account_rows(vendor, env, only=wired_ids)` and shows each row's `message` (the `level` says error or warning); `only` keeps a pass nobody uses from nagging.

Nothing here sends a message: whoever runs `status`, `ensure`, `instances expiry` (a person, or an agent that tells a person) sees the line. A cron that runs `instances expiry` and tells the account
holder on a non-zero exit is the way to be told without asking.

### A line across machines (0.7.0)

Callers on different machines cannot see each other's tickets, so each used to see a free box and send; the second waited inside the adapter's queue, unable to go to another box. The box now keeps the line
itself: `POST /cloudgpu/line {"label"}` takes a place (`state` "go" for the front one, else "waiting" with `ahead` and the label of who `running`), `GET /cloudgpu/line/ID` asks where it is, `DELETE` leaves,
and `GET /cloudgpu/line` lists who runs and who waits (token-checked, like every call but `/health`; ids are not listed). A ticket is dropped when nobody has asked about it for 60 s, so a machine that died
never blocks the line, and while any ticket is live `/health` counts the box as busy (it is not powered off or restarted under a caller that waits).

`Pool.take` does this by itself: once a take is the one for a free box on its own machine and the box is up, it takes a ticket on the box and goes when the box says "go"; a thread asks about the ticket
every 20 s while the take runs and the ticket is given up at the end. With several boxes a take that waits on one box's line tries the next, keeps its place where it waits, and gives up the others when it
goes. It says `waiting for its turn on BOX: 2 take(s) of other machines ahead (host/take-3 is running)` once on stderr (or through `on_wait`), and `python3 -m cloudgpu status` adds `line` (state, label, age) from the box.
**Limits:** the line orders callers that use it. An adapter older than 0.7.0 (reinstall it with `bundle` and `install.sh`), a line that does not answer, or a caller that sends a take without a ticket is not stopped
and not waited for: the take goes as before, and the adapter's own queue still serves it in order of arrival. The line is in the adapter's memory: an adapter restart forgets it (a restart already fails every
waiting take), and a caller whose ticket was forgotten joins again at the back. A ticket's label (this machine's name and the take's folder name) is sent to the box. A hold (above) is still per machine. Out of scope, as before:
autoscaling, or powering more boxes on to shorten a line.

## Giving the card's memory back after a take (0.13.0)

The picture models live in the ComfyUI process, not in the adapter (which is standard library only and has no torch), and ComfyUI keeps them on the card after a take: an idle box measured
46735 of 49140 MiB used, so a second model (the VoxCPM voice needs about 6 GB) could not load next to it. So after a take that leaves nothing accepted or queued, the adapter asks ComfyUI to unload its models
and free its memory (`POST /free` with `unload_models` and `free_memory`, the call the IndexTTS2 voice already makes before it runs). It is best effort: ComfyUI down, an older ComfyUI or a timeout never fails a take.
A take that is accepted or running holds it back, so a queue of takes frees once, after the last one. It is done after a failed take too (an out-of-memory take leaves the most behind).

**The cost:** the next take loads its models again (the H3 weights are several GB: seconds to a minute or more, measure it on your box). **Turn it off** (`CLOUDGPU_FREE_AFTER_TAKE=0` in the environment of
`start.sh`, or `--free-after-take 0`) on a box that only makes video and nothing else uses the card, or where takes come minutes apart in bursts you would rather keep warm. Default on.
A caller that wants the memory now (before it starts another model on the card) can also ask: `POST /cloudgpu/free` (token-checked, like every call but `/health`) answers `{"data":{"freed":bool}}`; `freed` is false
when a take is accepted or running, or ComfyUI did not answer. Callers can ignore the answer; asking works whatever the setting is. `vram_free_mib` in `/health` shows the effect. A caller waiting in the box's line
does not hold the free back (it has no take yet): its take loads the models again. **Not measured:** how much of the 46 GB ComfyUI gives back on a real box (`/free` also depends on the ComfyUI version).

## The audio-locked take (0.20.0)

**The problem.** In `minimax_h3_image_audio_to_video` the audio is only a reference: the model decodes a speech from its own latent. Measured by the operator's agent on a real 4090 box (2026-11-03; the same still, the same voice
audio, one scene, two runs each; not by this package, which cannot run a box): the sample correlation of the video's audio with the audio sent was 0.23 to 0.39, the transcript fidelity 0.72 to 0.77 with the best prompt and 0.00 to 0.17 with the old
default, and when 1.2 s of silence was put into the audio the model kept speaking and kept the mouth open. A lip-synced ad needs the lips to follow the audio it was given.

**The workflow.** `minimax_h3_image_audio_locked` has the unlocked workflow's fields and validation (`ref_image_0` and, for the other two locks, `ref_image_1`..`ref_image_8`, `ref_audio_0`, `audio_duration` 1..15, `resolution`, `seed`) and `lock`. It builds
the same model nodes from the same `models.json` names and the same size rules, and anchors the take's own audio. In every lock the audio is also the reference `<Audio 1>` (`ref_audios.ref_audio_0`), as the nodes' own descriptions ask.
**Words:** with `official` (the default) the request carries `line`, the words spoken in the audio (one plain line, at most 400 characters, no line break or other invisible or control character, no `<` `>`, no double-quote marks of any width, `«»` and `「」` included), and optionally `line_language` (default `Chinese`); the adapter builds the whole prompt (below) and `prompt` is refused
(`lock_prompt_unused`), because a prompt that is silently dropped is worse than an error. With `addguide` and `audiodrive` the prompt is the request's, word for word (no `prompt`: the unlocked workflow's default, which has no speech in it) and `line` is refused (`lock_line_unused`).

| `lock` | the nodes | where they come from | what the operator measured (4090, 2026-11-03) |
|---|---|---|---|
| `official` (default, 0.21.0) | `LTXVConcatAVLatent`, `VAEEncodeAudio`, `SetLatentNoiseMask`, `SolidMask`, `ImageFromBatch` | ComfyUI itself, core nodes only; needs a ComfyUI new enough for the audio-video latent (v0.35.0 or newer, by the design's reading of ComfyUI's source) | "E3", below: captions in 1 of 9 takes, audio correlation 0.999 to 1.000, silent gaps keep the mouth closed |
| `addguide` | `MiniMaxH3AddGuide` | ComfyUI itself (`comfy_extras.nodes_minimax_h3`), an official node | locked the audio (correlation 0.935); drew captions in 5 of 7 takes |
| `audiodrive` | `VRGDG_MiniMaxH3AudioDrive` | the third-party custom-node pack `comfyui-vrgamedevgirl`; **off by default, licence-restricted (below)** | correlation 0.945, fidelity 0.90 to 0.95; in the silence test the audio was true silence and the mouth closed; same duration, about 3 s more; no captions |

`addguide` and `audiodrive` stay for a caller that asks for them by name (`audiodrive` also needs the operator's switch); nothing asks for them by default, and `lock: custom`, which was reserved in 0.20.0 for this official lock, is gone (`lock_unknown`).

**The official lock, "E3" (0.21.0).** The graph is the operator's tested one (a data file, `tests/e3_lockaudio_api.json`, never run by this package, its spoken line replaced by a neutral sentence; a test holds the adapter's graph equal to it node for node, modulo the box's file names, the seed and the take's fields):
- The audio file is read once (`LoadAudio`) and used three times: encoded by the audio VAE into an audio latent (`VAEEncodeAudio`); as the reference `<Audio 1>` of the reference node (`ref_audios.ref_audio_0`, with `audio_vae`); and, untouched, as the audio of the finished video (`CreateVideo`). The audio is never decoded.
- The audio latent gets an all-zero mask (`SolidMask` 0.0 into `SetLatentNoiseMask`; 0 means keep) and `LTXVConcatAVLatent` puts it in place of the audio half of the reference node's latent. The mask is on the audio latent **only**: a mask on the whole audio-video latent would freeze the video. The sampler (`res_multistep`, `simple`, the turbo lora's steps, denoise 1.0, one seed) regenerates the video and keeps the audio as it is.
- `length` of the reference node is `17*ceil((frames-5)/17)+5` with at least 124 (the model's trained range is 124 to 362, 5.2 to 15.1 s), where `frames = round(audio_duration*24)`; the video is cut to `frames` (`ImageFromBatch`) and the adapter's usual final cut to `audio_duration` follows. So `audio_duration` must be the length of the audio sent; the width and height are the request's `resolution` (the measurements are 768x1344).
- The prompt is three lines, the second and third fixed:
  1. `[Shot 1] <Subject 1> (S1) whose appearance comes from <Picture 1> looks into the lens and says, <d>[Chinese] LINE</d>`
  2. `<Audio 1>: fully_copy - the original audio of <Audio 1> is the complete final track; <Subject 1> (S1) lip-syncs exactly to it.`
  3. `No captions, subtitles, logos or on-screen text appear.`
- Model file names are not in the lock code: the loaders of the finished graph are used as they are, so the names, the steps and the seed come from the box's `models.json` (`detect_models.py` prefers the `ref2v` turbo 4-step lora and, on a card that is not Blackwell, takes the int8 text encoder; the 4090 preset `models.4090.json` has exactly the names of the operator's run. An adapter started with no `models.json` uses the built-in PRO6000 names, whose lora is the `fl2v` one, which is not what E3 ran with).
- The nodes are found by name in ComfyUI's `/object_info` (all five, each with the inputs the graph wires); one missing or changed is `lock_node_missing` / `lock_node_incompatible` naming that node. The adapter does not read ComfyUI's version: to check a box by hand, `/system_stats` should say v0.35.0 or newer and `/object_info/LTXVConcatAVLatent` should describe the MiniMax H3 audio-video latent.
- One reference picture only (`lock_one_picture` for more): the run had one and the prompt names only `<Picture 1>`.

**What the operator measured for E3** (a 4090, 48 GB; relayed, not run by this package): captions drawn in 1 of 9 takes; the correlation of the video's audio with the audio sent 0.999 to 1.000; in the silence test the mouth stayed closed in the gaps; about 97 s for 4.8 s of audio and about 165 s for 7 s; peak memory about 48 GB on that 48 GB card (so do not plan a second model on the card during a take).
**A lesson that is written down so nobody repeats it:** the working hypothesis "the captions come from the dialogue text in the prompt" was **disproved**: a run without the line text in the prompt (variant A) still drew captions in 1 of 4 takes. Leaving the words out did not keep the screen clean, so the line stays in the prompt. What did stop the captions is not established: E3 differs from `addguide` in more than one thing (the audio kept as a latent besides being a reference, the three-line prompt), and no run isolates one of them. The hosted workflow this API imitates has no prompt at all, so there is no hosted wording to copy. The third line is a request, not a guarantee (1 take in 9 still had captions): a caller that needs none still looks at the take.

How the two older graphs differ from the unlocked one (the unlocked graph is not touched, a test holds it equal to the 0.19.0 graphs):
- `addguide`: one node, `MiniMaxH3AddGuide(positive = the reference node's conditioning, latent = its latent, frame_idx = 0, vae, audio_vae, audio = the take's audio)`; the guider takes the new conditioning. The sampler's latent and the decode of the audio from the sampled latent are as before.
  The node's definition: required `positive`, `latent`, `frame_idx` (default 0, a negative one counts from the tail); optional `vae`, `audio_vae`, `image`, `audio` (anchored from `frame_idx`, trimmed to the remaining duration); one output, the conditioning. No `image` is given: the picture is the reference node's.
- `audiodrive`: one node, `VRGDG_MiniMaxH3AudioDrive(av_latent = the reference node's latent, source_audio = the take's audio, audio_vae)`; the sampler starts from its first output, the video's audio is its second, and the decode of the audio is gone. This is the graph the operator measured, and a test holds it equal to that graph (modulo model names).

**No silent fallback.** A take of this workflow never runs any other graph than its lock's. The coded errors (HTTP 400 `RequestParameterIsWrong`, the words in `msg` starting with the code, before anything is fetched or queued; a take accepted while ComfyUI was
still starting and then found without its node is `FAILED` with the same words, never run as another graph):

| code | when |
|---|---|
| `lock_not_allowed` | `lock: audiodrive` and the operator has not switched it on (the licence words, below); also when ComfyUI is down or has the node |
| `lock_node_missing` | ComfyUI's `/object_info` has no node of that name (for `official` any of its five): the message names the node and, for audiodrive, the pack |
| `lock_node_incompatible` | the node is there but its definition lacks an input the graph wires (a changed version): the message names the input |
| `lock_state_unknown` | ComfyUI did not answer `/object_info` when the take ran |
| `lock_line_required` | `lock: official` (the default) and `line` is missing or empty |
| `lock_line_invalid` | `line` has a line break, an invisible or control character, `<` `>`, a double-quote mark or is over 400 characters, or `line_language` is not a language name in letters (single spaces or hyphens between words) |
| `lock_prompt_unused` | `lock: official` and a `prompt` was sent (the adapter builds the prompt from `line`) |
| `lock_line_unused` | `line` or `line_language` sent with `addguide` or `audiodrive` (they take `prompt`) |
| `lock_one_picture` | `lock: official` and `ref_image_1` or more |
| `lock_unknown` | any other value of `lock` (`custom` included) |
| `lock_wrong_workflow` | `lock`, `line` or `line_language` sent to a workflow that does not lock the audio (the unlocked one, Mode A, text to video, the voice): the caller asked for an anchored audio and would get an unanchored one |

**What `/health` says** (no token). The nodes are looked for by name in ComfyUI's `/object_info` when the adapter starts and again when `/health` asks (at most 30 s old); a take of a lock reads them afresh twice, when it is accepted and when it runs, so neither a refusal nor an acceptance is stale. While ComfyUI is down or silent nothing is known and no lock is offered.
```
"audio_lock": ["official", "addguide", "audiodrive"],   the lock values served right now: [] when none is
"audio_lock_detail": {
  "checked": true,                                false: ComfyUI did not answer /object_info
  "official":   {"node": "LTXVConcatAVLatent", "present": true, "usable": true, "module": "comfy_extras.nodes_lt", "problem": null,
                 "needs": ["LTXVConcatAVLatent", "VAEEncodeAudio", "SetLatentNoiseMask", "SolidMask", "ImageFromBatch"]},
  "addguide":   {"node": "MiniMaxH3AddGuide", "present": true, "usable": true, "module": "comfy_extras.nodes_minimax_h3", "problem": null},
  "audiodrive": {"node": "VRGDG_MiniMaxH3AudioDrive", "present": true, "usable": true, "allowed": true,
                 "status": "audiodrive (third-party, licence-restricted, enabled)",
                 "pack": "comfyui-vrgamedevgirl", "commit": "<40 hex>"|null, "version": "<pyproject version>"|null,
                 "licence": "VRGameDevGirl Community and Commercial License v1.0", "problem": null}}
```
On a bind other than loopback (`CLOUDGPU_BIND`, a vendor's public port) `commit` and `version` are null: `/health` answers anyone there. `present` is null when nothing is known; `problem` is the coded words a take of that lock would get (null when it can run); `status` of audiodrive is `audiodrive (third-party, licence-restricted, off: CLOUDGPU_ALLOW_NODE_VRGDG is not set)` until the operator switches it on.
`pack` is the custom-node folder ComfyUI reports for the node (`python_module`); `commit` is read from that folder's `.git` (no git is run), `version` from its `pyproject.toml`; both are null when they cannot be read. `python3 -m cloudgpu status` shows `box.audio_lock` (null: an adapter older than 0.20.0).
`install.sh` and `detect_models.py` (`--nodes-only`) report whether the pack is in `custom_nodes` and its commit; they never install or fetch it.

**The third-party pack and its licence: the owner decides.** What is recorded, and what is not:
- *The operator's reading* (relayed to this change; **the licence text was not read here**): `comfyui-vrgamedevgirl` is under the "VRGameDevGirl Community and Commercial License v1.0", not an open licence. Using it ourselves to make a client's ads is allowed (its section 3).
  Bundling its code into a paid product or service, offering an API or workflow endpoint that triggers it to third parties, or white-labelling or reselling needs a paid commercial licence (section 4). This package, a box image that carries the pack, and clients who run adcut against a box
  that triggers it likely fall under section 4. The licence allows independent implementations, which is what `lock: official` is (0.21.0). This is not legal advice.
- *What this package does about it*: it contains none of the pack's code (the graph refers to the node by name and nothing more), vendors, bundles and installs nothing of it, and serves the lock **only** when the box's operator sets `CLOUDGPU_ALLOW_NODE_VRGDG=1` for the adapter. Otherwise the take is refused with
  `lock_not_allowed`, which names the licence. Switching it on is the owner's decision after the licence question is settled with the author; the adapter does not decide it, and no consumer should set the variable on a box that serves anyone but the owner.
- *Pin it*: the pack is third-party code that runs inside ComfyUI with the box's rights. Install it on a box image once, at a commit someone looked at, note the commit that `/health` reports (`audio_lock_detail.audiodrive.commit`), save that box image, and never update the pack on a running box. The adapter reports the commit; it does not compare it with a pinned one.
- *Could not be verified here*: the licence text; the pack's source address, owner and history (the repository was deliberately not opened); the node's code and what exactly it does (only the operator's measurement and the input names of the operator's graph are known); whether a given use is a section 4 use; that the pack is safe to run.

**Not measured, not done.** No box was run for this change (0.20.0 or 0.21.0): every claim above about a real card is the operator's. The `official` graph is the operator's tested file, the adapter builds it in code and a test compares the two; what was run is one picture at 768x1344 with the 4090's model set, up to 7 s. Not tried: `official` at 480p or landscape, at 15 s (the top of the range), with the nvfp4 / PRO6000 model set, or on the `addguide` graph wired here from the node's definition (its audio is taken from the sampled latent as in the unlocked graph); either older lock with extra reference pictures. There are no price rows for the new workflow (`price` has none until `runs.jsonl` has a warm run of it; the operator saw about 3 s more per take with audiodrive,
which is not a measured row); `bench` does not take the new workflow yet.

## Landscape (0.15.0)

`480p横` (864x480) and `768p横` (1344x768) are the vertical sizes with the sides swapped, accepted by all three video workflows. A consumer measured `minimax_h3_lightx2v` at `768p横` on a rented
box: 1344x768, 24 fps, 5 s, about 145 s on the box (a vertical 768p take took about 143 s). `480p横` has no recorded run; the price rows come from recorded runs only, so landscape gets one once a run
is recorded (`runs.jsonl` names the resolution `480p`/`768p` for both orientations: same pixel count, same cost). The audio-driven default prompt says "wide screen video" instead of "vertical phone video" for a landscape size.
A box whose adapter was edited by hand to add these two sizes is overwritten by this version: nothing else is needed.

## How much a box works: `util` (0.14.0)

A consumer's own wall clock cannot say whether the box is the bottleneck: a take that took 143 s on the consumer's side may have been 40 s on the box, with a new one reaching it only every few minutes.
`python3 -m cloudgpu util [--since TIME] [BOX...]` reads each box's task records and prints, per box and workflow, the number of takes and how many failed, the time from creation to done (median and
slowest, of the takes that went well), the median and mean gap between consecutive creations, the busy share (the part of the span, first creation to last finish, in which a take was running on the
box), and the takes of the last 30 minutes. `TIME` is `90m`, `2h`, `1d`, a local date or date and time (`2026-10-29T14:30`) or seconds since 1970, so a change on the consumer's side can be judged by
running it before and after. `BOX` is a name in `CLOUDGPU_POOL` (`default` for a pool of one); none means all. The JSON is on stdout (one row per box and workflow, or a row with an `error` for a box that
does not answer); one line a person can read per row goes to stderr. It only reads: it powers nothing on, opens no tunnel and never touches a running take.

The records come from the adapter's new `GET /cloudgpu/tasks[?since=EPOCH]` (token-checked, read only; there was no way to list them before). It gives `workflow`, `status`, `created`, `started`, `done` of
each record and no task id. The adapter keeps a record for 24 h, so a report reaches back a day at most, and the adapter has to be installed on the box again (`bundle`, `install.sh`) for this: against an
older one `util` says so. The statistics are `cloudgpu.util.report(records, now, since)`, a pure function.

## An allowance for an agent: renting its own boxes (0.11.0)

By default an agent rents nothing: `create` and `release` are a person's, as above. A person can give an agent a limit to work inside, and then the agent may keep boxes up by itself.

```
python3 -m cloudgpu quota set --by NAME --max-instances 2 --max-cny-per-day 100 --price v-48g=3.2 --price pro6000-p=7   # a person; --by and the time are recorded
python3 -m cloudgpu quota show                                                                                        # free: the numbers, who set them, what is spent today
python3 -m cloudgpu instances create --agent --spec request.json [--yes]    # inside the allowance, or refused with the reason; a dry run says which
python3 -m cloudgpu instances release --agent ID [--yes]                    # only an instance this agent created, shut down, not wired in the settings
python3 -m cloudgpu instances ensure --count 2 --spec template.json [--specs v-48g,pro6000-p] [--data-centers westDC3,beijingDC2] [--patience 600] [--yes]
```

- **The allowance** is a file (`~/.cloudgpu/quota.json`, `CLOUDGPU_QUOTA_FILE`, beside it the ledger, `CLOUDGPU_STATE_DIR`). No file, or a 0, means the agent creates nothing. Numbers: how many
  instances the agent has made at once (`max_instances`), the CNY it may spend in a day (`max_cny_per_day`), and the price per hour the person assumes for each card spec (`--price`; a spec with
  no price is not created, because what it would spend is not known). Today's spend is **an upper bound** from the ledger: every instance the agent made counts at its price from its creation
  (or midnight) to its release, as if it ran all that time, and each new one has `reserve_hours` (default 2) of its price set aside first. The real price (the snapshot's) goes in the ledger
  when it is known; if it is above the assumed one the result says so.
- **The ledger** (`ledger.jsonl`, append only): time, action (`create`, `release`, `power-on`, `quota-set`, `gone`), instance, card spec, data center, price per hour, who triggered it
  (`CLOUDGPU_ACTOR`, else `agent` or `person`). The attempt is written before the vendor is called, so a ledger that cannot be written stops a create instead of leaving a bill with no record.
- **Release** by the agent needs three things: the ledger shows this agent created it (a box a person made is never released by the agent), it is shut down, and no setting wires it to a box
  (`CLOUDGPU_INSTANCE`, `CLOUDGPU_<NAME>_INSTANCE`: a take may be running there).
- **A create whose answer was lost** (a timeout after the request went out) is kept in the ledger as an attempt with no outcome and counts as one instance and its reserve until a person looks
  (`quota show` → `create_outcomes_unknown`; `instances list`). Creates and releases take a lock on the state directory, so two runs cannot both pass the same check. An instance the account
  does not list is written `gone` only 15 minutes after it was created (a listing can lag). `power-on` and `power-off` with `--agent` touch only the agent's own boxes (`power-on` also the
  ones wired in the settings, which are the operator's and not counted against the allowance: `ensure_up` powers those on by design).
- **`ensure --count N`** (the verb "have N boxes up"): lists the account, counts the boxes this setup knows (the ones wired in the settings plus the ledger's) that are running, powers on the ones that
  are off and billed by the hour, or are a pass that has not ended (a pass that has ended is reported for a person to renew, not touched), then creates what is still missing, inside the allowance, from the template
  for each card spec in order (default `v-48g`, the 4090 48G; `CLOUDGPU_ENSURE_SPECS`) at each data center in order (default `westDC3,beijingDC2`; `CLOUDGPU_ENSURE_DATA_CENTERS`).
  **Two steps, the second made by the first's page (0.18.0).** Step 1 is `instances stock` (above): every card and data center with the free count, from which the agent chooses a range and puts it in order. Step 2 is this
  verb with that order (`--specs`, `--data-centers`: the order is the preference, spec first, then data center). It reads the same page, makes a create only for the first spec and place in the order that shows a free card, **at once** (no second
  decision), and when none does it reads the page again every `--poll` seconds (default 30) until `--patience`, then takes the first card that appears. Nothing is asked of a card the page shows as none, so a wait leaves no
  failed create in the vendor's records and none in the ledger. The allowance still stands: the create is refused unless the quota covers it (a card price in `quota set --price`), which is what limits the spend. Without `--yes` the plan
  carries each place's `free` count. **Only where the vendor has no such page, or it cannot be read** (the call is undocumented), the old way is used: each place is tried blind and again after 20, 40, 80, 160, 300 seconds
  (at most `--patience` in all). A spec the page does not show at all is tried, not skipped. A power-on of a box that already exists is not gated (it may be tied to its own machine). Every other failure (no balance, a refused request, the allowance) is
  reported once and not retried. The AutoDL API page lists no error codes, so "no stock" is read from the words of the refusal (stock, insufficient, no available, 库存 ... but not balance or
  token): **check the first real "no card" answer and adjust `NO_STOCK_WORDS` in `vendors/autodl.py`**. The report: `running`, `powered_on`, `created`, `missing`, `failures` (what, where, kind,
  reason, tries), `now_running` and each new box's SSH door and price. Exit 0 when nothing is missing, 1 otherwise. Without `--yes` it prints the plan and the allowance's verdict and changes nothing.
- **What this does not stop.** Like `--yes`, `--agent` is a convention: a process that has the account token can call the vendor, or edit the quota file, itself. The allowance is a limit
  only when the agent cannot change the file (another user's file, mode 0644; a file others can write is not believed) and the account token is not the only thing between it and the money.
  The vendor's own balance is the hard stop: keep it as small as the allowance.

### A new box: what is automatic and what is not

A created instance is only the image. What `ensure` and `create --agent` do for it: create from the image with the start command `/root/h3host/start.sh` (so an adapter that is already in the image
starts by itself), wait until the account says `running`, read the SSH door and the price, write the ledger. What stays with a person, and why:

| step | automatic? | why |
|---|---|---|
| rent from an image or snapshot, start the adapter at boot | yes, if the image already has the adapter and `~/h3host/start.sh` | `start_command` on create |
| wait for `running`, read SSH door, price | yes | the account's status and snapshot (never a password) |
| put a key on the box, run `bundle` + `install.sh` | **no** | a new box has only the password the API deliberately does not return; a key must already be in the image |
| the adapter token (`CLOUDGPU_TOKEN`) | **no** | a secret: a person pipes it |
| wire it to the settings (`CLOUDGPU_<NAME>_INSTANCE`, `_SSH`, `_BASE`, `_TOKEN`) | **no** | the settings are the operator's |
| the voice service (VoxCPM, port 8190) | **no** | adcut's `docs/voxcpm.md`; not copied here |
| `python3 -m cloudgpu status` says `ready`, one measured take, a price confirmed | **no** | a paid take and a price are a person's |

The way to make the first row cover the rest: set one box up once by hand, put the restricted key in its `authorized_keys`, install the adapter and the voice service, save it as a private image,
and use that image as `application_uuid` in the template (whether AutoDL gives a saved image an application uuid that `create` takes has **not** been checked here: ask in the console before relying on it).
A box made from it then needs only the settings wired and `status` run.

## Safety

- The adapter checks the token on every call but `/health` (an empty token is refused at start and never matches); result files are at an unguessable name for one hour (after that the task reads `FAILED`, "result expired"). It listens on
  127.0.0.1 unless `CLOUDGPU_BIND` says otherwise, drops a client that stalls for 60 s, and fetches an input URL only from a **global** address (`ip.is_global`: not loopback, private,
  link-local, `100.64.0.0/10` where some clouds put their metadata address, or reserved), resolved once and connected to at that address, so a name cannot answer differently the second
  time; no redirect, no proxy, and every failure to fetch reads the same, so the box is no port scanner. `data:` URIs are taken.
- A token is never in an exception or a message of this package, and goes only to the host it was configured for: a redirect to another host or port is refused (urllib would forward
  the `Authorization` header), `http://` is accepted only for `127.0.0.1`, `::1` and `localhost`, and `bench` opens only an `https://` or local `http://` result URL, never `file://`,
  and keeps only letters, digits, `.`, `-`, `_` of NAME in the file name.
- **A vendor's public proxy usually has no login of its own.** A ComfyUI or a panel on a public port is open to anyone who has the address, and ComfyUI can run code through its
  nodes. Keep ComfyUI on 127.0.0.1 and put only the adapter on a public port, or use the tunnel (the restricted key reaches nothing else).
- Whatever a take sends (a person's still, a voice sample) goes to that box: tell the person, and keep the consent that covers it where your project keeps it.
- The tunnel is by key, never a password: nothing here accepts or stores one.
- **A third-party node is code the box runs, and may be licence-restricted.** The only one this package refers to is `VRGDG_MiniMaxH3AudioDrive` (the audio-locked workflow's `audiodrive` lock): off unless the operator sets `CLOUDGPU_ALLOW_NODE_VRGDG=1`, never installed or fetched by this package, to be pinned to a commit and kept in the box image ("The audio-locked take").

## Measure and price a new box

1. `python3 -m cloudgpu bench ENVFILE NAME WORKFLOW RES SECONDS IMAGE [AUDIO]` submits one take (**a paid take**: the box's hours, and it powers nothing on by itself; there is no `--yes`
   and no tty check, so only a person runs it) and prints the wall time; run each take twice (the first loads the models). The machine's own time is in the box's `~/h3host/tasks/*.json` (`started`, `finished`).
2. Append the warm runs to `cloudgpu/data/runs.jsonl` (box, card, `hourly_cny`, workflow, resolution, seconds, `machine_s`; `cold: true` on the first). Never edit a run.
3. `python3 -m cloudgpu price --box ID [--format tsv] [--provider NAME]` prints the rows (the worst warm run of each workflow and resolution, at full use). The package writes
   nothing: your project turns them into its price table, as pending until a person confirms.
