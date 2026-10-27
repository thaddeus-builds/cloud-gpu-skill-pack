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
AUTODL_DEV_TOKEN=…          the autodl backend: the account's developer token (console -> account -> settings), not the workflow token
```

The older `H3HOST_<NAME>` names are read second, and only when the new name is absent: a key that is present wins even when empty (`CLOUDGPU_AUTOOFF=` is off, it does not fall through
to `H3HOST_AUTOOFF`). Values are stripped of surrounding whitespace (a token read from a file keeps no newline). A consumer loads its own `.env` into the environment, or passes a
mapping: `Box(env)`; `python3 -m cloudgpu status --env-file FILE` and `bench` read a file and warn when other users can read it (chmod 600).
On the box, `start.sh` also takes `CLOUDGPU_BIND` (default 127.0.0.1), `CLOUDGPU_PUBLIC_BASE`, and leaves an adapter that has a take on it running (it restarts only at `busy` 0).

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

Workflows (the names autodl publishes): `minimax_h3_image_audio_to_video` (`ref_image_0`, `ref_audio_0`, `audio_duration` 1..15, `resolution`) and `minimax_h3_lightx2v` (Mode A:
`prompt`, `first_frame`, `last_frame`, `duration` 1..15, `resolution`, `seed`). `resolution` is `480p竖` or `768p竖` (1080p was never run on these cards). Inputs are http(s) URLs or
`data:` URIs. The result URL is `/files/<random>.mp4` (relative when the box has no public address: join it to the base you reached it at) and lives one hour: download at once.
A body the box will not run is HTTP 400 `RequestParameterIsWrong` before anything is queued. One take runs at a time; a take found waiting after a restart is failed, never re-run.

`GET /health` (no token) answers `{"ok", "comfy", "busy", "worker": "alive"|"dead", "card", "memory_mib"}` and, when asked with `?nonce=`, a `proof`. `ok` is false when the worker thread is dead. `busy` counts
a take from the moment it is accepted (while its inputs are still being fetched) until it is over **and collected** (the video downloaded, or the failure read), or 300 s after it finished.
`card` is the GPU the box runs on, as `nvidia-smi` names it with its memory (`NVIDIA GeForce RTX 4090, 49140 MiB`; `unknown` when it cannot say); `memory_mib` is the first card's memory as a number (`null` when unknown), so a consumer can check the size without parsing text: compare it with the box you declared
(`CLOUDGPU_BOX`); `python3 -m cloudgpu status` does and says so in `next` when they differ.

Errors are `CloudGpuError` with `kind`: `key_missing` (a setting or a key file is not right), `http` (a vendor API answered no), `unknown` (not a backend), `down` (the box
does not answer and nothing can start it), `timeout` (it did not come up); `next` lists what a person can do.

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
minute; every box down with none busy raises the last box's `CloudGpuError` instead of waiting for ever. The choice is written to the take's folder, and a take the caller says was sent
earlier runs on the box that holds it and on no other: only that box can give its result back. Threads of one process share one pool, so with two boxes two takes run at once.

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
The `--yes` gate is on the CLI (`instances power-on`, `power-off`, `create`, `release`): it is **a person's convention, not something enforced** (there is no tty check, any process can
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

## Measure and price a new box

1. `python3 -m cloudgpu bench ENVFILE NAME WORKFLOW RES SECONDS IMAGE [AUDIO]` submits one take (**a paid take**: the box's hours, and it powers nothing on by itself; there is no `--yes`
   and no tty check, so only a person runs it) and prints the wall time; run each take twice (the first loads the models). The machine's own time is in the box's `~/h3host/tasks/*.json` (`started`, `finished`).
2. Append the warm runs to `cloudgpu/data/runs.jsonl` (box, card, `hourly_cny`, workflow, resolution, seconds, `machine_s`; `cold: true` on the first). Never edit a run.
3. `python3 -m cloudgpu price --box ID [--format tsv] [--provider NAME]` prints the rows (the worst warm run of each workflow and resolution, at full use). The package writes
   nothing: your project turns them into its price table, as pending until a person confirms.
