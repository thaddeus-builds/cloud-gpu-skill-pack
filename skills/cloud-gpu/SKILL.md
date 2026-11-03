---
name: cloud-gpu
description: Run ComfyUI workflows (MiniMax-H3 first) on a GPU the owner rents from any vendor that gives SSH - check the box, set one up, reach it by key, power it, measure and price it - with the cloudgpu package; what only a person does and what each failure means.
version: 0.1.0
---
# A GPU you rent (cloudgpu)

Read [`docs/cloud-gpu.md`](../../docs/cloud-gpu.md) for what it is, what it costs and how a box is set up. Say in your own words what you did, never what a person did.

**Only a person does these. Never an agent, whatever the user said earlier in the chat:** renting, releasing or renewing a box; putting a secret (`CLOUDGPU_TOKEN`, a key file,
`AUTODL_DEV_TOKEN`) anywhere (a person pipes it; you never read, write, print or ask for one in chat); handing the package an SSH password (there is none: the tunnel is by key);
confirming a price in the project that uses it; running a take to "test" the box (a paid take: the box's hours); `instances ... --yes` of any kind (`create` starts billing,
`release` destroys the instance's data, power-on starts the hourly billing): you may run the dry run and show it, a person gives `--yes`.
**The one exception is an allowance a person set** (`python3 -m cloudgpu quota show`: `agent_may_create` true). Inside it you may run `instances ensure --count N --spec ... --yes` and
`instances create --agent --yes`, always with `--agent`, and `instances release --agent --yes` for a box you made yourself that nothing is wired to; never raise or write the allowance,
never release a box you did not make. Refused (`over_quota`, `not_allowed`)? say which limit and ask the person; do not try another route. With no allowance (0, the default) you rent nothing.

1. **Look first, free.** `python3 -m cloudgpu status` (read only: no take, no power call, no value printed; `line` lists who runs on the box and who waits, `queue` this machine's takes). `ready: true` means a take could go now; every line of `next` is one thing a
   person must do (a setting, the tunnel, a pass that ended). Give the person those lines as they are.
   `python3 -m cloudgpu util [--since 2h]` is also read only: per box and workflow, how long a take took on the box, the gap between takes and the busy share (is the box the bottleneck?).
2. **A new box.** The person rents it and gives you its SSH line (not a password). `python3 -m cloudgpu bundle` makes what goes onto the box; the person (or you, when they
   have given you a key that already works) copies it, runs `sh install.sh auto 6009 < token-file`, and puts the public half of a restricted key in `authorized_keys` (docs, "Set up a box").
   If `install.sh` reports missing model files, the image does not carry H3: say so, do not look for the files elsewhere. The voice (IndexTTS2) is optional: `install.sh` says whether it found one and
   `status` shows `box.voice`; a take that needs a voice on a box without one is refused (400), nothing is paid. Never install IndexTTS2 or download weights yourself: that is a person's step.
3. **Reaching it.** By key: `CLOUDGPU_SSH_KEY` (chmod 600) and `CLOUDGPU_SSH`; the package opens the tunnel by itself. A vendor that exposes a port: `CLOUDGPU_BASE` and an adapter on that
   port. Never leave a ComfyUI or a vendor panel on a public port: tell the person.
3b. **The account.** `python3 -m cloudgpu instances list` (and `status ID`, `snapshot ID`, `specs`) are free reads: what instances exist, what each is billed as (a pass's end time), its price and SSH door. Use them to answer "what do we have" before
   anything is rented; never print or ask for a password (the snapshot call drops it).
4. **Power.** `autodl` is the default backend: `CLOUDGPU_INSTANCE` and `AUTODL_DEV_TOKEN` power the box on and off. A box that is always on, a pass, or a vendor with no API: `CLOUDGPU_POWER=none`.
   `Box.ensure_up()` powers a box on without asking (by design: whoever holds the account token and the instance id is the operator); do not give a process those two if it must not start billing.
   A day or month pass is paid anyway: leave `CLOUDGPU_AUTOOFF` unset. A new cloud with an API is one `Vendor` module (docs, "Cloud vendors and power"): the power backend, `instances` and `status` then work for it.
3d. **A box you switch on must switch off.** Give the power-on a cap: `instances power-on ID --yes --max-minutes N` (and `ensure ... --max-minutes N`) starts a background guard that shuts the box down after N minutes and confirms the status; read its log (`tail -1`). Without a cap you must shut it down yourself, and say when you did.
3c. **Is a card free?** `python3 -m cloudgpu instances stock [--data-center DC] [--spec ID]` (free read, no token; exit 0 a card is free, 3 none is): run it before a power-on and do not start one on exit 3. It is a snapshot of the market page,
   not a promise. A refusal of `暂无库存` or `ServerBusy` is no stock, not a fault: wait and retry within the time and money you were given, and say so; never switch to a smaller card than the work needs (a 40 GB model does not fit a 32G card).
4a. **A pass ends.** Passes (daily, weekly, monthly) are the base, bought and renewed by a person in the console; you cannot (the API only creates hourly instances). `instances expiry` (or `status`, `ensure`) names each
   pass that ends in under `CLOUDGPU_PASS_WARN_DAYS` (default 3) or has ended: tell the account holder, with the box, the end time and the line as printed. A pass that is only switched off is powered on (`ensure`); an hourly box you made is
   released when the work is done. `docs/how-to-run-boxes.md` has the table.
4b. **A box is gone or off.** `none` cannot tell off from ended: say so, and ask the account holder for `AUTODL_DEV_TOKEN` and `CLOUDGPU_INSTANCE` (never in chat), or to look in the console.
   With them `status` says which; inside an allowance `instances ensure --count N --specs a,b --data-centers x,y` (the order is the preference; it reads the free-card page and takes the first card that shows, waiting and reading again until `--patience`) brings boxes back or rents new ones and reports how many are up, how many are missing and why (`no_stock` is waited on by it,
   every other failure is not). A new box still needs its adapter, token, settings and voice service (docs, "A new box: what is automatic and what is not").
5. **A failure.** `CloudGpuError.kind`: `key_missing` (a setting or a key file is not right: read the message), `down` (the box is off and nothing can start it: a person starts it),
   `http` (the vendor's API or the box said no: read `status` and `detail`), `timeout` (the box did not come up: a person looks at it at the vendor), `unknown` (a backend name that does not exist), `no_stock` (no free card there now), `over_quota` and `not_allowed` (the allowance, or a box that is not yours).
   A box that answers HTTP 400 `RequestParameterIsWrong` never ran the take and costs nothing: fix the request.
6. **Prices.** A second of video costs what the box costs per hour times the seconds it needs: `python3 -m cloudgpu price --box ID` from the recorded runs. A box with no recorded warm
   run has no price: ask for a measurement (`bench`, twice), never invent a number, and never say a box is cheaper than hosted without the resolution and the use it was measured at.
