"""The box as a consumer sees it: where it answers, the tunnel to it, whether it is up, and getting it up (and, when idle, off)."""

from __future__ import annotations

import atexit
import hmac
import json
import os
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping

from . import config, files, line, net, tunnel
from .errors import CloudGpuError
from .power import HOURLY_BILLING, NO_ACCOUNT_HINT, Power, backend

START_COMMAND = "/root/h3host/start.sh"          # a path on the box: written by onbox/install.sh (its folder is ~/h3host); a vendor's power-on hook runs it
UP_TIMEOUT_S = 900                               # power on, the vendor's own boot, ComfyUI up
DEFAULT_LOCAL_PORT, DEFAULT_REMOTE_PORT = 16009, 6009
OFF_STATUS = ("shutdown", "off")                 # the account's words for a box that is clearly off; "starting", "booting", anything unknown is not off: no second power-on
PROOF_EVERY_S = 60                               # at a local address the proof is checked again when the last one is older than this
API = "/api/v1/comfyui/comfyui_workflow"


class Box:
    def __init__(self, env: Mapping | None = None, *, spawn=tunnel.spawn, sleep=time.sleep):
        self.env = os.environ if env is None else env
        self._spawn, self._sleep = spawn, sleep
        self._up, self._ready, self._power = threading.Lock(), False, None
        self._proved, self._powered, self._hooked = None, False, False      # when the last proof passed (monotonic); this process sent the power-on; the exit hook is registered

    def get(self, name: str) -> str | None:
        return config.get(name, self.env)

    @property
    def power(self) -> Power:
        if self._power is None:
            self._power = backend(self.env)
        return self._power

    # ---- where it answers ----
    def root(self) -> str:
        b = (self.get("BASE") or "").rstrip("/")
        if not b and self.get("SSH_KEY"):
            b = f"http://127.0.0.1:{DEFAULT_LOCAL_PORT}"
        if not b.startswith(("https://", "http://")):
            raise CloudGpuError("key_missing", "set CLOUDGPU_BASE: where the box answers (its public https address, or http://127.0.0.1:PORT through the tunnel)")
        return net.check_url(b, "CLOUDGPU_BASE")

    def api_base(self) -> str:
        return self.root() + API

    def token(self) -> str:
        t = self.get("TOKEN")
        if not t:
            raise CloudGpuError("key_missing", "set CLOUDGPU_TOKEN: the secret the adapter was started with")
        if not net.header_safe(t):
            raise CloudGpuError("key_missing", "CLOUDGPU_TOKEN has a character that cannot be sent in an HTTP header (a newline inside it?)")
        if not self.reprove():
            raise CloudGpuError("down", "nothing answers at the box's local address any more (the tunnel is down?): the token was not sent",
                                ["run ensure_up() again, or python3 -m cloudgpu status"])
        return t

    def _needs_proof(self) -> bool:
        try:
            return bool(self.get("TOKEN")) and net.loopback(urllib.parse.urlparse(self.root()).hostname)
        except CloudGpuError:
            return False

    def reprove(self) -> bool:
        """At a local address (the tunnel) check the proof again when the last one passed more than PROOF_EVERY_S ago: the tunnel may have died and another process
        taken the port. True when no check was needed or it passed; False when nothing answers (and `ensure_up` must run again); a listener that does not prove
        the token is CloudGpuError(http). Either failure forgets that the box is ready. Called before every token is handed out and every ensure_up."""
        if not self._needs_proof() or (self._proved is not None and time.monotonic() - self._proved <= PROOF_EVERY_S):
            return True
        try:
            h = self.health()
        except CloudGpuError:
            self._ready, self._proved = False, None
            raise
        if h is None:
            self._ready, self._proved = False, None
            return False
        return True

    def health(self) -> dict | None:
        """The box's /health as a dict, None when nothing answers. At a local address (the tunnel) another process on this machine can hold the port, so with a token
        set the answer must carry `proof`, the HMAC-SHA256 of a random nonce under the token: only the adapter that was started with the token can make it. An
        answer without it is CloudGpuError(http) and the token is not sent to that listener (nothing here sends it before this passes)."""
        root = self.root()
        tok = self.get("TOKEN")
        nonce = secrets.token_hex(16) if tok and net.loopback(urllib.parse.urlparse(root).hostname) else None
        try:
            with net.urlopen(urllib.request.Request(root + "/health" + (f"?nonce={nonce}" if nonce else "")), timeout=15) as r:
                doc = json.loads(r.read())
        except (urllib.error.URLError, OSError, ValueError):
            return None
        if not isinstance(doc, dict):
            return None
        if nonce:
            want = hmac.new(tok.encode(), nonce.encode(), "sha256").hexdigest()
            got = doc.get("proof")
            if not (isinstance(got, str) and hmac.compare_digest(got.encode(), want.encode())):
                raise CloudGpuError("http", "something answers at the box's local address but does not prove it knows the token (another process on this machine holding the "
                                            "port, or an adapter older than 0.2.1): the token was not sent to it",
                                    ["look at what listens on that port (lsof -i)", "reinstall the adapter on the box from this version"])
            self._proved = time.monotonic()
        return doc

    # ---- the way to it ----
    def tunnel(self) -> bool:
        """Open the SSH tunnel by key when it is configured (CLOUDGPU_SSH_KEY and a local base) and nothing answers on its local port; True when it was
        opened. A box that is off or a door that is shut is not an error here: the caller's health loop says it."""
        key = self.get("SSH_KEY")
        u = urllib.parse.urlparse(self.root()) if key else None
        if not key or u.hostname not in ("127.0.0.1", "localhost") or self.health() is not None:
            return False
        if not os.path.isfile(key) or os.stat(key).st_mode & 0o077:
            raise CloudGpuError("key_missing", "CLOUDGPU_SSH_KEY must be a private key file only its owner can read (chmod 600)")
        known = self.get("KNOWN_HOSTS")
        if known and not os.path.isfile(known):
            raise CloudGpuError("key_missing", "CLOUDGPU_KNOWN_HOSTS must be a file holding the box's SSH host key (ssh-keyscan -p PORT HOST, checked another way)")
        try:
            local = u.port or DEFAULT_LOCAL_PORT
        except ValueError:
            raise CloudGpuError("key_missing", "CLOUDGPU_BASE has a port that is not a number from 1 to 65535") from None
        local, remote = tunnel.valid_port(local, "the port in CLOUDGPU_BASE"), tunnel.valid_port(self.get("REMOTE_PORT") or DEFAULT_REMOTE_PORT, "CLOUDGPU_REMOTE_PORT")
        spec = self.get("SSH")
        if spec:
            user, host, port = tunnel.parse_ssh(spec)
        else:
            door = self.power.ssh()
            if not door:
                raise CloudGpuError("key_missing", "set CLOUDGPU_SSH to [user@]host:port: the power backend does not say the box's SSH door")
            user, host, port = "root", door[0], door[1]
            tunnel.parse_ssh(f"{host}:{port}")
        return self._spawn(tunnel.argv(key, user, host, port, local, remote, known))

    def ensure_up(self) -> None:
        """Return when the box answers /health with ComfyUI up: open the tunnel when it is configured, and power the box on first when it is off and a
        backend can. By design this powers a box on WITHOUT asking a person: who may power boxes on is decided by whoever holds the account's token and
        CLOUDGPU_INSTANCE (the operator team), not by an agent or a customer, so a caller that must not cost money does not give the process those two.
        The `--yes` gate belongs to the CLI (`instances power-on`), which a person runs. With CLOUDGPU_AUTOOFF the box is powered off at exit, but only
        when THIS process powered it on."""
        with self._up:
            if self._ready and self.reprove():
                return
            self.tunnel()
            h = self.health()
            if not (h and h.get("comfy")):
                pw = self.power
                if pw.name == "none" or pw.status() in OFF_STATUS:       # a box that is starting (or whose state is unknown) is waited for, not powered on again
                    pw.power_on(START_COMMAND)               # `none` refuses with what a person must do
                    self._powered = True
                    self._register_power_off()               # from here, even a timeout or an error leaves the box to the exit hook
                end = time.time() + UP_TIMEOUT_S
                while not (h and h.get("comfy")):
                    if time.time() > end:
                        raise CloudGpuError("timeout", f"the box did not come up in {UP_TIMEOUT_S}s" + ("" if pw.configured() else "; " + NO_ACCOUNT_HINT),
                                            ["look at the box at its vendor; nothing was sent"], detail=pw.name)
                    self._sleep(10)
                    self.tunnel()                            # the door may have moved while it booted
                    h = self.health()
            self._ready = True

    # ---- the line of callers on other machines (line.py) ----
    def line_join(self, label: str = "") -> dict | None:
        return line.join(self, label)

    def line_poll(self, ticket: str) -> dict | None:
        return line.poll(self, ticket)

    def line_leave(self, ticket: str) -> None:
        line.leave(self, ticket)

    def line_list(self) -> list | None:
        return line.listing(self)

    # ---- the file store (files.py) ----
    def supports_files(self) -> bool:
        return files.supported(self)

    def ensure_file(self, path) -> str:
        """`sha256:<hex>` of a local file the box now has: sent once, named in every take after it."""
        return files.ensure(self, path)

    def _register_power_off(self) -> None:
        if not self._hooked and config.flag("AUTOOFF", self.env):
            self._hooked = True
            atexit.register(self._power_off_at_exit)

    def _power_off_at_exit(self) -> None:
        """The exit hook: only for a box THIS process powered on, and then only what `power_off_if_idle` allows."""
        if self._powered:
            self.power_off_if_idle()

    def power_off_if_idle(self) -> bool:
        """Power the box off when it is certain no take is waiting on it (another process may be using it) and the billing is by the hour (`payg`, exactly); True when it was sent.
        Anything unknown is "do not": /health that does not answer (a tunnel that is down, an adapter restarting), a worker that is not alive, no `busy` of 0,
        an account that does not say how the box is billed or says a word that is not `payg`. A pass is paid anyway, so it is left on."""
        pw = self.power
        if pw.name == "none" or not pw.configured():
            return False
        try:
            h = self.health()
            if not h or h.get("ok") is not True or h.get("busy") != 0:
                return False
            info = pw.pass_info()
            if not info or "error" in info or info.get("billing") != HOURLY_BILLING:
                return False
            return bool(pw.power_off())
        except CloudGpuError:
            return False
