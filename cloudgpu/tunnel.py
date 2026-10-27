"""The SSH tunnel from this machine to the adapter on the box: by key, no prompt, no shell, the pieces as separate arguments."""

from __future__ import annotations

import re
import subprocess

from .errors import CloudGpuError

HOST_RE, USER_RE = re.compile(r"^[A-Za-z0-9._-]{1,253}$"), re.compile(r"^[a-z_][a-z0-9_-]{0,31}$")
PORT_RE = re.compile(r"[0-9]{1,5}")             # ASCII digits only: str.isdigit() also takes superscripts that int() refuses


def valid_port(value, what: str = "a port") -> int:
    """`value` as a TCP port 1..65535, or CloudGpuError(key_missing)."""
    text = str(value).strip()
    if not PORT_RE.fullmatch(text) or not 0 < int(text) < 65536:
        raise CloudGpuError("key_missing", f"{what} must be a number from 1 to 65535")
    return int(text)


def parse_ssh(spec: str) -> tuple[str, str, int]:
    """(user, host, port) of `[user@]host:port`, or CloudGpuError(key_missing)."""
    user, _, rest = spec.rpartition("@")
    host, _, port = rest.rpartition(":")
    user = user or "root"
    if not (HOST_RE.fullmatch(host) and USER_RE.fullmatch(user) and PORT_RE.fullmatch(port) and 0 < int(port) < 65536):
        raise CloudGpuError("key_missing", f"the SSH setting must be [user@]host:port, not {spec!r}")
    return user, host, int(port)


def argv(key: str, user: str, host: str, port: int, local: int, remote: int, known_hosts: str | None = None) -> list[str]:
    """The ssh command that forwards 127.0.0.1:local to the box's 127.0.0.1:remote and goes to the background (-f): BatchMode (never asks for a
    password), only the given key, and the host key remembered the first time (`accept-new`: the first connection is trusted, a changed key is refused).
    With `known_hosts` (a file holding the box's host key, got another way) the key must be in it: nothing is trusted on first use."""
    hostkey = ["-o", "StrictHostKeyChecking=yes", "-o", f"UserKnownHostsFile={known_hosts}"] if known_hosts else ["-o", "StrictHostKeyChecking=accept-new"]
    return ["ssh", "-f", "-N", "-L", f"127.0.0.1:{local}:127.0.0.1:{remote}", "-p", str(port), "-i", key,
            "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes", *hostkey,
            "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=4", f"{user}@{host}"]


def spawn(args: list[str]) -> bool:
    """Run it; True when ssh went to the background. A box that is off or a door that is shut is False, never an exception."""
    try:
        return subprocess.run(args, capture_output=True, timeout=60).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False
