"""Where the settings come from: the environment, under CLOUDGPU_<NAME>, or the older H3HOST_<NAME> (read second: the first users of this
code named them so). Nothing here reads a file by itself: a consumer loads its own .env into the environment (or passes a mapping), and the
tools that take an ENVFILE argument use `read_env_file`. A value is never printed by this package."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

PREFIXES = ("CLOUDGPU_", "H3HOST_")
NAMES = ("BASE", "TOKEN", "BOX", "SSH_KEY", "SSH", "KNOWN_HOSTS", "REMOTE_PORT", "POWER", "INSTANCE", "AUTOOFF", "QUOTA_FILE", "LEDGER_FILE", "ACTOR", "ENSURE_SPECS", "ENSURE_DATA_CENTERS")       # AUTODL_DEV_TOKEN is the autodl backend's own


def get(name: str, env: Mapping | None = None) -> str | None:
    """The value, whitespace stripped, of the first of CLOUDGPU_<name>, H3HOST_<name> that is PRESENT; None when it is empty. A key that is present wins even when empty
    (`CLOUDGPU_AUTOOFF=` is off, it does not fall through to an older H3HOST_AUTOOFF). A token with a trailing newline is the usual way a secret breaks a header."""
    env = os.environ if env is None else env
    for p in PREFIXES:
        if p + name in env:
            v = env[p + name]
            return (str(v).strip() or None) if v is not None else None
    return None


def flag(name: str, env: Mapping | None = None) -> bool:
    """True when the setting is 1, true, yes or on; "0", "off", empty and unset are all off."""
    return (get(name, env) or "").lower() in ("1", "true", "yes", "on")


def present(env: Mapping | None = None) -> dict:
    """{name: 'set' | 'unset'} for every setting, and the backend's own token: names and whether, never a value."""
    env = os.environ if env is None else env
    out = {n: ("set" if get(n, env) else "unset") for n in NAMES}
    out["AUTODL_DEV_TOKEN"] = "set" if str(env.get("AUTODL_DEV_TOKEN") or "").strip() else "unset"
    return out


def read_env_file(path: str | Path) -> dict:
    """KEY=value lines of a file as a dict (comments and blanks skipped, one pair of quotes stripped). For the tools' ENVFILE arguments."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
                v = v[1:-1]
            out[k.strip()] = v
    return out


def env_file_warning(path: str | Path) -> str | None:
    """A sentence when the file that holds secrets can be read by group or others (chmod 600 it), else None. Never contains a value."""
    try:
        mode = Path(path).stat().st_mode & 0o777
    except OSError:
        return None
    if mode & 0o077:
        return f"the settings file {Path(path).name} is readable by other users (mode {mode:03o}): chmod 600 it"
    return None
