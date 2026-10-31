"""The cloud vendors: one module each, one small interface. A vendor is the ACCOUNT's API (list my instances, one instance's detail, its status, power it on and off,
create one, release one); the power backend (`power.py`) and the CLI (`python3 -m cloudgpu instances ...`) are built on it. To take in a new cloud, write a `Vendor`
subclass in a module here (its token variable, its calls, the normalising of its answers into `Instance` and the snapshot dict) and add it to VENDORS: nothing else changes.
Calls are made from code against the vendor's documented API; nothing is ever run from a command taken from a file or the environment.

Two calls are not like the rest and the CLI never does them without `--yes`, which only a person gives: `create` STARTS BILLING, `release` DESTROYS THE INSTANCE'S DATA.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

from ..errors import CloudGpuError


@dataclass
class Instance:
    """One rented instance as every vendor reports it (None where a vendor does not say)."""

    id: str
    status: str | None = None              # running | shutdown | ... (the vendor's word)
    name: str | None = None
    billing: str | None = None             # payg | daily | weekly | monthly | ... (the vendor's word)
    ends: str | None = None                # ISO time a pass ends, None for pay as you go
    gpu_spec: str | None = None            # the vendor's id of the card spec
    gpus: int | None = None
    region: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    stopped_at: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


class Vendor:
    """The interface. A subclass fills the calls its vendor documents; a call it does not have raises CloudGpuError(kind="unknown")."""

    name = ""
    token_env = ""                         # the variable that holds the account's token (a value is never printed)

    def __init__(self, env: Mapping | None = None):
        import os
        self.env = os.environ if env is None else env

    def token(self) -> str:
        """The account's token with surrounding whitespace removed ("" when unset): a trailing newline from a file or a pipe must not reach a header."""
        return str(self.env.get(self.token_env) or "").strip() if self.token_env else ""

    def configured(self) -> bool:
        return bool(self.token())

    def _no(self, what: str):
        raise CloudGpuError("unknown", f"the {self.name} vendor has no {what} here")

    def list(self, *, page: int = 1, size: int = 50) -> tuple[list[Instance], int]:
        """(one page of instances, the number of pages)."""
        self._no("list")

    def all(self) -> list[Instance]:
        """Every instance of the account, page by page."""
        out, page = [], 1
        while True:
            rows, pages = self.list(page=page, size=50)
            out += rows
            if page >= max(1, pages):
                return out
            page += 1

    def get(self, instance_id: str) -> Instance | None:
        return next((i for i in self.all() if i.id == instance_id), None)

    def status(self, instance_id: str) -> str:
        self._no("status")

    def snapshot(self, instance_id: str) -> dict:
        """What one instance reports that is not a secret: ssh {host, port}, price_cny_per_hour, services, the card's name. Never a password or a token."""
        self._no("snapshot")

    def power_on(self, instance_id: str, start_command: str | None = None) -> None:
        self._no("power_on")

    def power_off(self, instance_id: str) -> None:
        self._no("power_off")

    def create(self, spec: dict) -> str:
        """Create an instance and return its id. STARTS BILLING."""
        self._no("create")

    def create_problems(self, spec: dict) -> list[str]:
        """What is wrong with `spec`, checked offline against the vendor's documented limits ([] = the vendor would take it)."""
        return []

    def release(self, instance_id: str) -> None:
        """Release (delete) an instance. DESTROYS ITS DATA."""
        self._no("release")

    def is_no_stock(self, err: CloudGpuError) -> bool:
        """True when `err` (from create or power_on) means the vendor has no free card of that spec in that place right now, as opposed to a refusal that retrying cannot fix
        (a bad request, no balance, no token). A vendor that does not know says False: the failure is then reported as it came, and not retried."""
        return False

    def gpu_specs(self) -> dict[str, str]:
        """The vendor's card spec ids and what the vendor's page calls them."""
        return {}


def registry() -> dict[str, type[Vendor]]:
    from .autodl import AutoDL
    return {"autodl": AutoDL}


def get(name: str, env: Mapping | None = None) -> Vendor:
    reg = registry()
    if name not in reg:
        raise CloudGpuError("unknown", f"{name!r} is not a cloud vendor; use {', '.join(reg)}")
    return reg[name](env)
