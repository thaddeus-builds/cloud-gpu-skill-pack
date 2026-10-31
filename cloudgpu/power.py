"""How a box is powered: the account API of whichever cloud it is rented from says whether it is on, turns it on and off, and where its SSH door is.

A GPU rented from any vendor that gives SSH can run the adapter; what differs is only this. A power backend answers five questions (the class `Power`) and is chosen by
CLOUDGPU_POWER: `none` (an always-on box, a pass, or a vendor with no API: nothing is powered, the package only waits for the box to answer) or the name of a cloud
vendor (`cloudgpu/vendors/`: **autodl is built in and is the default**). Every vendor that is taken in gets its backend for free (`VendorPower`: the vendor plus the instance id
CLOUDGPU_INSTANCE); to take in a new cloud, write its `Vendor` (vendors/__init__.py) and nothing here changes. With autodl, CLOUDGPU_INSTANCE (pro-...) and AUTODL_DEV_TOKEN
power the box on and off; without them the default backend is "not configured": the first call says what to set, or to set CLOUDGPU_POWER=none.
"""

from __future__ import annotations

import time
from collections.abc import Mapping

from . import config, vendors
from .errors import CloudGpuError
from .vendors import Vendor

HOURLY_BILLING = "payg"                             # autodl's charge_type of a box paid by the hour: the only one that is ever powered off to save money
PASS_BILLING = ("daily", "weekly", "monthly")      # autodl's charge_type of a pass: paid whether the card works or not
DEFAULT_BACKEND = "autodl"
NO_ACCOUNT_HINT = ("with CLOUDGPU_POWER=none this package has no view of the cloud account, so it cannot tell a box that is switched off from one whose pass has ended or that was released. "
                   "To see which, and to start it, set AUTODL_DEV_TOKEN (the account's developer token: console -> account -> settings) and CLOUDGPU_INSTANCE (the pro-... number) "
                   "and unset CLOUDGPU_POWER; the person who rents the box (the account holder) has them and gives them to whoever runs this, never in chat")


class Power:
    """A box that is always on and reached as configured: the base of every backend, and what `none` is."""

    name = "none"

    def configured(self) -> bool:
        """True when this backend can act (its variables are set)."""
        return False

    def status(self) -> str | None:
        """"running" when the account says the box is on, another word when it is not, None when this backend cannot say."""
        return None

    def power_on(self, start_command: str) -> None:
        """Turn the box on; `start_command` (a path on the box) is run after boot where the vendor supports it. CloudGpuError when it cannot."""
        raise CloudGpuError("down", "the box does not answer: it may be switched off or its pass may have ended (or it was released), and " + NO_ACCOUNT_HINT,
                            ["the account holder looks at the box in the vendor's console: if it is off, start it; if it ended, renew it or rent another",
                             "or give this process the account token and instance id (docs/cloud-gpu.md, 'Power'), so the status is read and a box that is off is started"],
                            detail=self.name)

    def power_off(self) -> bool:
        """Turn the box off; False when this backend does not (a pass or an always-on box: nothing to save)."""
        return False

    def pass_info(self) -> dict:
        """{} or what the account says about the box's billing: billing (a word), ends (ISO time of a pass's end), hours_left."""
        return {}

    def ssh(self) -> tuple[str, int] | None:
        """(host, port) of the box's SSH door as the account says it now, or None (then CLOUDGPU_SSH names it)."""
        return None


class VendorPower(Power):
    """A power backend made from a cloud vendor (cloudgpu/vendors/) and an instance id (CLOUDGPU_INSTANCE): every vendor that is taken in gets one for free."""

    def __init__(self, vendor: Vendor, env: Mapping | None = None):
        self.vendor, self.name = vendor, vendor.name
        self.instance = config.get("INSTANCE", env)

    def configured(self) -> bool:
        return bool(self.instance and self.vendor.configured())

    def _need(self):
        if not self.configured():
            raise CloudGpuError("key_missing", f"the {self.name} power backend needs CLOUDGPU_INSTANCE and {self.vendor.token_env} (see docs/cloud-gpu.md, 'Power')",
                                ["set them, or set CLOUDGPU_POWER=none when the box is always on or a pass"])

    def status(self):
        self._need()
        return self.vendor.status(self.instance)

    def power_on(self, start_command):
        self._need()
        self.vendor.power_on(self.instance, start_command)

    def power_off(self):
        self._need()
        self.vendor.power_off(self.instance)
        return True

    def pass_info(self):
        self._need()
        i = self.vendor.get(self.instance)
        if i is None:
            return {"error": "the account lists no instance of that id"}
        out = {"billing": i.billing, "ends": i.ends, "status": i.status, "gpu_spec": i.gpu_spec, "region": i.region}
        if i.ends and i.billing in PASS_BILLING:
            out["hours_left"] = round((time.mktime(time.strptime(i.ends[:19], "%Y-%m-%dT%H:%M:%S")) - time.time()) / 3600, 1)
        return out

    def ssh(self):
        self._need()
        door = (self.vendor.snapshot(self.instance) or {}).get("ssh")
        return (door["host"], int(door["port"])) if door else None


BACKENDS = ("none", *vendors.registry())


def backend(env: Mapping | None = None) -> Power:
    """The backend CLOUDGPU_POWER names (none, or a cloud vendor's name); autodl when none is named. A vendor backend without its variables is not configured: the first
    call says what to set (or to set CLOUDGPU_POWER=none)."""
    name = config.get("POWER", env) or DEFAULT_BACKEND
    if name == "none":
        return Power()
    if name not in vendors.registry():
        raise CloudGpuError("unknown", f"CLOUDGPU_POWER {name!r} is not a power backend; use {', '.join(BACKENDS)}")
    return VendorPower(vendors.get(name, env), env)
