"""The one error this package raises. A consumer (a skill pack with its own error type) catches it and maps `kind`."""

from __future__ import annotations


class CloudGpuError(Exception):
    """`kind`: key_missing (a variable or a file is not set right), http (a vendor API or the box answered no), unknown (a name that is no
    backend), down (the box does not answer and nothing can turn it on), timeout (it did not come up),
    no_stock (the vendor has no free card of that spec in that place now: try another, or later), over_quota (the allowance a person gave the agent does not cover it),
    not_allowed (the agent may not do that to that instance: a person's box), starting (only from `ensure_up(wait=False)`: the box is on and a needed service says it is
    coming up; a `CloudGpuError` subclass, `BoxStarting`). `next`: what a person can do."""

    def __init__(self, kind: str, message: str, next: list[str] | None = None, *, status=None, detail: str = ""):
        super().__init__(message)
        self.kind, self.message, self.next, self.status, self.detail = kind, message, list(next or []), status, detail

    def __str__(self) -> str:
        return self.message


class BoxStarting(CloudGpuError):
    """`Box.ensure_up(wait=False)` found the box on and a needed service `starting`: not a failure, only "not yet". `detail` is the service."""

    def __init__(self, message: str, next: list[str] | None = None, *, detail: str = ""):
        super().__init__("starting", message, next, detail=detail)
