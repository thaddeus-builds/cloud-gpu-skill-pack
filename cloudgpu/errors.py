"""The one error this package raises. A consumer (a skill pack with its own error type) catches it and maps `kind`."""

from __future__ import annotations


class CloudGpuError(Exception):
    """`kind`: key_missing (a variable or a file is not set right), http (a vendor API or the box answered no), unknown (a name that is no
    backend), down (the box does not answer and nothing can turn it on), timeout (it did not come up). `next`: what a person can do."""

    def __init__(self, kind: str, message: str, next: list[str] | None = None, *, status=None, detail: str = ""):
        super().__init__(message)
        self.kind, self.message, self.next, self.status, self.detail = kind, message, list(next or []), status, detail

    def __str__(self) -> str:
        return self.message
