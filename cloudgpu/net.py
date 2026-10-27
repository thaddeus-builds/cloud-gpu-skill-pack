"""The guards on every call this package makes with a secret or to an address a box named: a redirect never leaves the host it started on (urllib would send the
Authorization header along), `http://` only to this machine, and a token fit to be a header value. Nothing here ever puts a token into a message."""

from __future__ import annotations

import urllib.parse
import urllib.request

from .errors import CloudGpuError

LOOPBACK = ("127.0.0.1", "::1", "localhost")
DEFAULT_PORTS = {"http": 80, "https": 443}


def loopback(host: str | None) -> bool:
    return (host or "").lower() in LOOPBACK


def check_url(url: str, what: str = "the address") -> str:
    """The URL when it is `https://` to anywhere or `http://` to this machine; otherwise CloudGpuError(key_missing). The message names the scheme, never the URL (a result URL is a secret)."""
    try:
        u = urllib.parse.urlparse(url)
        host = u.hostname
    except ValueError:
        raise CloudGpuError("key_missing", f"{what} is not a URL") from None
    if not host or (u.scheme != "https" and not (u.scheme == "http" and loopback(host))):
        raise CloudGpuError("key_missing", f"{what} must be https://, or http:// to this machine (127.0.0.1, ::1, localhost) only: a secret must not travel in clear "
                                           f"(this one is {u.scheme or 'no scheme'}://)")
    return url


def header_safe(value: str) -> bool:
    """True when `value` can be an HTTP header value: printable ASCII, no newline (http.client would raise a ValueError that carries the value)."""
    return value.isascii() and value.isprintable()


def _origin(url: str):
    u = urllib.parse.urlparse(url)
    return u.scheme, (u.hostname or "").lower(), u.port or DEFAULT_PORTS.get(u.scheme)


class SameOrigin(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only to the same scheme, host and port: urllib would otherwise forward the Authorization header to whoever the answer names."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if _origin(req.full_url) != _origin(urllib.parse.urljoin(req.full_url, newurl)):
            return None                                  # the 3xx then surfaces as an HTTPError
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def urlopen(req, timeout: float):
    """Open `req` with the redirect guard. A request to this machine (the tunnel) never goes through a proxy from the environment (http_proxy, HTTPS_PROXY): the proxy
    could not reach 127.0.0.1, and the token and the proof request would go to it."""
    url = req if isinstance(req, str) else req.full_url
    handlers = [SameOrigin()]
    if loopback(urllib.parse.urlparse(url).hostname):
        handlers.insert(0, urllib.request.ProxyHandler({}))
    return urllib.request.build_opener(*handlers).open(req, timeout=timeout)
