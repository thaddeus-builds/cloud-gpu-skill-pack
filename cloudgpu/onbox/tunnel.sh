#!/bin/sh
# Open (or reuse) the SSH tunnel from this machine to the box's adapter, with a key: no password, no prompt, so an agent can run it.
#   sh tunnel.sh HOST PORT KEYFILE [LOCAL_PORT=16009] [REMOTE_PORT=6009]
# HOST and PORT are the instance's SSH line (autodl console; the instance API's `proxy_host` and `ssh_port` say them too). KEYFILE is a private
# key whose public half the owner put in the box's /root/.ssh/authorized_keys. Prints the box's /health when the tunnel answers.
# CLOUDGPU_BASE is then http://127.0.0.1:LOCAL_PORT. A password login cannot be used here on purpose: an agent never types one.
# Any process on this machine can hold 127.0.0.1:LOCAL_PORT and read what is sent to it, so with CLOUDGPU_TOKEN set (the environment, never an argument) a listener is used only
# when it proves it knows the token: the adapter answers /health?nonce=N with proof = HMAC-SHA256(token, N). Exit 3: something else answers there. Without the token the
# listener cannot be checked and the script says so. CLOUDGPU_KNOWN_HOSTS: a file with the box's host key; then the key must be in it (otherwise the first connection is trusted).
set -e
HOST="${1:?host}"; PORT="${2:?ssh port}"; KEY="${3:?private key file}"; LP="${4:-16009}"; RP="${5:-6009}"
for n in "$PORT" "$LP" "$RP"; do
  case "$n" in ''|*[!0-9]*) echo "not a port number: $n" >&2; exit 2;; esac
  [ "$n" -ge 1 ] && [ "$n" -le 65535 ] || { echo "not a port number: $n" >&2; exit 2; }
done
case "$HOST" in ''|*[!A-Za-z0-9._-]*) echo "not a host name: $HOST" >&2; exit 2;; esac
[ -r "$KEY" ] || { echo "key file not readable: $KEY" >&2; exit 2; }
T="${CLOUDGPU_TOKEN:-${H3HOST_TOKEN:-}}"
PYC=""; for c in python3 python; do command -v "$c" >/dev/null 2>&1 && { PYC="$c"; break; }; done

# probe: prints /health and returns 0 when the listener answers (and, with a token, proves it); 1 when nothing answers; 3 when it answers but is not the adapter
probe() {
  if [ -n "$T" ] && [ -n "$PYC" ]; then
    CLOUDGPU_T="$T" "$PYC" - "$LP" <<'EOP'
import hmac, json, os, secrets, sys, urllib.request
nonce = secrets.token_hex(16)
try:
    raw = urllib.request.urlopen("http://127.0.0.1:%s/health?nonce=%s" % (sys.argv[1], nonce), timeout=5).read()
    doc = json.loads(raw)
except Exception:
    sys.exit(1)
want = hmac.new(os.environ["CLOUDGPU_T"].encode(), nonce.encode(), "sha256").hexdigest()
got = doc.get("proof") if isinstance(doc, dict) else None
if not (isinstance(got, str) and hmac.compare_digest(got.encode(), want.encode())):
    sys.exit(3)
doc.pop("proof", None)
print(json.dumps(doc))
EOP
  else
    curl -fs -m 5 "http://127.0.0.1:$LP/health" || return 1
    echo
    echo "(not verified: set CLOUDGPU_TOKEN so a listener that is not the adapter is refused)" >&2
  fi
}
verified() {
  rc=0; probe || rc=$?
  [ "$rc" -ne 3 ] || { echo "something on 127.0.0.1:$LP answers but does not prove it knows the token: another process holds the port; not using it, nothing was sent to it" >&2; exit 3; }
  return "$rc"
}

if verified; then exit 0; fi
KH=""; [ -z "${CLOUDGPU_KNOWN_HOSTS:-}" ] || KH="-o StrictHostKeyChecking=yes -o UserKnownHostsFile=$CLOUDGPU_KNOWN_HOSTS"
[ -n "$KH" ] || KH="-o StrictHostKeyChecking=accept-new"
# shellcheck disable=SC2086
ssh -f -N -L "127.0.0.1:$LP:127.0.0.1:$RP" -p "$PORT" -i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ExitOnForwardFailure=yes \
    $KH -o ServerAliveInterval=30 -o ServerAliveCountMax=4 "root@$HOST"
for i in 1 2 3 4 5 6 7 8 9 10; do
  if verified; then exit 0; fi
  sleep 1
done
echo "the tunnel is up but the adapter does not answer on the box's port $RP" >&2; exit 1
