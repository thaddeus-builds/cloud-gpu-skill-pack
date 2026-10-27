#!/bin/sh
# Run ON the GPU box (any vendor), from the folder `python3 -m cloudgpu bundle` made (copied there with scp and unpacked). Installs the adapter and starts it.
#   sh install.sh MODELS PORT < token-file        MODELS: auto (detect_models.py reads this image's files), or a json (models.4090.json)
#   sh install.sh auto 6009 < token
# The token (24+ characters) is read from stdin and written 0600; it is never an argument, so `ps` does not show it.
# Variables: CLOUDGPU_DIR (default $HOME/h3host: the folder name the first boxes got; START_COMMAND in box.py is its start.sh), CLOUDGPU_COMFY_ROOT (default $HOME/ComfyUI), CLOUDGPU_PYTHON (default: the first python found),
# CLOUDGPU_PUBLIC_BASE (https://host:port the callers reach it at, when the vendor exposes a port) and CLOUDGPU_BIND (default 127.0.0.1: all the SSH tunnel needs; 0.0.0.0 only for a
# vendor's public port). start.sh does not restart an adapter that has a take running. A systemd unit is installed when the box runs
# systemd, so the adapter starts at boot; a container without it relies on the vendor's power-on hook running $CLOUDGPU_DIR/start.sh.
set -e
MODELS="${1:?models: auto | a json file}"; PORT="${2:-6009}"
HERE="$(cd "$(dirname "$0")" && pwd)"
D="${CLOUDGPU_DIR:-$HOME/h3host}"; ROOT="${CLOUDGPU_COMFY_ROOT:-$HOME/ComfyUI}"
PY="${CLOUDGPU_PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3 python "$HOME/miniconda3/bin/python" /opt/conda/bin/python /usr/bin/python3; do
    if command -v "$c" >/dev/null 2>&1; then PY="$(command -v "$c")"; break; fi
  done
fi
[ -n "$PY" ] || { echo "no python found: set CLOUDGPU_PYTHON" >&2; exit 1; }
mkdir -p "$D"; chmod 700 "$D"
cp "$HERE/adapter.py" "$D/server.py"
if [ "$MODELS" = auto ]; then
  "$PY" "$HERE/detect_models.py" --comfy-root "$ROOT" --out "$D/models.json" || { echo "some model files were not found: see the report above" >&2; exit 1; }
else
  cp "$HERE/$MODELS" "$D/models.json"
fi
( umask 077; cat > "$D/token" )
[ "$(tr -d '\r\n' < "$D/token" | wc -c)" -ge 24 ] || { echo "the token is shorter than 24 characters" >&2; exit 1; }
cat > "$D/start.sh" <<EOS
#!/bin/sh
# (re)starts the adapter; the token comes from a 0600 file. Used by install.sh, the systemd unit and a vendor's power-on hook.
# An adapter that answers /health with a take on it (busy not 0) is left running: killing it would lose the take and its paid GPU time.
H="\$(curl -s -m 3 "127.0.0.1:$PORT/health" 2>/dev/null)"
if [ -n "\$H" ] && ! printf %s "\$H" | grep -Eq '"busy": *0[,}]'; then
  echo "the adapter has a take on it: not restarted (run start.sh again when /health says busy 0)"; echo "\$H"; exit 0
fi
pkill -f "$D/server.py" 2>/dev/null || true
cd "$D"
S=""; command -v setsid >/dev/null 2>&1 && S=setsid
CLOUDGPU_TOKEN="\$(cat "$D/token")" CLOUDGPU_PUBLIC_BASE="\${CLOUDGPU_PUBLIC_BASE:-${CLOUDGPU_PUBLIC_BASE:-}}" CLOUDGPU_BIND="\${CLOUDGPU_BIND:-${CLOUDGPU_BIND:-}}" \$S nohup "$PY" "$D/server.py" --port $PORT --comfy-root "$ROOT" --state "$D" --models "$D/models.json" > "$D/server.log" 2>&1 < /dev/null &
sleep 2; curl -s "127.0.0.1:$PORT/health"; echo
EOS
chmod 700 "$D/start.sh"
if [ "$(id -u)" = 0 ] && [ -d /run/systemd/system ]; then
  cat > /etc/systemd/system/cloudgpu.service <<EOU
[Unit]
Description=cloudgpu adapter
After=network-online.target
[Service]
Type=forking
ExecStart=$D/start.sh
Restart=no
[Install]
WantedBy=multi-user.target
EOU
  systemctl daemon-reload && systemctl enable cloudgpu.service >/dev/null 2>&1 && echo "systemd: cloudgpu.service enabled (starts at boot)"
fi
"$D/start.sh"
