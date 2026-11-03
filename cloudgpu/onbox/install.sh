#!/bin/sh
# Run ON the GPU box (any vendor), from the folder `python3 -m cloudgpu bundle` made (copied there with scp and unpacked). Installs the adapter and starts it.
#   sh install.sh MODELS PORT < token-file        MODELS: auto (detect_models.py reads this image's files), or a json (models.4090.json)
#   sh install.sh auto 6009 < token
# The token (24+ characters) is read from stdin and written 0600; it is never an argument, so `ps` does not show it.
# Variables: CLOUDGPU_DIR (default $HOME/h3host: the folder name the first boxes got; START_COMMAND in box.py is its start.sh), CLOUDGPU_COMFY_ROOT (default $HOME/ComfyUI), CLOUDGPU_PYTHON (default: the first python found),
# CLOUDGPU_TTS_DIR / CLOUDGPU_TTS_PYTHON (IndexTTS2's checkpoints folder and the Python of its environment, when they are not found by themselves: the voice is optional),
# CLOUDGPU_COMFY_PORT (ComfyUI's port, default 6006; the adapter is told it too), and the settings of services.py, which starts ComfyUI and VoxCPM after a boot (its header lists them: CLOUDGPU_COMFY_PYTHON,
# CLOUDGPU_COMFY_ARGS, CLOUDGPU_VOXCPM_CMD, CLOUDGPU_SERVICES_AUTOSTART=0 ...; the ones set here are written to $CLOUDGPU_DIR/services.json),
# CLOUDGPU_PUBLIC_BASE (https://host:port the callers reach it at, when the vendor exposes a port) and CLOUDGPU_BIND (default 127.0.0.1: all the SSH tunnel needs; 0.0.0.0 only for a
# vendor's public port), CLOUDGPU_ALLOW_NODE_VRGDG (unset or empty = off, the default; 1 = the owner allows the third-party audiodrive lock, see docs/cloud-gpu.md; the value given here is start.sh's default, and a value set when start.sh runs, 0 or empty included, wins). start.sh does not restart an adapter that has a take running. A systemd unit is installed when the box runs
# systemd, so the adapter starts at boot; a container without it relies on the vendor's power-on hook running $CLOUDGPU_DIR/start.sh.
set -e
MODELS="${1:?models: auto | a json file}"; PORT="${2:-6009}"
case "${CLOUDGPU_ALLOW_NODE_VRGDG:-}" in *[!A-Za-z0-9]*) echo "CLOUDGPU_ALLOW_NODE_VRGDG must be letters and digits only (1 switches the third-party audiodrive lock on, empty or 0 leaves it off)" >&2; exit 1;; esac
HERE="$(cd "$(dirname "$0")" && pwd)"
D="${CLOUDGPU_DIR:-$HOME/h3host}"; ROOT="${CLOUDGPU_COMFY_ROOT:-$HOME/ComfyUI}"
PY="${CLOUDGPU_PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3 python "$HOME/miniconda3/bin/python" /opt/conda/bin/python /usr/bin/python3; do
    if command -v "$c" >/dev/null 2>&1; then PY="$(command -v "$c")"; break; fi
  done
fi
[ -n "$PY" ] || { echo "no python found: set CLOUDGPU_PYTHON" >&2; exit 1; }
case "${CLOUDGPU_COMFY_PORT:-}" in
  '') ;;
  *[!0-9]*|??????*) echo "CLOUDGPU_COMFY_PORT must be a port number (1 to 65535)" >&2; exit 1;;
  *) [ "$CLOUDGPU_COMFY_PORT" -ge 1 ] && [ "$CLOUDGPU_COMFY_PORT" -le 65535 ] || { echo "CLOUDGPU_COMFY_PORT must be a port number (1 to 65535)" >&2; exit 1; };;
esac
mkdir -p "$D"; chmod 700 "$D"
cp "$HERE/adapter.py" "$D/server.py"
# ComfyUI and VoxCPM are started after a boot by services.py (0.22.0), run by start.sh below: ComfyUI's folder and the settings given here are written to services.json for it
SERVICES=""
if [ -f "$HERE/services.py" ]; then
  cp "$HERE/services.py" "$D/services.py"
  "$PY" "$D/services.py" configure --dir "$D" --comfy-root "$ROOT" --python "$PY" || echo "services: the settings could not be written (services.json); the defaults are used" >&2
  SERVICES=1
fi
if [ "$MODELS" = auto ]; then
  "$PY" "$HERE/detect_models.py" --comfy-root "$ROOT" --out "$D/models.json" || { echo "some model files were not found: see the report above" >&2; exit 1; }
else
  cp "$HERE/$MODELS" "$D/models.json"
fi
# The voice (IndexTTS2) is optional: installed only when the bundle carries tts_worker.py and its checkpoints and a Python that imports it are found (or named by
# CLOUDGPU_TTS_DIR and CLOUDGPU_TTS_PYTHON). Without it the adapter answers the voice workflow 400 and /health says "voice": false.
TTSFLAGS=""
if [ -f "$HERE/tts_worker.py" ] && [ -f "$HERE/detect_models.py" ]; then
  cp "$HERE/tts_worker.py" "$D/tts_worker.py"
  rm -f "$D/tts.json"
  "$PY" "$HERE/detect_models.py" --comfy-root "$ROOT" --tts-only --tts-out "$D/tts.json" >/dev/null 2>&1 || true
  if [ -f "$D/tts.json" ]; then
    TTSDIR="$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["dir"])' "$D/tts.json")"
    TTSPY="$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["python"])' "$D/tts.json")"
    TTSFLAGS="--tts-python '$TTSPY' --tts-dir '$TTSDIR'"
    echo "voice: IndexTTS2 found ($TTSDIR)"
  else
    echo "voice: IndexTTS2 not found on this box (optional; CLOUDGPU_TTS_DIR and CLOUDGPU_TTS_PYTHON name it)"
  fi
fi
# The audio-lock nodes (0.20.0) are only reported, never installed or fetched by this script: the default (official) lock and addguide use ComfyUI core nodes, and the third-party pack
# (comfyui-vrgamedevgirl, under a licence that is not open) is the owner's decision. The adapter serves its lock only with CLOUDGPU_ALLOW_NODE_VRGDG=1, which is NOT set by default.
if [ -f "$HERE/detect_models.py" ]; then
  "$PY" "$HERE/detect_models.py" --comfy-root "$ROOT" --nodes-only --text 2>/dev/null || true
fi
if [ -n "${CLOUDGPU_ALLOW_NODE_VRGDG:-}" ]; then
  echo "audio lock: CLOUDGPU_ALLOW_NODE_VRGDG=$CLOUDGPU_ALLOW_NODE_VRGDG is written into start.sh as this adapter's default (third-party node, licence-restricted: only the owner decides; 1 switches the audiodrive lock on)"
fi
( umask 077; cat > "$D/token" )
[ "$(tr -d '\r\n' < "$D/token" | wc -c)" -ge 24 ] || { echo "the token is shorter than 24 characters" >&2; exit 1; }
COMFYFLAG=""
[ -n "${CLOUDGPU_COMFY_PORT:-}" ] && COMFYFLAG="--comfy http://127.0.0.1:$CLOUDGPU_COMFY_PORT"
SVC=""
[ -n "$SERVICES" ] && SVC="[ -f \"$D/services.py\" ] && \"$PY\" \"$D/services.py\" up --detach --dir \"$D\" || true"
cat > "$D/start.sh" <<EOS
#!/bin/sh
# (re)starts the adapter, then ComfyUI and VoxCPM when they are down; the token comes from a 0600 file. Used by install.sh, the systemd unit and a vendor's power-on hook.
# An adapter that answers /health with a take on it (busy not 0) is left running: killing it would lose the take and its paid GPU time.
# ComfyUI and VoxCPM are started by services.py in the background (it returns at once): in that order, only what is not already running, VoxCPM's weights are not loaded.
# What each is doing: python3 $D/services.py status   (logs: $D/logs/)
H="\$(curl -s -m 3 "127.0.0.1:$PORT/health" 2>/dev/null)"
if [ -n "\$H" ] && ! printf %s "\$H" | grep -Eq '"busy": *0[,}]'; then
  echo "the adapter has a take on it: not restarted (run start.sh again when /health says busy 0)"; echo "\$H"
  $SVC
  exit 0
fi
pkill -f "$D/server.py" 2>/dev/null || true
cd "$D"
S=""; command -v setsid >/dev/null 2>&1 && S=setsid
CLOUDGPU_TOKEN="\$(cat "$D/token")" CLOUDGPU_PUBLIC_BASE="\${CLOUDGPU_PUBLIC_BASE:-${CLOUDGPU_PUBLIC_BASE:-}}" CLOUDGPU_BIND="\${CLOUDGPU_BIND:-${CLOUDGPU_BIND:-}}" CLOUDGPU_ALLOW_NODE_VRGDG="\${CLOUDGPU_ALLOW_NODE_VRGDG-${CLOUDGPU_ALLOW_NODE_VRGDG:-}}" \$S nohup "$PY" "$D/server.py" --port $PORT --comfy-root "$ROOT" --state "$D" --models "$D/models.json" $COMFYFLAG $TTSFLAGS > "$D/server.log" 2>&1 < /dev/null &
$SVC
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
