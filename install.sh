#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
MODELS_DIR="$SCRIPT_DIR/models/TTS"
HOOK_PY="$SCRIPT_DIR/src/manajeure/hook.py"
SETTINGS_FILE="$HOME/.claude/settings.json"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[+]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[x]${NC} $*"; }

# ── System dependencies (Linux only) ────────────────────────────────────────

if [[ "$(uname)" == "Linux" ]]; then
    for tool in xclip xdotool; do
        if ! command -v "$tool" &>/dev/null; then
            info "Installing $tool..."
            sudo apt install -y "$tool" 2>/dev/null || warn "Could not install $tool via apt — install it manually"
        else
            info "$tool already installed"
        fi
    done
fi

# ── Model files ──────────────────────────────────────────────────────────────

mkdir -p "$MODELS_DIR"

download() {
    local name="$1" url="$2" checksum="$3"
    local dest="$MODELS_DIR/$name"
    if [[ -f "$dest" ]]; then
        info "$name already exists, verifying checksum..."
        if echo "$checksum  $dest" | sha256sum -c --status 2>/dev/null; then
            info "$name checksum OK"
            return
        else
            warn "$name checksum mismatch, re-downloading"
        fi
    fi
    info "Downloading $name..."
    curl -L --progress-bar -o "$dest" "$url"
    if echo "$checksum  $dest" | sha256sum -c --status 2>/dev/null; then
        info "$name downloaded and verified"
    else
        error "$name checksum verification failed!"
        exit 1
    fi
}

# GLaDOS voice model (VITS/Piper)
download "glados.onnx" \
    "https://github.com/dnhkng/GlaDOS/releases/download/0.1/glados.onnx" \
    "17ea16dd18e1bac343090b8589042b4052f1e5456d42cad8842a4f110de25095"

# Kokoro voice model
download "kokoro-v1.0.fp16.onnx" \
    "https://github.com/dnhkng/GLaDOS/releases/download/0.1/kokoro-v1.0.fp16.onnx" \
    "c1610a859f3bdea01107e73e50100685af38fff88f5cd8e5c56df109ec880204"

download "kokoro-voices-v1.0.bin" \
    "https://github.com/dnhkng/GLaDOS/releases/download/0.1/kokoro-voices-v1.0.bin" \
    "c5adf5cc911e03b76fa5025c1c225b141310d0c4a721d6ed6e96e73309d0fd88"

# Shared phonemizer model
download "phomenizer_en.onnx" \
    "https://github.com/dnhkng/GlaDOS/releases/download/0.1/phomenizer_en.onnx" \
    "b64dbbeca8b350927a0b6ca5c4642e0230173034abd0b5bb72c07680d700c5a0"

# Small data files — bundled in repo under models/TTS/
for f in lang_phoneme_dict.pkl token_to_idx.pkl idx_to_token.pkl phoneme_to_id.pkl glados.json; do
    if [[ ! -f "$MODELS_DIR/$f" ]]; then
        error "Missing model data file: $MODELS_DIR/$f"
        error "These files should be included in the repository. Try: git checkout -- models/TTS/"
        exit 1
    fi
done
info "All model data files present"

# ── Python venv + dependencies ───────────────────────────────────────────────

if [[ ! -d "$VENV_DIR" ]]; then
    info "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
else
    info "Virtual environment already exists"
fi

info "Installing Python package (editable) into venv..."
"$VENV_DIR/bin/pip" install -e "$SCRIPT_DIR"

# ── Claude Code hook configuration ───────────────────────────────────────────

info "Configuring Claude Code Stop hook..."
mkdir -p "$(dirname "$SETTINGS_FILE")"

HOOK_CMD="$VENV_DIR/bin/python $HOOK_PY"

if [[ -f "$SETTINGS_FILE" ]]; then
    if grep -q "manajeure/hook.py" "$SETTINGS_FILE" 2>/dev/null; then
        info "Hook already configured in $SETTINGS_FILE"
    else
        python3 -c "
import json
with open('$SETTINGS_FILE') as f:
    settings = json.load(f)
hook_entry = {
    'hooks': [{
        'type': 'command',
        'command': '$HOOK_CMD',
        'timeout': 5,
        'async': True
    }]
}
settings.setdefault('hooks', {}).setdefault('Stop', []).append(hook_entry)
with open('$SETTINGS_FILE', 'w') as f:
    json.dump(settings, f, indent=2)
print('Hook added to existing settings')
"
    fi
else
    python3 -c "
import json
settings = {
    'hooks': {
        'Stop': [{
            'hooks': [{
                'type': 'command',
                'command': '$HOOK_CMD',
                'timeout': 5,
                'async': True
            }]
        }]
    }
}
with open('$SETTINGS_FILE', 'w') as f:
    json.dump(settings, f, indent=2)
print('Created settings with hook')
"
fi

# ── Done ─────────────────────────────────────────────────────────────────────

echo ""
info "Installation complete!"
echo ""
echo "  Usage:"
echo "    1. Run the daemon:    $VENV_DIR/bin/python $SCRIPT_DIR/src/manajeure/daemon.py [glados|kokoro] [options]"
echo "       (default: glados voice, Right Ctrl trigger, auto text injection)"
echo "    2. Open Claude Code (or any app) in another window"
echo "    3. Hold the trigger key to record, release to send"
echo "    4. Agent responses will be spoken automatically"
echo ""
echo "  Options:  --key KEY        trigger key (default: ctrl_r, try scroll_lock for games)"
echo "            --inject METHOD  xdotool|clipboard|auto (default: auto)"
echo "            --no-enter       skip Enter after injection"
echo "            --suppress       suppress trigger key from other apps (X11, adds latency)"
echo ""
