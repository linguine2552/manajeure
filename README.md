# manajeure

**Talk to your robot assistants.**

A minimal voice interface for AI agent orchestration. Hold a key, speak, release — your words get transcribed and typed into whatever window is focused. Agent responses get spoken back to you.

Works with Claude Code, Discord, Slack, your terminal — anything with a text input.

## How it works

```
[trigger key press]   → records audio from your mic
[trigger key release] → Whisper transcribes → injects text into focused window → submits
[Agent responds]      → Stop hook sends text to daemon → TTS speaks it
```

The trigger key defaults to Right Ctrl but can be changed with `--key` (e.g. `scroll_lock`, `pause`, `f13`).

## Voices

- **GLaDOS** (default) — VITS/Piper model, 22kHz, that familiar passive-aggressive AI voice
- **Kokoro** — multi-voice model, 24kHz, more natural sounding

## Quick start

```bash
git clone https://github.com/linguine2552/manajeure.git
cd manajeure
./install.sh   # downloads models (~300MB), sets up venv, configures Claude Code hook
```

Run the daemon:
```bash
.venv/bin/python src/manajeure/daemon.py                # GLaDOS voice (default)
.venv/bin/python src/manajeure/daemon.py kokoro          # Kokoro voice
.venv/bin/python src/manajeure/daemon.py --key scroll_lock --inject xdotool  # game-friendly
```

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `voice` | `glados` | TTS voice (`glados` or `kokoro`) |
| `--key KEY` | `ctrl_r` | Trigger key name (e.g. `scroll_lock`, `pause`, `f13`) |
| `--inject METHOD` | `auto` | Text injection: `xdotool`, `clipboard`, or `auto` |
| `--no-enter` | off | Skip Enter keypress after text injection |
| `--suppress` | off | Suppress trigger key from reaching other apps (X11 only) |

## Requirements

- Python 3.11+
- CUDA GPU (for Whisper STT + ONNX TTS inference)
- **Linux**: X11, `xclip`, `xdotool` (installed automatically by `install.sh`)
- **Windows**: works out of the box (uses native clipboard + pynput for key simulation)

## Architecture

~350 lines of Python total. No frameworks, no servers, no cloud APIs.

- **STT**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (base model, GPU, float16)
- **TTS**: GLaDOS VITS / [Kokoro](https://github.com/dnhkng/GLaDOS) ONNX on GPU via `onnxruntime-gpu`
- **Keyboard**: [pynput](https://github.com/moses-palmer/pynput) (cross-platform)
- **Audio I/O**: [sounddevice](https://python-sounddevice.readthedocs.io/)
- **Text injection**: `xdotool` or `xclip` + pynput (Linux) / native ctypes + pynput (Windows)
- **Agent hook**: Unix socket (Linux/macOS) or TCP loopback (Windows)

## Fullscreen games

The default Right Ctrl + clipboard paste doesn't work well in fullscreen games — the trigger key passes through, and `Ctrl+Shift+V` gets interpreted as raw keypresses. Use xdotool injection with a non-conflicting trigger key instead:

```bash
.venv/bin/python src/manajeure/daemon.py --key scroll_lock --inject xdotool --no-enter
```

- **`--key scroll_lock`** — most games don't bind Scroll Lock, so it won't interfere
- **`--inject xdotool`** — types text character-by-character with `--clearmodifiers`, bypassing clipboard entirely
- **`--no-enter`** — useful if the game's chat requires a different submit flow

If the trigger key still leaks through, add `--suppress` to grab all keyboard input and re-inject non-trigger keys. This adds slight latency and only works on X11.

## Disclaimer

100% vibe coded by a human and [Claude Code](https://claude.ai/claude-code). Built for fun and shipped accordingly.

## License

MIT
