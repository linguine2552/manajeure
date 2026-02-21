# manajeure

**Talk to your robot assistants.**

A minimal voice interface for AI agent orchestration. Hold a key, speak, release — your words get transcribed and typed into whatever window is focused. Agent responses get spoken back to you.

Works with Claude Code, Discord, Slack, your terminal — anything with a text input.

## How it works

```
[Right Ctrl press]  → records audio from your mic
[Right Ctrl release] → Whisper transcribes → pastes into focused window → submits
[Agent responds]     → Stop hook sends text to daemon → TTS speaks it
```

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
.venv/bin/python src/manajeure/daemon.py          # GLaDOS voice (default)
.venv/bin/python src/manajeure/daemon.py kokoro    # Kokoro voice
```

## Requirements

- Python 3.11+
- CUDA GPU (for Whisper STT + ONNX TTS inference)
- **Linux**: X11, `xclip` (installed automatically by `install.sh`)
- **Windows**: works out of the box (uses native clipboard + pynput for key simulation)

## Architecture

~350 lines of Python total. No frameworks, no servers, no cloud APIs.

- **STT**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (base model, GPU, float16)
- **TTS**: GLaDOS VITS / [Kokoro](https://github.com/dnhkng/GLaDOS) ONNX on GPU via `onnxruntime-gpu`
- **Keyboard**: [pynput](https://github.com/moses-palmer/pynput) (cross-platform)
- **Audio I/O**: [sounddevice](https://python-sounddevice.readthedocs.io/)
- **Text injection**: `xclip` + pynput (Linux) / native ctypes + pynput (Windows)
- **Agent hook**: Unix socket (Linux/macOS) or TCP loopback (Windows)

## Disclaimer

100% vibe coded by a human and [Claude Code](https://claude.ai/claude-code). Built for fun and shipped accordingly.

## License

MIT
