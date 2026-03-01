#!/usr/bin/env python3
"""manajeure daemon: voice interface for AI agents.

Hold a trigger key to record, release to transcribe and inject into the focused window.
Agent responses arrive via socket and are spoken via GLaDOS/Kokoro TTS.
"""

import argparse
import json
import os
import platform
import queue
import re
import selectors
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from pickle import load

import numpy as np
import onnxruntime as ort
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard
from pynput.keyboard import Controller as KeyboardController, Key

ort.set_default_logger_severity(4)

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "TTS"
RECORD_RATE = 16000
IS_WINDOWS = platform.system() == "Windows"

def _sock_path() -> str:
    if IS_WINDOWS:
        return os.path.join(os.environ.get("TEMP", "C:\\Temp"), "manajeure.sock")
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR", os.path.join(Path.home(), ".config", "manajeure"))
    os.makedirs(runtime_dir, exist_ok=True)
    return os.path.join(runtime_dir, "manajeure.sock")

SOCK_PATH = _sock_path()

# ── Markdown stripping ──────────────────────────────────────────────────────

def strip_markdown(text: str) -> str:
    text = re.sub(r"```[\s\S]*?```", " code omitted ", text)
    text = re.sub(r"`[^`]+`", "", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"^[\s]*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Cross-platform clipboard + paste ────────────────────────────────────────

_kb_ctl = KeyboardController()

# Resolved at startup by parse_args(); used by inject_text() / speak_text()
_inject_method: str = "auto"
_send_enter: bool = True
_volume: float = 1.0

def inject_text(text: str):
    """Inject text into the focused window, then optionally press Enter."""
    if IS_WINDOWS:
        _inject_text_windows(text)
    else:
        _inject_text_linux(text)
    if _send_enter:
        time.sleep(0.15)
        _kb_ctl.press(Key.enter)
        _kb_ctl.release(Key.enter)

def _inject_text_windows(text: str):
    _set_clipboard_win(text)
    with _kb_ctl.pressed(Key.ctrl):
        _kb_ctl.press('v')
        _kb_ctl.release('v')

def _inject_text_linux(text: str):
    method = _inject_method
    if method == "auto":
        method = "xdotool" if shutil.which("xdotool") else "clipboard"
    if method == "xdotool":
        if not _inject_xdotool(text):
            _inject_clipboard_linux(text)
    else:
        _inject_clipboard_linux(text)

def _inject_xdotool(text: str) -> bool:
    """Type text via xdotool --clearmodifiers. Returns True on success."""
    try:
        time.sleep(0.2)  # let trigger key release propagate
        subprocess.run(
            ["xdotool", "type", "--clearmodifiers", "--delay", "12", "--", text],
            check=True, timeout=10,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"manajeure: xdotool failed ({e}), falling back to clipboard", file=sys.stderr)
        return False

def _inject_clipboard_linux(text: str):
    subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode(), check=True)
    with _kb_ctl.pressed(Key.ctrl, Key.shift):
        _kb_ctl.press('v')
        _kb_ctl.release('v')

def _set_clipboard_win(text: str):
    import ctypes
    kernel32 = ctypes.windll.kernel32
    user32 = ctypes.windll.user32
    user32.OpenClipboard(0)
    user32.EmptyClipboard()
    data = text.encode("utf-16-le") + b"\x00\x00"
    h = kernel32.GlobalAlloc(0x0042, len(data))
    p = kernel32.GlobalLock(h)
    ctypes.memmove(p, data, len(data))
    kernel32.GlobalUnlock(h)
    user32.SetClipboardData(13, h)  # CF_UNICODETEXT = 13
    user32.CloseClipboard()


# ── Shared phonemizer ───────────────────────────────────────────────────────

def _pkl(p):
    with open(p, "rb") as f:
        return load(f)

def _ort_providers():
    return [p for p in ort.get_available_providers()
            if p not in ("TensorrtExecutionProvider", "CoreMLExecutionProvider")]


class Phonemizer:
    """ONNX-based grapheme-to-phoneme, shared by both TTS engines."""

    def __init__(self):
        self.ph_dict: dict = _pkl(MODELS_DIR / "lang_phoneme_dict.pkl")
        self.tok2idx: dict = _pkl(MODELS_DIR / "token_to_idx.pkl")
        self.idx2tok: dict = _pkl(MODELS_DIR / "idx_to_token.pkl")
        self.ph_sess = ort.InferenceSession(
            str(MODELS_DIR / "phomenizer_en.onnx"), providers=_ort_providers())

    def _encode_word(self, word):
        chars = [c for ch in word for c in [ch.lower()] * 3]
        seq = [self.tok2idx[c] for c in chars if c in self.tok2idx]
        return [self.tok2idx["<start>"], *seq, self.tok2idx["<end>"]]

    def phonemize(self, text: str) -> str:
        punc_set = set("().,:?!/– -")
        words = re.split(r"([().,:?!/– ])", text)
        words = [w for w in words if w]

        result_phons = []
        to_predict = []
        to_predict_idx = []

        for i, w in enumerate(words):
            if not w or w in punc_set:
                result_phons.append(w)
            elif w.lower() in self.ph_dict:
                result_phons.append(self.ph_dict[w.lower()])
            else:
                result_phons.append(None)
                to_predict.append(w)
                to_predict_idx.append(i)

        if to_predict:
            batch = [self._encode_word(w) for w in to_predict]
            padded = np.zeros((len(batch), 64), dtype=np.int64)
            for j, seq in enumerate(batch):
                length = min(len(seq), 64)
                padded[j, :length] = seq[:length]
            outs = self.ph_sess.run(None, {self.ph_sess.get_inputs()[0].name: padded})
            ids = np.argmax(outs[0], axis=2)
            for j, idx in enumerate(to_predict_idx):
                row = ids[j]
                mask = np.concatenate(([True], row[1:] != row[:-1]))
                row = row[mask]
                row = row[row != 0]
                stop = np.where(row == 2)[0]
                if len(stop):
                    row = row[:stop[0]]
                decoded = "".join(
                    self.idx2tok[t.item()] for t in row
                    if t.item() in self.idx2tok
                    and self.idx2tok[t.item()] not in ("_", "<end>", "<en_us>"))
                result_phons[idx] = decoded

        return "".join(p for p in result_phons if p is not None)


# ── GLaDOS TTS (VITS/Piper model) ──────────────────────────────────────────

class GladosTTS:
    SAMPLE_RATE = 22050

    def __init__(self, phonemizer: Phonemizer):
        self.phonemizer = phonemizer
        self.id_map = _pkl(MODELS_DIR / "phoneme_to_id.pkl")

        with open(MODELS_DIR / "glados.json") as f:
            config = json.load(f)
        self.noise_scale = config.get("inference", {}).get("noise_scale", 0.667)
        self.length_scale = config.get("inference", {}).get("length_scale", 1.0)
        self.noise_w = config.get("inference", {}).get("noise_w", 0.8)

        self.tts_sess = ort.InferenceSession(
            str(MODELS_DIR / "glados.onnx"), providers=_ort_providers())

    def synthesize(self, text: str) -> np.ndarray:
        phonemes = self.phonemizer.phonemize(text)
        ids = self._phonemes_to_ids(phonemes)
        if not ids:
            return np.array([], dtype=np.float32)
        return self._synthesize_ids(ids)

    def _phonemes_to_ids(self, phonemes: str) -> list[int]:
        ids: list[int] = list(self.id_map.get("^", [1]))  # BOS
        for p in phonemes:
            if p in self.id_map:
                ids.extend(self.id_map[p])
                ids.extend(self.id_map.get("_", [0]))  # PAD
        ids.extend(self.id_map.get("$", [2]))  # EOS
        return ids

    def _synthesize_ids(self, ids: list[int]) -> np.ndarray:
        phoneme_ids = np.expand_dims(np.array(ids, dtype=np.int64), 0)
        lengths = np.array([phoneme_ids.shape[1]], dtype=np.int64)
        scales = np.array([self.noise_scale, self.length_scale, self.noise_w], dtype=np.float32)
        audio = self.tts_sess.run(None, {
            "input": phoneme_ids,
            "input_lengths": lengths,
            "scales": scales,
            "sid": None,
        })[0].squeeze()  # (1,1,1,N) -> (N,)
        return audio.astype(np.float32)


# ── Kokoro TTS ──────────────────────────────────────────────────────────────

class KokoroTTS:
    MAX_PHONEME_LEN = 510
    VOICE = "af_alloy"
    SAMPLE_RATE = 24000

    def __init__(self, phonemizer: Phonemizer):
        self.phonemizer = phonemizer
        self.voices = np.load(MODELS_DIR / "kokoro-voices-v1.0.bin")
        self.vocab = self._build_vocab()
        self.tts_sess = ort.InferenceSession(
            str(MODELS_DIR / "kokoro-v1.0.fp16.onnx"), providers=_ort_providers())

    @staticmethod
    def _build_vocab():
        pad = "$"
        punc = ';:,.!?¡¿—…"«»"" '
        lett = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        ipa = ("ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻ"
               "ʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ")
        return {c: i for i, c in enumerate([pad, *punc, *lett, *ipa])}

    def synthesize(self, text: str) -> np.ndarray:
        phonemes = self.phonemizer.phonemize(text)
        ids = [i for i in map(self.vocab.get, phonemes) if i is not None]
        if not ids:
            return np.array([], dtype=np.float32)
        if len(ids) > self.MAX_PHONEME_LEN:
            ids = ids[:self.MAX_PHONEME_LEN]
        voice = self.voices[self.VOICE][len(ids)]
        tokens = [[0, *ids, 0]]
        audio = self.tts_sess.run(None, {
            "tokens": tokens,
            "style": voice,
            "speed": np.ones(1, dtype=np.float32),
        })[0]
        if len(audio) > 8000:
            return np.array(audio[:-8000], dtype=np.float32)
        return np.array(audio, dtype=np.float32)


# ── CLI ──────────────────────────────────────────────────────────────────────

def _resolve_trigger_key(name: str) -> keyboard.Key:
    """Map a CLI key name (e.g. 'scroll_lock') to a pynput Key enum member."""
    members = {k: v for k, v in Key.__members__.items()}
    if name in members:
        return members[name]
    print(f"manajeure: unknown key '{name}'", file=sys.stderr)
    print(f"  available keys: {', '.join(sorted(members))}", file=sys.stderr)
    sys.exit(1)

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="manajeure",
        description="Voice interface for AI agents",
    )
    p.add_argument("voice", nargs="?", default="glados", choices=["glados", "kokoro"],
                   help="TTS voice (default: glados)")
    p.add_argument("--key", default="ctrl_r", metavar="KEY",
                   help="trigger key name, e.g. scroll_lock, pause, f13 (default: ctrl_r)")
    p.add_argument("--no-enter", action="store_true",
                   help="skip Enter keypress after text injection")
    p.add_argument("--inject", default="auto", choices=["auto", "xdotool", "clipboard"],
                   help="text injection method (default: auto — tries xdotool, falls back to clipboard)")
    p.add_argument("--suppress", action="store_true",
                   help="suppress trigger key from reaching other apps (X11 only, adds latency)")
    p.add_argument("--no-tts", action="store_true",
                   help="disable TTS entirely — STT only, skips loading TTS models")
    p.add_argument("--volume", type=int, default=100, metavar="PCT",
                   help="TTS playback volume, 0-100 (default: 100)")
    return p.parse_args(argv)


# ── Main event loop ─────────────────────────────────────────────────────────

def main():
    global _inject_method, _send_enter, _volume

    args = parse_args()
    trigger_key = _resolve_trigger_key(args.key)
    _inject_method = args.inject
    _send_enter = not args.no_enter
    _volume = max(0, min(100, args.volume)) / 100.0

    print("manajeure: loading models...")
    whisper = WhisperModel("base", device="cuda", compute_type="float16")

    if args.no_tts:
        tts = None
        tts_rate = None
        print("manajeure: models loaded (STT only, TTS disabled)")
    else:
        phonemizer = Phonemizer()
        if args.voice == "glados":
            tts = GladosTTS(phonemizer)
        else:
            tts = KokoroTTS(phonemizer)
        tts_rate = tts.SAMPLE_RATE
        print(f"manajeure: models loaded (voice={args.voice}, rate={tts_rate}, volume={args.volume}%)")

    # Key events from pynput thread -> main thread
    key_events: queue.Queue[str] = queue.Queue()

    if args.suppress and not IS_WINDOWS:
        # Suppress mode: grab all keys, re-inject non-trigger keys
        def on_press(key):
            if key == trigger_key:
                key_events.put("press")
            else:
                try:
                    _kb_ctl.press(key)
                except Exception:
                    pass

        def on_release(key):
            if key == trigger_key:
                key_events.put("release")
            else:
                try:
                    _kb_ctl.release(key)
                except Exception:
                    pass

        listener = keyboard.Listener(on_press=on_press, on_release=on_release, suppress=True)
    else:
        def on_press(key):
            if key == trigger_key:
                key_events.put("press")

        def on_release(key):
            if key == trigger_key:
                key_events.put("release")

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)

    listener.start()
    suppress_note = " (suppress=on)" if args.suppress and not IS_WINDOWS else ""
    print(f"manajeure: keyboard listener started{suppress_note}")

    # Socket for TTS (Unix socket on Linux/macOS, TCP loopback on Windows)
    if IS_WINDOWS:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 51983))
        srv.listen(4)
        srv.setblocking(False)
        print("manajeure: listening on 127.0.0.1:51983")
    else:
        if os.path.exists(SOCK_PATH):
            os.unlink(SOCK_PATH)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(SOCK_PATH)
        os.chmod(SOCK_PATH, 0o600)  # owner-only access
        srv.listen(4)
        srv.setblocking(False)
        print(f"manajeure: listening on {SOCK_PATH}")

    sel = selectors.DefaultSelector()
    sel.register(srv.fileno(), selectors.EVENT_READ, "server")

    recording = False
    audio_chunks: list[np.ndarray] = []
    stream = None

    def start_recording():
        nonlocal recording, audio_chunks, stream
        sd.stop()
        audio_chunks = []
        stream = sd.InputStream(samplerate=RECORD_RATE, channels=1, dtype="float32",
                                callback=lambda data, *_: audio_chunks.append(data.copy()))
        stream.start()
        recording = True
        print("manajeure: recording...")

    def stop_recording_and_transcribe():
        nonlocal recording, stream
        if not recording:
            return
        stream.stop()
        stream.close()
        stream = None
        recording = False

        if not audio_chunks:
            print("manajeure: empty recording, skipping")
            return
        audio = np.concatenate(audio_chunks, axis=0).flatten()
        print(f"manajeure: transcribing {len(audio)/RECORD_RATE:.1f}s audio...")
        segments, _ = whisper.transcribe(audio, language="en")
        text = " ".join(seg.text for seg in segments).strip()
        if not text:
            print("manajeure: no speech detected")
            return
        print(f"manajeure: \"{text}\"")
        inject_text(text)

    def check_keyboard_interrupt() -> bool:
        try:
            event = key_events.get(timeout=0.02)
            if event == "press":
                sd.stop()
                start_recording()
                return True
        except queue.Empty:
            pass
        return False

    # Bluetooth A2DP wake-up: 500ms silence prepended to first chunk
    bt_pad = np.zeros(int(tts_rate * 0.5), dtype=np.float32) if tts else None

    def speak_text(text: str):
        if not tts:
            return
        cleaned = strip_markdown(text)
        if not cleaned:
            return
        sentences = re.split(r'(?<=[.!?])\s+', cleaned)
        first = True
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            try:
                audio = tts.synthesize(sentence)
                if len(audio) == 0:
                    continue
                if first:
                    audio = np.concatenate([bt_pad, audio])
                    first = False
                if _volume < 1.0:
                    audio = audio * _volume
                sd.play(audio, tts_rate)
                deadline = time.monotonic() + len(audio) / tts_rate
                while time.monotonic() < deadline:
                    if check_keyboard_interrupt():
                        return
            except Exception as e:
                print(f"manajeure: TTS error: {e}", file=sys.stderr)

    def handle_socket_client(conn):
        data = b""
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
        conn.close()
        text = data.decode("utf-8", errors="replace")
        if text:
            if tts:
                print(f"manajeure: speaking {len(text)} chars")
                speak_text(text)
            else:
                print(f"manajeure: received {len(text)} chars (TTS disabled, ignoring)")

    key_name = args.key.replace("_", " ").title()
    print(f"manajeure: ready — hold {key_name} to talk")

    try:
        while True:
            try:
                event = key_events.get(timeout=0.02)
                if event == "press" and not recording:
                    start_recording()
                elif event == "release" and recording:
                    stop_recording_and_transcribe()
            except queue.Empty:
                pass

            ready = sel.select(timeout=0)
            for key, _ in ready:
                if key.data == "server":
                    conn, _ = srv.accept()
                    handle_socket_client(conn)
    except KeyboardInterrupt:
        print("\nmanajeure: shutting down")
    finally:
        listener.stop()
        sel.close()
        srv.close()
        if not IS_WINDOWS and os.path.exists(SOCK_PATH):
            os.unlink(SOCK_PATH)


if __name__ == "__main__":
    main()
