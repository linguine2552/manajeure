#!/usr/bin/env python3
"""Claude Code Stop hook — sends assistant response to manajeure daemon via socket."""

import json
import os
import platform
import socket
import sys
from pathlib import Path

IS_WINDOWS = platform.system() == "Windows"

def _sock_path() -> str:
    if IS_WINDOWS:
        return os.path.join(os.environ.get("TEMP", "C:\\Temp"), "manajeure.sock")
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR", os.path.join(Path.home(), ".config", "manajeure"))
    return os.path.join(runtime_dir, "manajeure.sock")

def main():
    data = json.load(sys.stdin)
    text = data.get("last_assistant_message", "")
    if not text:
        return
    try:
        if IS_WINDOWS:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect(("127.0.0.1", 51983))
        else:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect(_sock_path())
        sock.sendall(text.encode("utf-8"))
        sock.close()
    except (ConnectionRefusedError, FileNotFoundError, OSError):
        pass  # daemon not running

if __name__ == "__main__":
    main()
