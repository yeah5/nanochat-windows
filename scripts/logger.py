"""Lightweight unified logger for console and optional file output."""

from __future__ import annotations

from datetime import datetime
import atexit
import os
from pathlib import Path
from threading import Lock
from typing import Dict, TextIO

_FILE_HANDLES: Dict[str, TextIO] = {}
_FILE_LOCK = Lock()
should_write_file = True
log_file_path = "training.log"



def _get_handle(file_path: str) -> TextIO:
    with _FILE_LOCK:
        handle = _FILE_HANDLES.get(file_path)
        if handle is None or handle.closed:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            handle = open(file_path, "a", encoding="utf-8")
            _FILE_HANDLES[file_path] = handle
        return handle


def log(message: str, debug_mode = True, write_file = True) -> None:
    """Print timestamped logs to stdout and optionally append to a log file."""
    # str = string to write
    # write_file = whether to write to file or just print
    if debug_mode:
        #do nothing if debug_mode is false
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        line = f"{timestamp} {message}"
        print(line)

        if write_file:
            try:
                handle = _get_handle(log_file_path)
                handle.write(line + "\n")
                handle.flush()
            except Exception as exc:  # noqa: BLE001
                print(f"{timestamp} Warning: failed to write log file '{log_file_path}': {exc}")


def _close_handles() -> None:
    with _FILE_LOCK:
        for handle in _FILE_HANDLES.values():
            try:
                handle.flush()
                handle.close()
            except Exception:
                pass
        _FILE_HANDLES.clear()


atexit.register(_close_handles)
