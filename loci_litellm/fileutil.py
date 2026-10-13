"""Small file helpers for the append-only files the plugin writes (cost ledger, delivery audit). stdlib only."""
from __future__ import annotations

import os
from pathlib import Path


def env_int(name: str, default: int) -> int:
    """An integer from the environment; unset, empty or malformed falls back to `default`."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def append_line(path: str, line: str, max_bytes: int = 0) -> None:
    """Append one line. When `max_bytes` > 0 and the file has reached it, rename it to `<path>.1`
    first (replacing an older one); 0 means unbounded.

    Rotation is unlocked: concurrent processes can lose a generation."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if max_bytes > 0:
        try:
            if target.stat().st_size >= max_bytes:
                os.replace(target, str(target) + ".1")
        except FileNotFoundError:
            pass
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(line if line.endswith("\n") else line + "\n")
