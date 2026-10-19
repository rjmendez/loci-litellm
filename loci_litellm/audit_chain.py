"""Append-only, hash-chained audit log with rotation, gzip and retention. stdlib only.

Every entry is one JSON line: {"seq", "ts", "kind", "data", "prev", "hash"} where
hash = sha256(prev + "\\n" + canonical_json({seq, ts, kind, data})) and prev is the previous entry's
hash (64 zeros for the first entry ever written). Changing, removing or reordering any line breaks the
chain at an exact entry index, reported by `verify()`. A side file `<name>.head` (written through a
temp file and a rename) holds the last (seq, hash) so that dropping whole trailing lines is detected too.

Files in the directory:
  <name>.jsonl                          the active segment
  <name>.<first_seq>.<last_ts>.jsonl    a closed segment (rotated at `segment_bytes`)
  <name>.<first_seq>.<last_ts>.jsonl.gz a closed segment older than `gzip_after_days`
Closed segments older than `delete_after_days` are deleted, and the oldest closed segments are deleted while
the total size is above `max_total_bytes` (default 256 MB). A deletion is itself recorded as a `retention`
entry. After a deletion the oldest retained entry is the start of the verified window: its `prev` is not
checked against anything.

A write that fails raises `AuditUnavailable`; callers decide whether that is fatal. This module has no
policy of its own.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

GENESIS = "0" * 64
DAY_S = 86400.0

_SEGMENT_RE = re.compile(r"^(?P<name>.+)\.(?P<first>\d{12})\.(?P<last_ts>\d+)\.jsonl(?P<gz>\.gz)?$")


class AuditUnavailable(Exception):
    """The audit chain could not be read or appended to."""


@dataclass(frozen=True)
class ChainCheck:
    ok: bool
    entries: int                 # entries checked before the break (all of them when ok)
    break_index: Optional[int]   # seq of the first bad entry; None when ok
    reason: str                  # "" when ok
    start_seq: int               # seq of the first retained entry (-1 when there are none)


def _canonical(seq: int, ts: float, kind: str, data: Any) -> str:
    return json.dumps({"seq": seq, "ts": ts, "kind": kind, "data": data}, sort_keys=True, separators=(",", ":"))


def entry_hash(prev: str, seq: int, ts: float, kind: str, data: Any) -> str:
    return hashlib.sha256((prev + "\n" + _canonical(seq, ts, kind, data)).encode("utf-8")).hexdigest()


def _open_text(path: Path, mode: str = "rt"):
    if path.name.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8")


class AuditChain:
    def __init__(self, directory: str, name: str = "audit", *, segment_bytes: int = 16 * 1024 * 1024,
                 max_total_bytes: int = 256 * 1024 * 1024, gzip_after_days: float = 7.0,
                 delete_after_days: float = 60.0, clock: Callable[[], float] = time.time) -> None:
        self.dir = Path(directory)
        self.name = name
        self.segment_bytes = segment_bytes
        self.max_total_bytes = max_total_bytes
        self.gzip_after_days = gzip_after_days
        self.delete_after_days = delete_after_days
        self._clock = clock
        self._lock = threading.Lock()
        self._loaded = False
        self._seq = -1
        self._hash = GENESIS
        self._active_first_seq: Optional[int] = None
        self._active_last_ts = 0.0

    # ------------------------------------------------------------------ paths
    @property
    def active_path(self) -> Path:
        return self.dir / f"{self.name}.jsonl"

    @property
    def head_path(self) -> Path:
        return self.dir / f"{self.name}.head"

    def segments(self) -> list[Path]:
        """Closed segments, oldest first."""
        found: list[tuple[int, Path]] = []
        try:
            names = list(self.dir.iterdir())
        except FileNotFoundError:
            return []
        for path in names:
            match = _SEGMENT_RE.match(path.name)
            if match and match.group("name") == self.name:
                found.append((int(match.group("first")), path))
        return [p for _, p in sorted(found)]

    def _all_files(self) -> list[Path]:
        files = self.segments()
        if self.active_path.exists():
            files.append(self.active_path)
        return files

    # ------------------------------------------------------------------ reading
    def entries(self) -> Iterator[dict]:
        """Every retained entry in order. Unparseable lines are skipped here; `verify()` reports them."""
        for path in self._all_files():
            try:
                with _open_text(path) as handle:
                    for line in handle:
                        try:
                            entry = json.loads(line)
                        except ValueError:
                            continue
                        if isinstance(entry, dict):
                            yield entry
            except OSError as exc:
                raise AuditUnavailable(f"cannot read {path}: {exc!r}") from exc

    def _read_head(self) -> Optional[dict]:
        try:
            head = json.loads(self.head_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            return {"seq": -2, "hash": ""}   # unreadable anchor: forces a verify failure
        return head if isinstance(head, dict) else {"seq": -2, "hash": ""}

    def verify(self) -> ChainCheck:
        expected_seq: Optional[int] = None
        prev: Optional[str] = None
        count = 0
        start = -1
        last_seq = -1
        hashes: dict[int, str] = {}
        for path in self._all_files():
            try:
                handle = _open_text(path)
            except OSError as exc:
                return ChainCheck(False, count, expected_seq if expected_seq is not None else 0,
                                  f"unreadable segment {path.name}: {exc!r}", start)
            with handle:
                try:
                    lines = handle.read().split("\n")
                except (OSError, EOFError) as exc:
                    return ChainCheck(False, count, expected_seq if expected_seq is not None else 0,
                                      f"unreadable segment {path.name}: {exc!r}", start)
            if lines and lines[-1] == "":
                lines.pop()
            for line in lines:
                try:
                    entry = json.loads(line)
                    seq, ts, kind, data = entry["seq"], entry["ts"], entry["kind"], entry["data"]
                    entry_prev, digest = entry["prev"], entry["hash"]
                    if not isinstance(seq, int) or isinstance(seq, bool):
                        raise ValueError("seq is not an integer")
                except (ValueError, KeyError, TypeError) as exc:
                    bad = expected_seq if expected_seq is not None else 0
                    return ChainCheck(False, count, bad, f"unparseable entry: {exc!r}", start)
                if expected_seq is None:
                    expected_seq, start = seq, seq
                    prev = entry_prev        # window start: its own prev is taken as given
                if seq != expected_seq:
                    return ChainCheck(False, count, expected_seq, f"seq {seq} where {expected_seq} was expected", start)
                if entry_prev != prev:
                    return ChainCheck(False, count, expected_seq, "prev hash does not match the previous entry", start)
                if entry_hash(entry_prev, seq, ts, kind, data) != digest:
                    return ChainCheck(False, count, expected_seq, "entry hash does not match its content", start)
                hashes[seq] = digest
                prev, last_seq = digest, seq
                expected_seq += 1
                count += 1
        head = self._read_head()
        if head is not None:
            head_seq = head.get("seq")
            if not isinstance(head_seq, int) or head_seq < -1:
                return ChainCheck(False, count, max(last_seq + 1, 0), "head anchor unreadable", start)
            if head_seq > last_seq and head_seq != -1:
                return ChainCheck(False, count, last_seq + 1, "chain is shorter than its head anchor (truncated)", start)
            if head_seq in hashes and hashes[head_seq] != head.get("hash"):
                return ChainCheck(False, count, head_seq, "entry at the head anchor was rewritten", start)
        return ChainCheck(True, count, None, "", start)

    # ------------------------------------------------------------------ writing
    def _load_tail(self) -> None:
        last: Optional[dict] = None
        first_active: Optional[dict] = None
        try:
            if self.active_path.exists():
                with open(self.active_path, "r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        entry = json.loads(line)
                        if first_active is None:
                            first_active = entry
                        last = entry
            if last is None:
                segments = self.segments()
                if segments:
                    with _open_text(segments[-1]) as handle:
                        for line in handle:
                            if line.strip():
                                last = json.loads(line)
            if last is not None:
                self._seq, self._hash = int(last["seq"]), str(last["hash"])
                self._active_last_ts = float(last["ts"])
            if first_active is not None:
                self._active_first_seq = int(first_active["seq"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise AuditUnavailable(f"audit chain tail unreadable: {exc!r}") from exc
        self._loaded = True

    def _write_head(self) -> None:
        tmp = self.head_path.with_name(self.head_path.name + ".tmp")
        tmp.write_text(json.dumps({"seq": self._seq, "hash": self._hash}, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, self.head_path)

    def _rotate_if_needed(self) -> None:
        try:
            size = self.active_path.stat().st_size
        except FileNotFoundError:
            return
        if size < self.segment_bytes or self._active_first_seq is None:
            return
        target = self.dir / f"{self.name}.{self._active_first_seq:012d}.{int(self._active_last_ts)}.jsonl"
        os.replace(self.active_path, target)
        self._active_first_seq = None

    def append(self, kind: str, data: Any) -> dict:
        """Append one entry and return it. Raises AuditUnavailable when the entry could not be written."""
        with self._lock:
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
                if not self._loaded:
                    self._load_tail()
                self._rotate_if_needed()
                ts = round(float(self._clock()), 3)
                seq = self._seq + 1
                digest = entry_hash(self._hash, seq, ts, kind, data)
                entry = {"seq": seq, "ts": ts, "kind": kind, "data": data, "prev": self._hash, "hash": digest}
                line = json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n"
                with open(self.active_path, "a", encoding="utf-8") as handle:
                    handle.write(line)
                    handle.flush()
                    os.fsync(handle.fileno())
                self._seq, self._hash, self._active_last_ts = seq, digest, ts
                if self._active_first_seq is None:
                    self._active_first_seq = seq
                self._write_head()
                return entry
            except AuditUnavailable:
                raise
            except (OSError, TypeError, ValueError) as exc:
                raise AuditUnavailable(f"audit append failed: {exc!r}") from exc

    # ------------------------------------------------------------------ retention
    def maintenance(self) -> dict:
        """Gzip closed segments older than gzip_after_days, delete those older than delete_after_days and then the
        oldest ones while the total size is above max_total_bytes. Returns what it did."""
        now = float(self._clock())
        gzipped: list[str] = []
        deleted: list[str] = []
        with self._lock:
            for path in self.segments():
                match = _SEGMENT_RE.match(path.name)
                if match is None:
                    continue
                age_days = (now - int(match.group("last_ts"))) / DAY_S
                try:
                    if age_days > self.delete_after_days:
                        path.unlink()
                        deleted.append(path.name)
                    elif age_days > self.gzip_after_days and not match.group("gz"):
                        tmp = path.with_name(path.name + ".gz.tmp")
                        with open(path, "rb") as src, gzip.open(tmp, "wb") as dst:
                            dst.write(src.read())
                        os.replace(tmp, path.with_name(path.name + ".gz"))
                        path.unlink()
                        gzipped.append(path.name)
                except OSError as exc:
                    raise AuditUnavailable(f"audit maintenance failed on {path.name}: {exc!r}") from exc
            while True:
                files = self._all_files()
                total = sum(p.stat().st_size for p in files)
                closed = self.segments()
                if total <= self.max_total_bytes or not closed:
                    break
                closed[0].unlink()
                deleted.append(closed[0].name)
        if deleted:
            self.append("retention", {"deleted": deleted, "reason": "age_or_size"})
        return {"gzipped": gzipped, "deleted": deleted}

    def total_bytes(self) -> int:
        return sum(p.stat().st_size for p in self._all_files())
