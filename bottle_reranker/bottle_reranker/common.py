"""Shared helpers: hashing, tabular IO, run records and the resume ledger.

Only the standard library is used here so the auditing stages (steps 1, 2, 5, 8)
run on a bare interpreter, without torch, OpenCV or pandas installed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


# ----------------------------------------------------------------- hashing
def sha256_file(path: str | os.PathLike[str], *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_unit_interval(text: str, seed: int) -> float:
    """The DINO retrieval split hash, reproduced bit-for-bit.

    ``dinov3_retrieval._stable_unit_interval`` decides which identities are held
    out for generalisation. The reranker must place an identity on the same side
    of that boundary, so this function has to stay byte-identical to it.
    """
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


# -------------------------------------------------------------- tabular IO
def read_csv_rows(path: str | os.PathLike[str], *, encoding: str = "utf-8") -> list[dict[str, str]]:
    with open(path, "r", encoding=encoding, newline="") as handle:
        return list(csv.DictReader(handle))


def iter_csv_rows(path: str | os.PathLike[str], *, encoding: str = "utf-8") -> Iterator[dict[str, str]]:
    with open(path, "r", encoding=encoding, newline="") as handle:
        yield from csv.DictReader(handle)


def csv_header(path: str | os.PathLike[str], *, encoding: str = "utf-8") -> list[str]:
    with open(path, "r", encoding=encoding, newline="") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def write_csv_rows(
    path: str | os.PathLike[str],
    rows: Sequence[dict[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> Path:
    """Write rows atomically so an interrupted run never leaves a half table."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        seen: dict[str, None] = {}
        for row in rows:
            for key in row:
                seen.setdefault(key, None)
        fieldnames = list(seen)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(target)
    return target


def write_json(path: str | os.PathLike[str], payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
    return target


def read_json(path: str | os.PathLike[str], default: Any = None) -> Any:
    target = Path(path)
    if not target.is_file():
        return default
    return json.loads(target.read_text(encoding="utf-8"))


def write_text(path: str | os.PathLike[str], text: str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


# ------------------------------------------------------------ resume ledger
class Ledger:
    """Append-only record of finished work items, for resumable stages.

    One JSON object per line. ``done`` is loaded once at start-up; a rerun skips
    every key already present, so an interrupted full build continues instead of
    redoing the expensive segmentation and crop rendering.
    """

    def __init__(self, path: str | os.PathLike[str], *, enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self._done: set[str] = set()
        self._handle = None
        if self.enabled and self.path.is_file():
            with open(self.path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._done.add(json.loads(line)["key"])
                    except (ValueError, KeyError):
                        # A torn last line from a hard kill: ignore it, the item
                        # is simply redone.
                        continue

    def __contains__(self, key: str) -> bool:
        return self.enabled and key in self._done

    def __len__(self) -> int:
        return len(self._done)

    def open(self) -> "Ledger":
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = open(self.path, "a", encoding="utf-8")
        return self

    def mark(self, key: str, **payload: Any) -> None:
        self._done.add(key)
        if self._handle is None:
            return
        self._handle.write(json.dumps({"key": key, **payload}, ensure_ascii=False, default=str) + "\n")
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "Ledger":
        return self.open()

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ------------------------------------------------------------- run records
def git_revision(root: str | os.PathLike[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def environment_record(root: str | os.PathLike[str]) -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("numpy", "cv2", "torch", "torchvision", "ultralytics", "PIL", "pandas", "transformers"):
        try:
            module = __import__(name)
        except Exception:  # noqa: BLE001 - absence is the information we want
            packages[name] = "missing"
        else:
            packages[name] = str(getattr(module, "__version__", "unknown"))
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_revision": git_revision(root),
        "packages": packages,
    }


def stage_record(
    stage: str,
    *,
    config_fingerprint: dict[str, Any],
    project_root: str | os.PathLike[str],
    **extra: Any,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "environment": environment_record(project_root),
        "config_fingerprint": config_fingerprint,
        **extra,
    }


# --------------------------------------------------------------- reporting
class Progress:
    """Minimal progress logging; no tqdm dependency for headless runs."""

    def __init__(self, total: int, label: str, *, every: int = 200) -> None:
        self.total = total
        self.label = label
        self.every = max(1, every)
        self.count = 0
        self.started = time.monotonic()

    def step(self, n: int = 1) -> None:
        self.count += n
        if self.count % self.every == 0 or self.count == self.total:
            elapsed = time.monotonic() - self.started
            rate = self.count / elapsed if elapsed > 0 else 0.0
            remaining = (self.total - self.count) / rate if rate > 0 else float("nan")
            print(
                f"  {self.label}: {self.count}/{self.total} "
                f"({100 * self.count / max(1, self.total):.1f}%) "
                f"{rate:.1f}/s eta {remaining / 60:.1f}min",
                flush=True,
            )


def percentage(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 2) if whole else 0.0


def summarise_counts(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
