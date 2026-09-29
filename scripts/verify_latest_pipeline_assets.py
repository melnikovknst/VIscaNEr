#!/usr/bin/env python3
"""Verify all files required by the latest five-stream inference pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models" / "five_stream_transformer" / "deployment_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures: list[str] = []
    for name, artifact in manifest["artifacts"].items():
        path = ROOT / artifact["path"]
        if not path.is_file():
            failures.append(f"{name}: missing {path}")
            continue
        actual = sha256(path)
        if actual != artifact["sha256"]:
            failures.append(f"{name}: expected {artifact['sha256']}, got {actual}")
        else:
            print(f"OK | {name} | {artifact['path']}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"READY | {manifest['pipeline']} | {manifest['gallery_identities']} identities")


if __name__ == "__main__":
    main()
