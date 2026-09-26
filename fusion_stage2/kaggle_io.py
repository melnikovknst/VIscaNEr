"""Kaggle input helpers for datasets containing archived directories.

Kaggle's CLI does not upload nested directories in its default ``skip`` mode.
With ``--dir-mode zip`` each directory is mounted as ``<name>.zip`` instead.
The Stage-II notebooks use this module to accept both representations:
already expanded directories and Kaggle-created ZIP/TAR archives.
"""

from __future__ import annotations

import shutil
import tarfile
import zipfile
from pathlib import Path


def _safe_destination(root: Path, member_name: str) -> Path:
    destination = (root / member_name).resolve()
    if destination != root.resolve() and root.resolve() not in destination.parents:
        raise RuntimeError(f"Unsafe archive member: {member_name}")
    return destination


def _extract_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as handle:
            for member in handle.infolist():
                _safe_destination(destination, member.filename)
            handle.extractall(destination)
        return
    if archive.name.endswith((".tar", ".tar.gz", ".tgz")):
        with tarfile.open(archive) as handle:
            for member in handle.getmembers():
                _safe_destination(destination, member.name)
            handle.extractall(destination)
        return
    raise ValueError(f"Unsupported dataset archive: {archive}")


def materialize_dataset_dirs(
    source_root: Path,
    directory_names: tuple[str, ...],
    cache_root: Path,
) -> Path:
    """Return a dataset root containing every requested directory.

    If the Kaggle input already exposes directories, it is returned unchanged.
    Otherwise metadata files are linked into a writable cache and archived
    directories are extracted there. Existing complete caches are reused.
    """

    source_root = source_root.resolve()
    if all((source_root / name).is_dir() for name in directory_names):
        return source_root

    target_root = cache_root / source_root.name
    complete_marker = target_root / ".complete"
    if complete_marker.is_file() and all(
        (target_root / name).is_dir() for name in directory_names
    ):
        return target_root

    shutil.rmtree(target_root, ignore_errors=True)
    target_root.mkdir(parents=True, exist_ok=True)
    for source in source_root.iterdir():
        if source.is_file() and not source.name.endswith(
            (".zip", ".tar", ".tar.gz", ".tgz")
        ):
            (target_root / source.name).symlink_to(source)

    for name in directory_names:
        source_directory = source_root / name
        target_directory = target_root / name
        if source_directory.is_dir():
            target_directory.symlink_to(source_directory, target_is_directory=True)
            continue
        archives = [
            source_root / f"{name}.zip",
            source_root / f"{name}.tar",
            source_root / f"{name}.tar.gz",
            source_root / f"{name}.tgz",
        ]
        archive = next((candidate for candidate in archives if candidate.is_file()), None)
        if archive is None:
            raise FileNotFoundError(
                f"Dataset {source_root.name} is missing both {name}/ and "
                f"{name}.zip (or TAR equivalent). Refresh the Kaggle dataset "
                "with --dir-mode zip."
            )
        print(f"SETUP | extracting {archive.name}...", flush=True)
        _extract_archive(archive, target_directory)

    complete_marker.write_text("ok\n", encoding="utf-8")
    return target_root
