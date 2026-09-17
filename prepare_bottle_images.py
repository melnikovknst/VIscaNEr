"""Build one flat, traceable directory of the 45k bottle-scene images."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATASETS_ROOT = PROJECT_ROOT / "datasets"
OUTPUT_DIR = DATASETS_ROOT / "bottle_images_45k"
MANIFEST_PATH = OUTPUT_DIR / "bottle_images_manifest.csv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

SOURCE_DIRECTORIES = {
    "wine_scanner_2": DATASETS_ROOT / "wine-scanner 2" / "data" / "trainset" / "images",
    "wine_scanner_3": DATASETS_ROOT / "wine-scanner 3" / "data" / "trainset" / "images",
}


def safe_component(value: str, max_length: int = 80) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "_" for character in value)
    return cleaned[:max_length].strip("_") or "image"


def build_flat_bottle_directory() -> tuple[int, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []
    target_names: set[str] = set()

    for source_name, source_root in SOURCE_DIRECTORIES.items():
        if not source_root.is_dir():
            raise FileNotFoundError(f"Bottle source directory not found: {source_root}")

        source_images = sorted(
            path
            for path in source_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        for source_path in source_images:
            relative_path = source_path.relative_to(source_root)
            wine_slug = source_path.parent.name
            digest = hashlib.sha1(
                f"{source_name}/{relative_path.as_posix()}".encode("utf-8")
            ).hexdigest()[:12]
            target_name = (
                f"{source_name}__{safe_component(wine_slug)}__"
                f"{safe_component(source_path.stem, 24)}__{digest}{source_path.suffix.lower()}"
            )
            if target_name in target_names:
                raise RuntimeError(f"Generated duplicate target name: {target_name}")
            target_names.add(target_name)

            target_path = OUTPUT_DIR / target_name
            if target_path.exists() or target_path.is_symlink():
                if not target_path.is_symlink() or target_path.resolve() != source_path.resolve():
                    raise RuntimeError(f"Unexpected existing output: {target_path}")
            else:
                target_path.symlink_to(source_path.resolve())

            manifest_rows.append(
                {
                    "source_dataset": source_name,
                    "source_path": str(source_path.resolve()),
                    "source_relative_path": relative_path.as_posix(),
                    "wine_slug": wine_slug,
                    "merged_path": str(target_path.absolute()),
                    "merged_filename": target_name,
                }
            )

    if len(manifest_rows) != 45_000:
        raise RuntimeError(f"Expected exactly 45,000 bottle images, found {len(manifest_rows):,}")

    temporary_manifest = MANIFEST_PATH.with_suffix(".tmp.csv")
    with temporary_manifest.open("w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    temporary_manifest.replace(MANIFEST_PATH)

    print(f"Bottle images ready: {len(manifest_rows):,}")
    print(f"Flat directory: {OUTPUT_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")
    return len(manifest_rows), MANIFEST_PATH


if __name__ == "__main__":
    build_flat_bottle_directory()
