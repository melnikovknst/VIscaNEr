#!/usr/bin/env python3
"""Run the complete local MPS evaluation and refresh the review queue."""

from __future__ import annotations

import subprocess
import sys

from .paths import DEFAULT_OUTPUT_DIR


def main() -> None:
    subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "bottle_dataset_curation.evaluate_all",
            "--include-fresh",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "bottle_dataset_curation.prepare",
            "--audit-csv",
            str(DEFAULT_OUTPUT_DIR / "all_retrieval_audit.csv"),
            "--skip-fresh",
        ],
        check=True,
    )
    print("FULL LOCAL CURATION QUEUE IS READY", flush=True)


if __name__ == "__main__":
    main()
