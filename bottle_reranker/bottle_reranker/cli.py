"""Command line entry point: ``python -m bottle_reranker.cli <stage>``."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from .config import Config, ConfigError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bottle_reranker",
        description="Prepare the bottle-level reranker dataset for VIscaNEr. "
                    "Nothing here trains an encoder.",
    )
    parser.add_argument("--config", help="path to the pipeline config (default: configs/bottle_reranker.yaml)")
    parser.add_argument("--project-root", help="override project_root from the config")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    inputs = subparsers.add_parser("inputs", help="step 1: check which inputs exist")
    inputs.add_argument("--hash", action="store_true", help="also hash small input files")

    subparsers.add_parser("identities", help="step 2: audit identities and correspondences")

    segment = subparsers.add_parser("segment", help="step 3: segment bottles and pick the target")
    segment.add_argument("--source", action="append", dest="sources")
    segment.add_argument("--limit", type=int)

    crops = subparsers.add_parser("crops", help="step 4: render bottle, mask, normalized and top views")
    crops.add_argument("--limit", type=int)
    crops.add_argument("--only", choices=["queries", "references"], action="append", dest="parts")

    splits = subparsers.add_parser("splits", help="step 5: build leakage-free splits")
    splits.add_argument("--phash", action="store_true",
                        help="also detect near-duplicate frames (slower, needs Pillow and numpy)")

    candidates = subparsers.add_parser("candidates", help="step 6a: mine DINO top-k candidates")
    candidates.add_argument("--view", default="bottle", choices=["bottle", "normalized", "top"])
    candidates.add_argument("--limit", type=int)
    candidates.add_argument("--batch-size", type=int, default=32)

    subparsers.add_parser("pairs", help="step 6b: build training pairs and triplets")

    pilot = subparsers.add_parser("pilot", help="step 7: render the review sheet")
    pilot.add_argument("--full", action="store_true", help="render every case, not a sample")
    pilot.add_argument("--summarise", action="store_true", help="fold reviewed verdicts into the headroom estimate")

    subparsers.add_parser("plan", help="step 8: rank confusion pairs and say what to photograph")

    package = subparsers.add_parser("package", help="step 9: manifest and Kaggle staging")
    package.add_argument("--no-images", action="store_true", help="stage tables only")

    subparsers.add_parser("report", help="roll every stage report into COVERAGE.md")

    audit = subparsers.add_parser("audit", help="run every stage that needs no model: 1, 2, 5, 8, report")
    audit.add_argument("--phash", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = Config.load(args.config, project_root=args.project_root)
    except ConfigError as error:
        print(f"config error: {error}", file=sys.stderr)
        return 2

    stage = args.stage
    result: Any = None
    try:
        if stage == "inputs":
            from . import step1_inputs

            result = step1_inputs.run(config, hash_small_files=args.hash)
        elif stage == "identities":
            from . import step2_identities

            result = step2_identities.run(config)
        elif stage == "segment":
            from . import step3_segment

            result = step3_segment.run(config, sources=args.sources, limit=args.limit)
        elif stage == "crops":
            from . import step4_crops

            result = step4_crops.run(config, limit=args.limit,
                                     parts=tuple(args.parts) if args.parts else ("references", "queries"))
        elif stage == "splits":
            from . import step5_splits

            result = step5_splits.run(config, compute_phash=args.phash)
        elif stage == "candidates":
            from . import step6_candidates

            result = step6_candidates.run(config, view=args.view, limit=args.limit, batch_size=args.batch_size)
        elif stage == "pairs":
            from . import step6_pairs

            result = step6_pairs.run(config)
        elif stage == "pilot":
            from . import step7_pilot

            result = step7_pilot.summarise(config) if args.summarise else step7_pilot.run(config, full=args.full)
        elif stage == "plan":
            from . import step8_gap_plan

            result = step8_gap_plan.run(config)
        elif stage == "package":
            from . import step9_package

            result = step9_package.run(config, copy_images=not args.no_images)
        elif stage == "report":
            from . import report as report_module

            result = report_module.run(config)
        elif stage == "audit":
            from . import report as report_module
            from . import step1_inputs, step2_identities, step5_splits, step8_gap_plan

            step1_inputs.run(config)
            step2_identities.run(config)
            step5_splits.run(config, compute_phash=args.phash)
            step8_gap_plan.run(config)
            result = report_module.run(config)
    except FileNotFoundError as error:
        print(f"missing input: {error}", file=sys.stderr)
        return 3
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    return 0 if result is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
