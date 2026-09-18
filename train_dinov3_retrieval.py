#!/usr/bin/env python3
"""Headless CLI for the DINOv3 wine-retrieval pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from dinov3_retrieval import (
    PipelineConfig,
    benchmark_model,
    choose_device,
    create_eval_loader,
    evaluate_splits,
    export_gallery_embeddings,
    load_trained_model,
    plot_retrieval_failures,
    prepare_retrieval_index,
    save_retrieval_audit,
    train_pipeline,
)


def load_config(path: str | Path, overrides: dict[str, Any]) -> PipelineConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    valid = {field.name for field in fields(PipelineConfig)}
    unknown = set(payload) - valid
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    payload.update({key: value for key, value in overrides.items() if value is not None})
    return PipelineConfig(**payload).resolved()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "train", "all", "evaluate", "validate", "benchmark"),
    )
    parser.add_argument("--config", default="configs/dinov3_retrieval.yaml")
    parser.add_argument("--project-root")
    parser.add_argument("--weights-path")
    parser.add_argument("--crops-metadata-path")
    parser.add_argument("--bottle-manifest-path")
    parser.add_argument("--crops-root")
    parser.add_argument("--refs-root")
    parser.add_argument("--index-path")
    parser.add_argument("--models-dir")
    parser.add_argument("--runs-dir")
    parser.add_argument("--run-name")
    parser.add_argument("--device")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--quick-smoke", action="store_true")
    parser.add_argument("--skip-file-validation", action="store_true")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument(
        "--checkpoint",
        help="Checkpoint used by evaluate/validate/benchmark; defaults to models_dir/best.pt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {
        key: getattr(args, key)
        for key in (
            "project_root",
            "weights_path",
            "crops_metadata_path",
            "bottle_manifest_path",
            "crops_root",
            "refs_root",
            "index_path",
            "models_dir",
            "runs_dir",
            "run_name",
            "device",
            "num_workers",
            "eval_batch_size",
        )
    }
    cfg = load_config(args.config, overrides)

    prepared_summary: dict[str, Any] | None = None
    if args.command in {"prepare", "all", "validate"}:
        _, prepared_summary = prepare_retrieval_index(
            cfg, validate_files=not args.skip_file_validation
        )
        # `all` and `validate` include this summary in their final JSON.  Print
        # it here only for the standalone command to avoid duplicate log blocks.
        if args.command == "prepare":
            print(json.dumps(prepared_summary, ensure_ascii=False, indent=2))
    if args.command in {"train", "all"}:
        result = train_pipeline(
            cfg,
            quick_smoke=args.quick_smoke,
            resume_checkpoint=args.resume_checkpoint,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.command in {"evaluate", "validate", "benchmark"}:
        index = pd.read_csv(cfg.index_path)
        device = choose_device(cfg.device)
        checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(cfg.models_dir) / "best.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Evaluation checkpoint does not exist: {checkpoint_path}")
        model, checkpoint = load_trained_model(checkpoint_path, cfg.weights_path, device)
        run_dir = Path(cfg.runs_dir) / cfg.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        if args.command in {"evaluate", "validate"}:
            metrics, details = evaluate_splits(model, index, cfg, device)
            output_prefix = "validation_only" if args.command == "validate" else "final"
            metrics_path = run_dir / f"{output_prefix}_metrics.json"
            gallery_path = run_dir / f"{output_prefix}_gallery_embeddings.pt"
            failures_path = run_dir / f"{output_prefix}_failures.png"
            metrics_path.write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            export_gallery_embeddings(details, gallery_path)
            plot_retrieval_failures(details, failures_path)
            audit_summary = save_retrieval_audit(
                details,
                run_dir / f"{output_prefix}_audit",
                n=8,
            )
            result = {
                "mode": args.command,
                "checkpoint_path": str(checkpoint_path.resolve()),
                "checkpoint_metrics_on_old_data": checkpoint.get("metrics"),
                "metrics_on_current_data": metrics,
                "current_data_summary": prepared_summary,
                "metrics_path": str(metrics_path.resolve()),
                "gallery_embeddings_path": str(gallery_path.resolve()),
                "failures_path": str(failures_path.resolve()),
                "audit": audit_summary,
            }
            (run_dir / f"{output_prefix}_summary.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            records = index[index["split"].eq("gallery")].head(max(cfg.eval_batch_size, 128))
            timings = benchmark_model(model, create_eval_loader(records, cfg), device)
            (run_dir / "benchmark.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
            print(json.dumps(timings, indent=2))


if __name__ == "__main__":
    main()
