"""Opt-in three-stage DINOv3 retrieval training with exact continuation.

The data preparation, model, losses and retrieval evaluation stay in
``dinov3_retrieval.py``.  This module only adds the long third phase where the
entire backbone is trainable, plus the checkpoint state required to continue a
Kaggle run without silently restarting an epoch or optimizer schedule.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import random
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import dinov3_retrieval as base


TRAINING_FORMAT = "full-finetune-v2"


def stage_plan(cfg: base.PipelineConfig) -> list[tuple[int, int, int, float, float]]:
    """Return stage, epochs, trainable-block mode, head LR and backbone LR."""
    return [
        (1, cfg.stage1_epochs, 0, cfg.head_lr, cfg.backbone_lr),
        (2, cfg.stage2_epochs, cfg.unfreeze_last_blocks, cfg.head_lr, cfg.backbone_lr),
        (3, cfg.stage3_epochs, -1, cfg.stage3_head_lr, cfg.stage3_backbone_lr),
    ]


def lr_factor(epoch: int, epochs: int, warmup: int) -> float:
    """Linear warm-up followed by cosine decay to five percent of peak LR."""
    if warmup and epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(epochs - warmup - 1, 1)
    progress = min(max(progress, 0.0), 1.0)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))


def atomic_save(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def trainability(model: base.DINOv3RetrievalModel) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for name, module in (
        ("backbone", model.backbone),
        ("projection", model.projection),
        ("classifier", model.classifier),
    ):
        result[name] = {
            "total": sum(parameter.numel() for parameter in module.parameters()),
            "trainable": sum(
                parameter.numel()
                for parameter in module.parameters()
                if parameter.requires_grad
            ),
        }
    return result


def _monitor_value(
    metrics: dict[str, dict[str, float]], cfg: base.PipelineConfig
) -> float:
    split_metrics = metrics.get(cfg.early_stopping_split)
    if split_metrics is None:
        raise ValueError(
            f"Early-stopping split is unavailable: {cfg.early_stopping_split}"
        )
    if cfg.early_stopping_metric not in split_metrics:
        raise ValueError(
            "Early-stopping metric is unavailable: "
            f"{cfg.early_stopping_split}.{cfg.early_stopping_metric}"
        )
    value = float(split_metrics[cfg.early_stopping_metric])
    if not math.isfinite(value):
        raise FloatingPointError(
            "Non-finite early-stopping metric: "
            f"{cfg.early_stopping_split}.{cfg.early_stopping_metric}={value}"
        )
    return value


def _nonzero_gradient_exists(module: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        and float(parameter.grad.detach().abs().max()) > 0.0
        for parameter in module.parameters()
    )


def smoke_test(
    cfg: base.PipelineConfig, index: pd.DataFrame, device: torch.device
) -> list[dict[str, Any]]:
    """Run one real forward/backward update for every trainability phase."""
    base.seed_everything(cfg.seed)
    model = base.DINOv3RetrievalModel(
        base.load_local_dinov3_backbone(cfg.weights_path, cfg.image_size),
        int(index["label_id"].max()) + 1,
        cfg.embedding_dim,
        cfg.projection_hidden_dim,
        cfg.ce_temperature,
    ).to(device)
    records = index[index["split"].eq("train")].reset_index(drop=True)
    selected = sorted(records["label_id"].unique())[
        : max(16, cfg.stage1_identities_per_batch)
    ]
    records = (
        records[records["label_id"].isin(selected)]
        .groupby("label_id", group_keys=False)
        .head(4)
        .reset_index(drop=True)
    )
    if records.empty:
        raise ValueError("Smoke test needs at least one training identity")
    test_cfg = replace(cfg, num_workers=0)
    results: list[dict[str, Any]] = []

    for stage, _, unfrozen, head_lr, backbone_lr in stage_plan(cfg):
        model.set_backbone_trainable(unfrozen)
        stage_cfg = replace(
            test_cfg, head_lr=head_lr, backbone_lr=backbone_lr
        )
        loader, _ = base.create_train_loader(records, stage_cfg, stage)
        batch = next(iter(loader))
        optimizer = base.build_optimizer(model, stage_cfg)
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=device.type == "cuda" and cfg.amp,
            init_scale=1024.0,
        )
        report = trainability(model)
        if stage == 3 and report["backbone"]["trainable"] != report["backbone"]["total"]:
            raise RuntimeError("Stage 3 did not unfreeze every backbone parameter")
        metrics = base.train_one_epoch(
            model, [batch], optimizer, device, stage_cfg, scaler
        )
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError(f"Non-finite smoke metrics: {metrics}")
        if stage == 3:
            for name, module in (
                ("backbone", model.backbone),
                ("projection", model.projection),
                ("classifier", model.classifier),
            ):
                if not _nonzero_gradient_exists(module):
                    raise RuntimeError(f"No finite non-zero stage-3 gradient in {name}")
        results.append(
            {"stage": stage, "trainability": report, "metrics": metrics}
        )
        del optimizer, scaler, loader, batch

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def _validate_full_config(cfg: base.PipelineConfig) -> None:
    if cfg.stage3_epochs <= 0:
        raise ValueError("Full training requires stage3_epochs > 0")
    if not 1 <= cfg.stage3_min_epochs <= cfg.stage3_epochs:
        raise ValueError(
            "Full training requires 1 <= stage3_min_epochs <= stage3_epochs"
        )
    if cfg.stage3_patience < 1:
        raise ValueError("stage3_patience must be positive")
    if not 0 <= cfg.stage3_warmup_epochs < cfg.stage3_epochs:
        raise ValueError("stage3_warmup_epochs must be in [0, stage3_epochs)")
    if cfg.max_session_hours <= 0:
        raise ValueError("max_session_hours must be positive")


def _resume_config_is_compatible(
    cfg: base.PipelineConfig, payload: dict[str, Any]
) -> None:
    allowed_changes = {
        "project_root",
        "weights_path",
        "crops_metadata_path",
        "bottle_manifest_path",
        "crops_root",
        "refs_root",
        "index_path",
        "split_summary_path",
        "models_dir",
        "runs_dir",
        "run_name",
        "device",
        "num_workers",
        "eval_batch_size",
        "max_session_hours",
    }
    saved = payload["config"]
    for key, value in asdict(cfg).items():
        if key not in allowed_changes and saved.get(key) != value:
            raise ValueError(f"Resume configuration changed: {key}")


def _rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    }


def _restore_rng(payload: dict[str, Any], device: torch.device) -> None:
    random.setstate(payload["rng_python"])
    np.random.set_state(payload["rng_numpy"])
    torch.set_rng_state(payload["rng_torch"])
    if device.type == "cuda" and payload.get("rng_cuda") is not None:
        torch.cuda.set_rng_state_all(payload["rng_cuda"])


def train_full_pipeline(
    config: base.PipelineConfig,
    quick_smoke: bool = False,
    resume_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    cfg = config.resolved()
    _validate_full_config(cfg)
    base.seed_everything(cfg.seed)
    device = base.choose_device(cfg.device)
    run_dir = Path(cfg.runs_dir) / cfg.run_name
    models_dir = Path(cfg.models_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Rebuild from current metadata so an old cached split can never be resumed
    # against different crops without detection. The disposable smoke run skips
    # the expensive 45k file inventory; the real run validates it once.
    print("Preparing full-training retrieval index...", flush=True)
    index, split_summary = base.prepare_retrieval_index(
        cfg, validate_files=not quick_smoke
    )
    print("Full-training retrieval index prepared", flush=True)
    index_sha256 = hashlib.sha256(Path(cfg.index_path).read_bytes()).hexdigest()
    if quick_smoke:
        result = {
            "quick_smoke": True,
            "device": str(device),
            "stages": smoke_test(cfg, index, device),
            "split_summary": split_summary,
        }
        (run_dir / "smoke_report.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result

    if resume_checkpoint is None and any(models_dir.glob("*.pt")):
        raise FileExistsError(
            "Fresh full training requires an empty models_dir; use a new run name "
            "or pass --resume-checkpoint explicitly"
        )

    print(f"Loading DINOv3 backbone from {cfg.weights_path}", flush=True)
    model = base.DINOv3RetrievalModel(
        base.load_local_dinov3_backbone(cfg.weights_path, cfg.image_size),
        int(index["label_id"].max()) + 1,
        cfg.embedding_dim,
        cfg.projection_hidden_dim,
        cfg.ce_temperature,
    ).to(device)
    print("DINOv3 backbone loaded", flush=True)
    resume_payload: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    global_epoch = 0
    best_monitor = -1.0
    best_full_monitor = -1.0
    train_records = index[index["split"].eq("train")].reset_index(drop=True)
    if index[index["split"].eq(cfg.early_stopping_split)].empty:
        raise ValueError(
            f"{cfg.early_stopping_split} is required for checkpoint selection"
        )

    if resume_checkpoint is not None:
        resume_path = Path(resume_checkpoint).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        resume_payload = torch.load(
            resume_path, map_location="cpu", weights_only=False
        )
        if resume_payload.get("training_format") != TRAINING_FORMAT:
            raise ValueError(
                f"Resume requires a {TRAINING_FORMAT} last.pt checkpoint"
            )
        if resume_payload["index_sha256"] != index_sha256:
            raise ValueError("Training split changed; exact continuation is unsafe")
        _resume_config_is_compatible(cfg, resume_payload)
        state = base.normalize_retrieval_checkpoint_state_dict(
            model, resume_payload["model_state_dict"]
        )
        model.load_state_dict(state, strict=True)
        history = list(resume_payload["history"])
        global_epoch = int(resume_payload["epoch"])
        best_monitor = float(resume_payload["best_monitor"])
        best_full_monitor = float(resume_payload["best_full_monitor"])
        for filename, required in (
            ("best.pt", True),
            ("best_full.pt", best_full_monitor >= 0.0),
        ):
            source = resume_path.parent / filename
            if required and not source.is_file():
                raise FileNotFoundError(
                    f"Resume also needs the sibling checkpoint: {source}"
                )
            destination = models_dir / filename
            if source.is_file() and source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
        _restore_rng(resume_payload, device)
        print(
            "Explicit continuation: "
            f"epoch={global_epoch}, stage={resume_payload['stage']}, "
            f"stage_epoch={resume_payload['stage_epoch']}",
            flush=True,
        )
    else:
        print(
            f"FRESH INITIALIZATION: {cfg.weights_path}; no trained checkpoint loaded",
            flush=True,
        )

    session_started = time.monotonic()
    stopped_for_time = False
    training_complete = False

    for stage, epochs, unfrozen, head_lr, backbone_lr in stage_plan(cfg):
        if epochs <= 0:
            continue
        if resume_payload is not None and (
            stage < int(resume_payload["stage"])
            or (
                stage == int(resume_payload["stage"])
                and bool(resume_payload["stage_complete"])
            )
        ):
            continue

        continuing = (
            resume_payload is not None
            and stage == int(resume_payload["stage"])
            and not bool(resume_payload["stage_complete"])
        )
        start_epoch = int(resume_payload["stage_epoch"]) if continuing else 0
        stage_best = float(resume_payload["stage_best"]) if continuing else -1.0
        no_improvement = int(resume_payload["no_improvement"]) if continuing else 0

        # Every new phase starts from this run's global best checkpoint.
        if stage > 1 and not continuing and (models_dir / "best.pt").is_file():
            best_payload = torch.load(
                models_dir / "best.pt", map_location="cpu", weights_only=False
            )
            state = base.normalize_retrieval_checkpoint_state_dict(
                model, best_payload["model_state_dict"]
            )
            model.load_state_dict(state, strict=True)
            del best_payload

        model.set_backbone_trainable(unfrozen)
        report = trainability(model)
        if stage == 3 and report["backbone"]["trainable"] != report["backbone"]["total"]:
            raise RuntimeError("Full training requested but the backbone is still frozen")
        print("\n" + "#" * 96, flush=True)
        print(
            f"STAGE {stage}/3 | max_epochs={epochs} "
            f"| train_rows={len(train_records)}",
            flush=True,
        )
        print(
            "TRAINABLE     | "
            f"backbone={report['backbone']['trainable']:,}/{report['backbone']['total']:,} "
            f"| projection={report['projection']['trainable']:,} "
            f"| classifier={report['classifier']['trainable']:,}",
            flush=True,
        )
        print("#" * 96, flush=True)

        stage_cfg = replace(cfg, head_lr=head_lr, backbone_lr=backbone_lr)
        optimizer = base.build_optimizer(model, stage_cfg)
        warmup = cfg.stage3_warmup_epochs if stage == 3 else min(1, epochs - 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: lr_factor(epoch, epochs, warmup)
        )
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=device.type == "cuda" and cfg.amp,
            init_scale=1024.0,
        )
        if continuing:
            optimizer.load_state_dict(resume_payload["optimizer"])
            scheduler.load_state_dict(resume_payload["scheduler"])
            scaler.load_state_dict(resume_payload["scaler"])
            _restore_rng(resume_payload, device)
        resume_payload = None

        for stage_epoch in range(start_epoch, epochs):
            global_epoch += 1
            epoch_started = time.monotonic()
            loader, sampler = base.create_train_loader(
                train_records, stage_cfg, stage, global_epoch
            )
            sampler.set_epoch(global_epoch)
            used_lrs = [float(group["lr"]) for group in optimizer.param_groups]
            current_backbone_lr = used_lrs[-1] if stage > 1 else 0.0
            backbone_lr_text = (
                f"{current_backbone_lr:.3e}"
                if current_backbone_lr > 0
                else "frozen"
            )
            print(
                f"\nSTART EPOCH {global_epoch} | stage {stage}/3 "
                f"| stage_epoch {stage_epoch + 1}/{epochs} "
                f"| head_lr={used_lrs[0]:.3e} "
                f"| backbone_lr={backbone_lr_text}",
                flush=True,
            )
            train_metrics = base.train_one_epoch(
                model, loader, optimizer, device, stage_cfg, scaler
            )
            del loader, sampler
            if not all(math.isfinite(value) for value in train_metrics.values()):
                raise FloatingPointError(
                    f"Non-finite training metrics: {train_metrics}"
                )

            val_metrics, _ = base.evaluate_splits(
                model,
                index,
                cfg,
                device,
                splits=("val_seen", "val_unseen"),
            )
            monitor = _monitor_value(val_metrics, cfg)
            scheduler.step()

            row: dict[str, Any] = {
                "epoch": global_epoch,
                "stage": stage,
                "stage_epoch": stage_epoch + 1,
                "elapsed_seconds": time.monotonic() - epoch_started,
                "head_lr": used_lrs[0],
                "backbone_lr": used_lrs[-1] if stage > 1 else 0.0,
                "early_stopping_split": cfg.early_stopping_split,
                "early_stopping_metric": cfg.early_stopping_metric,
                "early_stopping_value": monitor,
                **train_metrics,
            }
            for split, values in val_metrics.items():
                row.update(
                    {f"{split}_{key}": value for key, value in values.items()}
                )
            history.append(row)
            pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

            deployable = {
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "epoch": global_epoch,
                "stage": stage,
                "metrics": val_metrics,
                "monitor_split": cfg.early_stopping_split,
                "monitor_metric": cfg.early_stopping_metric,
                "monitor_value": monitor,
            }
            global_improved = monitor > best_monitor
            if global_improved:
                best_monitor = monitor
                atomic_save(deployable, models_dir / "best.pt")
            if stage == 3 and monitor > best_full_monitor:
                best_full_monitor = monitor
                atomic_save(deployable, models_dir / "best_full.pt")
            if monitor > stage_best:
                stage_best = monitor
                no_improvement = 0
            else:
                no_improvement += 1

            early_stop = (
                stage == 3
                and stage_epoch + 1 >= cfg.stage3_min_epochs
                and no_improvement >= cfg.stage3_patience
            )
            stage_complete = early_stop or stage_epoch + 1 == epochs
            continuation = {
                **deployable,
                "training_format": TRAINING_FORMAT,
                "index_sha256": index_sha256,
                "stage_epoch": stage_epoch + 1,
                "stage_complete": stage_complete,
                "best_monitor": best_monitor,
                "best_full_monitor": best_full_monitor,
                "stage_best": stage_best,
                "no_improvement": no_improvement,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "history": history,
                **_rng_state(device),
            }
            atomic_save(continuation, models_dir / "last.pt")
            del continuation, deployable
            base.print_epoch_report(
                row,
                val_metrics,
                stage_epochs=epochs,
                best_monitor=best_monitor,
                improved=global_improved,
                epochs_without_improvement=no_improvement,
                total_stages=3,
            )

            if stage == 3 and stage_complete:
                training_complete = True
            elapsed = time.monotonic() - session_started
            next_epoch_estimate = row["elapsed_seconds"] * (
                2.0 if stage == 2 and stage_complete else 1.25
            )
            if (
                not training_complete
                and elapsed + next_epoch_estimate >= cfg.max_session_hours * 3600
            ):
                stopped_for_time = True
                break
            if early_stop:
                print(
                    "Stage 3 early stopping: "
                    f"{no_improvement} epochs without improvement in "
                    f"{cfg.early_stopping_split}.{cfg.early_stopping_metric}",
                    flush=True,
                )
                break

        del optimizer, scheduler, scaler
        if stopped_for_time:
            break

    if not stopped_for_time:
        training_complete = True
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    best_model, best_payload = base.load_trained_model(
        models_dir / "best.pt", cfg.weights_path, device
    )
    final_metrics, details = base.evaluate_splits(
        best_model, index, cfg, device
    )
    prefix = "final" if training_complete else "interim"
    (run_dir / f"{prefix}_metrics.json").write_text(
        json.dumps(final_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    base.export_gallery_embeddings(details, run_dir / "gallery_embeddings.pt")
    base.plot_training_history(run_dir / "history.csv", run_dir / "training_curves.png")
    base.plot_retrieval_failures(details, run_dir / "retrieval_failures.png")
    base.save_retrieval_audit(details, run_dir / f"{prefix}_audit")

    result = {
        "training_complete": training_complete,
        "continuation_required": stopped_for_time,
        "stop_reason": (
            "session_budget" if stopped_for_time else "schedule_or_early_stopping"
        ),
        "epochs_completed": global_epoch,
        "monitor_split": cfg.early_stopping_split,
        "monitor_metric": cfg.early_stopping_metric,
        "best_checkpoint_stage": int(best_payload["stage"]),
        "best_checkpoint": str(models_dir / "best.pt"),
        "best_full_checkpoint": (
            str(models_dir / "best_full.pt")
            if (models_dir / "best_full.pt").is_file()
            else None
        ),
        "last_checkpoint": str(models_dir / "last.pt"),
        "run_dir": str(run_dir),
        "split_summary": split_summary,
        "final_metrics": final_metrics,
        "index_sha256": index_sha256,
        "quick_smoke": False,
    }
    (run_dir / "run_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result
