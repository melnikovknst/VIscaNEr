"""Ranking, conditional resolution, metrics, and qualitative audit figures."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps


METRIC_K = (1, 2, 5, 10)


@torch.inference_mode()
def rank_primary(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    true_label_ids: Iterable[int],
    slugs: list[str],
    top_k: int,
    batch_size: int,
) -> pd.DataFrame:
    if len(slugs) != len(gallery_embeddings):
        raise ValueError("Gallery slug order and embedding count differ")
    true_ids = torch.as_tensor(list(true_label_ids), dtype=torch.long)
    if len(query_embeddings) != len(true_ids):
        raise ValueError("Query embeddings and labels differ in length")
    rows: list[dict[str, Any]] = []
    for start in range(0, len(query_embeddings), batch_size):
        end = min(start + batch_size, len(query_embeddings))
        scores = query_embeddings[start:end] @ gallery_embeddings.T
        k = min(top_k, scores.shape[1])
        top_scores, top_indices = scores.topk(k=k, dim=1, largest=True, sorted=True)
        batch_true = true_ids[start:end]
        true_scores = scores.gather(1, batch_true[:, None]).squeeze(1)
        ranks = 1 + (scores > true_scores[:, None]).sum(dim=1)
        for local_index in range(end - start):
            indices = top_indices[local_index].tolist()
            values = top_scores[local_index].tolist()
            row: dict[str, Any] = {
                "b_true_rank": int(ranks[local_index]),
                "b_top1_label_id": int(indices[0]),
                "b_top2_label_id": int(indices[1]),
                "b_top1_slug": slugs[indices[0]],
                "b_top2_slug": slugs[indices[1]],
                "b_top1_similarity": float(values[0]),
                "b_top2_similarity": float(values[1]),
                "b_top1_top2_gap": float(values[0] - values[1]),
                "b_true_similarity": float(true_scores[local_index]),
            }
            for position, (label_id, value) in enumerate(zip(indices, values, strict=True), start=1):
                row[f"b_top{position}_label_id"] = int(label_id)
                row[f"b_top{position}_slug"] = slugs[label_id]
                row[f"b_top{position}_similarity"] = float(value)
            rows.append(row)
    return pd.DataFrame(rows)


def resolve_top_two(
    predictions: pd.DataFrame,
    resolver_query_embeddings: torch.Tensor,
    resolver_gallery_embeddings: torch.Tensor,
    resolver_valid: torch.Tensor,
) -> pd.DataFrame:
    """Reorder only B's two candidates using one whole-bottle DINO-S embedding."""

    result = predictions.copy()
    result["resolver_embedding_valid"] = resolver_valid.numpy().astype(bool)
    candidate_ids = torch.as_tensor(
        result[["b_top1_label_id", "b_top2_label_id"]].to_numpy(copy=True), dtype=torch.long
    )
    candidate_gallery = resolver_gallery_embeddings[candidate_ids]
    scores = torch.einsum("nd,nkd->nk", resolver_query_embeddings, candidate_gallery)
    score1 = scores[:, 0].numpy()
    score2 = scores[:, 1].numpy()
    choose_second = (score2 > score1) & result["resolver_embedding_valid"].to_numpy()
    result["s_candidate1_similarity"] = score1
    result["s_candidate2_similarity"] = score2
    result["s_candidate_gap"] = np.abs(score1 - score2)
    result["resolver_swapped"] = choose_second
    result["final_top1_label_id"] = np.where(
        choose_second, result["b_top2_label_id"], result["b_top1_label_id"]
    )
    result["final_top2_label_id"] = np.where(
        choose_second, result["b_top1_label_id"], result["b_top2_label_id"]
    )
    result["final_top1_slug"] = np.where(
        choose_second, result["b_top2_slug"], result["b_top1_slug"]
    )
    result["final_top2_slug"] = np.where(
        choose_second, result["b_top1_slug"], result["b_top2_slug"]
    )
    result["final_true_rank"] = result["b_true_rank"].astype(int)
    true_ids = result["label_id"].to_numpy()
    first_ids = result["final_top1_label_id"].to_numpy()
    second_ids = result["final_top2_label_id"].to_numpy()
    result.loc[true_ids == first_ids, "final_true_rank"] = 1
    result.loc[true_ids == second_ids, "final_true_rank"] = 2
    return result


def initialize_final_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    result = predictions.copy()
    result["resolver_embedding_valid"] = False
    result["resolver_swapped"] = False
    result["s_candidate1_similarity"] = np.nan
    result["s_candidate2_similarity"] = np.nan
    result["s_candidate_gap"] = np.nan
    result["final_top1_label_id"] = result["b_top1_label_id"]
    result["final_top2_label_id"] = result["b_top2_label_id"]
    result["final_top1_slug"] = result["b_top1_slug"]
    result["final_top2_slug"] = result["b_top2_slug"]
    result["final_true_rank"] = result["b_true_rank"]
    return result


def ranking_metrics(ranks: Iterable[int]) -> dict[str, float | int]:
    values = np.asarray(list(ranks), dtype=np.int64)
    if len(values) == 0:
        return {"num_queries": 0, "accuracy": math.nan, **{f"recall_at_{k}": math.nan for k in METRIC_K}}
    metrics: dict[str, float | int] = {
        "num_queries": int(len(values)),
        "accuracy": float(np.mean(values == 1)),
        "mrr": float(np.mean(1.0 / values)),
        "median_rank": float(np.median(values)),
        "mean_rank": float(np.mean(values)),
    }
    for k in METRIC_K:
        metrics[f"recall_at_{k}"] = float(np.mean(values <= k))
    return metrics


def make_metrics_table(predictions: pd.DataFrame) -> pd.DataFrame:
    scopes: dict[str, pd.Series] = {
        "all_eligible": pd.Series(True, index=predictions.index),
        "train_descriptive": predictions["primary_split"].eq("train"),
        "val_seen": predictions["primary_split"].eq("val_seen"),
        "val_unseen": predictions["primary_split"].eq("val_unseen"),
        "resolver_invoked": predictions["resolver_invoked"].astype(bool),
        "resolver_invoked_val_unseen": predictions["resolver_invoked"].astype(bool)
        & predictions["primary_split"].eq("val_unseen"),
    }
    for status in sorted(predictions.loc[predictions["resolver_invoked"], "bottle_status"].dropna().astype(str).unique()):
        scopes[f"resolver_status_{status}"] = (
            predictions["resolver_invoked"].astype(bool)
            & predictions["bottle_status"].astype(str).eq(status)
        )
    rows: list[dict[str, Any]] = []
    for scope, mask in scopes.items():
        subset = predictions.loc[mask]
        for stage, rank_column in (("dino_b", "b_true_rank"), ("cascade", "final_true_rank")):
            rows.append({"scope": scope, "stage": stage, **ranking_metrics(subset[rank_column])})
    return pd.DataFrame(rows)


def cascade_diagnostics(predictions: pd.DataFrame) -> dict[str, Any]:
    b_correct = predictions["b_true_rank"].eq(1)
    final_correct = predictions["final_true_rank"].eq(1)
    invoked = predictions["resolver_invoked"].astype(bool)
    return {
        "eligible_queries": int(len(predictions)),
        "resolver_invoked": int(invoked.sum()),
        "resolver_invocation_rate": float(invoked.mean()) if len(predictions) else 0.0,
        "ambiguous_without_bottle_crop": int(predictions["resolver_missing_bottle_crop"].sum()),
        "resolver_embedding_failures": int((invoked & ~predictions["resolver_embedding_valid"]).sum()),
        "resolver_swaps": int(predictions["resolver_swapped"].sum()),
        "fixed_top1_errors": int((~b_correct & final_correct).sum()),
        "harmed_top1_predictions": int((b_correct & ~final_correct).sum()),
        "net_top1_corrections": int(final_correct.sum() - b_correct.sum()),
        "truth_available_in_b_top2": int(predictions["b_true_rank"].le(2).sum()),
        "truth_outside_b_top2": int(predictions["b_true_rank"].gt(2).sum()),
        "b_top2_recall_ceiling_for_resolver": float(predictions["b_true_rank"].le(2).mean()),
    }


def _open_for_plot(path: Any, placeholder: str = "not available") -> tuple[Image.Image | None, str]:
    if not path or pd.isna(path):
        return None, placeholder
    try:
        with Image.open(str(path)) as image:
            return ImageOps.exif_transpose(image).convert("RGB").copy(), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}"


def _draw_panel(axis: plt.Axes, path: Any, title: str) -> None:
    image, error = _open_for_plot(path)
    axis.axis("off")
    if image is not None:
        axis.imshow(image)
    else:
        axis.text(0.5, 0.5, error, ha="center", va="center", wrap=True)
    axis.set_title(title, fontsize=8)


def _wrap_slug(value: Any, width: int = 32) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=True))


def save_case_sheets(
    cases: pd.DataFrame,
    category: str,
    output_dir: str | Path,
    label_ref_by_slug: dict[str, Path],
    bottle_ref_by_slug: dict[str, Path],
    max_examples: int,
    per_page: int = 4,
) -> list[str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = cases.head(max_examples)
    paths: list[str] = []
    if cases.empty:
        return paths
    for page, start in enumerate(range(0, len(cases), per_page), start=1):
        page_rows = cases.iloc[start : start + per_page]
        figure, axes = plt.subplots(len(page_rows), 6, figsize=(18, 4.2 * len(page_rows)), squeeze=False)
        for row_axes, (_, row) in zip(axes, page_rows.iterrows(), strict=True):
            gt = str(row["wine_slug"])
            b_top1 = str(row["b_top1_slug"])
            final = str(row["final_top1_slug"])
            _draw_panel(row_axes[0], row["original_path"], f"ORIGINAL\n{Path(row['source_relative_path']).name}")
            _draw_panel(
                row_axes[1],
                row["label_crop_path"],
                f"LABEL CROP\nB gap={row['b_top1_top2_gap']:.4f}",
            )
            _draw_panel(
                row_axes[2],
                row["bottle_crop_path"],
                f"BOTTLE CROP\nstatus={row['bottle_status']}",
            )
            _draw_panel(row_axes[3], label_ref_by_slug[gt], f"TRUE\n{_wrap_slug(gt)}")
            _draw_panel(
                row_axes[4],
                label_ref_by_slug[b_top1],
                f"DINO-B TOP-1 (rank={row['b_true_rank']})\n{_wrap_slug(b_top1)}\nsim={row['b_top1_similarity']:.4f}",
            )
            resolver_used = bool(row["resolver_invoked"] and row["resolver_embedding_valid"])
            final_ref = (
                bottle_ref_by_slug.get(final, label_ref_by_slug[final])
                if resolver_used
                else label_ref_by_slug[final]
            )
            resolver_text = (
                f"S={row['s_candidate1_similarity']:.4f}/{row['s_candidate2_similarity']:.4f}"
                if row["resolver_invoked"] and pd.notna(row["s_candidate1_similarity"])
                else "S not invoked"
            )
            _draw_panel(
                row_axes[5],
                final_ref,
                f"FINAL (rank={row['final_true_rank']})\n{_wrap_slug(final)}\n{resolver_text}",
            )
        figure.suptitle(f"DINO cascade audit — {category} — page {page}", fontsize=14, fontweight="bold")
        figure.tight_layout(rect=(0, 0, 1, 0.98))
        path = output_dir / f"{category}_{page:02d}.png"
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        paths.append(str(path))
    return paths


def save_audit_visualizations(
    predictions: pd.DataFrame,
    output_dir: str | Path,
    label_ref_by_slug: dict[str, Path],
    bottle_ref_by_slug: dict[str, Path],
    max_examples: int,
) -> dict[str, list[str]]:
    b_correct = predictions["b_true_rank"].eq(1)
    final_correct = predictions["final_true_rank"].eq(1)
    categories = {
        "worst_final_errors": predictions.loc[~final_correct].sort_values(
            ["final_true_rank", "b_top1_similarity"], ascending=[False, False]
        ),
        "truth_outside_b_top2": predictions.loc[predictions["b_true_rank"].gt(2)].sort_values(
            ["b_true_rank", "b_top1_similarity"], ascending=[False, False]
        ),
        "resolver_fixed": predictions.loc[~b_correct & final_correct].sort_values(
            "b_top1_top2_gap"
        ),
        "resolver_harmed": predictions.loc[b_correct & ~final_correct].sort_values(
            "b_top1_top2_gap"
        ),
        "truth_was_b_top2": predictions.loc[predictions["b_true_rank"].eq(2)].sort_values(
            "b_top1_top2_gap"
        ),
    }
    return {
        category: save_case_sheets(
            cases,
            category,
            output_dir,
            label_ref_by_slug,
            bottle_ref_by_slug,
            max_examples,
        )
        for category, cases in categories.items()
    }
