"""Step 6a - record what the current DINO actually does on every query.

The reranker exists to fix a specific failure mode, so the failures have to be
measured with the exact model that will be deployed in front of it. The stage
pins the checkpoint, the preprocessing and the gallery by hash, then writes the
full top-k for every query: the raw similarities, the top1-top2 margin, and the
rank of the correct answer.

Language matters here and the column names enforce it. The stored numbers are
**cosine similarities between L2-normalised embeddings**. They are not
probabilities that the candidate is correct, they are not calibrated, and
nothing downstream may treat them as such.

Queries are bucketed into the four cases that drive pair mining:

  correct_top1_close     top-1 right, top-2 within the close-call margin
  wrong_top1_correct_top2  top-1 wrong, the right answer is second
  correct_not_in_top2    the right answer is third or worse, or absent
  confident_correct      top-1 right with a clear margin

Every row also records whether the query image was part of the DINO training
data. Thresholds tuned on predictions the model has memorised look far better
than they are, so that exposure flag is mandatory reading for step 7.

The DINO implementation is imported read-only; nothing in it is modified.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Sequence

from .common import (
    percentage,
    read_csv_rows,
    sha256_file,
    stage_record,
    summarise_counts,
    write_csv_rows,
    write_json,
)
from .config import Config

CONFIG_FINGERPRINT_KEYS = (
    "dino.checkpoint",
    "dino.index_csv",
    "dino.retrieval_config",
    "dino.top_k",
    "dino.close_call_margin",
    "dino.gallery_includes_unseen_references",
)

CASE_CORRECT_CLOSE = "correct_top1_close"
CASE_WRONG_TOP1 = "wrong_top1_correct_top2"
CASE_NOT_IN_TOP2 = "correct_not_in_top2"
CASE_CONFIDENT = "confident_correct"

EXPOSURE_TRAIN = "dino_train"
EXPOSURE_HELD_OUT = "dino_held_out"
EXPOSURE_UNKNOWN = "unknown"


def _import_dino(config: Config):
    """Load dinov3_retrieval.py from the project root without copying it."""
    module_path = config.project_root / "dinov3_retrieval.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"dinov3_retrieval.py not found at {module_path}")
    if str(config.project_root) not in sys.path:
        sys.path.insert(0, str(config.project_root))
    spec = importlib.util.spec_from_file_location("dinov3_retrieval", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("dinov3_retrieval", module)
    spec.loader.exec_module(module)
    return module


def _dino_exposure(index_rows: list[dict[str, str]]) -> dict[str, str]:
    """source_path -> which DINO split that image was in.

    The retrieval index records exactly which crops trained the checkpoint.
    Without it the exposure of a query is unknown, and the report says so
    instead of assuming the optimistic case.
    """
    mapping: dict[str, str] = {}
    for row in index_rows:
        source = (row.get("source_path") or "").strip()
        if not source:
            continue
        split = (row.get("split") or "").strip()
        mapping[Path(source).as_posix()] = EXPOSURE_TRAIN if split == "train" else EXPOSURE_HELD_OUT
        mapping[Path(source).name] = mapping[Path(source).as_posix()]
    return mapping


def _classify(correct_rank: int | None, margin: float, close_margin: float) -> str:
    if correct_rank == 1:
        return CASE_CORRECT_CLOSE if margin < close_margin else CASE_CONFIDENT
    if correct_rank == 2:
        return CASE_WRONG_TOP1
    return CASE_NOT_IN_TOP2


def run(config: Config, *, view: str = "bottle", limit: int | None = None, batch_size: int = 32) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    dino = _import_dino(config)

    checkpoint_path = config.path("dino.checkpoint")
    index_path = config.path("dino.index_csv")
    retrieval_config_path = config.path("dino.retrieval_config")
    for required in (checkpoint_path, index_path, retrieval_config_path):
        if not required.is_file():
            raise FileNotFoundError(
                f"Required DINO input missing: {required}. Candidate mining cannot "
                "run without the exact checkpoint, gallery and preprocessing that "
                "produced the errors the reranker is meant to fix."
            )

    splits_csv = config.output_dir("audit", create=False) / "splits.csv"
    if not splits_csv.is_file():
        raise FileNotFoundError(f"{splits_csv} not found. Run step 5 first.")
    queries = read_csv_rows(splits_csv)
    if limit:
        queries = queries[:limit]

    refs_csv = config.output_dir("audit", create=False) / "reference_crops.csv"
    reference_rows = read_csv_rows(refs_csv) if refs_csv.is_file() else []
    reference_flags = {r["wine_slug"]: r.get("quality_flags", "") for r in reference_rows}

    # ---- the gallery ----------------------------------------------------
    # Catalog references of held-out identities may stay in the gallery: the
    # task protocol is a closed catalog and the scanner must be able to return
    # them. That is retrieval scope, and it never licenses training on a
    # held-out query photograph. Step 6b enforces the training side.
    refs_root = config.path("catalog.refs_rgb_root")
    gallery_paths = sorted(p for p in refs_root.iterdir() if p.is_file())
    if not bool(config.get("dino.gallery_includes_unseen_references", True)):
        from .common import stable_unit_interval

        seed = int(config.get("splits.seed"))
        fraction = float(config.get("splits.unseen_identity_fraction"))
        gallery_paths = [p for p in gallery_paths if stable_unit_interval(p.stem, seed) >= fraction]
    gallery_slugs = [p.stem for p in gallery_paths]

    device = dino.choose_device(str(config.get("segmentation.device", "auto")))
    weights_path = config.path("dino.retrieval_config")
    retrieval_cfg = dino.PipelineConfig.from_yaml(retrieval_config_path) if hasattr(dino.PipelineConfig, "from_yaml") else None
    backbone_weights = (
        config.resolve(retrieval_cfg.weights_path) if retrieval_cfg is not None
        else config.path("dino.backbone_weights", "models/dinov3/model.safetensors")
    )
    model, checkpoint = dino.load_trained_model(checkpoint_path, backbone_weights, device=device)
    image_size = int(checkpoint["config"]["image_size"])
    _, eval_transform = dino.build_transforms(image_size)

    class _Images(Dataset):
        def __init__(self, paths: Sequence[Path]) -> None:
            self.paths = list(paths)

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, index: int):
            return {"image": eval_transform(dino.open_rgb(self.paths[index])), "index": index}

    def embed(paths: Sequence[Path]):
        loader = DataLoader(_Images(paths), batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)
        chunks = []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                embeddings, _ = model(batch["image"].to(device))
                chunks.append(F.normalize(embeddings.float().cpu(), dim=-1))
        return torch.cat(chunks) if chunks else torch.empty(0)

    print(f"step6a: embedding {len(gallery_paths)} gallery references", flush=True)
    gallery = embed(gallery_paths)

    # Query images: the whole-bottle view when it exists, otherwise the frame
    # the DINO itself consumed. The view is recorded so a candidate table is
    # never compared against one built from a different view.
    crops_csv = config.output_dir("audit", create=False) / "query_crops.csv"
    crop_lookup: dict[str, str] = {}
    if crops_csv.is_file():
        for row in read_csv_rows(crops_csv):
            if row.get("status") == "ok" and row.get(f"{view}_path"):
                crop_lookup[f"{row['source']}:{row['image_relative_path']}"] = row[f"{view}_path"]

    resolved: list[tuple[dict[str, str], Path, str]] = []
    for row in queries:
        key = f"{row['source']}:{row['image_relative_path']}"
        if key in crop_lookup:
            resolved.append((row, config.output_root / crop_lookup[key], view))
        else:
            resolved.append((row, config.resolve(row["image_path"]), "source_frame"))

    print(f"step6a: embedding {len(resolved)} queries", flush=True)
    query_embeddings = embed([p for _, p, _ in resolved])

    top_k = int(config.get("dino.top_k"))
    close_margin = float(config.get("dino.close_call_margin"))
    exposure = _dino_exposure(read_csv_rows(index_path))
    slug_to_column = {slug: i for i, slug in enumerate(gallery_slugs)}

    rows: list[dict[str, Any]] = []
    similarities = query_embeddings @ gallery.T
    k = min(top_k, similarities.shape[1])
    scores, indices = torch.topk(similarities, k=k, dim=1)

    for position, (row, path, used_view) in enumerate(resolved):
        candidate_slugs = [gallery_slugs[i] for i in indices[position].tolist()]
        candidate_scores = [round(float(s), 6) for s in scores[position].tolist()]
        truth = row["wine_slug"]
        correct_rank = candidate_slugs.index(truth) + 1 if truth in candidate_slugs else None
        correct_similarity = (
            round(float(similarities[position, slug_to_column[truth]]), 6)
            if truth in slug_to_column else None
        )
        margin = candidate_scores[0] - candidate_scores[1] if len(candidate_scores) > 1 else float("inf")
        key = Path(row["image_relative_path"]).as_posix()
        record = {
            "source": row["source"],
            "image_relative_path": row["image_relative_path"],
            "query_view": used_view,
            "query_path": config.relative(path),
            "wine_slug": truth,
            "split": row["split"],
            "dino_unseen_identity": row["dino_unseen_identity"],
            "dino_exposure": exposure.get(key, exposure.get(Path(key).name, EXPOSURE_UNKNOWN)),
            "correct_rank": correct_rank if correct_rank else "",
            "correct_cosine_similarity": correct_similarity,
            "top1_cosine_similarity": candidate_scores[0],
            "top2_cosine_similarity": candidate_scores[1] if len(candidate_scores) > 1 else "",
            "top1_minus_top2_cosine": round(margin, 6) if margin != float("inf") else "",
            "case": _classify(correct_rank, margin, close_margin),
            "top1_reference_flags": reference_flags.get(candidate_slugs[0], ""),
        }
        for rank, (slug, score) in enumerate(zip(candidate_slugs, candidate_scores), start=1):
            record[f"cand{rank}_slug"] = slug
            record[f"cand{rank}_cosine_similarity"] = score
        rows.append(record)

    candidates_csv = config.output_dir("audit") / "dino_candidates.csv"
    write_csv_rows(candidates_csv, rows)

    by_case = summarise_counts(r["case"] for r in rows)
    by_split_case = summarise_counts(f"{r['split']}/{r['case']}" for r in rows)
    held_out = [r for r in rows if r["dino_exposure"] == EXPOSURE_HELD_OUT]

    def recall(pool: list[dict[str, Any]], at: int) -> float:
        if not pool:
            return 0.0
        hits = sum(1 for r in pool if r["correct_rank"] and int(r["correct_rank"]) <= at)
        return percentage(hits, len(pool))

    second_place = [r for r in rows if r["correct_rank"] == 2]

    report = stage_record(
        "step6_candidates",
        config_fingerprint=config.fingerprint(CONFIG_FINGERPRINT_KEYS),
        project_root=config.project_root,
        pinned_inputs={
            "checkpoint": config.relative(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "gallery_index": config.relative(index_path),
            "gallery_index_sha256": sha256_file(index_path),
            "gallery_size": len(gallery_slugs),
            "gallery_includes_unseen_references": bool(config.get("dino.gallery_includes_unseen_references")),
            "retrieval_config": config.relative(retrieval_config_path),
            "image_size": image_size,
            "query_view": view,
        },
        score_semantics=(
            "All *_cosine_similarity columns are cosine similarities between "
            "L2-normalised embeddings. They are not probabilities of being "
            "correct and are not calibrated."
        ),
        queries=len(rows),
        case_counts=by_case,
        case_counts_by_split=by_split_case,
        exposure_counts=summarise_counts(r["dino_exposure"] for r in rows),
        recall_all={"top1_percent": recall(rows, 1), "top2_percent": recall(rows, 2), "top10_percent": recall(rows, 10)},
        recall_on_dino_held_out={
            "queries": len(held_out),
            "top1_percent": recall(held_out, 1),
            "top2_percent": recall(held_out, 2),
            "note": (
                "Computed only on images the checkpoint did not train on. Figures "
                "over all queries include memorised training images and are "
                "optimistic by construction."
            ),
        },
        headroom={
            "correct_answer_at_rank_2": len(second_place),
            "correct_answer_at_rank_2_percent": percentage(len(second_place), max(1, len(rows))),
            "note": (
                "An upper bound on what a perfect binary reranker over (top1, top2) "
                "could recover. How much of it is actually reachable depends on "
                "whether a visible difference exists - that is the step 7 pilot."
            ),
        },
        outputs={"dino_candidates_csv": config.relative(candidates_csv)},
    )
    write_json(config.report_path("step6_candidates.json"), report)
    print(f"step6a: {len(rows)} queries; " + ", ".join(f"{k}={v}" for k, v in by_case.items()))
    return report
