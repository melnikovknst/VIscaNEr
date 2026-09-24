#!/usr/bin/env python3
"""Export reversible quarantine and a balanced hard-fine-tuning manifest."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import DEFAULT_OUTPUT_DIR, PROJECT_ROOT


REJECT_DECISIONS = {
    "reject_wrong_crop",
    "reject_wrong_label",
    "quarantine_visual_ambiguity",
}


def balanced_replay(correct: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if count <= 0 or correct.empty:
        return correct.head(0).copy()
    rng = np.random.default_rng(seed)
    shuffled = correct.assign(_random=rng.random(len(correct))).sort_values(
        ["true_slug", "_random"]
    )
    pieces: list[pd.DataFrame] = []
    per_identity = max(1, int(np.ceil(count / shuffled["true_slug"].nunique())))
    for _, group in shuffled.groupby("true_slug", sort=True):
        pieces.append(group.head(per_identity))
    result = pd.concat(pieces, ignore_index=True).sort_values("_random").head(count)
    return result.drop(columns="_random")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fresh-limit", type=int, default=1500)
    parser.add_argument("--replay-ratio", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Copy the exported subset into hard_finetune_dataset/images/<slug>/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_audit_path = args.output_dir / "all_retrieval_audit.csv"
    if not all_audit_path.is_file():
        raise FileNotFoundError(
            "Run `python -m bottle_dataset_curation.run_full_local` before export; "
            "the validation-only audit is not enough to build a training subset."
        )
    queue = pd.read_csv(args.output_dir / "review_queue.csv")
    decisions = pd.read_csv(args.output_dir / "curation_decisions.csv").fillna("")
    decisions = decisions.drop_duplicates("review_id", keep="last")
    reviewed = queue.merge(decisions, on="review_id", how="left")
    reviewed["decision"] = reviewed["decision"].fillna("")

    quarantine = reviewed[reviewed["decision"].isin(REJECT_DECISIONS)].copy()
    hard = reviewed[reviewed["decision"].eq("keep_hard")].copy()
    unresolved = reviewed[~reviewed["decision"].isin(REJECT_DECISIONS | {"keep_hard"})].copy()
    quarantine.to_csv(args.output_dir / "quarantine_manifest.csv", index=False)
    unresolved.to_csv(args.output_dir / "unresolved_manifest.csv", index=False)

    curated_master = pd.DataFrame()
    if all_audit_path.is_file():
        all_audit = pd.read_csv(all_audit_path)
        all_audit = all_audit[~all_audit["split"].eq("fresh_intake")].copy()
        rejected_paths = set(quarantine["local_crop_path"].dropna().astype(str))
        all_audit["curation_status"] = np.where(
            all_audit["query_path"].isin(rejected_paths), "quarantined", "kept"
        )
        curated_master = all_audit[
            [
                "query_path",
                "true_slug",
                "split",
                "rank",
                "curation_status",
            ]
        ].rename(columns={"query_path": "image_path", "true_slug": "wine_slug"})
        curated_master.to_csv(
            args.output_dir / "curated_master_manifest.csv", index=False
        )

    fresh = pd.read_csv(args.output_dir / "fresh_manifest.csv")
    fresh = fresh[
        fresh["status"].eq("candidate")
        & fresh["image_path"].map(lambda value: Path(str(value)).is_file())
    ].copy()
    if (args.output_dir / "all_retrieval_audit.csv").is_file():
        all_audit = pd.read_csv(args.output_dir / "all_retrieval_audit.csv")
        fresh_audit = all_audit[all_audit["split"].eq("fresh_intake")][
            ["query_path", "rank", "true_similarity", "top1_slug"]
        ].rename(columns={"query_path": "image_path", "rank": "model_rank"})
        fresh = fresh.merge(fresh_audit, on="image_path", how="left")
        fresh_decisions = reviewed[
            reviewed["split"].eq("fresh_intake")
        ].set_index("local_crop_path")["decision"].to_dict()
        fresh["review_decision"] = fresh["image_path"].map(fresh_decisions).fillna("")
        fresh = fresh[
            fresh["model_rank"].eq(1)
            | fresh["review_decision"].eq("keep_hard")
        ]
        fresh = fresh[~fresh["review_decision"].isin(REJECT_DECISIONS)]
        fresh = fresh.sort_values(
            ["model_rank", "reference_dhash_similarity"], ascending=[True, True]
        )
    else:
        fresh = fresh.sort_values("reference_dhash_similarity", ascending=True)
    fresh = fresh.head(args.fresh_limit).copy()

    hard_train = hard[hard["split"].eq("train")].copy()
    hard_manifest = pd.DataFrame(
        {
            "image_path": hard_train["local_crop_path"],
            "wine_slug": hard_train["true_slug"],
            "source_kind": "reviewed_hard_error",
            "source_split": hard_train["split"],
            "review_id": hard_train["review_id"],
        }
    )
    fresh_manifest = pd.DataFrame(
        {
            "image_path": fresh["image_path"],
            "wine_slug": fresh["wine_slug"],
            "source_kind": "fresh_archive",
            "source_split": "train_only",
            "review_id": "",
        }
    )

    replay = pd.DataFrame(columns=hard_manifest.columns)
    if all_audit_path.is_file():
        audit = pd.read_csv(all_audit_path)
        correct = audit[
            audit["split"].eq("train") & audit["rank"].eq(1) & audit["embedding_valid"]
        ].copy()
        replay_count = int(round((len(hard_manifest) + len(fresh_manifest)) * args.replay_ratio))
        replay_rows = balanced_replay(correct, replay_count, args.seed)
        replay = pd.DataFrame(
            {
                "image_path": replay_rows["query_path"],
                "wine_slug": replay_rows["true_slug"],
                "source_kind": "clean_replay",
                "source_split": replay_rows["split"],
                "review_id": "",
            }
        )

    fine_tune = pd.concat([hard_manifest, fresh_manifest, replay], ignore_index=True)
    fine_tune = fine_tune.drop_duplicates(["image_path", "wine_slug"]).reset_index(drop=True)
    fine_tune.to_csv(args.output_dir / "hard_finetune_manifest.csv", index=False)
    materialized_dir = args.output_dir / "hard_finetune_dataset"
    if args.materialize:
        if materialized_dir.exists():
            shutil.rmtree(materialized_dir)
        for row_number, row in fine_tune.iterrows():
            source = Path(str(row["image_path"]))
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = (
                materialized_dir
                / "images"
                / str(row["wine_slug"])
                / f"{row['source_kind']}__{row_number:06d}{source.suffix.lower()}"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        fine_tune.to_csv(materialized_dir / "manifest.csv", index=False)
        shutil.make_archive(
            str(args.output_dir / "hard_finetune_dataset"),
            "zip",
            root_dir=materialized_dir,
        )
    summary = {
        "review_queue": int(len(queue)),
        "reviewed_keep_hard": int(len(hard_manifest)),
        "reviewed_validation_kept_out_of_training": int(
            len(hard) - len(hard_train)
        ),
        "quarantined": int(len(quarantine)),
        "unresolved": int(len(unresolved)),
        "fresh_selected": int(len(fresh_manifest)),
        "clean_replay": int(len(replay)),
        "fine_tune_total": int(len(fine_tune)),
        "fine_tune_identities": int(fine_tune["wine_slug"].nunique()),
        "curated_master_rows": int(len(curated_master)),
        "curated_master_kept": int(
            curated_master["curation_status"].eq("kept").sum()
        )
        if not curated_master.empty
        else 0,
        "materialized": bool(args.materialize),
        "warning": (
            "This is a hard-fine-tuning subset, not a replacement for the immutable "
            "evaluation set. Quarantined files were not deleted."
        ),
    }
    (args.output_dir / "curation_export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
