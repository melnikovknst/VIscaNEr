"""Streamlit UI for grouped, persistent review of DINO bottle errors."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Streamlit executes this file as a script, so the repository root is not
# guaranteed to be on sys.path even when the command is run from that root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bottle_dataset_curation.paths import DEFAULT_OUTPUT_DIR


DECISIONS = {
    "keep_hard": "✅ Correct crop + correct label — keep as a hard example",
    "reject_wrong_crop": "🗑️ Wrong bottle / broken crop — quarantine",
    "reject_wrong_label": "🏷️ Wrong ground-truth identity — quarantine",
    "quarantine_visual_ambiguity": "👯 Human-indistinguishable identities — quarantine from strict Top-1",
    "skip": "⏭️ Not sure — leave unresolved",
}


def atomic_write(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def load_state(queue_path: Path, decisions_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    queue = pd.read_csv(queue_path)
    if decisions_path.is_file():
        decisions = pd.read_csv(decisions_path).fillna("")
    else:
        decisions = pd.DataFrame(columns=["review_id", "decision", "note", "decided_at"])
    decisions = decisions.drop_duplicates("review_id", keep="last")
    merged = queue.merge(decisions, on="review_id", how="left")
    for column in ("decision", "note", "decided_at"):
        merged[column] = merged[column].fillna("")
    return merged, decisions


def save_decision(
    decisions: pd.DataFrame,
    decisions_path: Path,
    review_ids: list[str],
    decision: str,
    note: str,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    replacement = pd.DataFrame(
        {
            "review_id": review_ids,
            "decision": decision,
            "note": note,
            "decided_at": now,
        }
    )
    decisions = decisions[~decisions["review_id"].isin(review_ids)]
    atomic_write(pd.concat([decisions, replacement], ignore_index=True), decisions_path)


def show_image(path: object, caption: str) -> None:
    candidate = Path(str(path))
    if candidate.is_file():
        st.image(str(candidate), caption=caption, width="stretch")
    else:
        st.error(f"Missing image: {candidate}")


def main() -> None:
    st.set_page_config(page_title="Bottle DINO dataset curation", layout="wide")
    st.title("Whole-bottle DINO error curation")
    output_dir = DEFAULT_OUTPUT_DIR
    queue_path = output_dir / "review_queue.csv"
    decisions_path = output_dir / "curation_decisions.csv"
    if not queue_path.is_file():
        st.error(f"Run the preparation command first. Missing: {queue_path}")
        st.stop()

    merged, decisions = load_state(queue_path, decisions_path)
    decided = merged["decision"].isin(set(DECISIONS) - {"skip"})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("All errors", len(merged))
    col2.metric("Reviewed", int(decided.sum()))
    col3.metric("Remaining", int((~decided).sum()))
    col4.metric("Confusion pairs", merged["confusion_pair"].nunique())
    st.progress(float(decided.mean()) if len(merged) else 1.0)

    with st.sidebar:
        st.header("Queue filters")
        unresolved_only = st.toggle("Unresolved only", value=True)
        bucket_options = sorted(merged["suggested_bucket"].dropna().unique())
        buckets = st.multiselect("Suggested bucket", bucket_options, default=bucket_options)
        split_options = sorted(merged["split"].dropna().unique())
        splits = st.multiselect("Split", split_options, default=split_options)
        min_pair = st.number_input("Minimum pair frequency", min_value=1, value=1)
        st.caption("Decisions are saved immediately to curation_decisions.csv")

    visible = merged[
        merged["suggested_bucket"].isin(buckets)
        & merged["split"].isin(splits)
        & merged["pair_frequency"].ge(min_pair)
    ].copy()
    if unresolved_only:
        visible = visible[~visible["decision"].isin(set(DECISIONS) - {"skip"})]
    if visible.empty:
        st.success("No items match the current filters.")
        st.stop()

    visible = visible.sort_values("review_order").reset_index(drop=True)
    if "cursor" not in st.session_state:
        st.session_state.cursor = 0
    st.session_state.cursor = min(st.session_state.cursor, len(visible) - 1)

    nav1, nav2, nav3 = st.columns([1, 4, 1])
    if nav1.button("← Previous", width="stretch"):
        st.session_state.cursor = max(0, st.session_state.cursor - 1)
        st.rerun()
    selected_order = nav2.number_input(
        "Item",
        min_value=1,
        max_value=len(visible),
        value=st.session_state.cursor + 1,
        step=1,
    )
    st.session_state.cursor = int(selected_order) - 1
    if nav3.button("Next →", width="stretch"):
        st.session_state.cursor = min(len(visible) - 1, st.session_state.cursor + 1)
        st.rerun()

    row = visible.iloc[st.session_state.cursor]
    st.subheader(
        f"{row['suggested_bucket']} · rank {int(row['rank'])} · "
        f"pair occurs {int(row['pair_frequency'])} times"
    )
    image_columns = st.columns(4)
    with image_columns[0]:
        show_image(row.get("local_original_path", ""), "ORIGINAL PHOTO")
    with image_columns[1]:
        show_image(row["local_crop_path"], f"MODEL CROP · detector={row.get('confidence', '')}")
    with image_columns[2]:
        show_image(row["true_reference_path"], f"TRUE · {row['true_slug']}")
    with image_columns[3]:
        show_image(
            row["predicted_reference_path"],
            f"PREDICTED · {row['top1_slug']}",
        )

    st.code(
        f"true={row['true_slug']}\n"
        f"pred={row['top1_slug']}\n"
        f"sim(pred)={row['top1_similarity']:.5f}  sim(true)={row['true_similarity']:.5f}  "
        f"margin={row['top1_top2_margin']:.5f}\n"
        f"reference dHash similarity={row['reference_dhash_similarity']:.3f}\n"
        f"source={row.get('source_relative_path', '')}"
    )

    note = st.text_input("Optional note", value=str(row.get("note", "")), key=row["review_id"])
    button_columns = st.columns(len(DECISIONS))
    for column, (decision, label) in zip(button_columns, DECISIONS.items(), strict=True):
        if column.button(label, key=f"{decision}-{row['review_id']}", width="stretch"):
            save_decision(decisions, decisions_path, [row["review_id"]], decision, note)
            st.session_state.cursor = min(st.session_state.cursor, max(0, len(visible) - 2))
            st.rerun()

    with st.expander("Review this entire confusion pair before a batch decision"):
        all_pair_rows = merged[merged["confusion_pair"].eq(row["confusion_pair"])]
        pair_rows = all_pair_rows.head(12)
        st.dataframe(
            pair_rows[
                [
                    "review_id",
                    "split",
                    "rank",
                    "true_slug",
                    "top1_slug",
                    "top1_top2_margin",
                    "decision",
                ]
            ],
            width="stretch",
            hide_index=True,
        )
        sample_columns = st.columns(min(4, len(pair_rows)))
        for column, (_, sample) in zip(sample_columns, pair_rows.head(4).iterrows()):
            with column:
                show_image(sample["local_crop_path"], f"rank={int(sample['rank'])}")
        confirm = st.checkbox(
            f"I inspected this pair; apply one decision to all {len(all_pair_rows)} rows",
            key=f"confirm-{row['confusion_pair']}",
        )
        batch_decision = st.selectbox(
            "Batch decision",
            ["keep_hard", "quarantine_visual_ambiguity"],
            format_func=lambda value: DECISIONS[value],
            key=f"batch-{row['confusion_pair']}",
        )
        if st.button("Apply to the whole pair", disabled=not confirm):
            all_pair_ids = all_pair_rows["review_id"].tolist()
            save_decision(decisions, decisions_path, all_pair_ids, batch_decision, note)
            st.rerun()


if __name__ == "__main__":
    main()
