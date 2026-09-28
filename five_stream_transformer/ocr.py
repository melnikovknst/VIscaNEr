"""Frozen EasyOCR extraction with local-only weights and a resumable cache."""

from __future__ import annotations

import gc
import time
from pathlib import Path

import pandas as pd


def fill_ocr_text(
    frame: pd.DataFrame,
    cache_csv: Path,
    source_groups: set[str],
    model_storage_directory: Path,
    gpu: bool = True,
    cache_every: int = 25,
) -> pd.DataFrame:
    """Run EasyOCR only where text is missing and the source is in scope.

    Model downloads are deliberately disabled. Kaggle must receive both EasyOCR
    weight files as an attached, versioned dataset before this function starts.
    """

    output = frame.copy()
    output["ocr_text"] = output["ocr_text"].fillna("").astype(str)
    cached: dict[str, str] = {}
    if cache_csv.is_file():
        prior = pd.read_csv(cache_csv).fillna("")
        cached = dict(zip(prior["sample_id"].astype(str), prior["ocr_text"].astype(str), strict=True))
        output.loc[output["sample_id"].isin(cached), "ocr_text"] = output.loc[
            output["sample_id"].isin(cached), "sample_id"
        ].map(cached)
    pending = output[
        output["source_group"].isin(source_groups)
        & output["ocr_text"].str.strip().eq("")
    ]
    if pending.empty:
        print(f"OCR | cache ready | non-empty={int(output['ocr_text'].str.strip().ne('').sum())}", flush=True)
        return output

    model_storage_directory = model_storage_directory.resolve()
    required_weights = {
        "craft_mlt_25k.pth": "2f8227d2def4037cdb3b34389dcf9ec1",
        "cyrillic_g2.pth": "19f85f43d9128a89ac21b8d6a06973fe",
    }
    missing = [name for name in required_weights if not (model_storage_directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing attached EasyOCR weights in {model_storage_directory}: {missing}. "
            "Runtime downloads are disabled."
        )

    import easyocr

    print(
        f"OCR | frozen EasyOCR ru+en | local_weights={model_storage_directory} "
        f"| download_enabled=False | pending={len(pending)} | cached={len(cached)} | gpu={gpu}",
        flush=True,
    )
    reader = easyocr.Reader(
        ["ru", "en"],
        gpu=gpu,
        model_storage_directory=str(model_storage_directory),
        download_enabled=False,
        verbose=True,
    )
    started = time.perf_counter()
    records = [{"sample_id": key, "ocr_text": value} for key, value in cached.items()]
    failures: list[str] = []
    total = len(pending)
    for number, (index, row) in enumerate(pending.iterrows(), start=1):
        try:
            pieces = reader.readtext(
                str(row["label_path"]), detail=0, paragraph=False,
                decoder="greedy", batch_size=16, workers=0,
            )
            text = " ".join(str(piece).strip() for piece in pieces if str(piece).strip())
        except Exception as exc:
            print(
                f"OCR | {row['sample_id']} first_attempt_error={type(exc).__name__}: {exc} "
                "| retrying batch_size=1",
                flush=True,
            )
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                pieces = reader.readtext(
                    str(row["label_path"]), detail=0, paragraph=False,
                    decoder="greedy", batch_size=1, workers=0,
                )
                text = " ".join(str(piece).strip() for piece in pieces if str(piece).strip())
            except Exception as retry_exc:
                print(
                    f"OCR | {row['sample_id']} failed={type(retry_exc).__name__}: {retry_exc}",
                    flush=True,
                )
                failures.append(str(row["sample_id"]))
                text = ""
        output.at[index, "ocr_text"] = text
        records.append({"sample_id": str(row["sample_id"]), "ocr_text": text})
        if number % cache_every == 0 or number == total:
            elapsed = time.perf_counter() - started
            rate = number / max(elapsed, 1e-9)
            eta_seconds = (total - number) / max(rate, 1e-9)
            print(
                f"OCR | {number}/{total} ({100.0 * number / total:5.1f}%) "
                f"| rate={rate:.2f} img/s | elapsed={elapsed/60:.1f}m "
                f"| eta={eta_seconds/60:.1f}m",
                flush=True,
            )
            cache_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(records).drop_duplicates("sample_id", keep="last").to_csv(cache_csv, index=False)
    del reader
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    print(
        f"OCR | complete | non_empty={int(output['ocr_text'].str.strip().ne('').sum())}/{len(output)} "
        f"| failures={len(failures)} | EasyOCR released before DINO loading",
        flush=True,
    )
    if len(failures) > max(10, int(0.01 * total)):
        raise RuntimeError(
            f"EasyOCR failed on {len(failures)}/{total} rows; first failures: {failures[:10]}. "
            f"Successful rows remain cached at {cache_csv}."
        )
    return output
