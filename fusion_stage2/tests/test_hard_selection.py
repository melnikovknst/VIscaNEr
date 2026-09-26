from __future__ import annotations

import numpy as np
import pandas as pd

from fusion_stage2.build_hard_dataset import choose_hard_rows


def test_hard_selection_respects_size_and_identity_cap() -> None:
    frame = pd.DataFrame(
        {
            "wine_slug": np.repeat([f"wine-{index}" for index in range(10)], 10),
            "hard_score": np.linspace(1.0, 0.0, 100),
            "label_true_rank": np.tile(np.arange(1, 11), 10),
            "bottle_true_rank": np.tile(np.arange(10, 0, -1), 10),
        }
    )
    selected = choose_hard_rows(frame, target=40, minimum_per_identity=2, maximum_per_identity=5)
    result = frame.loc[selected]
    assert len(result) == 40
    counts = result["wine_slug"].value_counts()
    assert counts.min() >= 2
    assert counts.max() <= 5
