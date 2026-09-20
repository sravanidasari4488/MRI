"""Bland–Altman statistics for paired volume measurements."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def bland_altman_stats(
    method_a: Sequence[float],
    method_b: Sequence[float],
) -> dict[str, float]:
    """
    Compute mean difference (bias) and 95% limits of agreement.

    ``method_a`` is typically segmented volume; ``method_b`` the reference
    (ellipsoid / manual).
    """
    a = np.asarray(method_a, dtype=np.float64)
    b = np.asarray(method_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("method_a and method_b must have the same shape")

    diff = a - b
    mean = (a + b) / 2.0
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    return {
        "bias": bias,
        "sd_diff": sd,
        "loa_lower": bias - 1.96 * sd,
        "loa_upper": bias + 1.96 * sd,
        "mean_of_means": float(np.mean(mean)),
        "n": float(len(diff)),
    }


def bland_altman_dataframe(
    method_a: Sequence[float],
    method_b: Sequence[float],
    *,
    id_col: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Per-case means and differences for plotting Bland–Altman charts."""
    a = np.asarray(method_a, dtype=np.float64)
    b = np.asarray(method_b, dtype=np.float64)
    df = pd.DataFrame(
        {
            "method_a": a,
            "method_b": b,
            "mean": (a + b) / 2.0,
            "diff": a - b,
        }
    )
    if id_col is not None:
        df.insert(0, "case_id", list(id_col))
    return df
