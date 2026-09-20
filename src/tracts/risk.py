"""Map tumor–tract distances to discrete risk classes."""

from __future__ import annotations

import math
from typing import Mapping

import pandas as pd


# Distance thresholds in millimeters (tune with clinical collaborators).
DEFAULT_THRESHOLDS = {
    "high": 2.0,  # abutting / involving
    "moderate": 5.0,
    # else low / remote
}


def classify_risk(
    distance_mm: float,
    *,
    high_mm: float = DEFAULT_THRESHOLDS["high"],
    moderate_mm: float = DEFAULT_THRESHOLDS["moderate"],
) -> str:
    """Return ``high``, ``moderate``, or ``low`` from minimum distance."""
    if not math.isfinite(distance_mm):
        return "unknown"
    if distance_mm <= high_mm:
        return "high"
    if distance_mm <= moderate_mm:
        return "moderate"
    return "low"


def risk_table(
    bundle_distances_mm: Mapping[str, float],
    *,
    high_mm: float = DEFAULT_THRESHOLDS["high"],
    moderate_mm: float = DEFAULT_THRESHOLDS["moderate"],
) -> pd.DataFrame:
    """Build a per-bundle distance + risk table."""
    rows = []
    for name, dist in bundle_distances_mm.items():
        rows.append(
            {
                "bundle": name,
                "min_distance_mm": dist,
                "risk": classify_risk(dist, high_mm=high_mm, moderate_mm=moderate_mm),
            }
        )
    return pd.DataFrame(rows).sort_values("min_distance_mm", ascending=True)
