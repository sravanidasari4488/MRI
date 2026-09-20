"""TractSeg integration, tumor–tract distance, and risk tiers."""

from .run_tractseg import (
    HIGHLIGHT_BUNDLES,
    MISSING_DTI_NOTE,
    TRACTSEG_BUNDLES,
    find_dti_input,
    run_tractseg,
    run_tractseg_for_case,
)
from .distance import (
    distances_to_tracts,
    distances_to_tractseg_dir,
    min_distance_mm,
)
from .risk import classify_risk, risk_table
from .risk_classifier import (
    RISK_DISTANCE_THRESHOLDS_MM,
    RISK_LABELS,
    TRACT_FUNCTION_LOOKUP,
    classify_distances,
    classify_tract_risk,
    lookup_tract_function,
)

__all__ = [
    "run_tractseg",
    "run_tractseg_for_case",
    "find_dti_input",
    "TRACTSEG_BUNDLES",
    "HIGHLIGHT_BUNDLES",
    "MISSING_DTI_NOTE",
    "min_distance_mm",
    "distances_to_tracts",
    "distances_to_tractseg_dir",
    "classify_risk",
    "risk_table",
    "RISK_DISTANCE_THRESHOLDS_MM",
    "RISK_LABELS",
    "TRACT_FUNCTION_LOOKUP",
    "classify_tract_risk",
    "classify_distances",
    "lookup_tract_function",
]
