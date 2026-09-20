"""Ellipsoid reference volumes and Bland–Altman agreement analysis."""

from .ellipsoid import (
    abc_over_2_volume_cm3,
    compare_segmentation_to_abc_over_2,
    compare_to_ellipsoid,
    diameters_from_manual,
    diameters_from_segmentation,
    ellipsoid_volume_cm3,
    principal_axis_bbox_diameters_mm,
    true_ellipsoid_volume_cm3,
)
from .bland_altman import bland_altman_stats, bland_altman_dataframe
from .compare import (
    collect_comparison_dataframe,
    error_by_sphericity,
    error_by_tumor_size,
    run_volume_comparison,
    summary_statistics,
)

__all__ = [
    "abc_over_2_volume_cm3",
    "true_ellipsoid_volume_cm3",
    "ellipsoid_volume_cm3",
    "diameters_from_manual",
    "diameters_from_segmentation",
    "principal_axis_bbox_diameters_mm",
    "compare_to_ellipsoid",
    "compare_segmentation_to_abc_over_2",
    "bland_altman_stats",
    "bland_altman_dataframe",
    "run_volume_comparison",
    "collect_comparison_dataframe",
    "summary_statistics",
    "error_by_tumor_size",
    "error_by_sphericity",
]
