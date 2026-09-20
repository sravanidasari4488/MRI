"""Clinical ellipsoid (ABC/2) tumor volume vs segmented volume.

The radiology rule-of-thumb ``V ≈ A × B × C / 2`` uses three orthogonal
diameters (mm). Diameters may be typed in by hand or estimated automatically
as the bounding-box extents of the segmentation along its three principal
axes — a stand-in for how a radiologist measures the lesion on 2D slices.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

logger = logging.getLogger(__name__)

FormulaName = Literal["abc_over_2", "true_ellipsoid"]


def abc_over_2_volume_cm3(a_mm: float, b_mm: float, c_mm: float) -> float:
    """
    Standard clinical ellipsoid volume estimate.

    ``V = A × B × C / 2`` with diameters in **millimeters**, returned in **cm³**
    (``mm³ / 1000``).
    """
    if min(a_mm, b_mm, c_mm) < 0:
        raise ValueError(f"Diameters must be non-negative, got {(a_mm, b_mm, c_mm)}")
    return float(a_mm * b_mm * c_mm / 2.0 / 1000.0)


def true_ellipsoid_volume_cm3(a_mm: float, b_mm: float, c_mm: float) -> float:
    """
    Geometric ellipsoid volume from diameters ``a, b, c`` (mm) → cm³.

    ``V = (4/3) π (a/2)(b/2)(c/2)`` ≡ ``π/6 × A × B × C``.
    """
    if min(a_mm, b_mm, c_mm) < 0:
        raise ValueError(f"Diameters must be non-negative, got {(a_mm, b_mm, c_mm)}")
    return float(
        (4.0 / 3.0) * math.pi * (a_mm / 2.0) * (b_mm / 2.0) * (c_mm / 2.0) / 1000.0
    )


def ellipsoid_volume_cm3(
    a_mm: float,
    b_mm: float,
    c_mm: float,
    *,
    formula: FormulaName = "abc_over_2",
) -> float:
    """
    Ellipsoid volume from three diameters (mm) → cm³.

    Default formula is the clinical ``ABC/2`` estimate. Pass
    ``formula="true_ellipsoid"`` for the geometric ``π ABC / 6`` volume.
    """
    if formula == "abc_over_2":
        return abc_over_2_volume_cm3(a_mm, b_mm, c_mm)
    if formula == "true_ellipsoid":
        return true_ellipsoid_volume_cm3(a_mm, b_mm, c_mm)
    raise ValueError(f"Unknown formula {formula!r}; use 'abc_over_2' or 'true_ellipsoid'")


@dataclass(frozen=True)
class EllipsoidEstimate:
    """Three diameters + clinical / geometric volumes."""

    a_mm: float
    b_mm: float
    c_mm: float
    volume_abc_over_2_cm3: float
    volume_true_ellipsoid_cm3: float
    diameter_source: str  # "manual" | "principal_axes"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def diameters_from_manual(
    a_mm: float,
    b_mm: float,
    c_mm: float,
    *,
    sort_descending: bool = True,
) -> EllipsoidEstimate:
    """Build an estimate from manually entered orthogonal diameters (mm)."""
    diameters = [float(a_mm), float(b_mm), float(c_mm)]
    if sort_descending:
        diameters.sort(reverse=True)
    a, b, c = diameters
    return EllipsoidEstimate(
        a_mm=a,
        b_mm=b,
        c_mm=c,
        volume_abc_over_2_cm3=abc_over_2_volume_cm3(a, b, c),
        volume_true_ellipsoid_cm3=true_ellipsoid_volume_cm3(a, b, c),
        diameter_source="manual",
    )


def principal_axis_bbox_diameters_mm(
    mask: np.ndarray,
    spacing_mm: tuple[float, float, float] | Sequence[float],
    *,
    label: int | None = None,
) -> tuple[float, float, float]:
    """
    Estimate radiologist-style diameters as PCA bounding-box extents (mm).

    Foreground voxel centers (scaled by ``spacing_mm``) are projected onto the
    three principal axes; each diameter is the outer extent along that axis
    (same convention as :func:`reconstruction.measurements.principal_axis_lengths_mm`).
    Sorted ``a ≥ b ≥ c``.
    """
    from reconstruction.measurements import (
        mask_coordinates_mm,
        principal_axis_lengths_mm,
    )
    from reconstruction.mesh import _binary_mask

    binary = _binary_mask(np.asanyarray(mask), label=label)
    if not np.any(binary):
        raise ValueError(f"Empty mask for label={label}")
    spacing = tuple(float(s) for s in spacing_mm)
    coords = mask_coordinates_mm(binary, spacing)
    lengths, _eigs = principal_axis_lengths_mm(coords, spacing_mm=spacing)
    return lengths


def diameters_from_segmentation(
    mask_nifti: str | Path,
    *,
    label: int | None = None,
) -> EllipsoidEstimate:
    """
    Auto-estimate A, B, C from a segmentation NIfTI via principal-axis extents.

    Simulates measuring the three orthogonal tumor diameters on 2D slices
    aligned with the lesion's long axes.
    """
    import nibabel as nib
    from reconstruction.mesh import voxel_spacing_mm

    mask_nifti = Path(mask_nifti)
    if not mask_nifti.is_file():
        raise FileNotFoundError(f"Mask NIfTI not found: {mask_nifti}")

    img = nib.load(str(mask_nifti))
    data = np.asanyarray(img.dataobj)
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]
    spacing = voxel_spacing_mm(mask_nifti)
    a, b, c = principal_axis_bbox_diameters_mm(data, spacing, label=label)
    logger.info(
        "PCA bbox diameters from %s: A=%.2f B=%.2f C=%.2f mm (spacing=%s)",
        mask_nifti.name,
        a,
        b,
        c,
        spacing,
    )
    return EllipsoidEstimate(
        a_mm=a,
        b_mm=b,
        c_mm=c,
        volume_abc_over_2_cm3=abc_over_2_volume_cm3(a, b, c),
        volume_true_ellipsoid_cm3=true_ellipsoid_volume_cm3(a, b, c),
        diameter_source="principal_axes",
    )


def compare_to_ellipsoid(
    segmented_cm3: float,
    a_mm: float,
    b_mm: float,
    c_mm: float,
    *,
    formula: FormulaName = "abc_over_2",
) -> dict[str, float]:
    """
    Compare segmented volume to the ellipsoid estimate from diameters.

    Default reference is clinical ``ABC/2``. Also reports the true-ellipsoid
    volume for context.
    """
    abc = abc_over_2_volume_cm3(a_mm, b_mm, c_mm)
    true_v = true_ellipsoid_volume_cm3(a_mm, b_mm, c_mm)
    ref = abc if formula == "abc_over_2" else true_v
    abs_err = float(segmented_cm3) - ref
    rel_err = abs_err / ref if ref != 0 else float("nan")
    return {
        "segmented_cm3": float(segmented_cm3),
        "ellipsoid_cm3": float(ref),
        "abc_over_2_cm3": float(abc),
        "true_ellipsoid_cm3": float(true_v),
        "a_mm": float(a_mm),
        "b_mm": float(b_mm),
        "c_mm": float(c_mm),
        "formula": formula,
        "abs_error_cm3": float(abs_err),
        "rel_error": float(rel_err),
    }


def compare_segmentation_to_abc_over_2(
    mask_nifti: str | Path,
    *,
    label: int | None = None,
    manual_diameters_mm: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    """
    Full comparison: voxel segmented volume vs ABC/2.

    Diameters come from ``manual_diameters_mm`` if given; otherwise from
    principal-axis bounding-box extents of the mask.
    """
    from reconstruction.volume import voxel_volume_cm3

    mask_nifti = Path(mask_nifti)
    segmented = voxel_volume_cm3(mask_nifti, label=label)

    if manual_diameters_mm is not None:
        est = diameters_from_manual(*manual_diameters_mm)
    else:
        est = diameters_from_segmentation(mask_nifti, label=label)

    cmp = compare_to_ellipsoid(
        segmented,
        est.a_mm,
        est.b_mm,
        est.c_mm,
        formula="abc_over_2",
    )
    return {
        **cmp,
        "diameter_source": est.diameter_source,
        "mask_nifti": str(mask_nifti),
        "label": label,
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description="Clinical ABC/2 ellipsoid volume (manual or PCA diameters)"
    )
    p.add_argument(
        "--mask",
        default=None,
        help="Segmentation NIfTI (auto diameters + voxel volume comparison)",
    )
    p.add_argument("--label", type=int, default=None)
    p.add_argument(
        "--diameters",
        nargs=3,
        type=float,
        metavar=("A", "B", "C"),
        default=None,
        help="Manual diameters in mm (overrides PCA when set with --mask)",
    )
    p.add_argument(
        "--formula",
        choices=("abc_over_2", "true_ellipsoid"),
        default="abc_over_2",
    )
    p.add_argument("-o", "--output-json", default=None)
    args = p.parse_args(argv)

    if args.mask is None and args.diameters is None:
        p.error("Provide --diameters A B C and/or --mask path")

    if args.mask is not None:
        result = compare_segmentation_to_abc_over_2(
            args.mask,
            label=args.label,
            manual_diameters_mm=tuple(args.diameters) if args.diameters else None,
        )
        # Recompute reference if user asked for true ellipsoid as primary.
        if args.formula != "abc_over_2":
            result = compare_to_ellipsoid(
                result["segmented_cm3"],
                result["a_mm"],
                result["b_mm"],
                result["c_mm"],
                formula=args.formula,
            )
            result["diameter_source"] = (
                "manual" if args.diameters else "principal_axes"
            )
            result["mask_nifti"] = str(args.mask)
    else:
        assert args.diameters is not None
        est = diameters_from_manual(*args.diameters)
        result = est.to_dict()
        result["ellipsoid_cm3"] = ellipsoid_volume_cm3(
            est.a_mm, est.b_mm, est.c_mm, formula=args.formula
        )
        result["formula"] = args.formula

    print(json.dumps(result, indent=2))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("Wrote → %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
