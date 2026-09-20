"""Minimum Euclidean distance (mm) between tumor boundary and tract masks.

Boundary voxels are extracted with ``scipy.ndimage``, converted to physical
millimeters via the NIfTI affine, then queried with a KD-tree.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _as_bool_mask(
    data: np.ndarray,
    *,
    label: int | None = None,
) -> np.ndarray:
    vol = np.asanyarray(data)
    while vol.ndim > 3 and vol.shape[-1] == 1:
        vol = vol[..., 0]
    if vol.ndim != 3:
        raise ValueError(f"Expected a 3-D mask, got shape {vol.shape}")
    if label is not None:
        return vol == label
    return vol > 0


def mask_boundary(
    binary: np.ndarray,
    *,
    connectivity: int = 1,
) -> np.ndarray:
    """
    Boolean boundary of a binary mask (surface voxels).

    A voxel is on the boundary if it is foreground and at least one neighbor
    (face-adjacent for ``connectivity=1``) is background — computed as
    ``mask XOR binary_erosion(mask)`` via ``scipy.ndimage``.
    """
    from scipy import ndimage as ndi

    binary = np.asarray(binary, dtype=bool)
    if not np.any(binary):
        return binary
    structure = ndi.generate_binary_structure(3, connectivity)
    eroded = ndi.binary_erosion(binary, structure=structure, border_value=0)
    boundary = binary & ~eroded
    # Thin structures that vanish under erosion: keep the solid as its own boundary.
    if not np.any(boundary):
        return binary
    return boundary


def voxel_indices_to_world_mm(
    indices_zyx_or_ijk: np.ndarray,
    affine: np.ndarray,
) -> np.ndarray:
    """
    Map integer voxel indices ``(N, 3)`` to world coordinates (mm).

    Uses voxel **centers** (index + 0.5) and ``nibabel.affines.apply_affine``.
    Index order matches nibabel array axes ``(i, j, k)`` ≡ ``argwhere`` order.
    """
    from nibabel.affines import apply_affine

    idx = np.asarray(indices_zyx_or_ijk, dtype=np.float64)
    if idx.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if idx.ndim != 2 or idx.shape[1] != 3:
        raise ValueError(f"indices must be (N, 3), got {idx.shape}")
    centers = idx + 0.5
    return np.asarray(apply_affine(np.asarray(affine, dtype=np.float64), centers), dtype=np.float64)


def boundary_points_mm(
    binary: np.ndarray,
    affine: np.ndarray,
    *,
    connectivity: int = 1,
) -> np.ndarray:
    """World-mm coordinates of mask boundary voxel centers."""
    boundary = mask_boundary(binary, connectivity=connectivity)
    indices = np.argwhere(boundary)
    return voxel_indices_to_world_mm(indices, affine)


def _min_kdtree_distance_mm(
    query_pts_mm: np.ndarray,
    ref_pts_mm: np.ndarray,
) -> float:
    """Nearest-neighbor min distance between two point clouds (mm)."""
    from scipy.spatial import cKDTree

    if query_pts_mm.shape[0] == 0 or ref_pts_mm.shape[0] == 0:
        return float("inf")
    # Build tree on the larger set for slightly better query balance.
    if ref_pts_mm.shape[0] >= query_pts_mm.shape[0]:
        tree = cKDTree(ref_pts_mm)
        dists, _ = tree.query(query_pts_mm, k=1, workers=-1)
    else:
        tree = cKDTree(query_pts_mm)
        dists, _ = tree.query(ref_pts_mm, k=1, workers=-1)
    return float(np.min(dists))


def min_distance_mm(
    tumor_nifti: str | Path,
    tract_nifti: str | Path,
    *,
    tumor_label: int | None = None,
    tract_label: int | None = None,
    connectivity: int = 1,
    use_tract_boundary: bool = True,
) -> float:
    """
    Minimum Euclidean distance (mm) between the **tumor mask boundary** and
    a tract mask.

    Boundary voxels → physical mm via each image's affine → ``cKDTree`` query.
    Returns ``0.0`` if the solid masks overlap (same grid) or the surface
    distance is effectively zero; ``inf`` if either mask is empty.
    """
    import nibabel as nib

    tumor_img = nib.load(str(tumor_nifti))
    tract_img = nib.load(str(tract_nifti))
    tumor = _as_bool_mask(np.asanyarray(tumor_img.dataobj), label=tumor_label)
    tract = _as_bool_mask(np.asanyarray(tract_img.dataobj), label=tract_label)

    if not np.any(tumor) or not np.any(tract):
        return float("inf")

    # Fast overlap check when volumes share the same lattice.
    if tumor.shape == tract.shape and np.allclose(tumor_img.affine, tract_img.affine):
        if np.any(tumor & tract):
            return 0.0

    tumor_pts = boundary_points_mm(tumor, tumor_img.affine, connectivity=connectivity)
    if use_tract_boundary:
        tract_pts = boundary_points_mm(tract, tract_img.affine, connectivity=connectivity)
    else:
        tract_pts = voxel_indices_to_world_mm(np.argwhere(tract), tract_img.affine)

    dist = _min_kdtree_distance_mm(tumor_pts, tract_pts)
    # Numerical / half-voxel tolerance → treat as contact.
    if dist < 1e-6:
        return 0.0
    return dist


def min_distance_arrays_mm(
    tumor_mask: np.ndarray,
    tumor_affine: np.ndarray,
    tract_mask: np.ndarray,
    tract_affine: np.ndarray,
    *,
    tumor_label: int | None = None,
    tract_label: int | None = None,
    connectivity: int = 1,
    use_tract_boundary: bool = True,
) -> float:
    """Same as :func:`min_distance_mm` but from in-memory arrays + affines."""
    tumor = _as_bool_mask(tumor_mask, label=tumor_label)
    tract = _as_bool_mask(tract_mask, label=tract_label)
    if not np.any(tumor) or not np.any(tract):
        return float("inf")
    if (
        tumor.shape == tract.shape
        and np.allclose(tumor_affine, tract_affine)
        and np.any(tumor & tract)
    ):
        return 0.0
    tumor_pts = boundary_points_mm(tumor, tumor_affine, connectivity=connectivity)
    if use_tract_boundary:
        tract_pts = boundary_points_mm(tract, tract_affine, connectivity=connectivity)
    else:
        tract_pts = voxel_indices_to_world_mm(np.argwhere(tract), tract_affine)
    dist = _min_kdtree_distance_mm(tumor_pts, tract_pts)
    return 0.0 if dist < 1e-6 else dist


def distances_to_tracts(
    tumor_nifti: str | Path,
    tract_masks: Mapping[str, str | Path],
    *,
    tumor_label: int | None = None,
    connectivity: int = 1,
    use_tract_boundary: bool = True,
) -> dict[str, float]:
    """
    Min boundary distance (mm) from one tumor mask to **each** tract mask.

    ``tract_masks`` maps bundle name → NIfTI path (e.g. TractSeg outputs).
    """
    import nibabel as nib

    tumor_img = nib.load(str(tumor_nifti))
    tumor = _as_bool_mask(np.asanyarray(tumor_img.dataobj), label=tumor_label)
    if not np.any(tumor):
        return {name: float("inf") for name in tract_masks}

    tumor_pts = boundary_points_mm(tumor, tumor_img.affine, connectivity=connectivity)
    tumor_affine = tumor_img.affine
    results: dict[str, float] = {}

    for name, path in tract_masks.items():
        tract_img = nib.load(str(path))
        tract = _as_bool_mask(np.asanyarray(tract_img.dataobj))
        if not np.any(tract):
            results[name] = float("inf")
            continue
        if (
            tumor.shape == tract.shape
            and np.allclose(tumor_affine, tract_img.affine)
            and np.any(tumor & tract)
        ):
            results[name] = 0.0
            continue
        if use_tract_boundary:
            tract_pts = boundary_points_mm(tract, tract_img.affine, connectivity=connectivity)
        else:
            tract_pts = voxel_indices_to_world_mm(np.argwhere(tract), tract_img.affine)
        dist = _min_kdtree_distance_mm(tumor_pts, tract_pts)
        results[name] = 0.0 if dist < 1e-6 else dist
        logger.info("%s: min distance = %.3f mm", name, results[name])

    return results


def distances_to_tractseg_dir(
    tumor_nifti: str | Path,
    bundle_dir: str | Path,
    *,
    bundles: Sequence[str] | None = None,
    tumor_label: int | None = None,
) -> dict[str, float]:
    """
    Distance from tumor to each ``*.nii.gz`` tract mask in a TractSeg
    ``bundle_segmentations`` directory.
    """
    from .run_tractseg import HIGHLIGHT_BUNDLES, collect_bundle_masks

    bundle_dir = Path(bundle_dir)
    # collect_bundle_masks expects the TractSeg output root (parent of bundle_segmentations)
    # or the bundle_segmentations folder itself.
    if bundle_dir.name == "bundle_segmentations":
        root = bundle_dir.parent
    else:
        root = bundle_dir
    found = collect_bundle_masks(root, bundles=bundles or list(HIGHLIGHT_BUNDLES))
    if not found and bundle_dir.is_dir():
        # Fallback: any NIfTI in the folder.
        found = {
            p.name.replace(".nii.gz", "").replace(".nii", ""): p
            for p in sorted(bundle_dir.glob("*.nii*"))
        }
    if not found:
        raise FileNotFoundError(f"No tract masks under {bundle_dir}")
    return distances_to_tracts(tumor_nifti, found, tumor_label=tumor_label)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description="Min tumor-boundary → tract distance (mm) via affine + KD-tree"
    )
    p.add_argument("--tumor", required=True, help="Tumor segmentation NIfTI")
    p.add_argument("--tract", default=None, help="Single tract mask NIfTI")
    p.add_argument(
        "--tract-dir",
        default=None,
        help="Directory of tract masks (e.g. tractseg/bundle_segmentations)",
    )
    p.add_argument("--tumor-label", type=int, default=None)
    p.add_argument("-o", "--output-json", default=None)
    args = p.parse_args(argv)

    if args.tract is None and args.tract_dir is None:
        p.error("Provide --tract or --tract-dir")

    if args.tract is not None:
        d = min_distance_mm(args.tumor, args.tract, tumor_label=args.tumor_label)
        result: dict[str, Any] = {"tumor": args.tumor, "tract": args.tract, "min_distance_mm": d}
    else:
        dists = distances_to_tractseg_dir(
            args.tumor, args.tract_dir, tumor_label=args.tumor_label
        )
        result = {
            "tumor": args.tumor,
            "tract_dir": args.tract_dir,
            "distances_mm": dists,
        }

    print(json.dumps(result, indent=2))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
