"""Marching-cubes surface extraction from tumor segmentation masks.

Vertices are scaled into physical millimeters by passing the NIfTI header
voxel spacing into ``skimage.measure.marching_cubes(..., spacing=...)``.
Meshes are written as ``.obj`` or ``.stl`` via trimesh.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_MESH_SUFFIXES = {".obj", ".stl"}


def voxel_spacing_mm(nifti_path: str | Path) -> tuple[float, float, float]:
    """
    Read voxel spacing (mm) from a NIfTI header (``pixdim`` / zooms).

    Returns ``(sx, sy, sz)`` matching the array axis order used by nibabel
    (same order passed to ``marching_cubes``).
    """
    import nibabel as nib

    img = nib.load(str(nifti_path))
    zooms = img.header.get_zooms()[:3]
    spacing = tuple(float(z) for z in zooms)
    if len(spacing) != 3 or any(s <= 0 for s in spacing):
        raise ValueError(f"Invalid NIfTI zooms {zooms!r} in {nifti_path}")
    return spacing  # type: ignore[return-value]


def _binary_mask(
    data: np.ndarray,
    *,
    label: int | None = None,
    threshold: float = 0.0,
) -> np.ndarray:
    """Build a boolean volume from exclusive labels or a soft / binary mask."""
    if label is not None:
        return data == label
    # Soft probabilities or already-binary maps.
    if np.issubdtype(data.dtype, np.floating):
        return data > threshold
    return data > 0


def mask_array_to_mesh(
    binary_or_label: np.ndarray,
    spacing_mm: tuple[float, float, float] | list[float] | np.ndarray,
    *,
    label: int | None = None,
    level: float = 0.5,
    step_size: int = 1,
    allow_degenerate: bool = False,
):
    """
    Run marching cubes on an in-memory mask with explicit voxel spacing (mm).

    Parameters
    ----------
    binary_or_label:
        3-D array (label map or binary / probability volume).
    spacing_mm:
        ``(sx, sy, sz)`` in millimeters from the NIfTI header — forwarded to
        ``marching_cubes(..., spacing=spacing_mm)`` so vertices are in mm.
    label:
        If set, keep only voxels equal to this label; otherwise any nonzero /
        above-zero soft mask.
    level:
        Iso-surface value for ``marching_cubes`` (0.5 for a 0/1 mask).

    Returns
    -------
    trimesh.Trimesh
        Triangle mesh with vertices in physical millimeters.
    """
    import trimesh
    from skimage.measure import marching_cubes

    volume = np.asanyarray(binary_or_label)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3-D mask, got shape {volume.shape}")

    spacing = tuple(float(s) for s in spacing_mm)
    if len(spacing) != 3 or any(s <= 0 for s in spacing):
        raise ValueError(f"spacing_mm must be 3 positive floats (mm), got {spacing_mm!r}")

    binary = _binary_mask(volume, label=label, threshold=0.0)
    if not np.any(binary):
        raise ValueError(f"Empty mask for label={label} (no foreground voxels)")

    # Float volume so ``level=0.5`` sits between background (0) and foreground (1).
    verts, faces, normals, _values = marching_cubes(
        binary.astype(np.float32),
        level=level,
        spacing=spacing,
        step_size=step_size,
        allow_degenerate=allow_degenerate,
    )
    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces,
        vertex_normals=normals,
        process=True,
    )
    logger.info(
        "Marching cubes: %d verts, %d faces, spacing_mm=%s, bbox_mm=%s",
        len(mesh.vertices),
        len(mesh.faces),
        spacing,
        np.round(mesh.bounds, 2).tolist(),
    )
    return mesh


def save_mesh(mesh, output_path: str | Path) -> Path:
    """
    Export a trimesh to ``.obj`` or ``.stl``.

    The format is chosen from the file suffix.
    """
    output_path = Path(output_path)
    suffix = output_path.suffix.lower()
    if suffix not in SUPPORTED_MESH_SUFFIXES:
        raise ValueError(
            f"Unsupported mesh format {suffix!r}; use one of "
            f"{sorted(SUPPORTED_MESH_SUFFIXES)}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path))
    logger.info("Wrote mesh → %s", output_path)
    return output_path


def mask_to_mesh(
    mask_nifti: str | Path,
    output_mesh: str | Path | None = None,
    *,
    label: int | None = None,
    level: float = 0.5,
    step_size: int = 1,
) -> Any:
    """
    Extract a triangular surface from a segmentation NIfTI with marching cubes.

    Voxel spacing is read from the NIfTI header (``header.get_zooms()``) and
    passed as ``spacing`` to ``skimage.measure.marching_cubes``, so mesh
    vertices are in **physical millimeters**.

    Parameters
    ----------
    mask_nifti:
        Path to a binary or labeled segmentation ``.nii`` / ``.nii.gz``.
    output_mesh:
        Optional path ending in ``.obj`` or ``.stl``. If omitted, the mesh is
        returned without writing a file.
    label:
        Exclusive label to extract (e.g. ``1``). ``None`` → any nonzero voxel.
    level:
        Iso-value for marching cubes (default ``0.5`` for binary masks).

    Returns
    -------
    trimesh.Trimesh
    """
    import nibabel as nib

    mask_nifti = Path(mask_nifti)
    if not mask_nifti.is_file():
        raise FileNotFoundError(f"Mask NIfTI not found: {mask_nifti}")

    img = nib.load(str(mask_nifti))
    data = np.asanyarray(img.dataobj)
    # Drop trailing time / channel dims if present (H, W, D[, 1]).
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected 3-D mask in {mask_nifti}, got shape {data.shape}")

    spacing = voxel_spacing_mm(mask_nifti)
    logger.info(
        "Building mesh from %s (shape=%s, spacing_mm=%s, label=%s)",
        mask_nifti.name,
        data.shape,
        spacing,
        label,
    )

    mesh = mask_array_to_mesh(
        data,
        spacing,
        label=label,
        level=level,
        step_size=step_size,
    )

    if output_mesh is not None:
        save_mesh(mesh, output_mesh)

    return mesh


def mesh_volume_cm3(mesh) -> float:
    """Return watertight mesh volume in cubic centimeters (vertices in mm)."""
    if not mesh.is_watertight:
        mesh.fill_holes()
    return float(abs(mesh.volume) / 1000.0)  # mm³ → cm³


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description="Marching-cubes mesh from a segmentation mask (vertices in mm)"
    )
    p.add_argument("mask_nifti", help="Path to segmentation .nii / .nii.gz")
    p.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output mesh path (.obj or .stl)",
    )
    p.add_argument(
        "--label",
        type=int,
        default=None,
        help="Exclusive label to extract (default: any nonzero)",
    )
    p.add_argument("--level", type=float, default=0.5)
    p.add_argument("--step-size", type=int, default=1)
    args = p.parse_args(argv)

    mesh = mask_to_mesh(
        args.mask_nifti,
        args.output,
        label=args.label,
        level=args.level,
        step_size=args.step_size,
    )
    try:
        vol = mesh_volume_cm3(mesh)
        logger.info("Mesh volume ≈ %.3f cm³", vol)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not compute mesh volume: %s", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
