"""Tumor morphometry from a mesh or a voxel mask with known spacing.

Computes:
  - volume (cm³)
  - surface area (mm²)
  - three principal-axis lengths via PCA on mask coordinates (mm) —
    usable later as ellipsoid diameters
  - sphericity index (Wadell; 1 = perfect sphere)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .mesh import (
    _binary_mask,
    mask_array_to_mesh,
    mask_to_mesh,
    mesh_volume_cm3,
    voxel_spacing_mm,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TumorMeasurements:
    """Scalar morphometrics for one tumor region."""

    volume_cm3: float
    surface_area_mm2: float
    principal_axis_lengths_mm: tuple[float, float, float]
    sphericity: float
    # Optional provenance / extras
    source: str = "mask"  # "mask" | "mesh"
    n_voxels: int | None = None
    spacing_mm: tuple[float, float, float] | None = None
    pca_eigenvalues_mm2: tuple[float, float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # JSON-friendly tuples
        d["principal_axis_lengths_mm"] = list(self.principal_axis_lengths_mm)
        if self.spacing_mm is not None:
            d["spacing_mm"] = list(self.spacing_mm)
        if self.pca_eigenvalues_mm2 is not None:
            d["pca_eigenvalues_mm2"] = list(self.pca_eigenvalues_mm2)
        return d


def sphericity_index(volume_cm3: float, surface_area_mm2: float) -> float:
    """
    Wadell sphericity: ``π^{1/3} (6 V)^{2/3} / A``.

    ``volume_cm3`` is converted to mm³; ``surface_area_mm2`` is already mm².
    A perfect sphere scores ``1``; irregular shapes score lower.
    """
    if volume_cm3 <= 0 or surface_area_mm2 <= 0:
        return float("nan")
    v_mm3 = float(volume_cm3) * 1000.0
    return float((math.pi ** (1.0 / 3.0) * (6.0 * v_mm3) ** (2.0 / 3.0)) / surface_area_mm2)


def mask_coordinates_mm(
    binary: np.ndarray,
    spacing_mm: tuple[float, float, float] | list[float] | np.ndarray,
) -> np.ndarray:
    """
    Physical coordinates (mm) of foreground voxel *centers*.

    Returns ``(N, 3)`` with columns aligned to array axes 0, 1, 2.
    """
    binary = np.asarray(binary, dtype=bool)
    if binary.ndim != 3:
        raise ValueError(f"Expected 3-D mask, got shape {binary.shape}")
    spacing = np.asarray(spacing_mm, dtype=np.float64).reshape(3)
    if np.any(spacing <= 0):
        raise ValueError(f"spacing_mm must be positive, got {spacing_mm!r}")

    idx = np.argwhere(binary)  # (N, 3) integer voxel indices
    if idx.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    # Voxel-center coordinates in mm.
    return (idx.astype(np.float64) + 0.5) * spacing


def principal_axis_lengths_mm(
    coords_mm: np.ndarray,
    *,
    spacing_mm: tuple[float, float, float] | list[float] | np.ndarray | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    PCA on point coordinates → three principal-axis **lengths** (mm).

    For each principal direction, the length is the extent of the point cloud
    along that axis (``max − min`` of the projection). When ``spacing_mm`` is
    given (voxel-**center** coordinates), one voxel width along that PC is
    added so lengths match the outer solid extent — suitable as ellipsoid
    diameters ``a ≥ b ≥ c`` for :mod:`validation.ellipsoid`.

    Also returns eigenvalues of the covariance (mm²), largest first.
    """
    pts = np.asarray(coords_mm, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"coords_mm must be (N, 3), got {pts.shape}")
    if pts.shape[0] < 2:
        zeros = (0.0, 0.0, 0.0)
        return zeros, zeros

    centered = pts - pts.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    spacing = (
        None
        if spacing_mm is None
        else np.asarray(spacing_mm, dtype=np.float64).reshape(3)
    )

    lengths: list[float] = []
    for k in range(3):
        proj = centered @ eigvecs[:, k]
        length = float(proj.max() - proj.min())
        if spacing is not None:
            # Expand center-to-center span to outer voxel faces along this PC.
            length += float(np.sum(np.abs(eigvecs[:, k]) * spacing))
        lengths.append(length)

    lengths_t = (lengths[0], lengths[1], lengths[2])
    eigs_t = (float(eigvals[0]), float(eigvals[1]), float(eigvals[2]))
    return lengths_t, eigs_t


def surface_area_mm2_from_mesh(mesh) -> float:
    """Triangle mesh surface area in mm² (vertices must already be in mm)."""
    return float(mesh.area)


def measure_from_mesh(mesh, *, source: str = "mesh") -> TumorMeasurements:
    """
    Morphometrics from a trimesh whose vertices are in millimeters.

    Volume uses the enclosed mesh volume; surface area uses mesh area.
    Principal axes are PCA on the **mesh vertices** (surface samples).
    """
    vol_cm3 = mesh_volume_cm3(mesh)
    area = surface_area_mm2_from_mesh(mesh)
    lengths, eigs = principal_axis_lengths_mm(np.asarray(mesh.vertices, dtype=np.float64))
    sph = sphericity_index(vol_cm3, area)
    return TumorMeasurements(
        volume_cm3=vol_cm3,
        surface_area_mm2=area,
        principal_axis_lengths_mm=lengths,
        sphericity=sph,
        source=source,
        pca_eigenvalues_mm2=eigs,
    )


def measure_from_mask_array(
    mask: np.ndarray,
    spacing_mm: tuple[float, float, float] | list[float] | np.ndarray,
    *,
    label: int | None = None,
    build_surface: bool = True,
) -> TumorMeasurements:
    """
    Morphometrics directly from a voxel mask with known spacing (mm).

    - Volume: voxel count × voxel volume → cm³
    - Surface area: marching-cubes mesh area (mm²), unless ``build_surface=False``
      (then area is NaN and sphericity is NaN)
    - Principal axes: PCA extents on foreground voxel centers in mm
    """
    spacing = tuple(float(s) for s in spacing_mm)
    if len(spacing) != 3:
        raise ValueError(f"spacing_mm must have length 3, got {spacing_mm!r}")

    binary = _binary_mask(np.asanyarray(mask), label=label)
    n_vox = int(binary.sum())
    if n_vox == 0:
        raise ValueError(f"Empty mask for label={label}")

    voxel_mm3 = float(np.prod(np.asarray(spacing, dtype=np.float64)))
    vol_cm3 = float(n_vox * voxel_mm3 / 1000.0)

    coords = mask_coordinates_mm(binary, spacing)
    lengths, eigs = principal_axis_lengths_mm(coords, spacing_mm=spacing)

    area = float("nan")
    if build_surface:
        # Pad so the iso-surface exists when the mask touches the array border
        # (marching_cubes needs values both below and above ``level``).
        padded = np.pad(binary.astype(np.uint8), pad_width=1, mode="constant", constant_values=0)
        mesh = mask_array_to_mesh(padded, spacing, label=None)
        area = surface_area_mm2_from_mesh(mesh)

    sph = sphericity_index(vol_cm3, area)
    return TumorMeasurements(
        volume_cm3=vol_cm3,
        surface_area_mm2=area,
        principal_axis_lengths_mm=lengths,
        sphericity=sph,
        source="mask",
        n_voxels=n_vox,
        spacing_mm=spacing,  # type: ignore[arg-type]
        pca_eigenvalues_mm2=eigs,
    )


def measure_from_mask_nifti(
    mask_nifti: str | Path,
    *,
    label: int | None = None,
    build_surface: bool = True,
) -> TumorMeasurements:
    """Load a segmentation NIfTI and compute morphometrics using header spacing."""
    import nibabel as nib

    mask_nifti = Path(mask_nifti)
    if not mask_nifti.is_file():
        raise FileNotFoundError(f"Mask NIfTI not found: {mask_nifti}")

    img = nib.load(str(mask_nifti))
    data = np.asanyarray(img.dataobj)
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]
    spacing = voxel_spacing_mm(mask_nifti)
    return measure_from_mask_array(
        data,
        spacing,
        label=label,
        build_surface=build_surface,
    )


def measure_tumor(
    mask_or_mesh: str | Path | Any,
    *,
    label: int | None = None,
    prefer: str = "mask",
) -> TumorMeasurements:
    """
    Convenience entry: path to mask NIfTI, mesh file (``.obj``/``.stl``), or
    an in-memory ``trimesh.Trimesh``.

    If ``prefer="mesh"`` and a mask path is given, builds a marching-cubes mesh
    first and measures from that (volume = mesh volume). Default ``prefer="mask"``
    uses voxel volume + PCA on voxels, with surface area from the mesh.
    """
    import trimesh

    # In-memory mesh
    if hasattr(mask_or_mesh, "vertices") and hasattr(mask_or_mesh, "faces"):
        return measure_from_mesh(mask_or_mesh)

    path = Path(mask_or_mesh)
    suffix = path.suffix.lower()
    if suffix in {".obj", ".stl", ".ply"}:
        mesh = trimesh.load(str(path), force="mesh")
        return measure_from_mesh(mesh, source="mesh_file")

    if prefer == "mesh":
        mesh = mask_to_mesh(path, output_mesh=None, label=label)
        # Still report voxel count / spacing from the mask for provenance.
        m = measure_from_mesh(mesh, source="mesh_from_mask")
        spacing = voxel_spacing_mm(path)
        import nibabel as nib

        data = np.asanyarray(nib.load(str(path)).dataobj)
        while data.ndim > 3 and data.shape[-1] == 1:
            data = data[..., 0]
        binary = _binary_mask(data, label=label)
        return TumorMeasurements(
            volume_cm3=m.volume_cm3,
            surface_area_mm2=m.surface_area_mm2,
            principal_axis_lengths_mm=m.principal_axis_lengths_mm,
            sphericity=m.sphericity,
            source="mesh_from_mask",
            n_voxels=int(binary.sum()),
            spacing_mm=spacing,
            pca_eigenvalues_mm2=m.pca_eigenvalues_mm2,
        )

    return measure_from_mask_nifti(path, label=label, build_surface=True)


def measurements_to_json(m: TumorMeasurements | Mapping[str, Any], path: str | Path) -> Path:
    """Write measurements dict to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = m.to_dict() if isinstance(m, TumorMeasurements) else dict(m)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote measurements → %s", path)
    return path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description="Tumor volume, surface area, PCA axes, and sphericity"
    )
    p.add_argument("input", help="Mask NIfTI or mesh (.obj/.stl)")
    p.add_argument("-o", "--output-json", default=None, help="Optional JSON output path")
    p.add_argument("--label", type=int, default=None)
    p.add_argument(
        "--prefer",
        choices=("mask", "mesh"),
        default="mask",
        help="For NIfTI input: voxel volume (mask) or enclosed mesh volume",
    )
    args = p.parse_args(argv)

    m = measure_tumor(args.input, label=args.label, prefer=args.prefer)
    a, b, c = m.principal_axis_lengths_mm
    logger.info(
        "volume=%.4f cm³  area=%.2f mm²  axes=(%.2f, %.2f, %.2f) mm  sphericity=%.4f",
        m.volume_cm3,
        m.surface_area_mm2,
        a,
        b,
        c,
        m.sphericity,
    )
    if args.output_json:
        measurements_to_json(m, args.output_json)
    else:
        print(json.dumps(m.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
