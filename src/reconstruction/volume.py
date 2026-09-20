"""Voxel-count volume from a segmentation mask."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def voxel_volume_cm3(
    mask_nifti: str | Path,
    *,
    label: int | None = None,
) -> float:
    """
    Compute volume as ``n_voxels * voxel_volume``, returned in cm³.

    Assumes NIfTI zooms are in millimeters.
    """
    import nibabel as nib

    img = nib.load(str(mask_nifti))
    data = np.asanyarray(img.dataobj)
    zooms = np.asarray(img.header.get_zooms()[:3], dtype=np.float64)
    voxel_mm3 = float(np.prod(zooms))

    binary = (data == label) if label is not None else (data > 0)
    return float(binary.sum() * voxel_mm3 / 1000.0)
