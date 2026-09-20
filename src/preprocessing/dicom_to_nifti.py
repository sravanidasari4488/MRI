"""Convert a DICOM series directory to a NIfTI volume."""

from __future__ import annotations

from pathlib import Path


def convert_dicom_series(
    dicom_dir: str | Path,
    output_nifti: str | Path,
    *,
    reorient: bool = True,
) -> Path:
    """
    Convert one DICOM series folder to a single ``.nii.gz`` file.

    Uses ``dicom2nifti`` when available; falls back guidance is in the README.
    """
    dicom_dir = Path(dicom_dir)
    output_nifti = Path(output_nifti)
    output_nifti.parent.mkdir(parents=True, exist_ok=True)

    import dicom2nifti

    # dicom2nifti writes into a directory; we normalize to a single target path.
    tmp_dir = output_nifti.parent / f".tmp_{output_nifti.stem}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        dicom2nifti.convert_directory(
            str(dicom_dir),
            str(tmp_dir),
            compression=True,
            reorient=reorient,
        )
        produced = sorted(tmp_dir.glob("*.nii*"))
        if not produced:
            raise FileNotFoundError(f"No NIfTI produced from {dicom_dir}")
        produced[0].replace(output_nifti)
    finally:
        for p in tmp_dir.glob("*"):
            p.unlink(missing_ok=True)
        tmp_dir.rmdir()

    return output_nifti
