"""Skull stripping via HD-BET, with a SimpleITK morphological fallback."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

import SimpleITK as sitk

logger = logging.getLogger(__name__)

SkullStripMethod = Literal["hd-bet", "simpleitk-otsu"]


def _default_mask_path(output_nifti: Path) -> Path:
    # Prefer *.nii.gz → *_brainmask.nii.gz over stripping only the last suffix.
    name = output_nifti.name
    if name.endswith(".nii.gz"):
        stem = name[: -len(".nii.gz")]
        return output_nifti.with_name(f"{stem}_brainmask.nii.gz")
    return output_nifti.with_name(f"{output_nifti.stem}_brainmask.nii.gz")


def hd_bet_available() -> bool:
    """Return True if the ``hd-bet`` CLI or ``HD_BET`` Python package is importable."""
    if shutil.which("hd-bet") is not None:
        return True
    try:
        import HD_BET  # noqa: F401

        return True
    except ImportError:
        return False


def _run_hd_bet_cli(
    input_nifti: Path,
    output_nifti: Path,
    *,
    device: str,
    save_mask: bool,
) -> None:
    cmd = [
        "hd-bet",
        "-i",
        str(input_nifti),
        "-o",
        str(output_nifti),
        "-device",
        device,
    ]
    if device == "cpu":
        cmd.append("--disable_tta")
    # Older CLIs use -s 1; newer keep mask by default. Pass when supported is best-effort.
    if save_mask:
        cmd.extend(["-s", "1"])

    logger.debug("Running HD-BET CLI: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        # Retry without -s if the flag is unrecognized.
        err = (exc.stderr or "") + (exc.stdout or "")
        if save_mask and "-s" in " ".join(cmd):
            cmd_retry = [c for c in cmd if c not in ("-s", "1")]
            logger.debug("Retrying HD-BET without -s: %s", " ".join(cmd_retry))
            proc = subprocess.run(cmd_retry, check=False, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"hd-bet failed ({proc.returncode}): {proc.stderr or proc.stdout or err}"
                ) from exc
        else:
            raise RuntimeError(f"hd-bet failed ({exc.returncode}): {err}") from exc


def _run_hd_bet_python(
    input_nifti: Path,
    output_nifti: Path,
    *,
    device: str,
    save_mask: bool,
) -> None:
    """Call HD-BET's Python API (API varies slightly across versions)."""
    try:
        from HD_BET.run import run_hd_bet
    except ImportError as exc:
        raise ImportError("HD_BET Python package not available") from exc

    kwargs_candidates = [
        dict(
            input_files=[str(input_nifti)],
            output_files=[str(output_nifti)],
            mode="fast" if device == "cpu" else "accurate",
            device=device,
            postprocess=True,
            do_tta=device != "cpu",
            keep_mask=save_mask,
            overwrite=True,
            bet=True,
        ),
        # Newer / alternate signatures sometimes take plain paths.
        dict(
            mr_file=str(input_nifti),
            out_file=str(output_nifti),
            device=device,
        ),
    ]

    last_err: Exception | None = None
    for kwargs in kwargs_candidates:
        try:
            run_hd_bet(**kwargs)
            return
        except TypeError as exc:
            last_err = exc
            continue
    # Positional fallback used by older HD-BET installs.
    try:
        run_hd_bet(
            [str(input_nifti)],
            [str(output_nifti)],
            "fast" if device == "cpu" else "accurate",
            None,
            device,
            True,
            device != "cpu",
            save_mask,
            True,
            True,
        )
        return
    except Exception as exc:  # noqa: BLE001
        last_err = exc
    raise RuntimeError(f"HD-BET Python API failed: {last_err}") from last_err


def _find_hd_bet_mask(output_nifti: Path) -> Path | None:
    """Locate the mask file HD-BET may have written next to the brain image."""
    parent = output_nifti.parent
    name = output_nifti.name
    stem = name[: -len(".nii.gz")] if name.endswith(".nii.gz") else output_nifti.stem
    candidates = [
        parent / f"{stem}_mask.nii.gz",
        parent / f"{stem}_mask.nii",
        parent / f"{name}_mask.nii.gz",
        parent / f"{stem}.nii.gz_mask.nii.gz",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _apply_mask_to_image(input_nifti: Path, mask_nifti: Path, output_nifti: Path) -> None:
    image = sitk.ReadImage(str(input_nifti))
    mask = sitk.ReadImage(str(mask_nifti))
    if mask.GetSize() != image.GetSize():
        mask = sitk.Resample(mask, image, sitk.Transform(), sitk.sitkNearestNeighbor, 0, mask.GetPixelID())
    mask_bin = sitk.Cast(mask > 0, image.GetPixelID())
    stripped = sitk.Cast(image, sitk.sitkFloat32) * sitk.Cast(mask_bin, sitk.sitkFloat32)
    stripped.CopyInformation(image)
    sitk.WriteImage(stripped, str(output_nifti))


def skull_strip_hd_bet(
    input_nifti: Path,
    output_nifti: Path,
    brain_mask_nifti: Path,
    *,
    device: str = "cuda",
) -> SkullStripMethod:
    """Run HD-BET (CLI preferred, Python API fallback) and normalize mask path."""
    output_nifti.parent.mkdir(parents=True, exist_ok=True)
    brain_mask_nifti.parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("hd-bet") is not None:
        _run_hd_bet_cli(input_nifti, output_nifti, device=device, save_mask=True)
    else:
        _run_hd_bet_python(input_nifti, output_nifti, device=device, save_mask=True)

    if not output_nifti.is_file():
        # Some versions write without .gz or use a slightly different name.
        alt = output_nifti.with_suffix("") if output_nifti.suffix == ".gz" else None
        candidates = [p for p in output_nifti.parent.glob(output_nifti.stem.split(".")[0] + "*.nii*") if p.is_file()]
        if not output_nifti.is_file() and candidates:
            shutil.move(str(candidates[0]), str(output_nifti))
        elif alt and Path(str(alt) + ".nii").is_file():
            shutil.move(str(alt) + ".nii", str(output_nifti))
        elif not output_nifti.is_file():
            raise FileNotFoundError(f"HD-BET did not produce output at {output_nifti}")

    mask_src = _find_hd_bet_mask(output_nifti)
    if mask_src is None:
        # Derive a mask from the stripped image if HD-BET omitted one.
        stripped = sitk.ReadImage(str(output_nifti))
        mask = sitk.Cast(stripped != 0, sitk.sitkUInt8)
        mask.CopyInformation(stripped)
        sitk.WriteImage(mask, str(brain_mask_nifti))
    elif mask_src.resolve() != brain_mask_nifti.resolve():
        shutil.copy2(mask_src, brain_mask_nifti)
        # Ensure stripped image is zero outside mask.
        if not output_nifti.is_file():
            _apply_mask_to_image(input_nifti, brain_mask_nifti, output_nifti)

    return "hd-bet"


def skull_strip_simpleitk(
    input_nifti: Path,
    output_nifti: Path,
    brain_mask_nifti: Path,
    *,
    closing_radius: int = 3,
    opening_radius: int = 1,
) -> SkullStripMethod:
    """
    Fallback brain extraction: Otsu threshold + morphology + largest component.

    Not as accurate as HD-BET; intended only when HD-BET is unavailable.
    """
    output_nifti.parent.mkdir(parents=True, exist_ok=True)
    brain_mask_nifti.parent.mkdir(parents=True, exist_ok=True)

    image = sitk.ReadImage(str(input_nifti))
    image_f = sitk.Cast(image, sitk.sitkFloat32)

    # Mild smoothing stabilizes Otsu on noisy MRI.
    smoothed = sitk.CurvatureFlow(image_f, timeStep=0.125, numberOfIterations=5)
    mask = sitk.OtsuThreshold(smoothed, 0, 1, 200)

    if opening_radius > 0:
        mask = sitk.BinaryMorphologicalOpening(mask, [opening_radius] * mask.GetDimension())
    if closing_radius > 0:
        mask = sitk.BinaryMorphologicalClosing(mask, [closing_radius] * mask.GetDimension())

    # Keep the largest connected foreground component (assumed brain).
    cc = sitk.ConnectedComponent(mask)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(cc)
    if stats.GetNumberOfLabels() == 0:
        raise RuntimeError(f"Otsu fallback produced an empty mask for {input_nifti}")

    largest = max(stats.GetLabels(), key=lambda lab: stats.GetPhysicalSize(lab))
    mask = sitk.Equal(cc, largest)
    # Fill holes inside the brain mask.
    mask = sitk.BinaryFillhole(mask)
    mask = sitk.Cast(mask, sitk.sitkUInt8)
    mask.CopyInformation(image)

    stripped = image_f * sitk.Cast(mask, sitk.sitkFloat32)
    stripped.CopyInformation(image)

    sitk.WriteImage(stripped, str(output_nifti))
    sitk.WriteImage(mask, str(brain_mask_nifti))
    return "simpleitk-otsu"


def skull_strip(
    input_nifti: str | Path,
    output_nifti: str | Path,
    brain_mask_nifti: str | Path | None = None,
    *,
    method: Literal["auto", "hd-bet", "simpleitk"] = "auto",
    device: str | None = None,
    case_id: str | None = None,
) -> tuple[Path, Path, SkullStripMethod]:
    """
    Skull-strip a NIfTI volume.

    Prefers HD-BET when ``method='auto'`` and HD-BET is installed; otherwise
    falls back to SimpleITK Otsu + morphological cleanup. Logs the method used.

    Returns
    -------
    (stripped_nifti, brain_mask_nifti, method_used)
    """
    input_nifti = Path(input_nifti)
    output_nifti = Path(output_nifti)
    if not input_nifti.is_file():
        raise FileNotFoundError(f"Input NIfTI not found: {input_nifti}")

    if brain_mask_nifti is None:
        brain_mask_nifti = _default_mask_path(output_nifti)
    else:
        brain_mask_nifti = Path(brain_mask_nifti)

    label = case_id or input_nifti.name

    if device is None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    use_hd_bet = method == "hd-bet" or (method == "auto" and hd_bet_available())
    if method == "hd-bet" and not hd_bet_available():
        raise ImportError(
            "HD-BET requested but not installed. Install with: pip install hd-bet"
        )

    try:
        if use_hd_bet:
            used = skull_strip_hd_bet(
                input_nifti,
                output_nifti,
                brain_mask_nifti,
                device=device,
            )
        else:
            if method == "auto":
                logger.warning(
                    "Case %s: HD-BET not installed; using SimpleITK Otsu fallback",
                    label,
                )
            used = skull_strip_simpleitk(input_nifti, output_nifti, brain_mask_nifti)
    except Exception:
        if method == "auto" and use_hd_bet:
            logger.exception(
                "Case %s: HD-BET failed; falling back to SimpleITK Otsu",
                label,
            )
            used = skull_strip_simpleitk(input_nifti, output_nifti, brain_mask_nifti)
        else:
            raise

    logger.info(
        "Case %s: skull stripping method=%s | brain=%s | mask=%s",
        label,
        used,
        output_nifti,
        brain_mask_nifti,
    )
    return output_nifti, brain_mask_nifti, used


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if len(sys.argv) < 3:
        print(
            f"Usage: {sys.argv[0]} INPUT.nii.gz OUTPUT_BRAIN.nii.gz [MASK.nii.gz]",
            file=sys.stderr,
        )
        sys.exit(2)
    mask_arg = sys.argv[3] if len(sys.argv) > 3 else None
    skull_strip(sys.argv[1], sys.argv[2], mask_arg)
