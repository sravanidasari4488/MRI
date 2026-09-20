"""N4 bias field correction via SimpleITK."""

from __future__ import annotations

import logging
from pathlib import Path

import SimpleITK as sitk

logger = logging.getLogger(__name__)


def n4_bias_correct(
    input_path: str | Path,
    output_path: str | Path,
    *,
    shrink_factor: int = 2,
    iterations: tuple[int, ...] = (50, 50, 30, 20),
    convergence_threshold: float = 1e-6,
    mask_image: sitk.Image | None = None,
    use_otsu_mask: bool = True,
) -> Path:
    """
    Apply SimpleITK ``N4BiasFieldCorrection`` to a NIfTI volume.

    Parameters
    ----------
    input_path:
        Input NIfTI path (``.nii`` / ``.nii.gz``).
    output_path:
        Destination NIfTI path for the bias-corrected image.
    shrink_factor:
        If ``> 1``, estimate the bias field on a downsampled volume (faster),
        then apply the field at full resolution.
    iterations:
        Maximum iterations per fitting level (passed to N4).
    convergence_threshold:
        N4 convergence threshold.
    mask_image:
        Optional binary mask in the same space as the input. If omitted and
        ``use_otsu_mask`` is True, an Otsu mask is computed automatically.
    use_otsu_mask:
        Build an Otsu foreground mask when ``mask_image`` is not provided.

    Returns
    -------
    Path
        ``output_path`` after writing the corrected volume.

    Notes
    -----
    Output spacing, origin, and direction are copied from the original input
    so geometry is preserved exactly.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input NIfTI not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    original = sitk.ReadImage(str(input_path))
    spacing = original.GetSpacing()
    origin = original.GetOrigin()
    direction = original.GetDirection()

    image = sitk.Cast(original, sitk.sitkFloat32)

    if mask_image is None and use_otsu_mask:
        mask_image = sitk.OtsuThreshold(image, 0, 1, 200)
        mask_image.CopyInformation(image)
    elif mask_image is not None:
        mask_image = sitk.Cast(mask_image, sitk.sitkUInt8)
        if (
            mask_image.GetSize() != image.GetSize()
            or mask_image.GetSpacing() != image.GetSpacing()
            or mask_image.GetOrigin() != image.GetOrigin()
            or mask_image.GetDirection() != image.GetDirection()
        ):
            raise ValueError(
                "mask_image must share size, spacing, origin, and direction with the input"
            )

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(list(iterations))
    corrector.SetConvergenceThreshold(convergence_threshold)

    if shrink_factor > 1:
        factors = [int(shrink_factor)] * image.GetDimension()
        image_s = sitk.Shrink(image, factors)
        if mask_image is not None:
            mask_s = sitk.Shrink(mask_image, factors)
            corrector.Execute(image_s, mask_s)
        else:
            corrector.Execute(image_s)
        # Reconstruct full-resolution corrected image from the estimated log-bias field.
        log_bias = corrector.GetLogBiasFieldAsImage(image)
        corrected = image / sitk.Exp(log_bias)
    else:
        if mask_image is not None:
            corrected = corrector.Execute(image, mask_image)
        else:
            corrected = corrector.Execute(image)

    # Explicitly restore original geometry (N4 / arithmetic can drop metadata).
    corrected.SetSpacing(spacing)
    corrected.SetOrigin(origin)
    corrected.SetDirection(direction)

    sitk.WriteImage(corrected, str(output_path))
    logger.info(
        "N4 bias correction: %s → %s (spacing=%s origin=%s)",
        input_path,
        output_path,
        spacing,
        origin,
    )
    return output_path


# Backward-compatible alias used elsewhere in the package.
def bias_correct(input_nifti: str | Path, output_nifti: str | Path, **kwargs) -> Path:
    return n4_bias_correct(input_nifti, output_nifti, **kwargs)
