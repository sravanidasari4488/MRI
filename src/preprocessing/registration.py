"""Rigid registration, 1 mm isotropic resampling, and full preprocess chain."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import SimpleITK as sitk

from .bias_correction import n4_bias_correct
from .dicom_loader import SeriesMeta, load_dicom_study
from .skull_strip import skull_strip

logger = logging.getLogger(__name__)

CORE_MODALITIES = ("t1", "t1c", "t2", "flair")
MovingModality = Literal["t1c", "t2", "flair"]


@dataclass
class PreprocessResult:
    """Paths produced by :func:`preprocess_patient_study`."""

    study_id: str
    output_dir: str
    dicom_series: list[dict[str, Any]] = field(default_factory=list)
    selected_nifti: dict[str, str] = field(default_factory=dict)
    bias_corrected: dict[str, str] = field(default_factory=dict)
    skull_stripped: dict[str, str] = field(default_factory=dict)
    brain_masks: dict[str, str] = field(default_factory=dict)
    registered: dict[str, str] = field(default_factory=dict)
    resampled_1mm: dict[str, str] = field(default_factory=dict)
    reference: str | None = None
    transforms: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def register_to_reference(
    moving_nifti: str | Path,
    fixed_nifti: str | Path,
    output_nifti: str | Path,
    *,
    transform_type: Literal["rigid", "affine"] = "rigid",
    transform_output: str | Path | None = None,
    number_of_iterations: int = 200,
    sampling_percentage: float = 0.25,
) -> Path:
    """
    Register ``moving`` to ``fixed`` with SimpleITK (rigid by default).

    Writes the resampled moving image in the fixed image's grid. Optionally
    saves the transform as ``.tfm``.
    """
    moving_nifti = Path(moving_nifti)
    fixed_nifti = Path(fixed_nifti)
    output_nifti = Path(output_nifti)
    output_nifti.parent.mkdir(parents=True, exist_ok=True)

    fixed = sitk.Cast(sitk.ReadImage(str(fixed_nifti)), sitk.sitkFloat32)
    moving = sitk.Cast(sitk.ReadImage(str(moving_nifti)), sitk.sitkFloat32)

    if transform_type == "rigid":
        initial_tx = sitk.Euler3DTransform()
    else:
        initial_tx = sitk.AffineTransform(3)

    initial = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        initial_tx,
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(sampling_percentage)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=number_of_iterations,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetInitialTransform(initial, inPlace=False)
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    transform = registration.Execute(fixed, moving)
    logger.info(
        "Registered %s → %s (%s), metric=%.6f, stop=%s",
        moving_nifti.name,
        fixed_nifti.name,
        transform_type,
        registration.GetMetricValue(),
        registration.GetOptimizerStopConditionDescription(),
    )

    resampled = sitk.Resample(
        moving,
        fixed,
        transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32,
    )
    sitk.WriteImage(resampled, str(output_nifti))

    if transform_output is not None:
        transform_output = Path(transform_output)
        transform_output.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteTransform(transform, str(transform_output))

    return output_nifti


def make_isotropic_reference(
    image: sitk.Image,
    *,
    spacing_mm: float = 1.0,
) -> sitk.Image:
    """
    Build an empty reference grid at ``spacing_mm`` isotropic covering ``image``.

    Preserves origin / direction; size is computed from physical extent.
    """
    old_spacing = image.GetSpacing()
    old_size = image.GetSize()
    new_spacing = (float(spacing_mm),) * image.GetDimension()
    new_size = [
        max(1, int(round(old_size[i] * old_spacing[i] / new_spacing[i])))
        for i in range(image.GetDimension())
    ]

    ref = sitk.Image(new_size, image.GetPixelID())
    ref.SetSpacing(new_spacing)
    ref.SetOrigin(image.GetOrigin())
    ref.SetDirection(image.GetDirection())
    return ref


def resample_to_isotropic(
    input_nifti: str | Path,
    output_nifti: str | Path,
    *,
    spacing_mm: float = 1.0,
    reference_image: sitk.Image | None = None,
    interpolator: int = sitk.sitkLinear,
    default_value: float = 0.0,
) -> Path:
    """Resample a NIfTI to isotropic spacing (default 1 mm)."""
    input_nifti = Path(input_nifti)
    output_nifti = Path(output_nifti)
    output_nifti.parent.mkdir(parents=True, exist_ok=True)

    image = sitk.Cast(sitk.ReadImage(str(input_nifti)), sitk.sitkFloat32)
    if reference_image is None:
        reference_image = make_isotropic_reference(image, spacing_mm=spacing_mm)

    resampled = sitk.Resample(
        image,
        reference_image,
        sitk.Transform(),
        interpolator,
        default_value,
        sitk.sitkFloat32,
    )
    sitk.WriteImage(resampled, str(output_nifti))
    logger.info(
        "Resampled %s → %s @ %s mm iso (size=%s)",
        input_nifti.name,
        output_nifti.name,
        spacing_mm,
        resampled.GetSize(),
    )
    return output_nifti


def register_modalities_to_reference(
    modality_paths: dict[str, str | Path],
    output_dir: str | Path,
    *,
    fixed_key: str = "t1",
    atlas_nifti: str | Path | None = None,
    spacing_mm: float = 1.0,
    transform_type: Literal["rigid", "affine"] = "rigid",
) -> dict[str, Path]:
    """
    Rigidly register T1c / T2 / FLAIR to T1 (or an atlas), then resample to 1 mm.

    ``modality_paths`` keys should include ``t1`` (unless ``atlas_nifti`` is set
    as the fixed image for all modalities including T1) and any of
    ``t1c``, ``t2``, ``flair``.

    Returns a dict of modality → path of the final 1 mm isotropic NIfTI.
    Intermediate registered (pre-isotropic) files are also written under
    ``output_dir/registered/``.
    """
    output_dir = Path(output_dir)
    reg_dir = output_dir / "registered"
    iso_dir = output_dir / "isotropic_1mm"
    tx_dir = output_dir / "transforms"
    for d in (reg_dir, iso_dir, tx_dir):
        d.mkdir(parents=True, exist_ok=True)

    paths = {k.lower(): Path(v) for k, v in modality_paths.items()}

    if atlas_nifti is not None:
        fixed_path = Path(atlas_nifti)
        fixed_label = "atlas"
    else:
        if fixed_key not in paths:
            raise KeyError(
                f"Fixed modality '{fixed_key}' missing from modality_paths "
                f"(have: {sorted(paths)})"
            )
        fixed_path = paths[fixed_key]
        fixed_label = fixed_key

    # Align fixed image into registered folder (identity) for a consistent set.
    fixed_reg = reg_dir / f"{fixed_label}_in_ref.nii.gz"
    if not fixed_reg.exists() or fixed_reg.resolve() != fixed_path.resolve():
        sitk.WriteImage(sitk.ReadImage(str(fixed_path)), str(fixed_reg))

    registered: dict[str, Path] = {fixed_label: fixed_reg}

    moving_keys = [k for k in ("t1", "t1c", "t2", "flair") if k in paths and k != fixed_label]
    # If atlas is fixed, also register T1.
    if atlas_nifti is not None and "t1" in paths:
        moving_keys = ["t1"] + [k for k in ("t1c", "t2", "flair") if k in paths]

    for key in moving_keys:
        out = reg_dir / f"{key}_to_{fixed_label}.nii.gz"
        tx = tx_dir / f"{key}_to_{fixed_label}.tfm"
        register_to_reference(
            paths[key],
            fixed_path,
            out,
            transform_type=transform_type,
            transform_output=tx,
        )
        registered[key] = out

    # Shared 1 mm grid from the fixed (already in ref space) image.
    fixed_img = sitk.ReadImage(str(registered[fixed_label]))
    iso_ref = make_isotropic_reference(fixed_img, spacing_mm=spacing_mm)

    isotropic: dict[str, Path] = {}
    for key, path in registered.items():
        out_iso = iso_dir / f"{key}_1mm.nii.gz"
        resample_to_isotropic(
            path,
            out_iso,
            spacing_mm=spacing_mm,
            reference_image=iso_ref,
        )
        isotropic[key] = out_iso

    return isotropic


def _pick_best_series(
    series: list[SeriesMeta],
    modality: str,
) -> SeriesMeta | None:
    """Prefer converted series with the most instances for a modality folder."""
    candidates = [
        s
        for s in series
        if s.inferred_mri_contrast == modality
        and s.nifti_path
        and Path(s.nifti_path).is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda s: s.num_instances)


def preprocess_nifti_modalities(
    modality_paths: dict[str, str | Path],
    output_dir: str | Path,
    *,
    study_id: str,
    atlas_nifti: str | Path | None = None,
    spacing_mm: float = 1.0,
    skip_bias_correction: bool = False,
    skip_skull_strip: bool = False,
    skull_strip_method: Literal["auto", "hd-bet", "simpleitk"] = "auto",
    stage_raw_dir: Path | None = None,
) -> PreprocessResult:
    """
    Run bias correction → skull strip → rigid register → 1 mm resample.

    ``modality_paths`` maps ``t1`` / ``t1c`` / ``t2`` / ``flair`` to NIfTI files.
    Intermediate outputs are written under ``output_dir`` for debugging.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dir_raw = stage_raw_dir or (output_dir / "01_nifti")
    dir_bias = output_dir / "02_bias_corrected"
    dir_strip = output_dir / "03_skull_stripped"
    dir_reg = output_dir / "04_registered"
    dir_iso = output_dir / "05_isotropic_1mm"
    for d in (dir_raw, dir_bias, dir_strip, dir_reg, dir_iso):
        d.mkdir(parents=True, exist_ok=True)

    selected = {k.lower(): Path(v) for k, v in modality_paths.items() if v is not None}
    result = PreprocessResult(study_id=study_id, output_dir=str(output_dir))
    for mod, path in selected.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing NIfTI for {mod}: {path}")
        result.selected_nifti[mod] = str(path)

    for mod in CORE_MODALITIES:
        if mod not in selected:
            result.notes.append(f"No usable {mod} volume found")

    if "t1" not in selected and atlas_nifti is None:
        raise FileNotFoundError(
            "T1 volume required as registration target (or pass atlas_nifti). "
            f"Found modalities: {sorted(selected)}"
        )

    after_bias: dict[str, Path] = {}
    for mod, path in selected.items():
        out = dir_bias / f"{mod}_n4.nii.gz"
        if skip_bias_correction:
            sitk.WriteImage(sitk.ReadImage(str(path)), str(out))
            logger.info("Case %s / %s: skipped N4 (copied raw)", study_id, mod)
        else:
            n4_bias_correct(path, out)
            logger.info("Case %s / %s: N4 bias correction → %s", study_id, mod, out)
        after_bias[mod] = out
        result.bias_corrected[mod] = str(out)

    after_strip: dict[str, Path] = {}
    for mod, path in after_bias.items():
        brain_out = dir_strip / f"{mod}_brain.nii.gz"
        mask_out = dir_strip / f"{mod}_brainmask.nii.gz"
        if skip_skull_strip:
            sitk.WriteImage(sitk.ReadImage(str(path)), str(brain_out))
            img = sitk.ReadImage(str(path))
            mask = sitk.Cast(img != 0, sitk.sitkUInt8)
            mask.CopyInformation(img)
            sitk.WriteImage(mask, str(mask_out))
            method = "skipped"
            logger.info("Case %s / %s: skipped skull strip", study_id, mod)
        else:
            _, _, method = skull_strip(
                path,
                brain_out,
                mask_out,
                method=skull_strip_method,
                case_id=f"{study_id}/{mod}",
            )
        after_strip[mod] = brain_out
        result.skull_stripped[mod] = str(brain_out)
        result.brain_masks[mod] = str(mask_out)
        result.notes.append(f"{mod} skull_strip_method={method}")

    reg_work = output_dir / "_reg_work"
    isotropic = register_modalities_to_reference(
        after_strip,
        reg_work,
        fixed_key="t1",
        atlas_nifti=atlas_nifti,
        spacing_mm=spacing_mm,
        transform_type="rigid",
    )

    for src in (reg_work / "registered").glob("*.nii.gz"):
        dest = dir_reg / src.name
        sitk.WriteImage(sitk.ReadImage(str(src)), str(dest))
        key = src.name.split("_")[0]
        result.registered[key] = str(dest)

    for src in (reg_work / "transforms").glob("*.tfm"):
        dest = dir_reg / src.name
        dest.write_bytes(src.read_bytes())
        result.transforms[src.stem] = str(dest)

    for mod, path in isotropic.items():
        dest = dir_iso / f"{mod}_1mm.nii.gz"
        sitk.WriteImage(sitk.ReadImage(str(path)), str(dest))
        result.resampled_1mm[mod] = str(dest)

    result.reference = str(atlas_nifti) if atlas_nifti is not None else result.skull_stripped.get("t1")

    manifest = output_dir / "preprocess_manifest.json"
    with manifest.open("w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2)
    logger.info(
        "Case %s: preprocess complete → %s (modalities=%s)",
        study_id,
        output_dir,
        sorted(result.resampled_1mm),
    )
    return result


def preprocess_patient_study(
    dicom_root: str | Path,
    output_dir: str | Path,
    *,
    study_id: str | None = None,
    atlas_nifti: str | Path | None = None,
    spacing_mm: float = 1.0,
    skip_bias_correction: bool = False,
    skip_skull_strip: bool = False,
    skull_strip_method: Literal["auto", "hd-bet", "simpleitk"] = "auto",
    dicom_backend: Literal["auto", "dicom2nifti", "simpleitk"] = "auto",
) -> PreprocessResult:
    """
    Full preprocessing chain for one patient study.

    Steps (each written under ``output_dir`` for debugging)::

        01_dicom_nifti/     DICOM → NIfTI by modality
        02_bias_corrected/  N4 bias correction
        03_skull_stripped/  brain extraction (+ masks)
        04_registered/      rigid align to T1 (or atlas)
        05_isotropic_1mm/   1 mm isotropic resample
        preprocess_manifest.json
    """
    dicom_root = Path(dicom_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dir_dicom = output_dir / "01_dicom_nifti"
    dir_dicom.mkdir(parents=True, exist_ok=True)

    series_list = load_dicom_study(
        dicom_root,
        dir_dicom,
        study_id=study_id,
        backend=dicom_backend,
    )
    resolved_study = study_id or (
        Path(series_list[0].nifti_path).parts[-3]
        if series_list and series_list[0].nifti_path
        else dicom_root.name
    )

    selected: dict[str, Path] = {}
    for mod in CORE_MODALITIES:
        best = _pick_best_series(series_list, mod)
        if best and best.nifti_path:
            selected[mod] = Path(best.nifti_path)
            logger.info("Selected %s: %s", mod, best.nifti_path)
        else:
            logger.warning("No usable %s series under %s", mod, dicom_root)

    result = preprocess_nifti_modalities(
        selected,
        output_dir,
        study_id=str(resolved_study),
        atlas_nifti=atlas_nifti,
        spacing_mm=spacing_mm,
        skip_bias_correction=skip_bias_correction,
        skip_skull_strip=skip_skull_strip,
        skull_strip_method=skull_strip_method,
        stage_raw_dir=dir_dicom,
    )
    result.dicom_series = [s.to_json_dict() for s in series_list]
    manifest = output_dir / "preprocess_manifest.json"
    with manifest.open("w", encoding="utf-8") as fh:
        json.dump(result.to_dict(), fh, indent=2)
    return result
