"""Load a DICOM study directory, group by series, and convert to NIfTI."""

from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

import pydicom
from pydicom.dataset import Dataset
from pydicom.errors import InvalidDicomError

logger = logging.getLogger(__name__)

ModalityFolder = Literal["t1", "t1c", "t2", "flair", "other"]

# Ordered rules: first match wins. Patterns are applied to SeriesDescription + ProtocolName.
_MODALITY_RULES: list[tuple[ModalityFolder, re.Pattern[str]]] = [
    ("flair", re.compile(r"\bflair\b|fluid[\s_-]*attenuated", re.I)),
    (
        "t1c",
        re.compile(
            r"t1[\s_-]*(c|ce|gd|gado)|t1.*\bpost\b|\bpost\b.*t1|"
            r"[\s_+]c\+|\+c\b|post[\s_-]*contrast|\bcontrast\b|\bgad+",
            re.I,
        ),
    ),
    ("t1", re.compile(r"\bt1\b|mprage|spgr|bravo", re.I)),
    ("t2", re.compile(r"\bt2\b(?![\s_-]*flair)|t2[_\s-]?w|space|cube", re.I)),
]


@dataclass
class SeriesMeta:
    """Metadata preserved from a DICOM series for the JSON sidecar."""

    series_instance_uid: str
    series_number: str | None = None
    series_description: str | None = None
    protocol_name: str | None = None
    modality: str | None = None
    inferred_mri_contrast: ModalityFolder = "other"
    study_instance_uid: str | None = None
    patient_id: str | None = None
    pixel_spacing: list[float] | None = None
    slice_thickness: float | None = None
    spacing_between_slices: float | None = None
    image_orientation_patient: list[float] | None = None
    image_position_patient: list[float] | None = None
    rows: int | None = None
    columns: int | None = None
    num_instances: int = 0
    conversion_backend: str | None = None
    nifti_path: str | None = None
    dicom_files: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def classify_modality(series_description: str | None, protocol_name: str | None) -> ModalityFolder:
    """Map SeriesDescription / ProtocolName to t1, t1c, t2, flair, or other."""
    text = " ".join(x for x in (series_description, protocol_name) if x).strip()
    if not text:
        return "other"
    for folder, pattern in _MODALITY_RULES:
        if pattern.search(text):
            return folder
    return "other"


def _as_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        return [float(v) for v in value]
    except TypeError:
        try:
            return [float(value)]
        except (TypeError, ValueError):
            return None


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_str(ds: Dataset, keyword: str) -> str | None:
    value = getattr(ds, keyword, None)
    if value is None:
        return None
    return str(value).strip() or None


def iter_dicom_files(root: Path) -> Iterable[Path]:
    """Yield candidate DICOM paths under ``root`` (skips obvious non-DICOM names)."""
    skip_suffixes = {".nii", ".nii.gz", ".json", ".txt", ".csv", ".png", ".jpg", ".jpeg"}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name == "dicomdir":
            continue
        if any(name.endswith(sfx) for sfx in skip_suffixes):
            continue
        yield path


def read_dicom_header(path: Path, *, force: bool = False) -> Dataset | None:
    """Read a DICOM header without pixel data; return None if not DICOM."""
    try:
        return pydicom.dcmread(str(path), stop_before_pixels=True, force=force)
    except (InvalidDicomError, OSError, ValueError) as exc:
        logger.debug("Skipping non-DICOM or unreadable file %s: %s", path, exc)
        return None


def group_series_by_uid(
    dicom_root: str | Path,
    *,
    force: bool = False,
) -> dict[str, list[tuple[Path, Dataset]]]:
    """
    Walk ``dicom_root`` and group files by ``SeriesInstanceUID``.

    Returns a mapping of UID → list of (path, header dataset).
    """
    dicom_root = Path(dicom_root)
    if not dicom_root.is_dir():
        raise NotADirectoryError(f"DICOM root is not a directory: {dicom_root}")

    groups: dict[str, list[tuple[Path, Dataset]]] = defaultdict(list)
    for path in iter_dicom_files(dicom_root):
        ds = read_dicom_header(path, force=force)
        if ds is None:
            continue
        uid = getattr(ds, "SeriesInstanceUID", None)
        if not uid:
            logger.warning("No SeriesInstanceUID in %s; skipping", path)
            continue
        groups[str(uid)].append((path, ds))

    logger.info("Found %d series under %s", len(groups), dicom_root)
    return dict(groups)


def extract_series_meta(
    uid: str,
    members: list[tuple[Path, Dataset]],
) -> SeriesMeta:
    """Build sidecar metadata from the first readable instance in a series."""
    members_sorted = sorted(
        members,
        key=lambda item: (
            int(getattr(item[1], "InstanceNumber", 0) or 0),
            str(item[0]),
        ),
    )
    path0, ds = members_sorted[0]
    desc = _safe_str(ds, "SeriesDescription")
    protocol = _safe_str(ds, "ProtocolName")
    contrast = classify_modality(desc, protocol)

    meta = SeriesMeta(
        series_instance_uid=uid,
        series_number=_safe_str(ds, "SeriesNumber"),
        series_description=desc,
        protocol_name=protocol,
        modality=_safe_str(ds, "Modality"),
        inferred_mri_contrast=contrast,
        study_instance_uid=_safe_str(ds, "StudyInstanceUID"),
        patient_id=_safe_str(ds, "PatientID"),
        pixel_spacing=_as_float_list(getattr(ds, "PixelSpacing", None)),
        slice_thickness=_as_float(getattr(ds, "SliceThickness", None)),
        spacing_between_slices=_as_float(getattr(ds, "SpacingBetweenSlices", None)),
        image_orientation_patient=_as_float_list(getattr(ds, "ImageOrientationPatient", None)),
        image_position_patient=_as_float_list(getattr(ds, "ImagePositionPatient", None)),
        rows=int(ds.Rows) if getattr(ds, "Rows", None) is not None else None,
        columns=int(ds.Columns) if getattr(ds, "Columns", None) is not None else None,
        num_instances=len(members_sorted),
        dicom_files=[str(p) for p, _ in members_sorted],
    )

    if meta.pixel_spacing is None:
        meta.notes.append("PixelSpacing missing")
    if meta.slice_thickness is None:
        meta.notes.append("SliceThickness missing")
    if meta.image_orientation_patient is None:
        meta.notes.append("ImageOrientationPatient missing")

    logger.info(
        "Series %s | contrast=%s | desc=%r | protocol=%r | n=%d | "
        "PixelSpacing=%s SliceThickness=%s IOP=%s",
        uid[:16] + "…",
        contrast,
        desc,
        protocol,
        meta.num_instances,
        meta.pixel_spacing,
        meta.slice_thickness,
        meta.image_orientation_patient,
    )
    # Avoid unused-var lint if path0 only used for potential future logging.
    _ = path0
    return meta


def _write_sidecar(meta: SeriesMeta, sidecar_path: Path) -> Path:
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    with sidecar_path.open("w", encoding="utf-8") as fh:
        json.dump(meta.to_json_dict(), fh, indent=2)
    logger.info("Wrote sidecar %s", sidecar_path)
    return sidecar_path


def _stage_series_files(members: list[tuple[Path, Dataset]], staging_dir: Path) -> Path:
    """Copy/symlink series files into a flat temp folder for converters."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    for i, (src, _) in enumerate(members):
        dest = staging_dir / f"img_{i:05d}{src.suffix if src.suffix else '.dcm'}"
        try:
            dest.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dest)
    return staging_dir


def convert_series_dicom2nifti(staging_dir: Path, output_nifti: Path) -> Path:
    """Convert a staged series directory with dicom2nifti."""
    import dicom2nifti
    import dicom2nifti.settings as d2n_settings

    d2n_settings.disable_validate_slice_increment()
    d2n_settings.disable_validate_orthogonal()

    output_nifti.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="d2n_") as tmp:
        tmp_path = Path(tmp)
        dicom2nifti.convert_directory(
            str(staging_dir),
            str(tmp_path),
            compression=True,
            reorient=True,
        )
        produced = sorted(tmp_path.glob("*.nii*"))
        if not produced:
            raise RuntimeError(f"dicom2nifti produced no NIfTI for {staging_dir}")
        shutil.move(str(produced[0]), str(output_nifti))
    return output_nifti


def convert_series_simpleitk(file_paths: list[Path], output_nifti: Path) -> Path:
    """Convert an ordered list of DICOM instance paths with SimpleITK."""
    import SimpleITK as sitk

    output_nifti.parent.mkdir(parents=True, exist_ok=True)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([str(p) for p in file_paths])
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()
    sitk.WriteImage(image, str(output_nifti))
    return output_nifti


def convert_series_to_nifti(
    members: list[tuple[Path, Dataset]],
    output_nifti: Path,
    *,
    backend: Literal["auto", "dicom2nifti", "simpleitk"] = "auto",
) -> tuple[Path, str]:
    """
    Convert one series to NIfTI.

    Tries ``dicom2nifti`` first when ``backend='auto'``, then SimpleITK.
    Returns ``(nifti_path, backend_used)``.
    """
    members_sorted = sorted(
        members,
        key=lambda item: (
            int(getattr(item[1], "InstanceNumber", 0) or 0),
            str(item[0]),
        ),
    )
    file_paths = [p for p, _ in members_sorted]
    output_nifti = Path(output_nifti)
    output_nifti.parent.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []

    def _try_d2n() -> Path:
        with tempfile.TemporaryDirectory(prefix="dcm_stage_") as tmp:
            staging = _stage_series_files(members_sorted, Path(tmp) / "series")
            return convert_series_dicom2nifti(staging, output_nifti)

    if backend in ("auto", "dicom2nifti"):
        try:
            return _try_d2n(), "dicom2nifti"
        except Exception as exc:  # noqa: BLE001 — fall back when auto
            errors.append(f"dicom2nifti: {exc}")
            logger.warning("dicom2nifti failed (%s); falling back to SimpleITK", exc)
            if backend == "dicom2nifti":
                raise

    if backend in ("auto", "simpleitk"):
        try:
            return convert_series_simpleitk(file_paths, output_nifti), "simpleitk"
        except Exception as exc:  # noqa: BLE001
            errors.append(f"simpleitk: {exc}")
            if backend == "simpleitk":
                raise

    raise RuntimeError("Series conversion failed: " + " | ".join(errors))


def _safe_slug(text: str | None, fallback: str) -> str:
    if not text:
        return fallback
    slug = re.sub(r"[^\w.\-]+", "_", text.strip())[:80].strip("_")
    return slug or fallback


def load_dicom_study(
    dicom_root: str | Path,
    output_dir: str | Path,
    *,
    study_id: str | None = None,
    backend: Literal["auto", "dicom2nifti", "simpleitk"] = "auto",
    force: bool = False,
    min_instances: int = 2,
) -> list[SeriesMeta]:
    """
    Convert all DICOM series under ``dicom_root`` to modality-organized NIfTI.

    Output layout::

        output_dir/
          <study_id>/
            t1/   series_*.nii.gz + series_*.json
            t1c/
            t2/
            flair/
            other/

    Each NIfTI is paired with a JSON sidecar containing PixelSpacing,
    SliceThickness, ImageOrientationPatient, and related series metadata.
    """
    dicom_root = Path(dicom_root)
    output_dir = Path(output_dir)
    groups = group_series_by_uid(dicom_root, force=force)

    if not groups:
        raise FileNotFoundError(f"No DICOM series found under {dicom_root}")

    # Prefer StudyInstanceUID / PatientID from the first series if study_id omitted.
    first_members = next(iter(groups.values()))
    first_ds = first_members[0][1]
    if study_id is None:
        study_id = (
            _safe_str(first_ds, "PatientID")
            or _safe_str(first_ds, "StudyID")
            or _safe_slug(_safe_str(first_ds, "StudyInstanceUID"), "study")
        )
    study_id = _safe_slug(study_id, "study")
    study_out = output_dir / study_id
    study_out.mkdir(parents=True, exist_ok=True)

    results: list[SeriesMeta] = []
    modality_counters: dict[str, int] = defaultdict(int)

    for uid, members in groups.items():
        meta = extract_series_meta(uid, members)
        if meta.num_instances < min_instances:
            meta.notes.append(f"Skipped: fewer than {min_instances} instances")
            logger.warning(
                "Skipping series %s (%s): only %d instance(s)",
                uid[:16],
                meta.series_description,
                meta.num_instances,
            )
            # Still write a sidecar under other/ for auditability.
            folder = study_out / meta.inferred_mri_contrast
            series_stem = (
                f"series_{_safe_slug(meta.series_number, 'na')}_"
                f"{_safe_slug(meta.series_description, uid[:8])}_skipped"
            )
            _write_sidecar(meta, folder / f"{series_stem}.json")
            results.append(meta)
            continue

        modality_counters[meta.inferred_mri_contrast] += 1
        idx = modality_counters[meta.inferred_mri_contrast]
        folder = study_out / meta.inferred_mri_contrast
        folder.mkdir(parents=True, exist_ok=True)

        series_stem = (
            f"{meta.inferred_mri_contrast}_{idx:02d}_"
            f"ser{_safe_slug(meta.series_number, 'na')}_"
            f"{_safe_slug(meta.series_description, uid[:8])}"
        )
        nifti_path = folder / f"{series_stem}.nii.gz"
        sidecar_path = folder / f"{series_stem}.json"

        try:
            out_path, used = convert_series_to_nifti(members, nifti_path, backend=backend)
            meta.conversion_backend = used
            meta.nifti_path = str(out_path)
            logger.info(
                "Converted %s → %s via %s",
                meta.series_description,
                out_path,
                used,
            )
        except Exception as exc:  # noqa: BLE001 — record failure in sidecar
            meta.notes.append(f"Conversion failed: {exc}")
            logger.exception("Failed to convert series %s", uid)
        finally:
            # Drop full file list from sidecar if huge; keep count via num_instances.
            if len(meta.dicom_files) > 50:
                meta.dicom_files = meta.dicom_files[:10] + ["… truncated …"]
            _write_sidecar(meta, sidecar_path)
            results.append(meta)

    summary_path = study_out / "series_index.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump([m.to_json_dict() for m in results], fh, indent=2)
    logger.info("Wrote study index %s (%d series)", summary_path, len(results))
    return results


# Friendly alias
load_dicom_directory = load_dicom_study
