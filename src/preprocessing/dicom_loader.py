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

# Ordered rules: first match wins on SeriesDescription + ProtocolName.
# FLAIR must come before generic T2 — vendor names often include "T2W_FLAIR".
# T1 matches "T1W_*" / "eT1W_*" via ``t1w`` (``\bt1\b`` fails there).
_FLAIR_RE = re.compile(
    r"flair|fluid[\s_-]*attenuated",
    re.I,
)
_T1C_NAME_RE = re.compile(
    r"t1(?:w)?[\s_-]*(?:c|ce|gd|gado)|"
    r"t1(?:w)?.{0,32}(?:post|\+c|c\+|contrast|gadolinium|\bgd\b)|"
    r"(?:post|\+c|c\+|contrast|gadolinium|\bgd\b).{0,32}t1(?:w)?|"
    r"post[\s_-]*contrast",
    re.I,
)
_T1_RE = re.compile(
    r"t1w(?:[\s_-]?ir)?|"  # T1W_SE, eT1W_SE, T1W_IR, t1w_ir
    r"t1[\s_-]ir|"  # t1_ir
    r"(?<![a-z0-9])t1(?![a-z0-9])|"  # bare t1 (not t1c / t1w — those handled above)
    r"mprage|spgr|bravo|fspgr|mp[\s_-]?rage",
    re.I,
)
_T2_RE = re.compile(
    r"t2w|"  # T2W_TSE, eT2W_TSE (FLAIR already handled above)
    r"(?<![a-z0-9])t2(?![a-z0-9w])|"
    r"(?<![a-z0-9])(?:space|cube)(?![a-z0-9])",
    re.I,
)

# Standalone contrast-agent name hints (with ContrastBolusAgent → promote T1 → T1c).
_CONTRAST_HINT_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:post|\+c|c\+|gd|gado|gadolinium|contrast)(?:[^a-z0-9]|$)",
    re.I,
)


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
    contrast_bolus_agent: str | None = None
    contrast_evidence: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def series_text(*parts: str | None) -> str:
    """Join non-empty description / protocol strings for modality matching."""
    return " ".join(x for x in parts if x).strip()


def contrast_bolus_agent_present(datasets: Iterable[Dataset]) -> tuple[bool, str | None]:
    """
    Return whether any instance has a non-empty ``ContrastBolusAgent`` (0018,0010).

    Also accepts common related tags when the primary agent string is empty.
    """
    related_keywords = (
        "ContrastBolusAgent",
        "ContrastBolusAgentSequence",
        "ContrastAgent",
        "ContrastBolusVolume",
        "ContrastBolusTotalDose",
    )
    for ds in datasets:
        for kw in related_keywords:
            if not hasattr(ds, kw):
                continue
            value = getattr(ds, kw, None)
            if value is None:
                continue
            # Sequence / multi-value → treat presence as positive if any element exists.
            if kw.endswith("Sequence"):
                try:
                    if len(value) > 0:
                        return True, f"{kw}(len={len(value)})"
                except TypeError:
                    return True, kw
                continue
            text = str(value).strip()
            if text and text.lower() not in {"none", "n/a", "na", "0", "0.0"}:
                return True, text
    return False, None


def name_suggests_contrast(text: str) -> bool:
    """True if series/protocol naming hints at post-contrast imaging."""
    return bool(text and _CONTRAST_HINT_RE.search(text))


def classify_modality(
    series_description: str | None,
    protocol_name: str | None,
    *,
    contrast_bolus_agent: str | None = None,
    has_contrast_bolus: bool | None = None,
) -> ModalityFolder:
    """
    Map SeriesDescription / ProtocolName (+ optional contrast tags) to a folder.

    Order: **FLAIR → T1c → T1 → T2** so that names like ``eT2W_FLAIR`` become
    ``flair`` rather than ``t2``. T1c uses name hints and/or DICOM contrast
    bolus evidence to promote a T1-like series.
    """
    text = series_text(series_description, protocol_name)
    if not text and not (has_contrast_bolus or contrast_bolus_agent):
        return "other"

    # Explicit FLAIR before any T2 rule (substring — underscores are word chars).
    if text and _FLAIR_RE.search(text):
        return "flair"

    bolus = bool(has_contrast_bolus) or bool(
        contrast_bolus_agent and str(contrast_bolus_agent).strip()
    )
    name_contrast = name_suggests_contrast(text) if text else False
    name_t1c = bool(text and _T1C_NAME_RE.search(text))
    name_t1 = bool(text and _T1_RE.search(text))

    # Post-contrast T1: name-based t1c, or T1-like + (bolus tag or contrast hint).
    if name_t1c or (name_t1 and (bolus or name_contrast)):
        return "t1c"
    if name_t1:
        return "t1"
    if text and _T2_RE.search(text):
        return "t2"
    return "other"


def refine_t1c_across_study(series_list: list[SeriesMeta]) -> list[SeriesMeta]:
    """
    Study-level T1c resolution.

    If multiple T1-family series exist and none were classified as ``t1c``
    (no ContrastBolusAgent / name hint), log a warning rather than silently
    promoting one to T1c.
    """
    t1_family = [
        s
        for s in series_list
        if s.inferred_mri_contrast in {"t1", "t1c"}
        or (s.series_description and _T1_RE.search(s.series_description))
        or (s.protocol_name and _T1_RE.search(s.protocol_name))
    ]
    t1c = [s for s in series_list if s.inferred_mri_contrast == "t1c"]
    t1_only = [s for s in series_list if s.inferred_mri_contrast == "t1"]

    if len(t1_family) >= 2 and not t1c and t1_only:
        descs = [s.series_description or s.series_instance_uid[:12] for s in t1_only]
        logger.warning(
            "T1c could not be determined: %d T1 series found (%s) but none have "
            "ContrastBolusAgent metadata or post-contrast name hints "
            "(post/+c/gd/contrast). Leaving them as t1 -- do not silently pick one.",
            len(t1_only),
            ", ".join(repr(d) for d in descs),
        )
        for s in t1_only:
            s.notes.append(
                "T1c undetermined: no ContrastBolusAgent / contrast name hint "
                "among multiple T1 series"
            )
    return series_list


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
    datasets = [m[1] for m in members_sorted]
    has_bolus, bolus_value = contrast_bolus_agent_present(datasets)
    evidence: list[str] = []
    if has_bolus and bolus_value:
        evidence.append(f"ContrastBolusAgent={bolus_value!r}")
    text = series_text(desc, protocol)
    if name_suggests_contrast(text):
        evidence.append("name_hint")

    contrast = classify_modality(
        desc,
        protocol,
        contrast_bolus_agent=bolus_value,
        has_contrast_bolus=has_bolus,
    )

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
        contrast_bolus_agent=bolus_value if has_bolus else None,
        contrast_evidence=evidence,
    )

    if meta.pixel_spacing is None:
        meta.notes.append("PixelSpacing missing")
    if meta.slice_thickness is None:
        meta.notes.append("SliceThickness missing")
    if meta.image_orientation_patient is None:
        meta.notes.append("ImageOrientationPatient missing")

    logger.info(
        "Series %s | contrast=%s | desc=%r | protocol=%r | n=%d | "
        "bolus=%s | PixelSpacing=%s SliceThickness=%s IOP=%s",
        uid[:16] + "…",
        contrast,
        desc,
        protocol,
        meta.num_instances,
        meta.contrast_bolus_agent,
        meta.pixel_spacing,
        meta.slice_thickness,
        meta.image_orientation_patient,
    )
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

    # Extract metadata for every series first so study-level T1c refinement
    # can run before NIfTI paths are assigned to modality folders.
    pending: list[tuple[str, list[tuple[Path, Dataset]], SeriesMeta]] = []
    for uid, members in groups.items():
        pending.append((uid, members, extract_series_meta(uid, members)))
    refine_t1c_across_study([meta for _, _, meta in pending])

    for uid, members, meta in pending:
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
