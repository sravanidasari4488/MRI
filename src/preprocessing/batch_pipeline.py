"""Batch preprocessing over real-patient DICOM studies and BraTS cases."""

from __future__ import annotations

import argparse
import logging
import re
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd
import SimpleITK as sitk

from .registration import (
    CORE_MODALITIES,
    preprocess_nifti_modalities,
    preprocess_patient_study,
)

logger = logging.getLogger(__name__)

# Kaggle BraTS2020 H5 channel order used by the ``volume_*_slice_*.h5`` pack.
BRATS_H5_CHANNEL_ORDER: tuple[str, ...] = ("flair", "t1", "t1c", "t2")

_NIFTI_MODALITY_PATTERNS: dict[str, re.Pattern[str]] = {
    "t1c": re.compile(r"(t1ce|t1gd|t1c\b|t1[_-]?ce)", re.I),
    "flair": re.compile(r"flair", re.I),
    "t2": re.compile(r"(?<![a-z])t2(?![a-z])", re.I),
    "t1": re.compile(r"(?<![a-z])t1(?!ce|gd|c)", re.I),
}


@dataclass
class StudyBatchRecord:
    """One row in the batch summary CSV."""

    dataset: str
    study_id: str
    status: Literal["success", "failed", "skipped"]
    source_path: str
    output_path: str = ""
    has_t1: bool = False
    has_t1c: bool = False
    has_t2: bool = False
    has_flair: bool = False
    modalities_found: str = ""
    modalities_processed: str = ""
    error: str = ""
    elapsed_sec: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BatchSummary:
    dataset: str
    records: list[StudyBatchRecord] = field(default_factory=list)

    @property
    def n_success(self) -> int:
        return sum(1 for r in self.records if r.status == "success")

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.records if r.status == "failed")

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.records])


_SKIP_DIR_NAMES = {".git", ".gitkeep", "__macosx", "__pycache__", ".ipynb_checkpoints"}
_STUDY_NAME_RE = re.compile(r"_MR_Study\d*", re.I)


def _is_skipped_dir(path: Path) -> bool:
    return any(part.lower() in _SKIP_DIR_NAMES for part in path.parts)


def _has_dicom_payload(path: Path) -> bool:
    """True if ``path`` has ``.dcm`` files directly or one series-folder deep."""
    try:
        for child in path.iterdir():
            if child.is_file() and child.suffix.lower() == ".dcm":
                return True
            if child.is_dir() and not _is_skipped_dir(child):
                if any(child.glob("*.dcm")):
                    return True
    except OSError:
        return False
    return False


def _looks_like_study_folder(path: Path) -> bool:
    """Heuristic: export-style ``*_MR_Study*`` name, or folder containing DICOM series."""
    if _is_skipped_dir(path) or not path.is_dir():
        return False
    if _STUDY_NAME_RE.search(path.name):
        return _has_dicom_payload(path)
    return _has_dicom_payload(path)


def discover_real_patient_studies(
    root: str | Path,
    *,
    limit: int | None = 100,
) -> list[Path]:
    """
    Discover patient study folders under ``data/real_patients``.

    Handles nested exports such as::

        real_patients/MRI DATA/MRI DATA/<study_id>_MR_Study1/<series_uid>/*.dcm

    Prefers folders matching ``*_MR_Study*``; otherwise keeps directories that
    directly contain DICOM series. Skips ``__MACOSX`` and placeholder dirs.
    """
    import os

    root = Path(root)
    if not root.is_dir():
        logger.warning("Real-patient root does not exist: %s", root)
        return []

    named_studies: list[Path] = []
    for dirpath, dirnames, _filenames in os.walk(root):
        # Prune junk trees early.
        dirnames[:] = [d for d in dirnames if d.lower() not in _SKIP_DIR_NAMES and not d.startswith(".")]
        base = Path(dirpath)
        if _is_skipped_dir(base):
            dirnames[:] = []
            continue
        if _STUDY_NAME_RE.search(base.name):
            named_studies.append(base)
            # Study folders contain series UIDs only — do not walk deeper for names.
            dirnames[:] = []

    if named_studies:
        # Light DICOM check: keep studies that have at least one .dcm under them.
        studies = []
        for path in named_studies:
            if any(path.glob("*/*.dcm")) or any(path.glob("*.dcm")):
                studies.append(path)
        studies = sorted(set(studies), key=lambda p: p.name.lower())
    else:
        studies = sorted(
            [p for p in root.iterdir() if p.is_dir() and _looks_like_study_folder(p)],
            key=lambda p: p.name.lower(),
        )
        if not studies:
            nested: list[Path] = []
            for mid in root.iterdir():
                if not mid.is_dir() or _is_skipped_dir(mid):
                    continue
                for child in mid.iterdir():
                    if child.is_dir() and _looks_like_study_folder(child):
                        nested.append(child)
            studies = sorted(set(nested), key=lambda p: p.name.lower())
        if not studies and _has_dicom_payload(root):
            studies = [root]

    # Drop obvious duplicates: prefer folder without trailing " 2" when both exist.
    by_base: dict[str, Path] = {}
    for path in studies:
        base_name = re.sub(r"\s+\d+$", "", path.name).strip()
        prev = by_base.get(base_name)
        if prev is None or (path.name == base_name and prev.name != base_name):
            by_base[base_name] = path
    studies = sorted(by_base.values(), key=lambda p: p.name.lower())

    if limit is not None:
        studies = studies[:limit]
    logger.info(
        "Discovered %d real-patient studies under %s (nested DICOM-aware)",
        len(studies),
        root,
    )
    return studies


def _find_brats_h5_data_dir(brats_root: Path) -> Path | None:
    """Locate the directory containing ``volume_*_slice_*.h5`` files."""
    candidates = [
        brats_root,
        brats_root / "BraTS2020_training_data" / "content" / "data",
        brats_root / "content" / "data",
        brats_root / "data",
    ]
    for cand in candidates:
        if cand.is_dir() and any(cand.glob("volume_*_slice_*.h5")):
            return cand
    # Recursive fallback (first hit).
    hits = list(brats_root.rglob("volume_*_slice_0.h5"))
    if hits:
        return hits[0].parent
    return None


def _discover_brats_nifti_cases(brats_root: Path) -> list[tuple[str, Path]]:
    """Official-style BraTS folders: one case dir with modality NIfTIs."""
    cases: list[tuple[str, Path]] = []
    for path in sorted(brats_root.rglob("*")):
        if not path.is_dir():
            continue
        niftis = list(path.glob("*.nii*"))
        if len(niftis) < 2:
            continue
        names = " ".join(p.name.lower() for p in niftis)
        if "t1" in names and ("flair" in names or "t2" in names):
            cases.append((path.name, path))
    return cases


def discover_brats_studies(
    brats_root: str | Path,
    *,
    limit: int | None = None,
) -> list[dict]:
    """
    Discover BraTS cases.

    Supports:
    - Kaggle H5 pack (``volume_<id>_slice_*.h5``)
    - Official NIfTI case folders (``BraTS20_Training_XXX/*.nii.gz``)

    Returns a list of dicts with keys ``study_id``, ``kind`` (``h5``|``nifti``),
    and ``path`` / ``volume_id``.
    """
    brats_root = Path(brats_root)
    if not brats_root.is_dir():
        logger.warning("BraTS root does not exist: %s", brats_root)
        return []

    studies: list[dict] = []

    h5_dir = _find_brats_h5_data_dir(brats_root)
    if h5_dir is not None:
        volume_ids = sorted(
            {
                int(m.group(1))
                for p in h5_dir.glob("volume_*_slice_0.h5")
                for m in [re.match(r"volume_(\d+)_slice_0\.h5$", p.name)]
                if m
            }
        )
        # Map volume index → BraTS subject ID when name_mapping.csv exists.
        id_map = _load_brats_volume_id_map(h5_dir)
        for vid in volume_ids:
            study_id = id_map.get(vid, f"volume_{vid:03d}")
            studies.append(
                {
                    "study_id": study_id,
                    "kind": "h5",
                    "path": h5_dir,
                    "volume_id": vid,
                }
            )
        logger.info("Discovered %d BraTS H5 volumes under %s", len(studies), h5_dir)

    nifti_cases = _discover_brats_nifti_cases(brats_root)
    # Avoid duplicating if H5 already covers the same tree heavily.
    if nifti_cases and h5_dir is None:
        for study_id, path in nifti_cases:
            studies.append({"study_id": study_id, "kind": "nifti", "path": path})
        logger.info("Discovered %d BraTS NIfTI cases under %s", len(nifti_cases), brats_root)

    if limit is not None:
        studies = studies[:limit]
    return studies


def _load_brats_volume_id_map(h5_dir: Path) -> dict[int, str]:
    """
    Best-effort map from H5 volume index → BraTS20 subject ID.

    ``meta_data.csv`` has volume indices; ``name_mapping.csv`` has subject IDs.
    When a direct join is unavailable, fall back to sorted subject order.
    """
    mapping: dict[int, str] = {}
    name_csv = h5_dir / "name_mapping.csv"
    meta_csv = h5_dir / "meta_data.csv"
    subjects: list[str] = []
    if name_csv.is_file():
        nm = pd.read_csv(name_csv)
        col = "BraTS_2020_subject_ID" if "BraTS_2020_subject_ID" in nm.columns else nm.columns[-1]
        subjects = [str(x) for x in nm[col].tolist()]

    if meta_csv.is_file() and subjects:
        meta = pd.read_csv(meta_csv, usecols=lambda c: c in {"volume", "slice"})
        volumes = sorted(meta["volume"].dropna().astype(int).unique().tolist())
        # Pair by sorted order when counts match (common for this pack).
        if len(volumes) == len(subjects):
            mapping = dict(zip(volumes, subjects))
            return mapping

    if subjects:
        # volume files are often 0..N-1 or 1..N; map by rank among discovered ids.
        return mapping
    return mapping


def convert_brats_h5_volume_to_nifti(
    h5_dir: str | Path,
    volume_id: int,
    output_dir: str | Path,
    *,
    channel_order: Sequence[str] = BRATS_H5_CHANNEL_ORDER,
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, Path]:
    """
    Stack ``volume_<id>_slice_*.h5`` slices into per-modality NIfTI volumes.

    Each H5 ``image`` array is ``(H, W, 4)``. Slices are stacked along Z.
    Also writes ``seg.nii.gz`` when a ``mask`` dataset is present.
    """
    import h5py

    h5_dir = Path(h5_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slice_files = sorted(
        h5_dir.glob(f"volume_{volume_id}_slice_*.h5"),
        key=lambda p: int(re.search(r"slice_(\d+)", p.name).group(1)),  # type: ignore[union-attr]
    )
    if not slice_files:
        raise FileNotFoundError(f"No H5 slices for volume_{volume_id} in {h5_dir}")

    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for path in slice_files:
        with h5py.File(path, "r") as handle:
            images.append(np.asarray(handle["image"]))
            if "mask" in handle:
                masks.append(np.asarray(handle["mask"]))

    # (Z, H, W, C)
    vol = np.stack(images, axis=0)
    if vol.ndim != 4 or vol.shape[-1] < len(channel_order):
        raise ValueError(f"Unexpected H5 image shape {vol.shape} for volume {volume_id}")

    paths: dict[str, Path] = {}
    for idx, mod in enumerate(channel_order):
        # SimpleITK expects (z, y, x)
        arr = np.ascontiguousarray(vol[:, :, :, idx].astype(np.float32))
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing(spacing_mm)
        out = output_dir / f"{mod}.nii.gz"
        sitk.WriteImage(img, str(out))
        paths[mod] = out

    if masks:
        mstack = np.stack(masks, axis=0)
        # Prefer a single-label seg: use argmax+1 over channels, or first channel.
        if mstack.ndim == 4:
            seg = np.argmax(mstack, axis=-1).astype(np.uint8)
            # Keep background 0: if all-zero voxel, leave 0.
            any_pos = mstack.max(axis=-1) > 0
            seg = np.where(any_pos, seg + 1, 0).astype(np.uint8)
        else:
            seg = mstack.astype(np.uint8)
        seg_img = sitk.GetImageFromArray(np.ascontiguousarray(seg))
        seg_img.SetSpacing(spacing_mm)
        seg_path = output_dir / "seg.nii.gz"
        sitk.WriteImage(seg_img, str(seg_path))
        paths["seg"] = seg_path

    return paths


def resolve_nifti_modalities(case_dir: str | Path) -> dict[str, Path]:
    """Infer t1/t1c/t2/flair NIfTI paths inside an official BraTS case folder."""
    case_dir = Path(case_dir)
    files = list(case_dir.glob("*.nii*"))
    found: dict[str, Path] = {}
    # Prefer longer / more specific patterns first (t1c before t1).
    for mod in ("t1c", "flair", "t2", "t1"):
        pat = _NIFTI_MODALITY_PATTERNS[mod]
        matches = [p for p in files if pat.search(p.name) and "seg" not in p.name.lower()]
        if matches:
            found[mod] = sorted(matches, key=lambda p: len(p.name))[0]
    return found


def _modalities_flags(mods: Iterable[str]) -> dict[str, bool]:
    s = {m.lower() for m in mods}
    return {
        "has_t1": "t1" in s,
        "has_t1c": "t1c" in s or "t1ce" in s,
        "has_t2": "t2" in s,
        "has_flair": "flair" in s,
    }


def write_summary_csv(records: Sequence[StudyBatchRecord], csv_path: str | Path) -> Path:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.to_dict() for r in records])
    df.to_csv(csv_path, index=False)
    logger.info(
        "Wrote summary CSV %s (success=%d failed=%d total=%d)",
        csv_path,
        int((df["status"] == "success").sum()) if len(df) else 0,
        int((df["status"] == "failed").sum()) if len(df) else 0,
        len(df),
    )
    return csv_path


def run_batch_real_patients(
    dicom_root: str | Path,
    output_root: str | Path,
    *,
    limit: int | None = 100,
    atlas_nifti: str | Path | None = None,
    skip_bias_correction: bool = False,
    skip_skull_strip: bool = False,
    skull_strip_method: Literal["auto", "hd-bet", "simpleitk"] = "auto",
    summary_csv: str | Path | None = None,
) -> BatchSummary:
    """Run ``preprocess_patient_study`` on up to ``limit`` real patient folders."""
    dicom_root = Path(dicom_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    studies = discover_real_patient_studies(dicom_root, limit=limit)
    summary = BatchSummary(dataset="real_patients")
    total = len(studies)

    if total == 0:
        logger.warning("No real-patient studies found under %s", dicom_root)
        if summary_csv:
            write_summary_csv(summary.records, summary_csv)
        return summary

    for idx, study_path in enumerate(studies, start=1):
        study_id = study_path.name
        out_dir = output_root / study_id
        logger.info(
            "[real %d/%d] Starting study %s from %s",
            idx,
            total,
            study_id,
            study_path,
        )
        t0 = time.perf_counter()
        try:
            result = preprocess_patient_study(
                study_path,
                out_dir,
                study_id=study_id,
                atlas_nifti=atlas_nifti,
                skip_bias_correction=skip_bias_correction,
                skip_skull_strip=skip_skull_strip,
                skull_strip_method=skull_strip_method,
            )
            mods_found = sorted(result.selected_nifti)
            mods_proc = sorted(result.resampled_1mm)
            flags = _modalities_flags(mods_found)
            summary.records.append(
                StudyBatchRecord(
                    dataset="real_patients",
                    study_id=study_id,
                    status="success",
                    source_path=str(study_path),
                    output_path=str(out_dir),
                    modalities_found=",".join(mods_found),
                    modalities_processed=",".join(mods_proc),
                    elapsed_sec=round(time.perf_counter() - t0, 2),
                    **flags,
                )
            )
            logger.info(
                "[real %d/%d] SUCCESS %s | modalities=%s | %.1fs",
                idx,
                total,
                study_id,
                mods_found,
                time.perf_counter() - t0,
            )
        except Exception as exc:  # noqa: BLE001 — continue batch
            err = f"{type(exc).__name__}: {exc}"
            logger.error(
                "[real %d/%d] FAILED %s: %s",
                idx,
                total,
                study_id,
                err,
            )
            logger.debug("Traceback for %s:\n%s", study_id, traceback.format_exc())
            summary.records.append(
                StudyBatchRecord(
                    dataset="real_patients",
                    study_id=study_id,
                    status="failed",
                    source_path=str(study_path),
                    output_path=str(out_dir),
                    error=err,
                    elapsed_sec=round(time.perf_counter() - t0, 2),
                )
            )

    csv_path = Path(summary_csv) if summary_csv else output_root / "batch_summary_real_patients.csv"
    write_summary_csv(summary.records, csv_path)
    logger.info(
        "Real-patient batch done: %d success, %d failed / %d total",
        summary.n_success,
        summary.n_failed,
        total,
    )
    return summary


def run_batch_brats(
    brats_root: str | Path,
    output_root: str | Path,
    *,
    limit: int | None = None,
    atlas_nifti: str | Path | None = None,
    skip_bias_correction: bool = False,
    skip_skull_strip: bool = False,
    skull_strip_method: Literal["auto", "hd-bet", "simpleitk"] = "auto",
    skip_registration: bool = True,
    summary_csv: str | Path | None = None,
) -> BatchSummary:
    """
    Preprocess all BraTS cases under ``brats_root``.

    BraTS volumes are already co-registered in template space, so rigid
    registration is skipped by default (identity copy into registered /
    isotropic stages still runs via ``preprocess_nifti_modalities`` with T1
    as fixed). Set ``skip_registration=False`` only if you also pass an atlas.
    """
    brats_root = Path(brats_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    studies = discover_brats_studies(brats_root, limit=limit)
    summary = BatchSummary(dataset="brats")
    total = len(studies)

    if total == 0:
        logger.warning("No BraTS studies found under %s", brats_root)
        if summary_csv:
            write_summary_csv(summary.records, summary_csv)
        return summary

    # Prefer one-shot cached H5→NIfTI (do not rebuild per case / epoch).
    nifti_cache: Path | None = None
    if any(s.get("kind") == "h5" for s in studies):
        try:
            from .h5_to_nifti import (
                convert_brats_h5_directory,
                get_cached_nifti_root,
                resolve_brats_h5_dir,
            )
            from config import load_config

            cfg = load_config()
            nifti_cache = get_cached_nifti_root(cfg.paths.processed)
            h5_dir = resolve_brats_h5_dir(brats_root)
            convert_brats_h5_directory(h5_dir, nifti_cache, force=False, limit=limit)
        except Exception as exc:  # noqa: BLE001
            logger.warning("BraTS H5→NIfTI cache step failed (%s); falling back per-case", exc)
            nifti_cache = None

    for idx, spec in enumerate(studies, start=1):
        study_id = str(spec["study_id"])
        out_dir = output_root / study_id
        logger.info("[brats %d/%d] Starting %s (%s)", idx, total, study_id, spec["kind"])
        t0 = time.perf_counter()
        try:
            raw_dir = out_dir / "01_nifti"
            raw_dir.mkdir(parents=True, exist_ok=True)

            if spec["kind"] == "h5":
                cached_case = None
                if nifti_cache is not None:
                    candidate = nifti_cache / study_id
                    if candidate.is_dir():
                        cached_case = candidate
                if cached_case is not None:
                    modality_paths = {
                        mod: cached_case / f"{mod}.nii.gz"
                        for mod in CORE_MODALITIES
                        if (cached_case / f"{mod}.nii.gz").is_file()
                    }
                    # Stage copies into raw_dir for a consistent debug layout.
                    staged = {}
                    for mod, src in modality_paths.items():
                        dest = raw_dir / f"{mod}.nii.gz"
                        if not dest.exists():
                            sitk.WriteImage(sitk.ReadImage(str(src)), str(dest))
                        staged[mod] = dest
                    modality_paths = staged
                else:
                    modality_paths = convert_brats_h5_volume_to_nifti(
                        spec["path"],
                        int(spec["volume_id"]),
                        raw_dir,
                    )
                    modality_paths = {
                        k: v for k, v in modality_paths.items() if k in CORE_MODALITIES
                    }
            else:
                modality_paths = resolve_nifti_modalities(spec["path"])
                for mod, src in modality_paths.items():
                    dest = raw_dir / f"{mod}.nii.gz"
                    if src.resolve() != dest.resolve():
                        sitk.WriteImage(sitk.ReadImage(str(src)), str(dest))
                    modality_paths[mod] = dest

            mods_found = sorted(modality_paths)
            if not mods_found:
                raise FileNotFoundError(f"No modalities resolved for {study_id}")

            # BraTS is already aligned; still run N4 / strip / 1mm resample.
            # Registration to T1 is effectively identity when volumes share space.
            _ = skip_registration  # reserved flag for future atlas-only path
            result = preprocess_nifti_modalities(
                modality_paths,
                out_dir,
                study_id=study_id,
                atlas_nifti=atlas_nifti,
                skip_bias_correction=skip_bias_correction,
                skip_skull_strip=skip_skull_strip,
                skull_strip_method=skull_strip_method,
                stage_raw_dir=raw_dir,
            )
            mods_proc = sorted(result.resampled_1mm)
            flags = _modalities_flags(mods_found)
            summary.records.append(
                StudyBatchRecord(
                    dataset="brats",
                    study_id=study_id,
                    status="success",
                    source_path=str(spec["path"]),
                    output_path=str(out_dir),
                    modalities_found=",".join(mods_found),
                    modalities_processed=",".join(mods_proc),
                    elapsed_sec=round(time.perf_counter() - t0, 2),
                    **flags,
                )
            )
            logger.info(
                "[brats %d/%d] SUCCESS %s | modalities=%s | %.1fs",
                idx,
                total,
                study_id,
                mods_found,
                time.perf_counter() - t0,
            )
        except Exception as exc:  # noqa: BLE001 — continue batch
            err = f"{type(exc).__name__}: {exc}"
            logger.error("[brats %d/%d] FAILED %s: %s", idx, total, study_id, err)
            logger.debug("Traceback for %s:\n%s", study_id, traceback.format_exc())
            summary.records.append(
                StudyBatchRecord(
                    dataset="brats",
                    study_id=study_id,
                    status="failed",
                    source_path=str(spec.get("path", "")),
                    output_path=str(out_dir),
                    error=err,
                    elapsed_sec=round(time.perf_counter() - t0, 2),
                )
            )

    csv_path = Path(summary_csv) if summary_csv else output_root / "batch_summary_brats.csv"
    write_summary_csv(summary.records, csv_path)
    logger.info(
        "BraTS batch done: %d success, %d failed / %d total",
        summary.n_success,
        summary.n_failed,
        total,
    )
    return summary


def run_all_batches(
    *,
    real_dicom_root: str | Path,
    brats_root: str | Path,
    processed_root: str | Path,
    real_limit: int | None = 100,
    brats_limit: int | None = None,
    **kwargs,
) -> tuple[BatchSummary, BatchSummary]:
    """Run real-patient (≤100) and BraTS batches; write per-dataset + combined CSVs."""
    processed_root = Path(processed_root)
    real_out = processed_root / "real_patients"
    brats_out = processed_root / "brats"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    real_summary = run_batch_real_patients(
        real_dicom_root,
        real_out,
        limit=real_limit,
        summary_csv=real_out / f"batch_summary_real_patients_{stamp}.csv",
        **kwargs,
    )
    brats_summary = run_batch_brats(
        brats_root,
        brats_out,
        limit=brats_limit,
        summary_csv=brats_out / f"batch_summary_brats_{stamp}.csv",
        **kwargs,
    )

    combined = list(real_summary.records) + list(brats_summary.records)
    write_summary_csv(combined, processed_root / f"batch_summary_all_{stamp}.csv")
    return real_summary, brats_summary


def _build_arg_parser() -> argparse.ArgumentParser:
    try:
        from config import load_config

        cfg = load_config()
        default_real = str(cfg.paths.raw_dicom)
        default_brats = str(cfg.paths.brats)
        default_processed = str(cfg.paths.processed)
    except Exception:  # noqa: BLE001
        default_real = "data/real_patients"
        default_brats = "data/brats"
        default_processed = "data/processed"

    p = argparse.ArgumentParser(description="Batch MRI preprocessing (real patients + BraTS)")
    p.add_argument("--dataset", choices=("real", "brats", "all"), default="all")
    p.add_argument("--real-root", default=default_real)
    p.add_argument("--brats-root", default=default_brats)
    p.add_argument("--output-root", default=default_processed)
    p.add_argument("--real-limit", type=int, default=100)
    p.add_argument("--brats-limit", type=int, default=None)
    p.add_argument("--skip-bias-correction", action="store_true")
    p.add_argument("--skip-skull-strip", action="store_true")
    p.add_argument(
        "--skull-strip-method",
        choices=("auto", "hd-bet", "simpleitk"),
        default="auto",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    common = dict(
        skip_bias_correction=args.skip_bias_correction,
        skip_skull_strip=args.skip_skull_strip,
        skull_strip_method=args.skull_strip_method,
    )
    if args.dataset == "real":
        run_batch_real_patients(
            args.real_root,
            Path(args.output_root) / "real_patients",
            limit=args.real_limit,
            **common,
        )
    elif args.dataset == "brats":
        run_batch_brats(
            args.brats_root,
            Path(args.output_root) / "brats",
            limit=args.brats_limit,
            **common,
        )
    else:
        run_all_batches(
            real_dicom_root=args.real_root,
            brats_root=args.brats_root,
            processed_root=args.output_root,
            real_limit=args.real_limit,
            brats_limit=args.brats_limit,
            **common,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
