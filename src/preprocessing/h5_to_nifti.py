"""Convert BraTS per-slice HDF5 files into cached per-patient NIfTI volumes.

Filename pattern (confirmed by scanning the dataset)::

    volume_{patient_id}_slice_{slice_index}.h5

Each file contains:
  - ``image``: (240, 240, 4) — FLAIR, T1, T1ce, T2
  - ``mask``:  (240, 240, 3) — tumor sub-region channels

Spacing: H5 datasets and accompanying CSVs (``meta_data.csv``,
``name_mapping.csv``, ``survival_info.csv``) carry **no** slice
thickness / pixel-spacing attributes. BraTS is nominally 1 mm isotropic;
because that is not recorded in this pack we write an **identity affine**
and record the decision in the conversion manifest.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import nibabel as nib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Confirmed on-disk naming (see module docstring).
_SLICE_RE = re.compile(r"^volume_(\d+)_slice_(\d+)\.h5$", re.IGNORECASE)

# Channel order used by the Kaggle BraTS2020 H5 pack (image[..., c]).
# Filenames use ``t1c`` to match the rest of this project's modality keys.
MODALITY_ORDER: tuple[str, ...] = ("flair", "t1", "t1c", "t2")

CACHE_MARKER = ".h5_to_nifti_complete.json"
DEFAULT_OUTPUT_NAME = "brats_nifti"


@dataclass
class PatientSliceGroup:
    patient_id: int
    slices: dict[int, Path] = field(default_factory=dict)

    @property
    def slice_indices(self) -> list[int]:
        return sorted(self.slices)

    @property
    def n_slices(self) -> int:
        return len(self.slices)


@dataclass
class ConvertReport:
    patient_key: str
    volume_id: int
    status: str  # success | skipped_cached | failed | skipped_bad_slices
    n_slices: int = 0
    expected_slices: int | None = None
    missing_slices: list[int] = field(default_factory=list)
    duplicate_slices: list[int] = field(default_factory=list)
    output_dir: str = ""
    error: str = ""
    affine_note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def inspect_h5_naming(h5_dir: str | Path, *, sample_size: int = 20) -> dict:
    """
    Inspect filenames and return the confirmed naming pattern summary.

    Call this (or rely on the log from :func:`scan_h5_directory`) before
    assuming a parse format.
    """
    h5_dir = Path(h5_dir)
    files = sorted(h5_dir.glob("volume_*_slice_*.h5"))
    if not files:
        files = sorted(h5_dir.glob("*.h5"))
    sample = files[:sample_size]
    matched = 0
    examples = []
    for path in sample:
        m = _SLICE_RE.match(path.name)
        examples.append({"name": path.name, "matched": bool(m), "groups": m.groups() if m else None})
        if m:
            matched += 1
    summary = {
        "directory": str(h5_dir),
        "n_h5_files": len(files),
        "sample_size": len(sample),
        "sample_matched": matched,
        "pattern": r"volume_{patient_id}_slice_{slice_index}.h5",
        "examples": examples,
    }
    logger.info(
        "H5 naming inspection: %d/%d sample files match %s (total .h5=%d)",
        matched,
        len(sample),
        summary["pattern"],
        len(files),
    )
    for ex in examples[:5]:
        logger.info("  example: %s → %s", ex["name"], ex["groups"])
    return summary


def probe_spacing_metadata(h5_dir: str | Path) -> dict:
    """
    Check H5 attributes and CSV sidecars for spacing / thickness.

    Returns a dict describing what was found. When nothing is present,
    ``use_identity_affine`` is True.
    """
    h5_dir = Path(h5_dir)
    result: dict = {
        "h5_file_attrs": {},
        "h5_dataset_attrs": {},
        "csv_columns": {},
        "spacing_mm": None,
        "slice_thickness_mm": None,
        "use_identity_affine": True,
        "note": "",
    }

    sample = next(iter(sorted(h5_dir.glob("volume_*_slice_*.h5"))), None)
    if sample is not None:
        with h5py.File(sample, "r") as handle:
            result["h5_file_attrs"] = {k: _jsonable(v) for k, v in handle.attrs.items()}
            ds_attrs = {}
            for key in handle.keys():
                ds_attrs[key] = {k: _jsonable(v) for k, v in handle[key].attrs.items()}
            result["h5_dataset_attrs"] = ds_attrs

    spacing_keys = re.compile(
        r"spacing|thickness|pixel|zooms|resolution|voxel",
        re.I,
    )
    for csv_name in ("meta_data.csv", "name_mapping.csv", "survival_info.csv"):
        csv_path = h5_dir / csv_name
        if not csv_path.is_file():
            continue
        cols = list(pd.read_csv(csv_path, nrows=0).columns)
        result["csv_columns"][csv_name] = cols
        hit = [c for c in cols if spacing_keys.search(str(c))]
        if hit:
            result["note"] += f" CSV {csv_name} has possible spacing cols {hit};"
            # No known numeric spacing in this pack; leave identity unless values exist.
            df = pd.read_csv(csv_path, nrows=5)
            for c in hit:
                if pd.api.types.is_numeric_dtype(df[c]):
                    result["spacing_mm"] = float(df[c].iloc[0])
                    result["use_identity_affine"] = False

    if result["use_identity_affine"]:
        result["note"] = (
            "No slice spacing / thickness found in H5 attrs or CSVs. "
            "BraTS is nominally 1 mm isotropic; using identity affine "
            "(voxel index == world mm if spacing were 1 mm)."
        )
    logger.info("Spacing probe: %s", result["note"])
    return result


def _jsonable(value):
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def scan_h5_directory(h5_dir: str | Path) -> dict[int, PatientSliceGroup]:
    """Group ``volume_*_slice_*.h5`` files by patient (volume) ID."""
    h5_dir = Path(h5_dir)
    if not h5_dir.is_dir():
        raise NotADirectoryError(f"BraTS H5 directory not found: {h5_dir}")

    inspect_h5_naming(h5_dir)
    groups: dict[int, PatientSliceGroup] = {}
    unmatched = 0

    for path in sorted(h5_dir.glob("*.h5")):
        m = _SLICE_RE.match(path.name)
        if not m:
            unmatched += 1
            logger.debug("Skipping unmatched H5 name: %s", path.name)
            continue
        volume_id = int(m.group(1))
        slice_idx = int(m.group(2))
        group = groups.get(volume_id)
        if group is None:
            group = PatientSliceGroup(patient_id=volume_id)
            groups[volume_id] = group
        if slice_idx in group.slices:
            logger.warning(
                "Duplicate slice file for volume %d slice %d: %s (keeping %s)",
                volume_id,
                slice_idx,
                path.name,
                group.slices[slice_idx].name,
            )
        else:
            group.slices[slice_idx] = path

    if unmatched:
        logger.warning("Unmatched .h5 filenames skipped: %d", unmatched)
    logger.info("Grouped %d patients from %s", len(groups), h5_dir)
    return groups


def validate_slice_indices(group: PatientSliceGroup) -> tuple[list[int], list[int], list[int]]:
    """
    Return ``(sorted_indices, missing, duplicates)``.

    ``duplicates`` is empty when using a dict keyed by slice index; retained
    for API completeness if callers pass multi-maps later.
    """
    indices = group.slice_indices
    if not indices:
        return [], [], []
    lo, hi = indices[0], indices[-1]
    expected = list(range(lo, hi + 1))
    present = set(indices)
    missing = [i for i in expected if i not in present]
    # Gaps relative to a dense 0..max range (BraTS H5 pack uses 0..154).
    if lo > 0:
        missing = list(range(0, lo)) + missing
    duplicates: list[int] = []
    return indices, missing, duplicates


def load_patient_volume(
    group: PatientSliceGroup,
    *,
    modality_order: Sequence[str] = MODALITY_ORDER,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[int]]:
    """
    Load and stack all slices for one patient.

    Returns
    -------
    modalities:
        Mapping modality name → array shaped ``(Z, H, W)`` float32.
    mask:
        Array shaped ``(Z, H, W)`` uint8 (merged sub-region labels 1/2/4).
    sorted_indices:
        Slice indices in the order stacked along Z.
    """
    indices, missing, _dup = validate_slice_indices(group)
    if missing:
        raise ValueError(
            f"volume_{group.patient_id}: missing slices {missing[:20]}"
            + ("…" if len(missing) > 20 else "")
        )
    if not indices:
        raise ValueError(f"volume_{group.patient_id}: no slices")

    # Detect out-of-order filenames vs numeric sort (we always sort numerically).
    raw_names = [group.slices[i].name for i in indices]
    if raw_names != sorted(raw_names, key=lambda n: int(_SLICE_RE.match(n).group(2))):  # type: ignore[union-attr]
        logger.warning(
            "volume_%d: slice files were not in lexical order; stacking by numeric slice index",
            group.patient_id,
        )

    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for idx in indices:
        path = group.slices[idx]
        with h5py.File(path, "r") as handle:
            if "image" not in handle or "mask" not in handle:
                raise KeyError(f"{path.name}: expected 'image' and 'mask' datasets")
            img = np.asarray(handle["image"])
            msk = np.asarray(handle["mask"])
        if img.shape[-1] < len(modality_order):
            raise ValueError(f"{path.name}: image shape {img.shape}, need C>={len(modality_order)}")
        if msk.ndim != 3 or msk.shape[-1] < 1:
            raise ValueError(f"{path.name}: unexpected mask shape {msk.shape}")
        images.append(img)
        masks.append(msk)

    vol = np.stack(images, axis=0)  # (Z, H, W, C)
    mask_vol = np.stack(masks, axis=0)  # (Z, H, W, Cmask)

    modalities = {
        name: np.ascontiguousarray(vol[:, :, :, c], dtype=np.float32)
        for c, name in enumerate(modality_order)
    }
    mask_3d = _merge_mask_channels(mask_vol)
    return modalities, mask_3d, indices


def _merge_mask_channels(mask_vol: np.ndarray) -> np.ndarray:
    """
    Merge (Z,H,W,3) sub-region channels into one BraTS-like label map.

    Channel 0 → label 1, channel 1 → label 2, channel 2 → label 4.
    Higher-index channels overwrite on overlap (enhancing preferred last).
    """
    z, h, w, c = mask_vol.shape
    out = np.zeros((z, h, w), dtype=np.uint8)
    labels = (1, 2, 4)
    for i in range(min(c, 3)):
        out[mask_vol[:, :, :, i] > 0] = labels[i]
    return out


def _affine_from_probe(probe: dict) -> tuple[np.ndarray, str]:
    if not probe.get("use_identity_affine", True) and probe.get("spacing_mm"):
        s = float(probe["spacing_mm"])
        aff = np.diag([s, s, s, 1.0]).astype(np.float64)
        return aff, f"diagonal spacing={s} mm from metadata"
    # Identity: confirmed no spacing metadata in this H5 pack.
    return np.eye(4, dtype=np.float64), probe.get("note", "identity affine")


def _load_subject_id_map(h5_dir: Path) -> dict[int, str]:
    """Map volume index → BraTS20 subject ID when CSVs allow a stable join."""
    name_csv = h5_dir / "name_mapping.csv"
    meta_csv = h5_dir / "meta_data.csv"
    if not name_csv.is_file():
        return {}
    nm = pd.read_csv(name_csv)
    col = "BraTS_2020_subject_ID" if "BraTS_2020_subject_ID" in nm.columns else nm.columns[-1]
    subjects = [str(x) for x in nm[col].tolist()]
    if meta_csv.is_file():
        meta = pd.read_csv(meta_csv, usecols=lambda c: c in {"volume", "slice"})
        volumes = sorted(meta["volume"].dropna().astype(int).unique().tolist())
        if len(volumes) == len(subjects):
            return dict(zip(volumes, subjects))
    return {}


def patient_output_dir(output_root: Path, volume_id: int, subject_id: str | None) -> Path:
    key = subject_id or f"volume_{volume_id:03d}"
    return output_root / key


def patient_is_cached(out_dir: Path, modality_order: Sequence[str] = MODALITY_ORDER) -> bool:
    if not out_dir.is_dir():
        return False
    needed = [f"{m}.nii.gz" for m in modality_order] + ["mask.nii.gz"]
    return all((out_dir / name).is_file() for name in needed)


def save_patient_niftis(
    out_dir: Path,
    modalities: dict[str, np.ndarray],
    mask: np.ndarray,
    affine: np.ndarray,
) -> list[Path]:
    """Write modality + mask NIfTIs. Arrays are (Z,H,W); nibabel uses that axis order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, arr in modalities.items():
        path = out_dir / f"{name}.nii.gz"
        # nibabel expects (X,Y,Z); our stack is (Z,H,W). Transpose to (H,W,Z)
        # so the third axis is slice — conventional for axial BraTS-like data.
        xyz = np.transpose(arr, (1, 2, 0))
        nib.save(nib.Nifti1Image(xyz, affine), str(path))
        written.append(path)
    mask_xyz = np.transpose(mask, (1, 2, 0))
    mask_path = out_dir / "mask.nii.gz"
    nib.save(nib.Nifti1Image(mask_xyz, affine), str(mask_path))
    written.append(mask_path)
    return written


def convert_patient(
    group: PatientSliceGroup,
    output_root: str | Path,
    *,
    affine: np.ndarray,
    affine_note: str,
    subject_id: str | None = None,
    force: bool = False,
    modality_order: Sequence[str] = MODALITY_ORDER,
) -> ConvertReport:
    """Convert one patient group to NIfTI; skip if cached unless ``force``."""
    output_root = Path(output_root)
    key = subject_id or f"volume_{group.patient_id:03d}"
    out_dir = patient_output_dir(output_root, group.patient_id, subject_id)

    indices, missing, duplicates = validate_slice_indices(group)
    if missing or duplicates:
        msg = (
            f"missing={missing[:30]}{'…' if len(missing) > 30 else ''}; "
            f"duplicates={duplicates}"
        )
        logger.error(
            "Patient %s (volume_%d): bad slice set — %s. Skipping (no malformed volume written).",
            key,
            group.patient_id,
            msg,
        )
        return ConvertReport(
            patient_key=key,
            volume_id=group.patient_id,
            status="skipped_bad_slices",
            n_slices=len(indices),
            expected_slices=(indices[-1] - indices[0] + 1) if indices else None,
            missing_slices=missing,
            duplicate_slices=duplicates,
            error=msg,
            affine_note=affine_note,
        )

    if patient_is_cached(out_dir, modality_order) and not force:
        logger.info("Patient %s: cached NIfTI found at %s — skip", key, out_dir)
        return ConvertReport(
            patient_key=key,
            volume_id=group.patient_id,
            status="skipped_cached",
            n_slices=len(indices),
            expected_slices=len(indices),
            output_dir=str(out_dir),
            affine_note=affine_note,
        )

    try:
        modalities, mask, sorted_idx = load_patient_volume(group, modality_order=modality_order)
        # Extra out-of-order guard: sorted_idx must be contiguous increasing.
        if sorted_idx != list(range(sorted_idx[0], sorted_idx[0] + len(sorted_idx))):
            raise ValueError(f"out-of-order or gapped indices after sort: {sorted_idx[:10]}…")
        save_patient_niftis(out_dir, modalities, mask, affine)
        meta = {
            "volume_id": group.patient_id,
            "subject_id": key,
            "n_slices": len(sorted_idx),
            "slice_range": [int(sorted_idx[0]), int(sorted_idx[-1])],
            "modalities": list(modality_order),
            "mask_labels": {"1": "channel0", "2": "channel1", "4": "channel2"},
            "affine": affine.tolist(),
            "affine_note": affine_note,
            "shape_zyx": list(modalities[modality_order[0]].shape),
        }
        with (out_dir / "conversion_meta.json").open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        logger.info(
            "Patient %s: wrote %d modalities + mask (%d slices) → %s",
            key,
            len(modalities),
            len(sorted_idx),
            out_dir,
        )
        return ConvertReport(
            patient_key=key,
            volume_id=group.patient_id,
            status="success",
            n_slices=len(sorted_idx),
            expected_slices=len(sorted_idx),
            output_dir=str(out_dir),
            affine_note=affine_note,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Patient %s failed: %s", key, exc)
        return ConvertReport(
            patient_key=key,
            volume_id=group.patient_id,
            status="failed",
            n_slices=len(indices),
            error=f"{type(exc).__name__}: {exc}",
            affine_note=affine_note,
        )


def convert_brats_h5_directory(
    h5_dir: str | Path,
    output_dir: str | Path,
    *,
    force: bool = False,
    limit: int | None = None,
    modality_order: Sequence[str] = MODALITY_ORDER,
) -> list[ConvertReport]:
    """
    Convert all BraTS H5 patients to NIfTI once and cache under ``output_dir``.

    Re-running is a no-op for patients that already have a full NIfTI set,
    unless ``force=True``. A root marker ``.h5_to_nifti_complete.json`` is
    written when the full directory conversion finishes successfully enough
    to use as a training cache (individual failures are listed inside).
    """
    h5_dir = Path(h5_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    marker = output_dir / CACHE_MARKER
    if marker.is_file() and not force:
        try:
            prev = json.loads(marker.read_text(encoding="utf-8"))
            partial = prev.get("limit") is not None
            incomplete = int(prev.get("n_success_or_cached", 0)) < int(
                prev.get("n_patients_in_h5_dir", prev.get("n_patients", 0))
            )
            if partial or incomplete:
                logger.warning(
                    "Ignoring incomplete cache marker at %s (limit=%s, ok=%s, expected=%s)",
                    marker,
                    prev.get("limit"),
                    prev.get("n_success_or_cached"),
                    prev.get("n_patients_in_h5_dir"),
                )
            else:
                logger.info(
                    "BraTS H5→NIfTI cache hit at %s (%s patients). "
                    "Not re-converting (pass force=True to rebuild).",
                    output_dir,
                    prev.get("n_success_or_cached"),
                )
                return [ConvertReport(**r) for r in prev.get("reports", [])] or [
                    ConvertReport(
                        patient_key="*",
                        volume_id=-1,
                        status="skipped_cached",
                        output_dir=str(output_dir),
                        affine_note=prev.get("affine_note", ""),
                    )
                ]
        except (json.JSONDecodeError, TypeError, KeyError, ValueError):
            logger.warning("Corrupt cache marker %s — rebuilding", marker)

    probe = probe_spacing_metadata(h5_dir)
    affine, affine_note = _affine_from_probe(probe)
    id_map = _load_subject_id_map(h5_dir)
    groups = scan_h5_directory(h5_dir)
    volume_ids = sorted(groups)
    if limit is not None:
        volume_ids = volume_ids[:limit]

    reports: list[ConvertReport] = []
    t0 = time.perf_counter()
    total = len(volume_ids)
    for i, vid in enumerate(volume_ids, start=1):
        subject = id_map.get(vid)
        logger.info("[%d/%d] Converting volume_%d → %s", i, total, vid, subject or f"volume_{vid:03d}")
        reports.append(
            convert_patient(
                groups[vid],
                output_dir,
                affine=affine,
                affine_note=affine_note,
                subject_id=subject,
                force=force,
                modality_order=modality_order,
            )
        )

    n_ok = sum(1 for r in reports if r.status in {"success", "skipped_cached"})
    n_bad = sum(1 for r in reports if r.status in {"failed", "skipped_bad_slices"})
    summary = {
        "h5_dir": str(h5_dir),
        "output_dir": str(output_dir),
        "n_patients": total,
        "n_patients_in_h5_dir": len(groups),
        "limit": limit,
        "n_success_or_cached": n_ok,
        "n_failed_or_bad_slices": n_bad,
        "affine_note": affine_note,
        "affine": affine.tolist(),
        "spacing_probe": probe,
        "modality_order": list(modality_order),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "reports": [r.to_dict() for r in reports],
    }
    report_path = output_dir / "h5_to_nifti_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Full-directory cache marker only when converting every patient (limit is None).
    if n_bad == 0 and total > 0 and limit is None and total == len(groups):
        marker.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("Wrote full-cache marker %s", marker)
    elif n_bad:
        logger.warning(
            "Conversion finished with %d bad patients — full cache marker NOT written. See %s",
            n_bad,
            report_path,
        )
    else:
        logger.info(
            "Partial conversion (limit=%s) — full cache marker NOT written. See %s",
            limit,
            report_path,
        )

    csv_path = output_dir / "h5_to_nifti_summary.csv"
    pd.DataFrame([r.to_dict() for r in reports]).to_csv(csv_path, index=False)
    logger.info(
        "H5→NIfTI done: %d ok/cached, %d failed/bad, %.1fs → %s",
        n_ok,
        n_bad,
        time.perf_counter() - t0,
        output_dir,
    )
    return reports


def resolve_brats_h5_dir(brats_root: str | Path) -> Path:
    """Locate the directory that contains ``volume_*_slice_*.h5``."""
    brats_root = Path(brats_root)
    candidates = [
        brats_root,
        brats_root / "BraTS2020_training_data" / "content" / "data",
        brats_root / "content" / "data",
        brats_root / "data",
    ]
    for cand in candidates:
        if cand.is_dir() and any(cand.glob("volume_*_slice_0.h5")):
            return cand
    hits = list(brats_root.rglob("volume_*_slice_0.h5"))
    if hits:
        return hits[0].parent
    raise FileNotFoundError(f"No BraTS H5 slices under {brats_root}")


def get_cached_nifti_root(
    processed_root: str | Path,
    *,
    subdir: str = DEFAULT_OUTPUT_NAME,
) -> Path:
    """Return the cache directory path (may not exist yet)."""
    return Path(processed_root) / subdir


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        from config import load_config

        cfg = load_config()
        default_h5 = str(cfg.paths.brats)
        default_out = str(get_cached_nifti_root(cfg.paths.processed))
    except Exception:  # noqa: BLE001
        default_h5 = "data/brats"
        default_out = "data/processed/brats_nifti"

    parser = argparse.ArgumentParser(description="Convert BraTS H5 slices to cached NIfTI")
    parser.add_argument("--h5-dir", default=default_h5)
    parser.add_argument("--output-dir", default=default_out)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    h5 = resolve_brats_h5_dir(args.h5_dir)
    convert_brats_h5_directory(h5, args.output_dir, force=args.force, limit=args.limit)
