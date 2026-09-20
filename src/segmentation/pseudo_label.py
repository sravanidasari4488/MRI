"""Generate BraTS-pretrained pseudo-labels for real patient studies.

Saves NIfTI label maps next to the preprocessed scans so they can be opened
in ITK-SNAP or 3D Slicer for manual correction.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)

from .dataset import IMAGE_KEY, MODALITIES
from .evaluate import load_model_from_checkpoint, predict_regions
from .model import OUT_REGION_NAMES
from .train import list_preprocessed_cases

logger = logging.getLogger(__name__)

# Preferred scan to open under the label in ITK-SNAP / Slicer.
REFERENCE_MODALITY = "t1c"


@dataclass
class PseudoLabelResult:
    study_id: str
    status: str  # success | skipped | failed
    source_dir: str = ""
    output_dir: str = ""
    seg_path: str = ""
    ref_path: str = ""
    modalities_used: list[str] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def get_infer_transforms(
    *,
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Compose:
    """Deterministic image-only transform chain (no crop / no label)."""
    return Compose(
        [
            LoadImaged(keys=[IMAGE_KEY], image_only=False),
            EnsureChannelFirstd(keys=[IMAGE_KEY]),
            Orientationd(keys=[IMAGE_KEY], axcodes="RAS"),
            Spacingd(keys=[IMAGE_KEY], pixdim=pixdim, mode="bilinear"),
            NormalizeIntensityd(keys=[IMAGE_KEY], nonzero=True, channel_wise=True),
            EnsureTyped(keys=[IMAGE_KEY]),
        ]
    )


def _find_modality_nifti(case_dir: Path, modality: str) -> Path | None:
    """Locate a modality NIfTI under common preprocess stage folders."""
    modality = modality.lower()
    candidates = [
        case_dir / "05_isotropic_1mm" / f"{modality}_1mm.nii.gz",
        case_dir / "05_isotropic_1mm" / f"{modality}.nii.gz",
        case_dir / "04_registered" / f"{modality}_to_t1.nii.gz",
        case_dir / "04_registered" / f"{modality}_in_ref.nii.gz",
        case_dir / "03_skull_stripped" / f"{modality}_brain.nii.gz",
        case_dir / "02_bias_corrected" / f"{modality}_n4.nii.gz",
        case_dir / "01_nifti" / f"{modality}.nii.gz",
        case_dir / f"{modality}.nii.gz",
        case_dir / f"{modality}_1mm.nii.gz",
    ]
    # Also accept BraTS-style t1ce alias.
    if modality == "t1c":
        candidates[0:0] = [
            case_dir / "05_isotropic_1mm" / "t1ce_1mm.nii.gz",
            case_dir / "05_isotropic_1mm" / "t1ce.nii.gz",
            case_dir / "01_nifti" / "t1ce.nii.gz",
            case_dir / "t1ce.nii.gz",
        ]
    for path in candidates:
        if path.is_file():
            return path
    # Fuzzy search inside case_dir.
    hits = sorted(case_dir.rglob(f"*{modality}*.nii*"))
    hits = [p for p in hits if "pseudo" not in p.name.lower() and "seg" not in p.name.lower()]
    return hits[0] if hits else None


def discover_real_study_modalities(
    case_dir: str | Path,
    *,
    modalities: Sequence[str] = MODALITIES,
) -> dict[str, Path] | None:
    """
    Return modality→path for a preprocessed real-patient study, or None if incomplete.
    """
    case_dir = Path(case_dir)
    found: dict[str, Path] = {}
    for mod in modalities:
        path = _find_modality_nifti(case_dir, mod)
        if path is not None:
            found[mod] = path
    # Need all 4 channels for the BraTS SegResNet.
    if len(found) < len(modalities):
        missing = [m for m in modalities if m not in found]
        logger.debug("Study %s missing modalities: %s", case_dir.name, missing)
        return None
    return found


def regions_to_brats_labelmap(probs: torch.Tensor, *, threshold: float = 0.5) -> np.ndarray:
    """
    Convert ET/TC/WT probabilities ``(3, H, W, D)`` to a BraTS-style label map.

    Labels (ITK-SNAP / Slicer friendly integers)::

        0 background
        1 necrotic / non-enhancing core  (TC \\ ET)
        2 edema                          (WT \\ TC)
        4 enhancing tumor                (ET)
    """
    if probs.ndim == 4 and probs.shape[0] == 3:
        p = probs
    elif probs.ndim == 5 and probs.shape[0] == 1 and probs.shape[1] == 3:
        p = probs[0]
    else:
        raise ValueError(f"Expected (3,H,W,D) or (1,3,H,W,D) probs, got {tuple(probs.shape)}")

    et = (p[0] > threshold).cpu().numpy()
    tc = (p[1] > threshold).cpu().numpy()
    wt = (p[2] > threshold).cpu().numpy()

    label = np.zeros(wt.shape, dtype=np.uint8)
    label[wt] = 2
    label[tc] = 1
    label[et] = 4
    return label


def _meta_affine(image_tensor: torch.Tensor) -> np.ndarray:
    aff = getattr(image_tensor, "affine", None)
    if aff is not None:
        return np.asarray(aff.detach().cpu() if torch.is_tensor(aff) else aff, dtype=np.float64)
    meta = getattr(image_tensor, "meta", None)
    if isinstance(meta, dict) and "affine" in meta:
        return np.asarray(meta["affine"], dtype=np.float64)
    return np.eye(4, dtype=np.float64)


def save_nifti(data: np.ndarray, affine: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


def pseudo_label_study(
    case_dir: str | Path,
    checkpoint: str | Path,
    *,
    output_dir: str | Path | None = None,
    device: torch.device | None = None,
    model: torch.nn.Module | None = None,
    roi_size: tuple[int, int, int] = (96, 96, 96),
    threshold: float = 0.5,
    use_amp: bool = False,
    overwrite: bool = False,
) -> PseudoLabelResult:
    """
    Run BraTS checkpoint inference on one preprocessed real-patient study.

    Writes into ``output_dir`` (default: ``<case_dir>/pseudo_labels/``)::

        pseudo_seg.nii.gz       BraTS multi-class labels (1 / 2 / 4)
        pseudo_et.nii.gz        binary enhancing
        pseudo_tc.nii.gz        binary tumor core
        pseudo_wt.nii.gz        binary whole tumor
        pseudo_ref_t1c.nii.gz   reference MRI in the same grid (for overlay)
        pseudo_label_meta.json  run metadata
    """
    case_dir = Path(case_dir)
    study_id = case_dir.name
    out = Path(output_dir) if output_dir is not None else case_dir / "pseudo_labels"
    seg_path = out / "pseudo_seg.nii.gz"

    modalities = discover_real_study_modalities(case_dir)
    if modalities is None:
        return PseudoLabelResult(
            study_id=study_id,
            status="skipped",
            source_dir=str(case_dir),
            error="Missing one or more of flair/t1/t1c/t2 NIfTIs",
        )

    if seg_path.is_file() and not overwrite:
        logger.info("Skip %s — existing %s (pass overwrite=True to redo)", study_id, seg_path)
        return PseudoLabelResult(
            study_id=study_id,
            status="skipped",
            source_dir=str(case_dir),
            output_dir=str(out),
            seg_path=str(seg_path),
            modalities_used=sorted(modalities),
            error="already exists",
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model, _ = load_model_from_checkpoint(checkpoint, device=device)

    # Stack in dataset channel order: flair, t1, t1c, t2
    data = {
        IMAGE_KEY: [str(modalities[m]) for m in MODALITIES],
        "case_id": study_id,
    }
    batch = get_infer_transforms()(data)
    image = batch[IMAGE_KEY]
    if not torch.is_tensor(image):
        image = torch.as_tensor(image)
    affine = _meta_affine(image)

    with torch.no_grad():
        probs = predict_regions(
            model,
            image.unsqueeze(0).to(device),
            roi_size=roi_size,
            use_amp=use_amp and device.type == "cuda",
            device=device,
        )[0]  # (3, H, W, D)

    label = regions_to_brats_labelmap(probs, threshold=threshold)
    et = (probs[0] > threshold).cpu().numpy().astype(np.uint8)
    tc = (probs[1] > threshold).cpu().numpy().astype(np.uint8)
    wt = (probs[2] > threshold).cpu().numpy().astype(np.uint8)

    out.mkdir(parents=True, exist_ok=True)
    save_nifti(label, affine, seg_path)
    save_nifti(et, affine, out / "pseudo_et.nii.gz")
    save_nifti(tc, affine, out / "pseudo_tc.nii.gz")
    save_nifti(wt, affine, out / "pseudo_wt.nii.gz")

    # Reference MRI in the *same* grid as the label (channel-first → drop channel).
    ref_idx = list(MODALITIES).index(REFERENCE_MODALITY) if REFERENCE_MODALITY in MODALITIES else 0
    ref_vol = image[ref_idx].detach().cpu().numpy().astype(np.float32)
    ref_path = out / f"pseudo_ref_{REFERENCE_MODALITY}.nii.gz"
    save_nifti(ref_vol, affine, ref_path)

    # Convenience copies next to isotropic scans when that folder exists.
    iso_dir = case_dir / "05_isotropic_1mm"
    if iso_dir.is_dir():
        for src, name in (
            (seg_path, "pseudo_seg.nii.gz"),
            (ref_path, f"pseudo_ref_{REFERENCE_MODALITY}.nii.gz"),
        ):
            dest = iso_dir / name
            shutil.copy2(src, dest)

    meta = {
        "study_id": study_id,
        "checkpoint": str(checkpoint),
        "modalities": {k: str(v) for k, v in modalities.items()},
        "regions": list(OUT_REGION_NAMES),
        "label_legend": {"0": "background", "1": "NCR/NET", "2": "ED", "4": "ET"},
        "threshold": threshold,
        "roi_size": list(roi_size),
        "affine": affine.tolist(),
        "shape": list(label.shape),
        "itk_snap_hint": (
            f"Open {ref_path.name} as main image and {seg_path.name} as segmentation "
            "(or load both from 05_isotropic_1mm/)."
        ),
    }
    (out / "pseudo_label_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    logger.info(
        "Pseudo-labeled %s → %s (voxels ET=%d TC=%d WT=%d)",
        study_id,
        seg_path,
        int(et.sum()),
        int(tc.sum()),
        int(wt.sum()),
    )
    return PseudoLabelResult(
        study_id=study_id,
        status="success",
        source_dir=str(case_dir),
        output_dir=str(out),
        seg_path=str(seg_path),
        ref_path=str(ref_path),
        modalities_used=list(MODALITIES),
    )


def run_pseudo_label_all(
    real_processed_root: str | Path,
    checkpoint: str | Path,
    *,
    device: str | None = None,
    roi_size: tuple[int, int, int] = (96, 96, 96),
    threshold: float = 0.5,
    overwrite: bool = False,
    limit: int | None = None,
    summary_csv: str | Path | None = None,
) -> pd.DataFrame:
    """
    Pseudo-label every preprocessed real-patient study under ``real_processed_root``.
    """
    root = Path(real_processed_root)
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    cases = list_preprocessed_cases(root)
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        logger.warning("No preprocessed real-patient studies under %s", root)
        df = pd.DataFrame(columns=list(PseudoLabelResult(study_id="", status="").to_dict()))
        if summary_csv:
            Path(summary_csv).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(summary_csv, index=False)
        return df

    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device
    device_t = torch.device(device_str)
    use_amp = device_t.type == "cuda"
    model, _ = load_model_from_checkpoint(checkpoint, device=device_t)

    results: list[PseudoLabelResult] = []
    logger.info(
        "Pseudo-labeling %d real studies from %s with %s on %s",
        len(cases),
        root,
        checkpoint,
        device_str,
    )
    for i, case_dir in enumerate(cases, start=1):
        logger.info("[%d/%d] %s", i, len(cases), case_dir.name)
        try:
            results.append(
                pseudo_label_study(
                    case_dir,
                    checkpoint,
                    device=device_t,
                    model=model,
                    roi_size=roi_size,
                    threshold=threshold,
                    use_amp=use_amp,
                    overwrite=overwrite,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed %s", case_dir.name)
            results.append(
                PseudoLabelResult(
                    study_id=case_dir.name,
                    status="failed",
                    source_dir=str(case_dir),
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    df = pd.DataFrame([r.to_dict() for r in results])
    out_csv = (
        Path(summary_csv)
        if summary_csv is not None
        else root / "pseudo_label_summary.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    n_ok = int((df["status"] == "success").sum()) if len(df) else 0
    n_skip = int((df["status"] == "skipped").sum()) if len(df) else 0
    n_fail = int((df["status"] == "failed").sum()) if len(df) else 0
    logger.info(
        "Pseudo-label done: success=%d skipped=%d failed=%d → %s",
        n_ok,
        n_skip,
        n_fail,
        out_csv,
    )
    return df


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        from config import load_config

        cfg = load_config()
        default_root = str(cfg.paths.processed_real_patients)
        default_ckpt = str(cfg.paths.checkpoints / "brats_pretrain" / "best_model.pt")
    except Exception:  # noqa: BLE001
        default_root = "data/processed/real_patients"
        default_ckpt = "models/checkpoints/brats_pretrain/best_model.pt"

    smoke = Path("models/checkpoints/brats_pretrain_smoke/best_model.pt")
    if not Path(default_ckpt).is_file() and smoke.is_file():
        default_ckpt = str(smoke)

    p = argparse.ArgumentParser(
        description="Pseudo-label real patient MRI with a BraTS-pretrained model"
    )
    p.add_argument("--real-root", default=default_root)
    p.add_argument("--checkpoint", default=default_ckpt)
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--summary-csv", default=None)
    args = p.parse_args(argv)

    run_pseudo_label_all(
        args.real_root,
        args.checkpoint,
        device=args.device,
        limit=args.limit,
        overwrite=args.overwrite,
        threshold=args.threshold,
        summary_csv=args.summary_csv,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
