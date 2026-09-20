"""Sanity-check Dice (numpy vs MONAI) and overlay plots for BraTS cases."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.data import MetaTensor
from monai.metrics import DiceMetric
from monai.transforms import Compose

from .dataset import IMAGE_KEY, LABEL_KEY, build_brats_data_dicts, get_val_transforms
from .evaluate import load_model_from_checkpoint, predict_regions
from .model import OUT_REGION_NAMES
from .train import exclusive_labels_to_regions

logger = logging.getLogger(__name__)

RegionName = Literal["et", "tc", "wt"]
REGION_INDEX = {name: i for i, name in enumerate(OUT_REGION_NAMES)}


def numpy_dice(pred: np.ndarray, gt: np.ndarray, *, eps: float = 1e-8) -> float:
    """Manual Dice: ``2 * |P ∩ G| / (|P| + |G|)`` on binary arrays."""
    p = (pred > 0).astype(np.float64).ravel()
    g = (gt > 0).astype(np.float64).ravel()
    inter = float(np.sum(p * g))
    denom = float(np.sum(p) + np.sum(g))
    if denom <= 0.0:
        return 1.0 if inter == 0.0 else 0.0
    return float((2.0 * inter + eps) / (denom + eps))


def monai_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """MONAI ``DiceMetric`` on a single binary volume pair."""
    # DiceMetric expects (B, C, H, W, D)
    p = torch.from_numpy((pred > 0).astype(np.float32))[None, None]
    g = torch.from_numpy((gt > 0).astype(np.float32))[None, None]
    metric = DiceMetric(include_background=True, reduction="mean")
    metric(y_pred=p, y=g)
    score = metric.aggregate()
    metric.reset()
    return float(score.item() if torch.is_tensor(score) else score)


def assert_dice_match(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> tuple[float, float]:
    """Compute numpy + MONAI Dice and assert they agree within tolerance."""
    d_np = numpy_dice(pred, gt)
    d_monai = monai_dice(pred, gt)
    if not np.isclose(d_np, d_monai, atol=atol, rtol=rtol):
        print(
            f"Dice mismatch — numpy={d_np:.8f}  monai={d_monai:.8f}  "
            f"|diff|={abs(d_np - d_monai):.8e} (atol={atol}, rtol={rtol})"
        )
        raise AssertionError(
            f"Manual Dice ({d_np:.8f}) != MONAI DiceMetric ({d_monai:.8f})"
        )
    return d_np, d_monai


def _as_affine(obj) -> np.ndarray:
    if isinstance(obj, MetaTensor) and getattr(obj, "affine", None) is not None:
        return np.asarray(obj.affine.detach().cpu(), dtype=np.float64)
    if hasattr(obj, "affine") and obj.affine is not None:
        return np.asarray(obj.affine, dtype=np.float64)
    meta = getattr(obj, "meta", None)
    if isinstance(meta, dict) and "affine" in meta:
        return np.asarray(meta["affine"], dtype=np.float64)
    return np.eye(4, dtype=np.float64)


def check_geometry(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    pred_affine: np.ndarray,
    gt_affine: np.ndarray,
    atol: float = 1e-5,
) -> None:
    """Raise a clear error if shape or affine differ (orientation/resample bugs)."""
    if pred.shape != gt.shape:
        raise ValueError(
            f"Prediction/GT shape mismatch: pred={pred.shape} gt={gt.shape}. "
            "This usually means orientation or resampling diverged before comparison."
        )
    pred_aff = np.asarray(pred_affine, dtype=np.float64)
    gt_aff = np.asarray(gt_affine, dtype=np.float64)
    if pred_aff.shape != (4, 4) or gt_aff.shape != (4, 4):
        raise ValueError(
            f"Expected 4x4 affines, got pred={pred_aff.shape} gt={gt_aff.shape}"
        )
    if not np.allclose(pred_aff, gt_aff, atol=atol, rtol=0.0):
        raise ValueError(
            "Prediction/GT affine mismatch (orientation/spacing diverged).\n"
            f"pred affine:\n{pred_aff}\n"
            f"gt affine:\n{gt_aff}\n"
            f"max |diff|={np.max(np.abs(pred_aff - gt_aff)):.6e}"
        )


def _volume_zyx(arr: np.ndarray) -> np.ndarray:
    """Normalize to (Z, Y, X) for axial slicing."""
    a = np.asarray(arr)
    if a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {a.shape}")
    # MONAI channel-first spatial is (H, W, D) ≈ (Y, X, Z) after EnsureChannelFirst;
    # our BraTS nifti load often ends as (H, W, D). Axial = last axis.
    return np.transpose(a, (2, 0, 1))  # (Z, Y, X)


def tumor_slice_indices(mask_zyx: np.ndarray, n: int = 4) -> list[int]:
    """Pick ``n`` evenly spaced axial indices that intersect the tumor mask."""
    zs = np.where(np.any(mask_zyx > 0, axis=(1, 2)))[0]
    if len(zs) == 0:
        # Fall back to middle of volume.
        zmid = mask_zyx.shape[0] // 2
        return [max(0, zmid - 3), max(0, zmid - 1), zmid, min(mask_zyx.shape[0] - 1, zmid + 2)][:n]
    if len(zs) <= n:
        return [int(z) for z in zs]
    picks = np.linspace(0, len(zs) - 1, n)
    return [int(zs[int(round(i))]) for i in picks]


def plot_overlay_grid(
    mri_zyx: np.ndarray,
    pred_zyx: np.ndarray,
    gt_zyx: np.ndarray,
    slice_indices: list[int],
    output_png: str | Path,
    *,
    title: str = "",
    pred_color: str = "tab:red",
    gt_color: str = "tab:cyan",
) -> Path:
    """Save a 1×N PNG grid of axial MRI slices with pred/GT contours."""
    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    n = len(slice_indices)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4), squeeze=False)
    for ax, z in zip(axes[0], slice_indices):
        sl = mri_zyx[z]
        # Robust display window
        lo, hi = np.percentile(sl[sl > 0], [1, 99]) if np.any(sl > 0) else (0.0, 1.0)
        ax.imshow(sl, cmap="gray", vmin=lo, vmax=hi, origin="lower")
        if np.any(gt_zyx[z] > 0):
            ax.contour(gt_zyx[z] > 0, levels=[0.5], colors=[gt_color], linewidths=1.2)
        if np.any(pred_zyx[z] > 0):
            ax.contour(pred_zyx[z] > 0, levels=[0.5], colors=[pred_color], linewidths=1.2)
        ax.set_title(f"z={z}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Legend proxies
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=gt_color, lw=2, label="ground truth"),
        Line2D([0], [0], color=pred_color, lw=2, label="prediction"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(output_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote overlay grid → %s", output_png)
    return output_png


def load_case_batch(case_id: str, data_dir: str | Path) -> dict:
    """Load one BraTS case with the deterministic val transform (full volume)."""
    data_dir = Path(data_dir)
    dicts = build_brats_data_dicts(data_dir)
    match = [d for d in dicts if d.get("case_id") == case_id]
    if not match:
        raise FileNotFoundError(f"Case {case_id!r} not found under {data_dir}")
    tf: Compose = get_val_transforms(spatial_size=None)
    return tf(match[0])


def region_masks_from_batch(
    batch: dict,
    model: torch.nn.Module,
    *,
    region: RegionName,
    device: torch.device,
    roi_size: tuple[int, int, int] = (96, 96, 96),
    use_amp: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return ``(mri, pred_bin, gt_bin, affine)`` for one region.

    ``mri`` / masks are (H, W, D) in the transformed MONAI space.
    """
    if region not in REGION_INDEX:
        raise ValueError(f"region must be one of {list(REGION_INDEX)}, got {region!r}")

    image = batch[IMAGE_KEY]
    label = batch[LABEL_KEY]
    if not torch.is_tensor(image):
        image = torch.as_tensor(image)
    if not torch.is_tensor(label):
        label = torch.as_tensor(label)

    # Add batch dim
    images = image.unsqueeze(0).to(device)
    labels = label.unsqueeze(0).to(device)
    regions_gt = exclusive_labels_to_regions(labels)
    probs = predict_regions(
        model,
        images,
        roi_size=roi_size,
        use_amp=use_amp and device.type == "cuda",
        device=device,
    )
    idx = REGION_INDEX[region]
    pred = (probs[0, idx] > 0.5).detach().cpu().numpy().astype(np.uint8)
    gt = (regions_gt[0, idx] > 0.5).detach().cpu().numpy().astype(np.uint8)

    # Prefer T1c / T1 channel for display (index 2 = t1c in FLAIR,T1,T1c,T2)
    img_np = image.detach().cpu().numpy()
    ch = min(2, img_np.shape[0] - 1)
    mri = img_np[ch]

    affine = _as_affine(label)
    return mri, pred, gt, affine


def sanity_check_case(
    case_id: str,
    *,
    data_dir: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    region: RegionName = "wt",
    device: str | None = None,
    dice_atol: float = 1e-4,
    tag: str = "",
) -> dict:
    """
    Full sanity check for one case: geometry, Dice (numpy vs MONAI), overlay PNG.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device
    device_t = torch.device(device_str)

    model, _meta = load_model_from_checkpoint(checkpoint, device=device_t)
    batch = load_case_batch(case_id, data_dir)
    mri, pred, gt, affine = region_masks_from_batch(
        batch,
        model,
        region=region,
        device=device_t,
    )

    # Geometry: pred and GT must share shape + affine (same transformed space).
    check_geometry(pred, gt, pred_affine=affine, gt_affine=affine)

    # Also verify against on-disk mask affine if available (resampled space may differ;
    # we only require pred/gt pair agreement for the metric comparison).
    case_mask = Path(data_dir) / case_id / "mask.nii.gz"
    if case_mask.is_file():
        disk = nib.load(str(case_mask))
        logger.info(
            "On-disk mask shape=%s affine_diag=%s (pred/gt compared in transform space %s)",
            disk.shape,
            np.diag(disk.affine)[:3],
            pred.shape,
        )

    d_np, d_monai = assert_dice_match(pred, gt, atol=dice_atol)
    logger.info(
        "Case %s region=%s  Dice numpy=%.6f  MONAI=%.6f  (match OK)",
        case_id,
        region,
        d_np,
        d_monai,
    )

    # Save binary NIfTIs for inspection
    pred_nii = output_dir / f"{case_id}_{region}_pred.nii.gz"
    gt_nii = output_dir / f"{case_id}_{region}_gt.nii.gz"
    nib.save(nib.Nifti1Image(pred.astype(np.uint8), affine), str(pred_nii))
    nib.save(nib.Nifti1Image(gt.astype(np.uint8), affine), str(gt_nii))

    mri_z = _volume_zyx(mri)
    pred_z = _volume_zyx(pred)
    gt_z = _volume_zyx(gt)
    union = ((pred_z > 0) | (gt_z > 0)).astype(np.uint8)
    slices = tumor_slice_indices(union, n=4)
    tag_part = f"_{tag}" if tag else ""
    png = output_dir / f"{case_id}_{region}{tag_part}_overlay.png"
    plot_overlay_grid(
        mri_z,
        pred_z,
        gt_z,
        slices,
        png,
        title=f"{case_id} · {region.upper()} · Dice={d_np:.3f} {tag}".strip(),
    )

    return {
        "case_id": case_id,
        "region": region,
        "dice_numpy": d_np,
        "dice_monai": d_monai,
        "shape": list(pred.shape),
        "overlay_png": str(png),
        "pred_nii": str(pred_nii),
        "gt_nii": str(gt_nii),
        "tag": tag,
    }


def pick_best_worst_cases(
    metrics_csv: str | Path,
    *,
    score_col: str = "dice_mean",
) -> tuple[str, str, float, float]:
    """Return ``(best_case_id, worst_case_id, best_score, worst_score)``."""
    df = pd.read_csv(metrics_csv)
    if score_col not in df.columns:
        raise KeyError(f"{score_col!r} not in CSV columns {list(df.columns)}")
    if "case_id" not in df.columns:
        raise KeyError("metrics CSV must include case_id")
    if df.empty:
        raise ValueError(f"Empty metrics CSV: {metrics_csv}")

    ranked = df.dropna(subset=[score_col]).sort_values(score_col, ascending=False)
    if ranked.empty:
        raise ValueError(f"No finite values in {score_col}")
    best = ranked.iloc[0]
    worst = ranked.iloc[-1]
    return (
        str(best["case_id"]),
        str(worst["case_id"]),
        float(best[score_col]),
        float(worst[score_col]),
    )


def run_best_worst_sanity_checks(
    metrics_csv: str | Path,
    *,
    data_dir: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    region: RegionName = "wt",
    score_col: str = "dice_mean",
    device: str | None = None,
) -> list[dict]:
    """Sanity-check the best-Dice and worst-Dice cases from an eval CSV."""
    best_id, worst_id, best_s, worst_s = pick_best_worst_cases(
        metrics_csv, score_col=score_col
    )
    logger.info(
        "From %s: best=%s (%.4f)  worst=%s (%.4f)  region=%s",
        metrics_csv,
        best_id,
        best_s,
        worst_id,
        worst_s,
        region,
    )
    results = []
    results.append(
        sanity_check_case(
            best_id,
            data_dir=data_dir,
            checkpoint=checkpoint,
            output_dir=output_dir,
            region=region,
            device=device,
            tag="best",
        )
    )
    if worst_id != best_id:
        results.append(
            sanity_check_case(
                worst_id,
                data_dir=data_dir,
                checkpoint=checkpoint,
                output_dir=output_dir,
                region=region,
                device=device,
                tag="worst",
            )
        )
    else:
        logger.warning(
            "Best and worst case are the same (%s) — CSV likely has a single row; "
            "ran one sanity check only.",
            best_id,
        )
    return results


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        from config import load_config

        cfg = load_config()
        default_data = str(cfg.paths.processed_brats_nifti)
        default_ckpt = str(cfg.paths.checkpoints / "brats_pretrain" / "best_model.pt")
        default_csv = str(cfg.paths.processed / "metrics" / "brats_val_metrics.csv")
        default_out = str(cfg.paths.processed / "metrics" / "sanity_checks")
    except Exception:  # noqa: BLE001
        default_data = "data/processed/brats_nifti"
        default_ckpt = "models/checkpoints/brats_pretrain/best_model.pt"
        default_csv = "data/processed/metrics/brats_val_metrics.csv"
        default_out = "data/processed/metrics/sanity_checks"

    smoke = Path("models/checkpoints/brats_pretrain_smoke/best_model.pt")
    if not Path(default_ckpt).is_file() and smoke.is_file():
        default_ckpt = str(smoke)
    if not Path(default_csv).is_file():
        alt = Path("data/processed/metrics/brats_val_metrics_smoke.csv")
        if alt.is_file():
            default_csv = str(alt)

    p = argparse.ArgumentParser(description="BraTS prediction sanity checks (Dice + overlays)")
    p.add_argument("--metrics-csv", default=default_csv)
    p.add_argument("--data-dir", default=default_data)
    p.add_argument("--checkpoint", default=default_ckpt)
    p.add_argument("--output-dir", default=default_out)
    p.add_argument("--region", choices=list(REGION_INDEX), default="wt")
    p.add_argument("--score-col", default="dice_mean")
    p.add_argument("--device", default=None)
    p.add_argument("--case-id", default=None, help="If set, check only this case")
    args = p.parse_args(argv)

    if args.case_id:
        sanity_check_case(
            args.case_id,
            data_dir=args.data_dir,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            region=args.region,
            device=args.device,
        )
    else:
        run_best_worst_sanity_checks(
            args.metrics_csv,
            data_dir=args.data_dir,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            region=args.region,
            score_col=args.score_col,
            device=args.device,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
