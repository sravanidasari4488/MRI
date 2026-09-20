"""Evaluate a BraTS checkpoint: per-class Dice, Hausdorff, per-case CSV."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from monai.data import DataLoader, Dataset, list_data_collate
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from torch.amp import autocast

from .dataset import IMAGE_KEY, LABEL_KEY, build_brats_data_dicts, get_val_transforms
from .model import OUT_REGION_NAMES, build_model
from .train import exclusive_labels_to_regions, split_train_val

logger = logging.getLogger(__name__)


def load_model_from_checkpoint(
    checkpoint: str | Path,
    *,
    device: torch.device,
    architecture: str = "segresnet",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load model weights from a training checkpoint (``best_model.pt``)."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    meta = blob.get("meta", {}) if isinstance(blob, dict) else {}
    model_info = blob.get("model_info", {}) if isinstance(blob, dict) else {}
    arch = model_info.get("architecture", architecture)

    model = build_model(arch if arch in {"segresnet", "unet"} else "segresnet")
    state = blob["model_state"] if isinstance(blob, dict) and "model_state" in blob else blob
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    logger.info(
        "Loaded checkpoint %s (epoch=%s best_metric=%s)",
        checkpoint,
        blob.get("epoch") if isinstance(blob, dict) else "?",
        blob.get("best_metric") if isinstance(blob, dict) else "?",
    )
    return model, meta if isinstance(meta, dict) else {}


@torch.no_grad()
def predict_regions(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    roi_size: tuple[int, int, int] = (96, 96, 96),
    sw_batch_size: int = 1,
    overlap: float = 0.5,
    use_amp: bool = False,
    device: torch.device,
) -> torch.Tensor:
    """Sliding-window inference → sigmoid probabilities ``(B, 3, H, W, D)``."""

    def _predictor(x: torch.Tensor) -> torch.Tensor:
        with autocast(device_type=device.type, enabled=use_amp):
            return model(x)

    logits = sliding_window_inference(
        inputs=images,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=_predictor,
        overlap=overlap,
        mode="gaussian",
    )
    return torch.sigmoid(logits)


def _safe_float(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if x != x:  # NaN
        return float("nan")
    return x


def evaluate_checkpoint(
    checkpoint: str | Path,
    data_dir: str | Path,
    output_csv: str | Path,
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
    device: str | None = None,
    roi_size: tuple[int, int, int] = (96, 96, 96),
    sw_batch_size: int = 1,
    overlap: float = 0.5,
    amp: bool | None = None,
    hd_percentile: float = 95.0,
    summary_json: str | Path | None = None,
    max_cases: int | None = None,
) -> pd.DataFrame:
    """
    Run validation-set inference and write per-case metrics CSV.

    Metrics (per region ET / TC / WT):
      - Dice
      - Hausdorff distance (default HD95, millimeters if spacing is 1 mm)

    Uses the same 80/20 split as training (``val_ratio=0.2``, ``seed=42``).
    """
    data_dir = Path(data_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device
    device_t = torch.device(device_str)
    use_amp = bool(amp) if amp is not None else device_t.type == "cuda"
    if use_amp and device_t.type != "cuda":
        use_amp = False

    data_dicts = build_brats_data_dicts(data_dir)
    if not data_dicts:
        raise FileNotFoundError(f"No BraTS cases under {data_dir}")

    _train_files, val_files = split_train_val(data_dicts, val_ratio=val_ratio, seed=seed)
    if not val_files:
        raise RuntimeError("Validation split is empty")
    if max_cases is not None:
        val_files = val_files[: max(1, max_cases)]

    # Full volumes for Hausdorff (no center crop).
    val_ds = Dataset(
        data=val_files,
        transform=get_val_transforms(spatial_size=None),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=list_data_collate,
    )

    model, meta = load_model_from_checkpoint(checkpoint, device=device_t)

    dice_metric = DiceMetric(include_background=True, reduction="none")
    hd_metric = HausdorffDistanceMetric(
        include_background=True,
        reduction="none",
        percentile=hd_percentile,
    )

    rows: list[dict[str, Any]] = []
    logger.info(
        "Evaluating %d val cases from %s on %s (AMP=%s)",
        len(val_files),
        data_dir,
        device_str,
        use_amp,
    )

    for i, batch in enumerate(val_loader, start=1):
        case_id = batch.get("case_id", [f"case_{i}"])[0]
        if isinstance(case_id, (list, tuple)):
            case_id = case_id[0]
        case_id = str(case_id)

        images = batch[IMAGE_KEY].to(device_t)
        labels = batch[LABEL_KEY].to(device_t)
        regions = exclusive_labels_to_regions(labels)

        probs = predict_regions(
            model,
            images,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            use_amp=use_amp,
            device=device_t,
        )
        preds = (probs > 0.5).float()

        # Metrics expect (B, C, H, W, D)
        dice_metric(y_pred=preds, y=regions)
        hd_metric(y_pred=preds, y=regions)
        dice_vals = dice_metric.aggregate()
        hd_vals = hd_metric.aggregate()
        dice_metric.reset()
        hd_metric.reset()

        # Shapes: (B, C) → take batch 0
        if dice_vals.ndim == 1:
            dice_row = dice_vals
            hd_row = hd_vals
        else:
            dice_row = dice_vals[0]
            hd_row = hd_vals[0]

        row: dict[str, Any] = {"case_id": case_id}
        for c, name in enumerate(OUT_REGION_NAMES):
            row[f"dice_{name}"] = _safe_float(dice_row[c].item())
            row[f"hd{int(hd_percentile)}_{name}"] = _safe_float(hd_row[c].item())
        row["dice_mean"] = float(
            sum(row[f"dice_{n}"] for n in OUT_REGION_NAMES) / len(OUT_REGION_NAMES)
        )
        rows.append(row)
        logger.info(
            "[%d/%d] %s  Dice ET=%.3f TC=%.3f WT=%.3f  mean=%.3f",
            i,
            len(val_files),
            case_id,
            row["dice_et"],
            row["dice_tc"],
            row["dice_wt"],
            row["dice_mean"],
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    logger.info("Wrote per-case metrics → %s", output_csv)

    summary = {
        "checkpoint": str(checkpoint),
        "data_dir": str(data_dir),
        "n_val": len(df),
        "val_ratio": val_ratio,
        "seed": seed,
        "hd_percentile": hd_percentile,
        "device": device_str,
        "train_meta": meta,
        "mean_dice_et": float(df["dice_et"].mean()) if len(df) else float("nan"),
        "mean_dice_tc": float(df["dice_tc"].mean()) if len(df) else float("nan"),
        "mean_dice_wt": float(df["dice_wt"].mean()) if len(df) else float("nan"),
        "mean_dice": float(df["dice_mean"].mean()) if len(df) else float("nan"),
        f"mean_hd{int(hd_percentile)}_et": float(df[f"hd{int(hd_percentile)}_et"].mean())
        if len(df)
        else float("nan"),
        f"mean_hd{int(hd_percentile)}_tc": float(df[f"hd{int(hd_percentile)}_tc"].mean())
        if len(df)
        else float("nan"),
        f"mean_hd{int(hd_percentile)}_wt": float(df[f"hd{int(hd_percentile)}_wt"].mean())
        if len(df)
        else float("nan"),
    }
    logger.info(
        "Val summary — Dice ET=%.4f TC=%.4f WT=%.4f mean=%.4f | HD%.0f ET=%.2f TC=%.2f WT=%.2f",
        summary["mean_dice_et"],
        summary["mean_dice_tc"],
        summary["mean_dice_wt"],
        summary["mean_dice"],
        hd_percentile,
        summary[f"mean_hd{int(hd_percentile)}_et"],
        summary[f"mean_hd{int(hd_percentile)}_tc"],
        summary[f"mean_hd{int(hd_percentile)}_wt"],
    )

    if summary_json is None:
        summary_json = output_csv.with_suffix(".summary.json")
    else:
        summary_json = Path(summary_json)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote summary → %s", summary_json)
    return df


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        from config import load_config

        cfg = load_config()
        default_data = str(cfg.paths.processed_brats_nifti)
        default_ckpt = str(cfg.paths.checkpoints / "brats_pretrain" / "best_model.pt")
        default_csv = str(cfg.paths.processed / "metrics" / "brats_val_metrics.csv")
    except Exception:  # noqa: BLE001
        default_data = "data/processed/brats_nifti"
        default_ckpt = "models/checkpoints/brats_pretrain/best_model.pt"
        default_csv = "data/processed/metrics/brats_val_metrics.csv"

    # Prefer smoke checkpoint if full pretrain is missing.
    smoke = Path("models/checkpoints/brats_pretrain_smoke/best_model.pt")
    if not Path(default_ckpt).is_file() and smoke.is_file():
        default_ckpt = str(smoke)

    p = argparse.ArgumentParser(description="Evaluate BraTS SegResNet checkpoint")
    p.add_argument("--checkpoint", default=default_ckpt)
    p.add_argument("--data-dir", default=default_data)
    p.add_argument("--output-csv", default=default_csv)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--hd-percentile", type=float, default=95.0)
    args = p.parse_args(argv)

    evaluate_checkpoint(
        args.checkpoint,
        args.data_dir,
        args.output_csv,
        val_ratio=args.val_ratio,
        seed=args.seed,
        device=args.device,
        hd_percentile=args.hd_percentile,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
