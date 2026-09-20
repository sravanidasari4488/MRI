"""Domain-adaptation report: BraTS-only vs fine-tuned on held-out real cases."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader, Dataset, list_data_collate
from monai.metrics import DiceMetric

from .dataset import IMAGE_KEY, LABEL_KEY, get_val_transforms
from .evaluate import _safe_float, load_model_from_checkpoint, predict_regions
from .finetune import discover_corrected_cases, split_train_holdout
from .model import OUT_REGION_NAMES
from .train import exclusive_labels_to_regions

logger = logging.getLogger(__name__)

# Isotropic 1 mm preprocessing → 1 voxel = 1 mm³ = 0.001 mL.
VOXEL_ML = 1.0e-3


def resolve_holdout_cases(
    corrected_root: str | Path,
    *,
    finetune_dir: str | Path | None = None,
    n_holdout: int = 4,
    seed: int = 42,
) -> list[dict]:
    """
    Resolve held-out real test cases for the domain-gap comparison.

    Prefer ``finetune_split.json`` from a prior fine-tune run so the report
    uses the **exact** cases never seen in fine-tuning. Otherwise recompute
    the same split via :func:`split_train_holdout`.
    """
    corrected_root = Path(corrected_root)
    all_cases = discover_corrected_cases(corrected_root)
    if not all_cases:
        raise FileNotFoundError(
            f"No corrected cases with modalities + mask under {corrected_root}"
        )
    by_id = {d["case_id"]: d for d in all_cases}

    split_path = None
    if finetune_dir is not None:
        split_path = Path(finetune_dir) / "finetune_split.json"
    if split_path is not None and split_path.is_file():
        blob = json.loads(split_path.read_text(encoding="utf-8"))
        holdout_ids = list(blob.get("holdout_test", []))
        missing = [c for c in holdout_ids if c not in by_id]
        if missing:
            raise FileNotFoundError(
                f"Holdout IDs from {split_path} not found under {corrected_root}: {missing}"
            )
        holdout = [by_id[c] for c in holdout_ids]
        logger.info(
            "Using holdout from %s (%d cases): %s",
            split_path,
            len(holdout),
            holdout_ids,
        )
        return holdout

    _, holdout = split_train_holdout(all_cases, n_holdout=n_holdout, seed=seed)
    if not holdout:
        raise RuntimeError("Holdout split is empty — need more corrected cases")
    logger.info(
        "No finetune_split.json; recomputed holdout (n=%d, seed=%d): %s",
        len(holdout),
        seed,
        [d["case_id"] for d in holdout],
    )
    return holdout


def _region_volumes_ml(binary: torch.Tensor) -> dict[str, float]:
    """Per-region volume in mL for a ``(1, 3, H, W, D)`` binary map."""
    dims = tuple(range(2, binary.ndim))
    counts = torch.sum(binary.float(), dim=dims)[0]  # (3,)
    return {
        name: float(counts[i].item()) * VOXEL_ML for i, name in enumerate(OUT_REGION_NAMES)
    }


def evaluate_checkpoint_on_holdout(
    checkpoint: str | Path,
    holdout_files: Sequence[dict],
    *,
    model_tag: str,
    device: torch.device,
    use_amp: bool = False,
    roi_size: tuple[int, int, int] = (96, 96, 96),
    sw_batch_size: int = 1,
    overlap: float = 0.5,
) -> pd.DataFrame:
    """
    Run one checkpoint on holdout cases; return per-case Dice + volume error.

    Volume error (per region):
      - ``vol_pred_*_ml`` / ``vol_gt_*_ml``
      - ``vol_abs_err_*_ml`` = |pred − gt|
      - ``vol_rel_err_*`` = |pred − gt| / max(gt, ε)  (fraction)
    """
    ds = Dataset(data=list(holdout_files), transform=get_val_transforms(spatial_size=None))
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=list_data_collate,
    )
    model, meta = load_model_from_checkpoint(checkpoint, device=device)
    dice_metric = DiceMetric(include_background=True, reduction="none")

    rows: list[dict[str, Any]] = []
    for i, (batch, meta_dict) in enumerate(zip(loader, holdout_files), start=1):
        case_id = batch.get("case_id", [meta_dict.get("case_id", f"case_{i}")])[0]
        if isinstance(case_id, (list, tuple)):
            case_id = case_id[0]
        case_id = str(case_id)

        images = batch[IMAGE_KEY].to(device)
        labels = batch[LABEL_KEY].to(device)
        regions = exclusive_labels_to_regions(labels)

        probs = predict_regions(
            model,
            images,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            use_amp=use_amp,
            device=device,
        )
        preds = (probs > 0.5).float()

        dice_metric(y_pred=preds, y=regions)
        dice_vals = dice_metric.aggregate()
        dice_metric.reset()
        dice_row = dice_vals if dice_vals.ndim == 1 else dice_vals[0]

        pred_vol = _region_volumes_ml(preds)
        gt_vol = _region_volumes_ml(regions)

        row: dict[str, Any] = {
            "case_id": case_id,
            "model": model_tag,
            "checkpoint": str(checkpoint),
        }
        for c, name in enumerate(OUT_REGION_NAMES):
            row[f"dice_{name}"] = _safe_float(dice_row[c].item())
            vp, vg = pred_vol[name], gt_vol[name]
            row[f"vol_pred_{name}_ml"] = vp
            row[f"vol_gt_{name}_ml"] = vg
            row[f"vol_abs_err_{name}_ml"] = abs(vp - vg)
            row[f"vol_rel_err_{name}"] = abs(vp - vg) / max(vg, 1e-6)

        row["dice_mean"] = float(
            sum(row[f"dice_{n}"] for n in OUT_REGION_NAMES) / len(OUT_REGION_NAMES)
        )
        row["vol_abs_err_mean_ml"] = float(
            sum(row[f"vol_abs_err_{n}_ml"] for n in OUT_REGION_NAMES)
            / len(OUT_REGION_NAMES)
        )
        row["vol_rel_err_mean"] = float(
            sum(row[f"vol_rel_err_{n}"] for n in OUT_REGION_NAMES) / len(OUT_REGION_NAMES)
        )
        rows.append(row)
        logger.info(
            "[%s %d/%d] %s  Dice mean=%.3f  vol_rel_err mean=%.3f",
            model_tag,
            i,
            len(holdout_files),
            case_id,
            row["dice_mean"],
            row["vol_rel_err_mean"],
        )

    df = pd.DataFrame(rows)
    df.attrs["train_meta"] = meta
    return df


def build_comparison_table(
    brats_df: pd.DataFrame,
    finetuned_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge BraTS-only vs fine-tuned metrics and add delta columns (after − before)."""
    left = brats_df.drop(columns=["model", "checkpoint"], errors="ignore").add_prefix(
        "brats_"
    )
    left = left.rename(columns={"brats_case_id": "case_id"})
    right = finetuned_df.drop(columns=["model", "checkpoint"], errors="ignore").add_prefix(
        "ft_"
    )
    right = right.rename(columns={"ft_case_id": "case_id"})

    merged = left.merge(right, on="case_id", how="inner")
    if merged.empty:
        raise RuntimeError("No overlapping case_ids between BraTS and fine-tuned evals")

    for name in OUT_REGION_NAMES:
        merged[f"delta_dice_{name}"] = merged[f"ft_dice_{name}"] - merged[f"brats_dice_{name}"]
        merged[f"delta_vol_abs_err_{name}_ml"] = (
            merged[f"ft_vol_abs_err_{name}_ml"] - merged[f"brats_vol_abs_err_{name}_ml"]
        )
        merged[f"delta_vol_rel_err_{name}"] = (
            merged[f"ft_vol_rel_err_{name}"] - merged[f"brats_vol_rel_err_{name}"]
        )

    merged["delta_dice_mean"] = merged["ft_dice_mean"] - merged["brats_dice_mean"]
    merged["delta_vol_abs_err_mean_ml"] = (
        merged["ft_vol_abs_err_mean_ml"] - merged["brats_vol_abs_err_mean_ml"]
    )
    merged["delta_vol_rel_err_mean"] = (
        merged["ft_vol_rel_err_mean"] - merged["brats_vol_rel_err_mean"]
    )
    return merged


def summary_means(comparison: pd.DataFrame) -> dict[str, float]:
    """Aggregate mean metrics across holdout cases for the report."""
    keys = [
        "brats_dice_mean",
        "ft_dice_mean",
        "delta_dice_mean",
        "brats_vol_rel_err_mean",
        "ft_vol_rel_err_mean",
        "delta_vol_rel_err_mean",
        "brats_vol_abs_err_mean_ml",
        "ft_vol_abs_err_mean_ml",
        "delta_vol_abs_err_mean_ml",
    ]
    for name in OUT_REGION_NAMES:
        keys.extend(
            [
                f"brats_dice_{name}",
                f"ft_dice_{name}",
                f"delta_dice_{name}",
                f"brats_vol_rel_err_{name}",
                f"ft_vol_rel_err_{name}",
                f"delta_vol_rel_err_{name}",
            ]
        )
    out: dict[str, float] = {}
    for k in keys:
        if k in comparison.columns:
            out[k] = float(comparison[k].mean())
    return out


def comparison_table_markdown(means: dict[str, float]) -> str:
    """Human-readable Markdown summary table (domain-adaptation contribution)."""
    lines = [
        "# Domain-gap report: BraTS-only → fine-tuned",
        "",
        "Held-out real patient test set (never used in fine-tuning).",
        "",
        "| Metric | BraTS-only | Fine-tuned | Δ (after − before) |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| Mean Dice | {means.get('brats_dice_mean', float('nan')):.4f} | "
            f"{means.get('ft_dice_mean', float('nan')):.4f} | "
            f"{means.get('delta_dice_mean', float('nan')):+.4f} |"
        ),
    ]
    for name in OUT_REGION_NAMES:
        lines.append(
            f"| Dice {name.upper()} | {means.get(f'brats_dice_{name}', float('nan')):.4f} | "
            f"{means.get(f'ft_dice_{name}', float('nan')):.4f} | "
            f"{means.get(f'delta_dice_{name}', float('nan')):+.4f} |"
        )
    lines.append(
        f"| Mean |V| rel. error | {means.get('brats_vol_rel_err_mean', float('nan')):.4f} | "
        f"{means.get('ft_vol_rel_err_mean', float('nan')):.4f} | "
        f"{means.get('delta_vol_rel_err_mean', float('nan')):+.4f} |"
    )
    for name in OUT_REGION_NAMES:
        lines.append(
            f"| |V| rel. err. {name.upper()} | "
            f"{means.get(f'brats_vol_rel_err_{name}', float('nan')):.4f} | "
            f"{means.get(f'ft_vol_rel_err_{name}', float('nan')):.4f} | "
            f"{means.get(f'delta_vol_rel_err_{name}', float('nan')):+.4f} |"
        )
    lines.append(
        f"| Mean |V| abs. error (mL) | "
        f"{means.get('brats_vol_abs_err_mean_ml', float('nan')):.3f} | "
        f"{means.get('ft_vol_abs_err_mean_ml', float('nan')):.3f} | "
        f"{means.get('delta_vol_abs_err_mean_ml', float('nan')):+.3f} |"
    )
    lines.append("")
    lines.append(
        "Positive Δ Dice = improvement after fine-tuning. "
        "Negative Δ volume error = smaller error after fine-tuning."
    )
    lines.append("")
    return "\n".join(lines)


def plot_domain_gap_bars(
    means: dict[str, float],
    output_path: str | Path,
    *,
    title: str = "Domain adaptation: BraTS-only vs fine-tuned (holdout)",
) -> Path:
    """
    Grouped bar chart: Dice (higher better) and relative volume error
    (lower better) before/after fine-tuning.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    regions = list(OUT_REGION_NAMES) + ["mean"]
    x = np.arange(len(regions))
    width = 0.35

    dice_before = [
        means.get(f"brats_dice_{r}" if r != "mean" else "brats_dice_mean", float("nan"))
        for r in regions
    ]
    dice_after = [
        means.get(f"ft_dice_{r}" if r != "mean" else "ft_dice_mean", float("nan"))
        for r in regions
    ]
    vol_before = [
        means.get(
            f"brats_vol_rel_err_{r}" if r != "mean" else "brats_vol_rel_err_mean",
            float("nan"),
        )
        for r in regions
    ]
    vol_after = [
        means.get(
            f"ft_vol_rel_err_{r}" if r != "mean" else "ft_vol_rel_err_mean",
            float("nan"),
        )
        for r in regions
    ]

    labels = [r.upper() for r in regions]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

    ax0 = axes[0]
    b0 = ax0.bar(x - width / 2, dice_before, width, label="BraTS-only", color="#4C78A8")
    b1 = ax0.bar(x + width / 2, dice_after, width, label="Fine-tuned", color="#F58518")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)
    ax0.set_ylim(0.0, 1.05)
    ax0.set_ylabel("Dice")
    ax0.set_title("Dice (↑ better)")
    ax0.legend(frameon=False, loc="lower right")
    ax0.axhline(0, color="#888", linewidth=0.5)
    for bars in (b0, b1):
        for bar in bars:
            h = bar.get_height()
            if h == h:  # not NaN
                ax0.annotate(
                    f"{h:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax1 = axes[1]
    b2 = ax1.bar(x - width / 2, vol_before, width, label="BraTS-only", color="#4C78A8")
    b3 = ax1.bar(x + width / 2, vol_after, width, label="Fine-tuned", color="#F58518")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Relative volume error")
    ax1.set_title("Volume error (↓ better)")
    ax1.legend(frameon=False, loc="upper right")
    for bars in (b2, b3):
        for bar in bars:
            h = bar.get_height()
            if h == h:
                ax1.annotate(
                    f"{h:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    fig.suptitle(title, fontsize=12)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    logger.info("Wrote bar chart → %s", output_path)
    return output_path


def run_domain_gap_report(
    corrected_root: str | Path,
    brats_checkpoint: str | Path,
    finetuned_checkpoint: str | Path,
    output_dir: str | Path,
    *,
    finetune_dir: str | Path | None = None,
    n_holdout: int = 4,
    seed: int = 42,
    device: str | None = None,
    amp: bool | None = None,
    roi_size: tuple[int, int, int] = (96, 96, 96),
) -> dict[str, Path]:
    """
    Compare BraTS-only vs fine-tuned checkpoints on held-out real cases.

    Writes:
      - ``per_case_brats.csv`` / ``per_case_finetuned.csv``
      - ``comparison_table.csv`` (per-case deltas)
      - ``comparison_summary.md`` + ``comparison_summary.json``
      - ``domain_gap_bars.png``
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device
    device_t = torch.device(device_str)
    use_amp = bool(amp) if amp is not None else device_t.type == "cuda"
    if use_amp and device_t.type != "cuda":
        use_amp = False

    # Default finetune_dir to the fine-tuned checkpoint parent for split JSON.
    if finetune_dir is None:
        finetune_dir = Path(finetuned_checkpoint).parent

    holdout = resolve_holdout_cases(
        corrected_root,
        finetune_dir=finetune_dir,
        n_holdout=n_holdout,
        seed=seed,
    )

    brats_df = evaluate_checkpoint_on_holdout(
        brats_checkpoint,
        holdout,
        model_tag="brats_only",
        device=device_t,
        use_amp=use_amp,
        roi_size=roi_size,
    )
    ft_df = evaluate_checkpoint_on_holdout(
        finetuned_checkpoint,
        holdout,
        model_tag="finetuned",
        device=device_t,
        use_amp=use_amp,
        roi_size=roi_size,
    )

    brats_csv = output_dir / "per_case_brats.csv"
    ft_csv = output_dir / "per_case_finetuned.csv"
    brats_df.to_csv(brats_csv, index=False)
    ft_df.to_csv(ft_csv, index=False)

    comparison = build_comparison_table(brats_df, ft_df)
    comparison_csv = output_dir / "comparison_table.csv"
    comparison.to_csv(comparison_csv, index=False)

    means = summary_means(comparison)
    md = comparison_table_markdown(means)
    md_path = output_dir / "comparison_summary.md"
    md_path.write_text(md, encoding="utf-8")
    print(md)

    chart_path = plot_domain_gap_bars(means, output_dir / "domain_gap_bars.png")

    summary = {
        "corrected_root": str(corrected_root),
        "brats_checkpoint": str(brats_checkpoint),
        "finetuned_checkpoint": str(finetuned_checkpoint),
        "holdout_ids": [d["case_id"] for d in holdout],
        "n_holdout": len(holdout),
        "device": device_str,
        "means": means,
        "artifacts": {
            "per_case_brats": str(brats_csv),
            "per_case_finetuned": str(ft_csv),
            "comparison_table": str(comparison_csv),
            "comparison_summary_md": str(md_path),
            "bar_chart": str(chart_path),
        },
    }
    summary_json = output_dir / "comparison_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "Domain gap — Dice %.4f → %.4f (Δ%+.4f); vol_rel_err %.4f → %.4f (Δ%+.4f)",
        means.get("brats_dice_mean", float("nan")),
        means.get("ft_dice_mean", float("nan")),
        means.get("delta_dice_mean", float("nan")),
        means.get("brats_vol_rel_err_mean", float("nan")),
        means.get("ft_vol_rel_err_mean", float("nan")),
        means.get("delta_vol_rel_err_mean", float("nan")),
    )

    return {
        "per_case_brats": brats_csv,
        "per_case_finetuned": ft_csv,
        "comparison_table": comparison_csv,
        "comparison_summary_md": md_path,
        "comparison_summary_json": summary_json,
        "bar_chart": chart_path,
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        from config import load_config

        cfg = load_config()
        default_data = str(cfg.paths.processed / "real_corrected")
        default_brats = str(cfg.paths.checkpoints / "brats_pretrain" / "best_model.pt")
        default_ft = str(cfg.paths.checkpoints / "real_finetune" / "best_model.pt")
        default_out = str(cfg.paths.processed / "metrics" / "domain_gap")
        default_ft_dir = str(cfg.paths.checkpoints / "real_finetune")
    except Exception:  # noqa: BLE001
        default_data = "data/processed/real_corrected"
        default_brats = "models/checkpoints/brats_pretrain/best_model.pt"
        default_ft = "models/checkpoints/real_finetune/best_model.pt"
        default_out = "data/processed/metrics/domain_gap"
        default_ft_dir = "models/checkpoints/real_finetune"

    smoke = Path("models/checkpoints/brats_pretrain_smoke/best_model.pt")
    if not Path(default_brats).is_file() and smoke.is_file():
        default_brats = str(smoke)

    p = argparse.ArgumentParser(
        description="Domain-gap report: BraTS-only vs fine-tuned on holdout real cases"
    )
    p.add_argument("--corrected-root", default=default_data)
    p.add_argument("--brats-checkpoint", default=default_brats)
    p.add_argument("--finetuned-checkpoint", default=default_ft)
    p.add_argument(
        "--finetune-dir",
        default=default_ft_dir,
        help="Directory with finetune_split.json (holdout IDs)",
    )
    p.add_argument("--output-dir", default=default_out)
    p.add_argument("--n-holdout", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    run_domain_gap_report(
        args.corrected_root,
        args.brats_checkpoint,
        args.finetuned_checkpoint,
        args.output_dir,
        finetune_dir=args.finetune_dir,
        n_holdout=args.n_holdout,
        seed=args.seed,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
