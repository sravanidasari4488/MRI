"""Fine-tune a BraTS-pretrained SegResNet on manually corrected real cases."""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch
from monai.data import DataLoader, Dataset, list_data_collate
from monai.losses import DiceLoss
from monai.networks.nets import SegResNet, UNet
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from .dataset import (
    IMAGE_KEY,
    LABEL_KEY,
    MODALITIES,
    get_train_transforms,
    get_val_transforms,
)
from .evaluate import load_model_from_checkpoint
from .model import OUT_REGION_NAMES, describe_model
from .pseudo_label import _find_modality_nifti
from .train import (
    exclusive_labels_to_regions,
    save_checkpoint,
    train_one_epoch,
    validate_one_epoch,
)

logger = logging.getLogger(__name__)

# Default fine-tune LR is 10× lower than config pretrain (1e-4 → 1e-5).
DEFAULT_PRETRAIN_LR = 1.0e-4
DEFAULT_FINETUNE_LR = 1.0e-5


def discover_corrected_cases(
    corrected_root: str | Path,
    *,
    modalities: Sequence[str] = MODALITIES,
) -> list[dict]:
    """
    Build MONAI data dicts from a directory of manually corrected real cases.

    Expected layout (either)::

        corrected_root/<case_id>/{flair,t1,t1c,t2}.nii.gz + mask.nii.gz
        corrected_root/<case_id>/05_isotropic_1mm/{mod}_1mm.nii.gz
            + pseudo_labels/pseudo_seg.nii.gz   (after manual edit)

    Label files accepted: ``mask.nii.gz``, ``seg.nii.gz``, ``pseudo_seg.nii.gz``,
    ``label.nii.gz`` (under the case dir, ``pseudo_labels/``, or ``05_isotropic_1mm/``).
    """
    root = Path(corrected_root)
    if not root.is_dir():
        raise NotADirectoryError(f"Corrected-case root not found: {root}")

    records: list[dict] = []
    for case_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        image_paths: list[str] = []
        ok = True
        for mod in modalities:
            path = _find_modality_nifti(case_dir, mod)
            if path is None:
                logger.warning("Skip %s — missing modality %s", case_dir.name, mod)
                ok = False
                break
            image_paths.append(str(path))
        if not ok:
            continue

        label = _find_label_nifti(case_dir)
        if label is None:
            logger.warning("Skip %s — no corrected mask/seg NIfTI found", case_dir.name)
            continue

        records.append(
            {
                IMAGE_KEY: image_paths,
                LABEL_KEY: str(label),
                "case_id": case_dir.name,
            }
        )

    logger.info("Found %d manually corrected cases under %s", len(records), root)
    return records


def _find_label_nifti(case_dir: Path) -> Path | None:
    names = (
        "mask.nii.gz",
        "seg.nii.gz",
        "label.nii.gz",
        "pseudo_seg.nii.gz",
        "corrected_seg.nii.gz",
    )
    search_dirs = (
        case_dir,
        case_dir / "pseudo_labels",
        case_dir / "05_isotropic_1mm",
        case_dir / "labels",
    )
    for d in search_dirs:
        for name in names:
            path = d / name
            if path.is_file():
                return path
    # Fuzzy fallback
    hits = sorted(case_dir.rglob("*seg*.nii*")) + sorted(case_dir.rglob("*mask*.nii*"))
    hits = [p for p in hits if p.is_file()]
    return hits[0] if hits else None


def split_train_holdout(
    data_dicts: Sequence[dict],
    *,
    n_holdout: int = 4,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Hold out ``n_holdout`` cases (clamped to 3–5 when possible) as a fine-tune
    **test** set never used for training/validation updates.

    Remaining cases are the fine-tune training pool.
    """
    items = list(data_dicts)
    n = len(items)
    if n == 0:
        return [], []

    # Prefer 3–5 holdout cases; shrink if the pool is tiny.
    n_hold = int(n_holdout)
    n_hold = max(3, min(5, n_hold)) if n >= 6 else max(1, min(n_hold, n - 1)) if n >= 2 else 0
    if n_hold >= n:
        n_hold = max(0, n - 1)

    rng = random.Random(seed)
    rng.shuffle(items)
    holdout = items[:n_hold]
    train = items[n_hold:]
    logger.info(
        "Fine-tune split: train=%d  holdout_test=%d (requested=%d, seed=%d)  holdout_ids=%s",
        len(train),
        len(holdout),
        n_holdout,
        seed,
        [d.get("case_id") for d in holdout],
    )
    return train, holdout


def freeze_early_encoder(
    model: torch.nn.Module,
    *,
    n_down_stages: int = 2,
) -> list[str]:
    """
    Freeze early encoder weights (optional fine-tune regularization).

    SegResNet: freezes ``convInit`` and the first ``n_down_stages`` of
    ``down_layers``. UNet: freezes the first ``n_down_stages`` encoder blocks
    in ``model.model`` when present.

    Returns names of frozen parameter tensors.
    """
    frozen: list[str] = []

    def _freeze(module: torch.nn.Module, prefix: str) -> None:
        for name, param in module.named_parameters():
            param.requires_grad = False
            frozen.append(f"{prefix}.{name}" if prefix else name)

    if isinstance(model, SegResNet):
        if hasattr(model, "convInit"):
            _freeze(model.convInit, "convInit")
        if hasattr(model, "down_layers"):
            n = max(0, min(int(n_down_stages), len(model.down_layers)))
            for i in range(n):
                _freeze(model.down_layers[i], f"down_layers.{i}")
    elif isinstance(model, UNet):
        # MONAI UNet stores Sequential blocks in ``model``.
        seq = getattr(model, "model", None)
        if seq is not None and len(seq) > 0:
            n = max(1, min(int(n_down_stages), len(seq) // 2))
            for i in range(n):
                _freeze(seq[i], f"model.{i}")
    else:
        # Generic fallback: freeze parameters whose names look encoder-like.
        for name, param in model.named_parameters():
            if any(k in name for k in ("convInit", "down_layers", "encoder")):
                param.requires_grad = False
                frozen.append(name)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Froze %d early-encoder tensors (%d/%d params still trainable)",
        len(frozen),
        n_train,
        n_total,
    )
    return frozen


def finetune_real_cases(
    corrected_root: str | Path,
    pretrained_checkpoint: str | Path,
    output_dir: str | Path,
    *,
    max_epochs: int = 30,
    batch_size: int = 1,
    learning_rate: float = DEFAULT_FINETUNE_LR,
    pretrain_lr: float = DEFAULT_PRETRAIN_LR,
    n_holdout: int = 4,
    seed: int = 42,
    freeze_encoder: bool = False,
    freeze_encoder_stages: int = 2,
    spatial_size: tuple[int, int, int] = (96, 96, 96),
    num_workers: int = 0,
    device: str | None = None,
    amp: bool | None = None,
    log_dir: str | Path | None = None,
) -> dict[str, Path]:
    """
    Fine-tune BraTS weights on manually corrected real patient cases.

    Parameters
    ----------
    freeze_encoder:
        If True, freeze ``convInit`` + early ``down_layers`` (SegResNet).
    learning_rate:
        Fine-tune LR (default ``1e-5``, lower than pretrain ``1e-4``).
    n_holdout:
        Number of cases held out as a **test** set (clamped toward 3–5).

    Returns
    -------
    Paths to ``best_model.pt``, ``last_model.pt``, holdout metrics CSV, and
    split JSON.
    """
    corrected_root = Path(corrected_root)
    pretrained_checkpoint = Path(pretrained_checkpoint)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(log_dir) if log_dir is not None else output_dir / "tb"
    log_dir.mkdir(parents=True, exist_ok=True)

    if learning_rate >= pretrain_lr:
        logger.warning(
            "Fine-tune LR (%.2e) is not lower than pretrain LR (%.2e); "
            "typical fine-tune uses ~10× smaller LR.",
            learning_rate,
            pretrain_lr,
        )

    data_dicts = discover_corrected_cases(corrected_root)
    if len(data_dicts) < 2:
        raise FileNotFoundError(
            f"Need at least 2 corrected cases with 4 modalities + mask under "
            f"{corrected_root} (found {len(data_dicts)})."
        )

    train_files, holdout_files = split_train_holdout(
        data_dicts, n_holdout=n_holdout, seed=seed
    )
    if not train_files:
        raise RuntimeError("No training cases left after holdout split")

    # Small internal val slice from train pool (for best-checkpoint selection),
    # keeping holdout untouched until final test evaluation.
    if len(train_files) >= 4:
        rng = random.Random(seed + 1)
        shuffled = list(train_files)
        rng.shuffle(shuffled)
        n_val = max(1, len(shuffled) // 5)
        val_files = shuffled[:n_val]
        fit_files = shuffled[n_val:]
    else:
        fit_files = train_files
        val_files = train_files[:1]  # monitor only; still not the holdout test set

    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device
    device_t = torch.device(device_str)
    use_amp = bool(amp) if amp is not None else device_t.type == "cuda"
    if use_amp and device_t.type != "cuda":
        use_amp = False

    model, pre_meta = load_model_from_checkpoint(pretrained_checkpoint, device=device_t)
    model.train()
    frozen_names: list[str] = []
    if freeze_encoder:
        frozen_names = freeze_early_encoder(model, n_down_stages=freeze_encoder_stages)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("All parameters frozen — nothing to fine-tune")

    train_ds = Dataset(
        data=fit_files,
        transform=get_train_transforms(spatial_size=spatial_size, num_samples=2),
    )
    val_ds = Dataset(
        data=val_files,
        transform=get_val_transforms(spatial_size=spatial_size),
    )
    holdout_ds = Dataset(
        data=holdout_files,
        transform=get_val_transforms(spatial_size=None),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=list_data_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=list_data_collate,
    )
    holdout_loader = DataLoader(
        holdout_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=list_data_collate,
    )

    loss_fn = DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    optimizer = torch.optim.Adam(trainable, lr=learning_rate, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=learning_rate * 0.1)
    scaler = GradScaler("cuda", enabled=use_amp)
    writer = SummaryWriter(log_dir=str(log_dir))

    best_metric = -1.0
    best_path = output_dir / "best_model.pt"
    last_path = output_dir / "last_model.pt"
    split_path = output_dir / "finetune_split.json"
    split_path.write_text(
        json.dumps(
            {
                "train_fit": [d["case_id"] for d in fit_files],
                "train_val_monitor": [d["case_id"] for d in val_files],
                "holdout_test": [d["case_id"] for d in holdout_files],
                "seed": seed,
                "n_holdout": len(holdout_files),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    meta: dict[str, Any] = {
        "stage": "real_finetune",
        "pretrained_checkpoint": str(pretrained_checkpoint),
        "pretrained_meta": pre_meta,
        "corrected_root": str(corrected_root),
        "n_train_fit": len(fit_files),
        "n_val_monitor": len(val_files),
        "n_holdout_test": len(holdout_files),
        "holdout_ids": [d["case_id"] for d in holdout_files],
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "pretrain_lr": pretrain_lr,
        "freeze_encoder": freeze_encoder,
        "freeze_encoder_stages": freeze_encoder_stages if freeze_encoder else 0,
        "frozen_param_count": len(frozen_names),
        "regions": list(OUT_REGION_NAMES),
        "amp": use_amp,
        "device": device_str,
    }

    logger.info(
        "Fine-tuning: epochs=%d lr=%.2e (pretrain_lr=%.2e) freeze_encoder=%s "
        "fit=%d val_mon=%d holdout=%d device=%s",
        max_epochs,
        learning_rate,
        pretrain_lr,
        freeze_encoder,
        len(fit_files),
        len(val_files),
        len(holdout_files),
        device_str,
    )

    for epoch in range(1, max_epochs + 1):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device_t,
            scaler,
            use_amp=use_amp,
        )
        val_loss, val_dice, per_region = validate_one_epoch(
            model,
            val_loader,
            loss_fn,
            device_t,
            use_amp=use_amp,
        )
        scheduler.step()
        lr_now = float(optimizer.param_groups[0]["lr"])
        elapsed = time.perf_counter() - t0

        writer.add_scalar("finetune/loss_train", train_loss, epoch)
        writer.add_scalar("finetune/loss_val", val_loss, epoch)
        writer.add_scalar("finetune/dice_val", val_dice, epoch)
        writer.add_scalar("finetune/lr", lr_now, epoch)
        for name, score in per_region.items():
            writer.add_scalar(f"finetune/dice_val_{name}", score, epoch)

        logger.info(
            "FT epoch %d/%d  train=%.4f  val=%.4f  dice=%.4f  "
            "ET=%.3f TC=%.3f WT=%.3f  lr=%.2e  (%.1fs)",
            epoch,
            max_epochs,
            train_loss,
            val_loss,
            val_dice,
            per_region.get("et", 0.0),
            per_region.get("tc", 0.0),
            per_region.get("wt", 0.0),
            lr_now,
            elapsed,
        )

        save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=best_metric,
            meta=meta,
        )
        if val_dice > best_metric:
            best_metric = val_dice
            meta_best = {**meta, "best_epoch": epoch, "best_val_dice": best_metric}
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_metric,
                meta=meta_best,
            )
            logger.info("New best fine-tune val Dice=%.4f → %s", best_metric, best_path)

    writer.close()
    if not best_path.is_file():
        save_checkpoint(
            best_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=max_epochs,
            best_metric=best_metric,
            meta=meta,
        )

    # Final evaluation on holdout test set (never used in training).
    holdout_csv = output_dir / "holdout_test_metrics.csv"
    holdout_rows = _eval_holdout_cases(
        model,
        holdout_loader,
        holdout_files,
        device_t,
        use_amp=use_amp,
        loss_fn=loss_fn,
    )
    pd.DataFrame(holdout_rows).to_csv(holdout_csv, index=False)
    if holdout_rows:
        mean_dice = float(
            sum(r["dice_mean"] for r in holdout_rows) / len(holdout_rows)
        )
        logger.info(
            "Holdout test mean Dice=%.4f over %d cases → %s",
            mean_dice,
            len(holdout_rows),
            holdout_csv,
        )

    summary = {
        **meta,
        "best_val_dice": best_metric,
        "holdout_csv": str(holdout_csv),
        "model_info": describe_model(model),
    }
    (output_dir / "finetune_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    return {
        "best_model": best_path,
        "last_model": last_path,
        "holdout_csv": holdout_csv,
        "split_json": split_path,
    }


@torch.no_grad()
def _eval_holdout_cases(
    model: torch.nn.Module,
    loader: DataLoader,
    file_dicts: Sequence[dict],
    device: torch.device,
    *,
    use_amp: bool,
    loss_fn: DiceLoss,
) -> list[dict[str, Any]]:
    """Per-case Dice on the holdout test set (center-crop-free val transforms)."""
    from monai.inferers import sliding_window_inference
    from torch.amp import autocast

    model.eval()
    rows: list[dict[str, Any]] = []
    # Loader may use full-volume transforms; use sliding window for safety.
    for batch, meta in zip(loader, file_dicts):
        case_id = str(batch.get("case_id", [meta.get("case_id", "?")])[0])
        images = batch[IMAGE_KEY].to(device)
        labels = batch[LABEL_KEY].to(device)
        regions = exclusive_labels_to_regions(labels)

        def _pred(x: torch.Tensor) -> torch.Tensor:
            with autocast(device_type=device.type, enabled=use_amp):
                return model(x)

        with autocast(device_type=device.type, enabled=use_amp):
            if tuple(images.shape[2:]) == tuple(regions.shape[2:]) and max(images.shape[2:]) <= 128:
                logits = model(images)
            else:
                logits = sliding_window_inference(
                    images,
                    roi_size=(96, 96, 96),
                    sw_batch_size=1,
                    predictor=_pred,
                    overlap=0.5,
                )
            loss = loss_fn(logits, regions)
            probs = torch.sigmoid(logits)

        pred = (probs > 0.5).float()
        dims = tuple(range(2, pred.ndim))
        inter = torch.sum(pred * regions, dim=dims)
        denom = torch.sum(pred, dim=dims) + torch.sum(regions, dim=dims)
        dice = (2.0 * inter + 1e-5) / (denom + 1e-5)  # (B, 3)
        d = dice[0]
        row = {
            "case_id": case_id,
            "loss": float(loss.item()),
            "dice_et": float(d[0].item()),
            "dice_tc": float(d[1].item()),
            "dice_wt": float(d[2].item()),
            "dice_mean": float(d.mean().item()),
        }
        rows.append(row)
        logger.info(
            "Holdout %s  Dice ET=%.3f TC=%.3f WT=%.3f mean=%.3f",
            case_id,
            row["dice_et"],
            row["dice_tc"],
            row["dice_wt"],
            row["dice_mean"],
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        from config import load_config

        cfg = load_config()
        default_data = str(cfg.paths.processed / "real_corrected")
        default_ckpt = str(cfg.paths.checkpoints / "brats_pretrain" / "best_model.pt")
        default_out = str(cfg.paths.checkpoints / "real_finetune")
        default_epochs = cfg.finetune.epochs
        default_bs = cfg.finetune.batch_size
        default_lr = cfg.finetune.learning_rate
        default_pre_lr = cfg.pretrain.learning_rate
    except Exception:  # noqa: BLE001
        default_data = "data/processed/real_corrected"
        default_ckpt = "models/checkpoints/brats_pretrain/best_model.pt"
        default_out = "models/checkpoints/real_finetune"
        default_epochs = 30
        default_bs = 1
        default_lr = DEFAULT_FINETUNE_LR
        default_pre_lr = DEFAULT_PRETRAIN_LR

    smoke = Path("models/checkpoints/brats_pretrain_smoke/best_model.pt")
    if not Path(default_ckpt).is_file() and smoke.is_file():
        default_ckpt = str(smoke)

    p = argparse.ArgumentParser(
        description="Fine-tune BraTS SegResNet on manually corrected real cases"
    )
    p.add_argument("--corrected-root", default=default_data)
    p.add_argument("--checkpoint", default=default_ckpt)
    p.add_argument("--output-dir", default=default_out)
    p.add_argument("--epochs", type=int, default=default_epochs)
    p.add_argument("--batch-size", type=int, default=default_bs)
    p.add_argument("--lr", type=float, default=default_lr, help="Fine-tune LR (<< pretrain)")
    p.add_argument("--pretrain-lr", type=float, default=default_pre_lr)
    p.add_argument("--n-holdout", type=int, default=4, help="Test holdout size (aim 3–5)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze convInit + early down_layers during fine-tuning",
    )
    p.add_argument("--freeze-encoder-stages", type=int, default=2)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    finetune_real_cases(
        args.corrected_root,
        args.checkpoint,
        args.output_dir,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        pretrain_lr=args.pretrain_lr,
        n_holdout=args.n_holdout,
        seed=args.seed,
        freeze_encoder=args.freeze_encoder,
        freeze_encoder_stages=args.freeze_encoder_stages,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
