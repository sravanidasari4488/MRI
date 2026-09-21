"""MONAI BraTS training loop: DiceLoss, Adam, AMP, checkpoints, TensorBoard."""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from monai.data import DataLoader, Dataset, list_data_collate
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from .dataset import (
    IMAGE_KEY,
    LABEL_KEY,
    build_brats_data_dicts,
    get_train_transforms,
    get_val_transforms,
)
from .model import OUT_REGION_NAMES, build_model, describe_model

logger = logging.getLogger(__name__)


def _default_model(in_channels: int = 4, out_channels: int = 3):
    return build_model("segresnet", in_channels=in_channels, out_channels=out_channels)


def list_preprocessed_cases(processed_root: str | Path) -> list[Path]:
    """Return case folders under a processed dataset root that look trainable."""
    root = Path(processed_root)
    if not root.is_dir():
        return []
    cases: list[Path] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or path.name.startswith("."):
            continue
        iso = path / "05_isotropic_1mm"
        nifti_stage = path / "01_nifti"
        if iso.is_dir() or nifti_stage.is_dir() or any(path.glob("**/*.nii*")):
            cases.append(path)
    return cases


def split_train_val(
    data_dicts: Sequence[dict],
    *,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Split BraTS case dicts into train / validation (default **80/20**).

    Deterministic given ``seed``. Ensures at least one train case when possible.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    items = list(data_dicts)
    rng = random.Random(seed)
    rng.shuffle(items)

    n_val = int(round(len(items) * val_ratio)) if items else 0
    if items and n_val == 0 and val_ratio > 0:
        n_val = 1
    if n_val >= len(items) and len(items) > 1:
        n_val = len(items) - 1

    val_files = items[:n_val]
    train_files = items[n_val:]
    logger.info(
        "Train/val split 80/20-style (val_ratio=%.2f): train=%d val=%d (seed=%d)",
        val_ratio,
        len(train_files),
        len(val_files),
        seed,
    )
    return train_files, val_files


def exclusive_labels_to_regions(label: torch.Tensor) -> torch.Tensor:
    """
    Convert exclusive voxel labels to overlapping BraTS region channels.

    Expects labels remapped by the dataset transforms:
      0=bg, 1=NCR, 2=ED, 3=ET

    Returns float tensor ``(B, 3, H, W, D)`` ordered as ET, TC, WT
    (matching ``OUT_REGION_NAMES`` / SegResNet heads).
    """
    if label.ndim == 5 and label.shape[1] == 1:
        lab = label[:, 0]
    elif label.ndim == 4:
        lab = label
    else:
        raise ValueError(f"Unexpected label shape {tuple(label.shape)}")

    et = lab == 3
    tc = (lab == 1) | (lab == 3)
    wt = (lab == 1) | (lab == 2) | (lab == 3)
    return torch.stack([et, tc, wt], dim=1).to(dtype=torch.float32)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: DiceLoss,
    device: torch.device,
    scaler: GradScaler | None,
    *,
    use_amp: bool,
) -> float:
    model.train()
    running = 0.0
    n_steps = 0
    for batch in loader:
        images = batch[IMAGE_KEY].to(device)
        labels = batch[LABEL_KEY].to(device)
        regions = exclusive_labels_to_regions(labels)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, regions)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running += float(loss.detach().item())
        n_steps += 1

    return running / max(n_steps, 1)


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: DiceLoss,
    device: torch.device,
    *,
    use_amp: bool,
) -> tuple[float, float, dict[str, float]]:
    """Return ``(val_loss, mean_dice, per_region_dice)``."""
    model.eval()
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    running_loss = 0.0
    n_steps = 0
    region_scores: dict[str, list[float]] = {name: [] for name in OUT_REGION_NAMES}

    for batch in loader:
        images = batch[IMAGE_KEY].to(device)
        labels = batch[LABEL_KEY].to(device)
        regions = exclusive_labels_to_regions(labels)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = loss_fn(logits, regions)
            probs = torch.sigmoid(logits)

        running_loss += float(loss.item())
        n_steps += 1
        dice_metric(y_pred=(probs > 0.5).float(), y=regions)

        # Per-region batch dice for TensorBoard.
        dims = tuple(range(2, probs.ndim))
        pred_bin = (probs > 0.5).float()
        inter = torch.sum(pred_bin * regions, dim=dims)
        denom = torch.sum(pred_bin, dim=dims) + torch.sum(regions, dim=dims)
        per = (2.0 * inter + 1e-5) / (denom + 1e-5)  # (B, 3)
        for i, name in enumerate(OUT_REGION_NAMES):
            region_scores[name].append(float(per[:, i].mean().item()))

    agg = dice_metric.aggregate()
    dice_metric.reset()
    if torch.is_tensor(agg):
        mean_dice = float(agg.mean().item())
    else:
        mean_dice = float(agg)

    per_region = {
        name: (sum(vals) / len(vals) if vals else 0.0) for name, vals in region_scores.items()
    }
    return running_loss / max(n_steps, 1), mean_dice, per_region


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    epoch: int,
    best_metric: float,
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "best_metric": best_metric,
            "meta": meta,
            "model_info": describe_model(model),
        },
        path,
    )


def save_resume_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: GradScaler | None,
    epoch: int,
    best_metric: float,
    use_amp: bool,
) -> None:
    """
    Write a full training-resume snapshot (``checkpoint.pt``).

    Separate from ``best_model.pt`` / ``last_model.pt``, which remain the
    inference-oriented artifacts produced by :func:`save_checkpoint`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    blob: dict[str, Any] = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_metric": float(best_metric),
    }
    if use_amp and scaler is not None:
        blob["scaler_state_dict"] = scaler.state_dict()
    torch.save(blob, path)


def load_resume_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: GradScaler | None,
    device: torch.device,
) -> tuple[int, float]:
    """
    Restore model / optimizer / scheduler / scaler from ``checkpoint.pt``.

    Returns ``(start_epoch, best_metric)`` where ``start_epoch`` is the next
    epoch to run (``saved_epoch + 1``).
    """
    blob = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(blob, dict):
        raise ValueError(f"Resume checkpoint is not a dict: {path}")

    model_state = blob.get("model_state_dict", blob.get("model_state"))
    optim_state = blob.get("optimizer_state_dict", blob.get("optimizer_state"))
    sched_state = blob.get("scheduler_state_dict", blob.get("scheduler_state"))
    if model_state is None:
        raise KeyError(f"No model_state_dict in resume checkpoint: {path}")

    model.load_state_dict(model_state)
    if optim_state is not None:
        optimizer.load_state_dict(optim_state)
    if sched_state is not None:
        scheduler.load_state_dict(sched_state)

    scaler_state = blob.get("scaler_state_dict")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    completed_epoch = int(blob.get("epoch", 0))
    best_metric = float(blob.get("best_metric", -1.0))
    start_epoch = completed_epoch + 1
    return start_epoch, best_metric


def train_brats(
    data_dir: str | Path,
    output_dir: str | Path,
    *,
    max_epochs: int = 50,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    spatial_size: tuple[int, int, int] = (96, 96, 96),
    device: str | None = None,
    amp: bool | None = None,
    log_dir: str | Path | None = None,
    resume_from: str | Path | None = None,
) -> Path:
    """
    Pretrain SegResNet on cached BraTS NIfTI with a full MONAI training loop.

    - Loss: ``DiceLoss(sigmoid=True)`` on ET / TC / WT region maps
    - Optim: Adam + CosineAnnealingLR
    - AMP mixed precision when CUDA is available
    - Saves ``best_model.pt`` on best validation mean Dice
    - Saves resumable ``checkpoint.pt`` every epoch
    - TensorBoard logs under ``output_dir/tb`` (or ``log_dir``)
    - Cases split **80/20** train/val via :func:`split_train_val`
    - Optional ``resume_from`` path to a prior ``checkpoint.pt``
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(log_dir) if log_dir is not None else output_dir / "tb"
    log_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device
    device_t = torch.device(device_str)
    use_amp = bool(amp) if amp is not None else device_t.type == "cuda"
    if use_amp and device_t.type != "cuda":
        logger.warning("AMP requested but device=%s — disabling AMP", device_str)
        use_amp = False

    # Prefer converted NIfTI layout; fall back to scanning data_dir itself.
    nifti_root = data_dir
    data_dicts = []
    try:
        data_dicts = build_brats_data_dicts(nifti_root)
    except (NotADirectoryError, FileNotFoundError):
        data_dicts = []

    if not data_dicts:
        raise FileNotFoundError(
            f"No BraTS NIfTI cases with flair/t1/t1c/t2 + mask under {nifti_root}. "
            "Run preprocessing.h5_to_nifti first (cache at data/processed/brats_nifti)."
        )

    train_files, val_files = split_train_val(data_dicts, val_ratio=val_ratio, seed=seed)
    if not train_files:
        raise RuntimeError("Train split is empty — need more BraTS cases")

    train_ds = Dataset(
        data=train_files,
        transform=get_train_transforms(spatial_size=spatial_size, num_samples=2),
    )
    val_ds = Dataset(
        data=val_files if val_files else train_files[:1],
        transform=get_val_transforms(spatial_size=spatial_size),
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

    model = _default_model().to(device_t)
    loss_fn = DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=learning_rate * 0.01)
    scaler = GradScaler("cuda", enabled=use_amp)
    writer = SummaryWriter(log_dir=str(log_dir))

    best_metric = -1.0
    start_epoch = 1
    best_path = output_dir / "best_model.pt"
    last_path = output_dir / "last_model.pt"
    resume_path = output_dir / "checkpoint.pt"
    meta = {
        "stage": "brats_pretrain",
        "data_dir": str(data_dir),
        "n_train": len(train_files),
        "n_val": len(val_files),
        "val_ratio": val_ratio,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "loss": "DiceLoss(sigmoid=True)",
        "regions": list(OUT_REGION_NAMES),
        "amp": use_amp,
        "device": device_str,
    }

    if resume_from is not None:
        resume_file = Path(resume_from)
        if resume_file.is_file():
            start_epoch, best_metric = load_resume_checkpoint(
                resume_file,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler if use_amp else None,
                device=device_t,
            )
            logger.info(
                "Resuming training from epoch %d, best_metric=%.4f (loaded %s)",
                start_epoch,
                best_metric,
                resume_file,
            )
            if start_epoch > max_epochs:
                logger.warning(
                    "Resume start_epoch=%d exceeds max_epochs=%d — nothing to train",
                    start_epoch,
                    max_epochs,
                )
        else:
            logger.warning(
                "Resume checkpoint not found at %s — starting fresh from epoch 0",
                resume_file,
            )

    logger.info(
        "Starting BraTS training: epochs=%d (from %d) batch=%d lr=%s device=%s "
        "amp=%s train=%d val=%d",
        max_epochs,
        start_epoch,
        batch_size,
        learning_rate,
        device_str,
        use_amp,
        len(train_files),
        len(val_files),
    )

    for epoch in range(start_epoch, max_epochs + 1):
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

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("dice/val_mean", val_dice, epoch)
        writer.add_scalar("lr", lr_now, epoch)
        for name, score in per_region.items():
            writer.add_scalar(f"dice/val_{name}", score, epoch)

        logger.info(
            "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_dice=%.4f  "
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
            logger.info("New best val Dice=%.4f — saved %s", best_metric, best_path)

        # Full resumable snapshot every epoch (in addition to best/last).
        save_resume_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler if use_amp else None,
            epoch=epoch,
            best_metric=best_metric,
            use_amp=use_amp,
        )

    writer.close()
    if not best_path.is_file():
        # Degenerate val (e.g. empty) — keep last weights as best.
        save_checkpoint(
            best_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=max_epochs,
            best_metric=best_metric,
            meta=meta,
        )
    logger.info("BraTS training finished. Best val Dice=%.4f → %s", best_metric, best_path)
    return best_path


def fine_tune(
    pretrained_ckpt: str | Path,
    real_data_dir: str | Path,
    output_dir: str | Path,
    *,
    max_epochs: int = 30,
    learning_rate: float = 1e-5,
    batch_size: int = 1,
    device: str | None = None,
    freeze_encoder: bool = False,
    n_holdout: int = 4,
    **kwargs: Any,
) -> Path:
    """
    Fine-tune a BraTS-pretrained checkpoint on manually corrected real cases.

    Delegates to :func:`segmentation.finetune.finetune_real_cases` (lower LR,
    optional early-encoder freeze, 3–5 case holdout test set).
    """
    from .finetune import finetune_real_cases

    paths = finetune_real_cases(
        real_data_dir,
        pretrained_ckpt,
        output_dir,
        max_epochs=max_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        freeze_encoder=freeze_encoder,
        n_holdout=n_holdout,
        device=device,
        **kwargs,
    )
    return paths["best_model"]


def run_training_from_config(
    config_path: str | Path | None = None,
    *,
    preprocess: bool = False,
    real_limit: int | None = 100,
    brats_limit: int | None = None,
    skip_bias_correction: bool = False,
    skip_skull_strip: bool = False,
) -> dict[str, Path]:
    """
    Full training path wired to project config:

    1. (optional) Preprocess real-patient studies + BraTS H5→NIfTI cache
    2. Pretrain on ``data/processed/brats_nifti`` (80/20 split)
    3. Fine-tune hook on ``data/processed/real_patients``
    """
    from config import load_config
    from preprocessing.batch_pipeline import run_batch_brats, run_batch_real_patients

    cfg = load_config(config_path)
    paths = cfg.paths

    if preprocess:
        logger.info(
            "Preprocessing real patients from %s (limit=%s) and BraTS from %s",
            paths.raw_dicom,
            real_limit,
            paths.brats,
        )
        from preprocessing.h5_to_nifti import convert_brats_h5_directory, resolve_brats_h5_dir

        convert_brats_h5_directory(
            resolve_brats_h5_dir(paths.brats),
            paths.processed_brats_nifti,
            force=False,
            limit=brats_limit,
        )
        run_batch_real_patients(
            paths.raw_dicom,
            paths.processed_real_patients,
            limit=real_limit,
            skip_bias_correction=skip_bias_correction,
            skip_skull_strip=skip_skull_strip,
        )
        run_batch_brats(
            paths.brats,
            paths.processed_brats,
            limit=brats_limit,
            skip_bias_correction=skip_bias_correction,
            skip_skull_strip=skip_skull_strip,
        )

    brats_data = (
        paths.processed_brats_nifti
        if paths.processed_brats_nifti.is_dir()
        else paths.processed_brats
        if paths.processed_brats.is_dir()
        else paths.brats
    )
    # Prefer manually corrected masks; fall back to processed real studies.
    corrected = paths.processed / "real_corrected"
    real_data = corrected if corrected.is_dir() else paths.processed_real_patients

    pretrain_dir = paths.checkpoints / "brats_pretrain"
    finetune_dir = paths.checkpoints / "real_finetune"

    brats_ckpt = train_brats(
        brats_data,
        pretrain_dir,
        max_epochs=cfg.pretrain.epochs,
        batch_size=cfg.pretrain.batch_size,
        learning_rate=cfg.pretrain.learning_rate,
        val_ratio=0.2,
    )
    ft_ckpt = fine_tune(
        brats_ckpt,
        real_data,
        finetune_dir,
        max_epochs=cfg.finetune.epochs,
        batch_size=cfg.finetune.batch_size,
        learning_rate=cfg.finetune.learning_rate,
    )
    return {"brats_pretrain": brats_ckpt, "finetune": ft_ckpt}


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        from config import load_config

        cfg = load_config()
        default_data = str(cfg.paths.processed_brats_nifti)
        default_out = str(cfg.paths.checkpoints / "brats_pretrain")
        default_epochs = cfg.pretrain.epochs
        default_bs = cfg.pretrain.batch_size
        default_lr = cfg.pretrain.learning_rate
    except Exception:  # noqa: BLE001
        default_data = "data/processed/brats_nifti"
        default_out = "models/checkpoints/brats_pretrain"
        default_epochs = 50
        default_bs = 1
        default_lr = 1e-4

    p = argparse.ArgumentParser(description="Train BraTS SegResNet (MONAI)")
    p.add_argument("--data-dir", default=default_data)
    p.add_argument("--output-dir", default=default_out)
    p.add_argument("--epochs", type=int, default=default_epochs)
    p.add_argument("--batch-size", type=int, default=default_bs)
    p.add_argument("--lr", type=float, default=default_lr)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--resume-from",
        default=None,
        help="Path to checkpoint.pt to resume training (model/optim/scheduler/scaler)",
    )
    args = p.parse_args()

    train_brats(
        args.data_dir,
        args.output_dir,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_ratio=args.val_ratio,
        seed=args.seed,
        resume_from=args.resume_from,
    )
