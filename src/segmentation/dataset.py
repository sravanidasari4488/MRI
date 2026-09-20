"""MONAI dataset + transforms for cached BraTS NIfTI volumes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from monai.data import CacheDataset, Dataset, list_data_collate
from monai.transforms import (
    CenterSpatialCropd,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapLabelValued,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    Spacingd,
    SpatialPadd,
)

logger = logging.getLogger(__name__)

# Matches ``preprocessing.h5_to_nifti.MODALITY_ORDER`` / on-disk filenames.
MODALITIES: tuple[str, ...] = ("flair", "t1", "t1c", "t2")
LABEL_KEY = "label"
IMAGE_KEY = "image"

# H5→NIfTI merge wrote channel0→1, channel1→2, channel2→4. Remap to contiguous
# class indices 1..3 for standard multi-class Dice / CrossEntropy.
_MASK_LABEL_FROM = (0, 1, 2, 4)
_MASK_LABEL_TO = (0, 1, 2, 3)


def build_brats_data_dicts(
    nifti_root: str | Path,
    *,
    modalities: Sequence[str] = MODALITIES,
    require_mask: bool = True,
) -> list[dict]:
    """
    Scan ``data/processed/brats_nifti`` (or similar) for case folders.

    Each dict has::

        {
          "image": [flair.nii.gz, t1.nii.gz, t1c.nii.gz, t2.nii.gz],
          "label": mask.nii.gz,
          "case_id": "BraTS20_Training_001",
        }

    ``LoadImaged`` stacks the image list into a multi-channel array.
    """
    nifti_root = Path(nifti_root)
    if not nifti_root.is_dir():
        raise NotADirectoryError(f"BraTS NIfTI root not found: {nifti_root}")

    records: list[dict] = []
    for case_dir in sorted(p for p in nifti_root.iterdir() if p.is_dir()):
        image_paths = [case_dir / f"{m}.nii.gz" for m in modalities]
        if not all(p.is_file() for p in image_paths):
            missing = [p.name for p in image_paths if not p.is_file()]
            logger.warning("Skipping %s — missing modalities %s", case_dir.name, missing)
            continue
        label_path = case_dir / "mask.nii.gz"
        if require_mask and not label_path.is_file():
            logger.warning("Skipping %s — missing mask.nii.gz", case_dir.name)
            continue
        item = {
            IMAGE_KEY: [str(p) for p in image_paths],
            "case_id": case_dir.name,
        }
        if label_path.is_file():
            item[LABEL_KEY] = str(label_path)
        records.append(item)

    logger.info("Found %d BraTS NIfTI cases under %s", len(records), nifti_root)
    return records


def get_train_transforms(
    *,
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0),
    spatial_size: tuple[int, int, int] = (96, 96, 96),
    num_samples: int = 2,
    pos: float = 1.0,
    neg: float = 1.0,
    remap_mask_labels: bool = True,
) -> Compose:
    """
    Training transform chain (with random positive/negative crops).

    LoadImaged → EnsureChannelFirstd → Orientationd → Spacingd →
    (optional label remap) → NormalizeIntensityd → SpatialPadd →
    RandCropByPosNegLabeld → EnsureTyped
    """
    keys = [IMAGE_KEY, LABEL_KEY]
    transforms: list = [
        LoadImaged(keys=keys, image_only=False),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(
            keys=keys,
            pixdim=pixdim,
            mode=("bilinear", "nearest"),
        ),
    ]
    if remap_mask_labels:
        transforms.append(
            MapLabelValued(
                keys=[LABEL_KEY],
                orig_labels=list(_MASK_LABEL_FROM),
                target_labels=list(_MASK_LABEL_TO),
            )
        )
    transforms.extend(
        [
            NormalizeIntensityd(keys=[IMAGE_KEY], nonzero=True, channel_wise=True),
            # Guarantee crop ROI fits even if a volume is slightly smaller.
            SpatialPadd(keys=keys, spatial_size=spatial_size),
            RandCropByPosNegLabeld(
                keys=keys,
                label_key=LABEL_KEY,
                spatial_size=spatial_size,
                pos=pos,
                neg=neg,
                num_samples=num_samples,
                image_key=IMAGE_KEY,
            ),
            EnsureTyped(keys=keys),
        ]
    )
    return Compose(transforms)


def get_val_transforms(
    *,
    pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0),
    spatial_size: tuple[int, int, int] | None = (96, 96, 96),
    remap_mask_labels: bool = True,
) -> Compose:
    """
    Deterministic validation transform chain (no random crops).

    Same load / spacing / normalize path as training. If ``spatial_size`` is
    set, applies ``SpatialPadd`` + ``CenterSpatialCropd`` for a fixed ROI;
    full-volume sliding-window inference can set ``spatial_size=None``.
    """
    keys = [IMAGE_KEY, LABEL_KEY]
    transforms: list = [
        LoadImaged(keys=keys, image_only=False),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(
            keys=keys,
            pixdim=pixdim,
            mode=("bilinear", "nearest"),
        ),
    ]
    if remap_mask_labels:
        transforms.append(
            MapLabelValued(
                keys=[LABEL_KEY],
                orig_labels=list(_MASK_LABEL_FROM),
                target_labels=list(_MASK_LABEL_TO),
            )
        )
    transforms.append(
        NormalizeIntensityd(keys=[IMAGE_KEY], nonzero=True, channel_wise=True)
    )
    if spatial_size is not None:
        transforms.append(SpatialPadd(keys=keys, spatial_size=spatial_size))
        transforms.append(CenterSpatialCropd(keys=keys, roi_size=spatial_size))
    transforms.append(EnsureTyped(keys=keys))
    return Compose(transforms)


def split_brats_dicts(
    data_dicts: list[dict],
    *,
    val_frac: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Deterministic train/val split by case order after shuffling with ``seed``."""
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be in (0, 1)")
    import random

    items = list(data_dicts)
    rng = random.Random(seed)
    rng.shuffle(items)
    n_val = max(1, int(round(len(items) * val_frac))) if items else 0
    val = items[:n_val]
    train = items[n_val:]
    if not train and val:
        # Tiny sets: keep at least one train case.
        train, val = val[:1], val[1:]
    logger.info("Split BraTS cases: train=%d val=%d", len(train), len(val))
    return train, val


def create_brats_datasets(
    nifti_root: str | Path,
    *,
    val_frac: float = 0.2,
    seed: int = 42,
    cache_rate: float = 0.0,
    train_spatial_size: tuple[int, int, int] = (96, 96, 96),
    val_spatial_size: tuple[int, int, int] | None = (96, 96, 96),
    num_samples: int = 2,
) -> tuple[Dataset, Dataset]:
    """
    Build MONAI train/val ``Dataset`` (or ``CacheDataset`` if ``cache_rate`` > 0).

    Returns ``(train_ds, val_ds)``.
    """
    data_dicts = build_brats_data_dicts(nifti_root)
    if not data_dicts:
        raise FileNotFoundError(f"No complete BraTS cases under {nifti_root}")

    train_files, val_files = split_brats_dicts(data_dicts, val_frac=val_frac, seed=seed)
    train_tf = get_train_transforms(
        spatial_size=train_spatial_size,
        num_samples=num_samples,
    )
    val_tf = get_val_transforms(spatial_size=val_spatial_size)

    if cache_rate > 0.0:
        train_ds: Dataset = CacheDataset(
            data=train_files,
            transform=train_tf,
            cache_rate=cache_rate,
            num_workers=0,
        )
        val_ds: Dataset = CacheDataset(
            data=val_files,
            transform=val_tf,
            cache_rate=min(1.0, cache_rate),
            num_workers=0,
        )
    else:
        train_ds = Dataset(data=train_files, transform=train_tf)
        val_ds = Dataset(data=val_files, transform=val_tf)

    return train_ds, val_ds


def create_brats_dataloaders(
    nifti_root: str | Path,
    *,
    batch_size: int = 1,
    val_frac: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    cache_rate: float = 0.0,
    **dataset_kwargs,
):
    """Convenience wrapper returning ``(train_loader, val_loader)``."""
    from torch.utils.data import DataLoader

    train_ds, val_ds = create_brats_datasets(
        nifti_root,
        val_frac=val_frac,
        seed=seed,
        cache_rate=cache_rate,
        **dataset_kwargs,
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
    return train_loader, val_loader
