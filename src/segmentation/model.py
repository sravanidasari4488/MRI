"""MONAI segmentation backbones for BraTS multi-modal MRI."""

from __future__ import annotations

from typing import Literal

import torch
from monai.networks.nets import SegResNet, UNet
from torch import nn

# Input channel order must match stacked NIfTI modalities in ``dataset.py`` /
# ``h5_to_nifti.MODALITY_ORDER``: FLAIR, T1, T1c, T2.
IN_CHANNELS = 4
IN_MODALITIES: tuple[str, ...] = ("flair", "t1", "t1c", "t2")

# BraTS *region* heads (overlapping, typically trained with sigmoid + Dice):
#   0 → ET  enhancing tumor          (original label 4 / remapped 3)
#   1 → TC  tumor core               (labels 1 ∪ 4)
#   2 → WT  whole tumor / edema+core (labels 1 ∪ 2 ∪ 4)
OUT_CHANNELS = 3
OUT_REGION_NAMES: tuple[str, ...] = ("et", "tc", "wt")

# Exclusive voxel labels after ``MapLabelValued`` in dataset.py:
#   0 background, 1 channel0/NCR, 2 channel1/ED, 3 channel2/ET
EXCLUSIVE_LABEL_NAMES: tuple[str, ...] = ("bg", "ncr", "ed", "et")

ArchitectureName = Literal["segresnet", "unet"]


def build_segresnet(
    *,
    in_channels: int = IN_CHANNELS,
    out_channels: int = OUT_CHANNELS,
    init_filters: int = 16,
    dropout_prob: float = 0.2,
) -> SegResNet:
    """
    SegResNet configured for BraTS: 4 MRI channels → 3 region logits.

    Use ``sigmoid`` activation (or ``DiceLoss(sigmoid=True)``) when
    ``out_channels == 3`` for overlapping ET / TC / WT targets.
    """
    return SegResNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        init_filters=init_filters,
        dropout_prob=dropout_prob,
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
    )


def build_unet(
    *,
    in_channels: int = IN_CHANNELS,
    out_channels: int = OUT_CHANNELS,
) -> UNet:
    """3D UNet fallback with the same BraTS I/O channel counts."""
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )


def build_model(
    architecture: ArchitectureName = "segresnet",
    *,
    in_channels: int = IN_CHANNELS,
    out_channels: int = OUT_CHANNELS,
    **kwargs,
) -> nn.Module:
    """
    Factory for BraTS segmentation networks.

    Parameters
    ----------
    architecture:
        ``\"segresnet\"`` (default) or ``\"unet\"``.
    in_channels:
        Must be 4 for T1 / T1c / T2 / FLAIR (order as stacked in the dataset).
    out_channels:
        ``3`` for BraTS region heads (ET, TC, WT). Use ``4`` only if training
        exclusive softmax classes (bg + NCR + ED + ET).
    """
    if in_channels != IN_CHANNELS:
        # Allow override but keep the BraTS default explicit in call sites.
        pass
    if architecture == "segresnet":
        return build_segresnet(
            in_channels=in_channels,
            out_channels=out_channels,
            **{k: v for k, v in kwargs.items() if k in {"init_filters", "dropout_prob"}},
        )
    if architecture == "unet":
        return build_unet(in_channels=in_channels, out_channels=out_channels)
    raise ValueError(f"Unknown architecture {architecture!r}; use 'segresnet' or 'unet'")


def describe_model(model: nn.Module) -> dict:
    """Return a small summary dict (params + I/O hints) for logging/checkpoints."""
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    meta: dict = {
        "class_name": model.__class__.__name__,
        "trainable_params": int(n_params),
        "in_modalities": list(IN_MODALITIES),
        "out_regions": list(OUT_REGION_NAMES),
        "in_channels": IN_CHANNELS,
        "out_channels": OUT_CHANNELS,
    }
    if isinstance(model, SegResNet):
        meta["architecture"] = "segresnet"
    elif isinstance(model, UNet):
        meta["architecture"] = "unet"
    else:
        meta["architecture"] = model.__class__.__name__.lower()
    return meta


def forward_regions(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """
    Run the network and return raw logits ``(B, 3, H, W, D)``.

    Callers apply ``sigmoid`` for ET/TC/WT multi-label decoding, or
    ``softmax`` if the head was built with exclusive ``out_channels=4``.
    """
    return model(images)
