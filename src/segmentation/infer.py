"""Run MONAI segmentation inference on a preprocessed multimodal case."""

from __future__ import annotations

from pathlib import Path

from .dataset import MODALITIES
from .model import IN_CHANNELS, OUT_CHANNELS, build_model


def run_inference(
    checkpoint: str | Path,
    image_paths: dict[str, str | Path],
    output_mask: str | Path,
    *,
    device: str | None = None,
) -> Path:
    """
    Infer tumor labels for one case.

    ``image_paths`` keys are modality names, e.g.
    ``{"flair": ..., "t1": ..., "t2": ...}`` (order ``MODALITIES``).
    Writes a multi-label (or multi-channel) NIfTI mask.
    """
    import numpy as np
    import nibabel as nib
    import torch
    from monai.inferers import sliding_window_inference

    checkpoint = Path(checkpoint)
    output_mask = Path(output_mask)
    output_mask.parent.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    arrays = []
    affine = None
    for key in MODALITIES:
        if key not in image_paths:
            raise KeyError(f"Missing modality '{key}' in image_paths (need {list(MODALITIES)})")
        img = nib.load(str(image_paths[key]))
        if affine is None:
            affine = img.affine
        arrays.append(np.asanyarray(img.dataobj, dtype=np.float32))

    volume = np.stack(arrays, axis=0)[None, ...]  # (1, C, H, W, D)
    tensor = torch.from_numpy(volume).to(device_t)

    model = build_model("segresnet", in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(
        device_t
    )
    blob = torch.load(checkpoint, map_location=device_t, weights_only=False)
    state = blob["model_state"] if isinstance(blob, dict) and "model_state" in blob else blob
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        logits = sliding_window_inference(
            tensor,
            roi_size=(96, 96, 96),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )
        # Region heads → BraTS-style exclusive map via argmax over sigmoid probs
        # is not ideal; keep legacy argmax for this thin helper.
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    nib.save(nib.Nifti1Image(pred, affine), str(output_mask))
    return output_mask
