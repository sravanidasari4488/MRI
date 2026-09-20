"""Run MONAI segmentation inference on a preprocessed multimodal case."""

from __future__ import annotations

from pathlib import Path


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
    ``{"t1": ..., "t1ce": ..., "t2": ..., "flair": ...}``.
    Writes a multi-label (or multi-channel) NIfTI mask.
    """
    import numpy as np
    import nibabel as nib
    import torch
    from monai.networks.nets import UNet
    from monai.inferers import sliding_window_inference

    checkpoint = Path(checkpoint)
    output_mask = Path(output_mask)
    output_mask.parent.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    modalities = ["t1", "t1ce", "t2", "flair"]
    arrays = []
    affine = None
    for key in modalities:
        if key not in image_paths:
            raise KeyError(f"Missing modality '{key}' in image_paths")
        img = nib.load(str(image_paths[key]))
        if affine is None:
            affine = img.affine
        arrays.append(np.asanyarray(img.dataobj, dtype=np.float32))

    volume = np.stack(arrays, axis=0)[None, ...]  # (1, C, H, W, D) or similar
    tensor = torch.from_numpy(volume).to(device)

    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    blob = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(blob["model_state"])
    model.eval()

    with torch.no_grad():
        logits = sliding_window_inference(
            tensor,
            roi_size=(96, 96, 96),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    nib.save(nib.Nifti1Image(pred, affine), str(output_mask))
    return output_mask
