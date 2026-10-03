"""
Side-by-side check: does a real patient look like BraTS *to the model*?

Loads one BraTS case and one real case through the exact transforms used in
training / inference (Orientation RAS → 1 mm → z-score), then saves a PNG grid
of mid-slices (axial / coronal / sagittal) per channel. Look for:

  * Orientation — in every column, BraTS and real brains should face the same
    way (nose to the same side, top of head up). If BraTS is mirrored, upside
    down, or shown in a different plane, training orientation ≠ inference.
  * Channel order — the row labelled FLAIR should look like FLAIR in BOTH
    (dark CSF/ventricles, bright edema); T2 = bright CSF; T1 = dark CSF,
    bright white matter. If a BraTS row looks like a different sequence than
    its label, the H5 channel order assumption is wrong.
  * Background — real background should be 0 (black) like BraTS; a visible
    skull/scalp ring on the real case means skull-stripping failed.

Usage (Colab, from the repo root)::

    python -m src.segmentation.check_orientation \
        --brats_case /content/drive/MyDrive/capstone/processed/brats_nifti/volume_1 \
        --real_case  /content/drive/MyDrive/capstone/processed/real_patients/<study_id> \
        --out orientation_check.png

``--brats_case`` is a cached H5→NIfTI folder ({flair,t1,t1c,t2,mask}.nii.gz).
``--real_case`` is a preprocessed study folder (with 05_isotropic_1mm/).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

from .dataset import IMAGE_KEY, LABEL_KEY, MODALITIES, get_val_transforms
from .pseudo_label import _meta_affine, discover_real_study_modalities, get_infer_transforms

logger = logging.getLogger(__name__)

# BraTS rows also show t1c (not a model input) to help identify the channels.
BRATS_DISPLAY = ("flair", "t1", "t1c", "t2")


def _np(x) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _mid_slices(vol: np.ndarray, center: tuple[int, int, int]):
    """Return (axial, coronal, sagittal) 2-D slices for a RAS-ordered volume.

    Each slice is transposed so that, with origin='lower':
      axial    → x (R) horizontal, y (A) up
      coronal  → x (R) horizontal, z (S) up
      sagittal → y (A) horizontal, z (S) up
    """
    cx, cy, cz = center
    return vol[:, :, cz].T, vol[:, cy, :].T, vol[cx, :, :].T


def _brain_center(vol4: np.ndarray) -> tuple[int, int, int]:
    """Center of the non-zero (brain) region, so slices cut through the brain."""
    nz = np.argwhere(np.any(vol4 != 0, axis=0))
    if len(nz) == 0:
        return tuple(s // 2 for s in vol4.shape[1:])  # type: ignore[return-value]
    return tuple(int(v) for v in (nz.min(0) + nz.max(0)) // 2)  # type: ignore[return-value]


def _describe(name: str, paths: list[str], image, label=None) -> dict:
    raw = nib.load(paths[0])
    info = {
        "name": name,
        "file_axcodes": "".join(nib.aff2axcodes(raw.affine)),
        "file_affine_is_identity": bool(np.allclose(raw.affine, np.eye(4))),
        "file_shape": tuple(int(s) for s in raw.shape[:3]),
        "model_shape": tuple(int(s) for s in image.shape[1:]),
        "model_axcodes": "".join(nib.aff2axcodes(_meta_affine(image))),
        "background_fraction": float(np.mean(np.all(_np(image) == 0, axis=0))),
    }
    if label is not None:
        info["label_values"] = sorted(int(v) for v in np.unique(_np(label)))
    return info


def load_brats(case_dir: Path):
    names = [m for m in BRATS_DISPLAY if (case_dir / f"{m}.nii.gz").is_file()]
    paths = [str(case_dir / f"{m}.nii.gz") for m in names]
    label = case_dir / "mask.nii.gz"
    if not label.is_file():
        raise FileNotFoundError(f"No mask.nii.gz in {case_dir}")
    batch = get_val_transforms(spatial_size=None)({IMAGE_KEY: paths, LABEL_KEY: str(label)})
    return names, paths, batch[IMAGE_KEY], batch[LABEL_KEY]


def load_real(case_dir: Path):
    mods = discover_real_study_modalities(case_dir)
    if mods is None:
        raise FileNotFoundError(f"{case_dir} is missing one of {list(MODALITIES)}")
    paths = [str(mods[m]) for m in MODALITIES]
    batch = get_infer_transforms()({IMAGE_KEY: paths})
    return list(MODALITIES), paths, batch[IMAGE_KEY]


def make_figure(brats, real, out_png: Path) -> Path:
    b_names, _bp, b_img, b_lab = brats
    r_names, _rp, r_img = real
    b_vol, r_vol, b_seg = _np(b_img), _np(r_img), _np(b_lab)[0]
    b_c, r_c = _brain_center(b_vol), _brain_center(r_vol)

    rows = [("BraTS", n, b_vol[i], b_c, b_seg) for i, n in enumerate(b_names)]
    rows += [("Real", n, r_vol[i], r_c, None) for i, n in enumerate(r_names)]
    planes = (
        ("Axial", "R →", "A →"),
        ("Coronal", "R →", "S →"),
        ("Sagittal", "A →", "S →"),
    )

    fig, axes = plt.subplots(len(rows), 3, figsize=(10, 3.1 * len(rows)))
    for r, (src, name, vol, center, seg) in enumerate(rows):
        lo, hi = np.percentile(vol[vol != 0], [1, 99]) if np.any(vol != 0) else (0, 1)
        segs = _mid_slices(seg, center) if seg is not None else (None, None, None)
        for c, ((plane, xl, yl), sl) in enumerate(zip(planes, _mid_slices(vol, center))):
            ax = axes[r, c]
            ax.imshow(sl, cmap="gray", origin="lower", vmin=lo, vmax=hi)
            if segs[c] is not None and np.any(segs[c] > 0):
                ax.contour(segs[c] > 0, levels=[0.5], colors="red", linewidths=0.8, origin="lower")
            ax.set_title(f"{src} · {name.upper()} · {plane}", fontsize=9,
                         color="#1F3A5F" if src == "BraTS" else "#7A1F1F")
            ax.set_xlabel(xl, fontsize=8)
            ax.set_ylabel(yl, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        "Model's-eye view (after Orientation RAS + 1 mm + z-score)\n"
        "Same plane should face the same way in BraTS and Real. Red = BraTS tumor label.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="BraTS vs real: orientation and channel check")
    p.add_argument("--brats_case", required=True, type=Path, help="Cached BraTS NIfTI case folder")
    p.add_argument("--real_case", required=True, type=Path, help="Preprocessed real study folder")
    p.add_argument("--out", default=Path("orientation_check.png"), type=Path)
    args = p.parse_args(argv)

    brats = load_brats(args.brats_case)
    real = load_real(args.real_case)

    print("\n=== What the files and the model see ===")
    for info in (_describe("BraTS", brats[1], brats[2], brats[3]), _describe("Real", real[1], real[2])):
        for k, v in info.items():
            print(f"  {k:24s} {v}")
        print()
    print("If BraTS 'file_affine_is_identity' is True, its true orientation is unknown:")
    print("compare the PNG rows below to see whether it matches the real scan.\n")

    out = make_figure(brats, real, args.out)
    print(f"Saved → {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
