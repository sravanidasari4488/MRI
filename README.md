# Brain Tumor 3D Reconstruction + Tract-Risk

Core pipeline for converting clinical MRI (DICOM) into tumor segmentations, validated 3D meshes, and white-matter tract proximity risk scores.

**In scope:** DICOM → preprocessing → BraTS pretrain → fine-tune on real data → 3D reconstruction → volume validation → tract-proximity risk → basic visualization.

**Stretch (later):** anatomy labeling, full tissue segmentation, polished XAI interface.

## Project layout

```
.
├── pyproject.toml
├── requirements.txt
├── README.md
├── data/
│   ├── brats/           # BraTS public training volumes
│   ├── real_patients/   # de-identified clinical DICOM / NIfTI
│   └── processed/       # intermediate NIfTI, masks, meshes, metrics
├── notebooks/           # exploratory analysis
└── src/
    ├── preprocessing/   # DICOM→NIfTI, N4, skull strip, registration
    ├── segmentation/    # MONAI train / infer (BraTS → fine-tune)
    ├── reconstruction/  # marching cubes, mesh volume
    ├── validation/      # ellipsoid reference, Bland–Altman
    ├── tracts/          # TractSeg, distance, risk tiers
    └── visualization/   # Plotly 3D views
```

Install in editable mode from the project root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -U pip
pip install -e .
# or: pip install -r requirements.txt && pip install -e .
```

## Pipeline stages

### 1. Preprocessing (`src/preprocessing`)

| Step | Purpose |
|------|---------|
| DICOM → NIfTI | Convert series with `dicom2nifti` / `pydicom` + `nibabel` |
| Bias correction | N4ITK via SimpleITK |
| Skull stripping | Brain mask for intensity / registration focus |
| Registration | Align T1ce / FLAIR / T2 / T1 (and DWI when available) to a common space |

Outputs land under `data/processed/<case_id>/`.

### 2. Segmentation (`src/segmentation`)

1. **Pretrain** a MONAI 3D U-Net (or equivalent) on BraTS (`data/brats/`).
2. **Fine-tune** on labeled real cases (`data/real_patients/` → processed masks).
3. **Infer** whole-tumor / enhancing / edema labels for new cases.

Checkpoint paths and configs live beside training scripts; inference writes NIfTI masks into `data/processed/`.

### 3. Reconstruction (`src/reconstruction`)

- Extract tumor surface with **marching cubes** (`scikit-image`).
- Build a mesh (`trimesh`) and compute **mesh / voxel volume** (mm³ / cm³).

### 4. Volume validation (`src/validation`)

- Compare segmented volume to an **ellipsoid** (or radiology) reference.
- Summarize agreement with **Bland–Altman** plots and bias / LoA stats (`pandas` / `scipy`).

### 5. Tract-proximity risk (`src/tracts`)

- Run **TractSeg** (+ DIPY utilities) to obtain major white-matter bundles.
- Compute **minimum distance** from tumor surface (or mask) to each tract.
- Map distances to a simple **risk class** (e.g. abutting / near / remote).

### 6. Visualization (`src/visualization`)

- Plotly 3D: MRI context (optional), tumor mesh, selected tracts, distance annotations.
- Notebooks under `notebooks/` for case review.

## Data directories

| Path | Contents |
|------|----------|
| `data/brats/` | Public BraTS NIfTI for pretraining |
| `data/real_patients/` | Clinical DICOM or converted NIfTI (PHI removed) |
| `data/processed/` | Preprocessed images, preds, meshes, CSVs |

A local BraTS archive may already exist at `Downloads/archive/BraTS2020_training_data`; symlink or copy into `data/brats/` as needed.

## Suggested workflow

```text
DICOM series
    → preprocessing.dicom_to_nifti / bias / skull / register
    → segmentation.infer (after BraTS pretrain + fine-tune)
    → reconstruction.marching_cubes + volume
    → validation.ellipsoid + bland_altman
    → tracts.tractseg + distance + risk
    → visualization.plotly_3d
```

## Stretch goals

- Anatomy / structure labeling beyond tumor classes  
- Full tissue / multi-label brain segmentation  
- Explainability (saliency / occlusion) and a polished review UI  

## Notes

- Keep PHI out of git; treat `data/real_patients/` as local-only.
- GPU recommended for MONAI training; CPU is fine for small reconstruction / Plotly demos.
- TractSeg and DIPY may need extra system deps (e.g. FSL-related tooling depending on your TractSeg install path)—see upstream docs if bundling fails.
