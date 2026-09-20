"""Load and validate project configuration from ``config.yaml``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class PathsConfig(BaseModel):
    """Filesystem locations for data and model artifacts."""

    raw_dicom: Path = Field(description="Raw clinical DICOM (or unconverted) patient data")
    brats: Path = Field(description="BraTS training data root")
    processed: Path = Field(description="Preprocessed images, masks, meshes, metrics")
    checkpoints: Path = Field(description="Model checkpoint directory")

    @field_validator("raw_dicom", "brats", "processed", "checkpoints", mode="before")
    @classmethod
    def _coerce_path(cls, value: Any) -> Path:
        return Path(value)

    def resolve_against(self, root: Path) -> PathsConfig:
        """Return a copy with relative paths resolved against ``root``."""

        def _resolve(p: Path) -> Path:
            return p if p.is_absolute() else (root / p).resolve()

        return PathsConfig(
            raw_dicom=_resolve(self.raw_dicom),
            brats=_resolve(self.brats),
            processed=_resolve(self.processed),
            checkpoints=_resolve(self.checkpoints),
        )

    @property
    def processed_brats(self) -> Path:
        """BraTS cases after full preprocessing pipeline."""
        return self.processed / "brats"

    @property
    def processed_brats_nifti(self) -> Path:
        """One-shot H5→NIfTI cache (do not rebuild each training epoch)."""
        return self.processed / "brats_nifti"

    @property
    def processed_real_patients(self) -> Path:
        """Real-patient cases after preprocessing (fine-tune input)."""
        return self.processed / "real_patients"


class TrainHyperParams(BaseModel):
    """Shared training hyperparameters for one stage."""

    batch_size: int = Field(ge=1)
    learning_rate: float = Field(gt=0.0)
    epochs: int = Field(ge=1)


class AppConfig(BaseModel):
    """Top-level validated application config."""

    paths: PathsConfig
    pretrain: TrainHyperParams
    finetune: TrainHyperParams
    project_root: Path = Field(default=PROJECT_ROOT, exclude=True)

    @model_validator(mode="after")
    def _resolve_paths(self) -> AppConfig:
        object.__setattr__(
            self,
            "paths",
            self.paths.resolve_against(self.project_root),
        )
        return self


def load_config(path: str | Path | None = None) -> AppConfig:
    """
    Load ``config.yaml``, validate with Pydantic, and resolve relative paths.

    Parameters
    ----------
    path:
        Config file path. Defaults to ``<project_root>/config.yaml``.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}")

    root = config_path.resolve().parent
    return AppConfig(project_root=root, **raw)
