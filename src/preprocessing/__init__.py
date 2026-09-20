"""DICOM conversion, bias correction, skull stripping, and registration."""

from .dicom_to_nifti import convert_dicom_series
from .dicom_loader import (
    classify_modality,
    group_series_by_uid,
    load_dicom_directory,
    load_dicom_study,
)
from .bias_correction import n4_bias_correct
from .skull_strip import skull_strip
from .registration import (
    preprocess_nifti_modalities,
    preprocess_patient_study,
    register_modalities_to_reference,
    register_to_reference,
    resample_to_isotropic,
)
from .batch_pipeline import (
    run_all_batches,
    run_batch_brats,
    run_batch_real_patients,
)
from .h5_to_nifti import convert_brats_h5_directory, resolve_brats_h5_dir

__all__ = [
    "convert_dicom_series",
    "classify_modality",
    "group_series_by_uid",
    "load_dicom_directory",
    "load_dicom_study",
    "n4_bias_correct",
    "skull_strip",
    "register_to_reference",
    "register_modalities_to_reference",
    "resample_to_isotropic",
    "preprocess_nifti_modalities",
    "preprocess_patient_study",
    "run_batch_real_patients",
    "run_batch_brats",
    "run_all_batches",
    "convert_brats_h5_directory",
    "resolve_brats_h5_dir",
]
