"""Unit tests for DICOM series → t1 / t1c / t2 / flair classification."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow ``python tests/test_dicom_modality_classify.py`` from repo root.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pydicom.dataset import Dataset

from preprocessing.dicom_loader import (
    SeriesMeta,
    classify_modality,
    contrast_bolus_agent_present,
    refine_t1c_across_study,
)

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None  # type: ignore[assignment]


# Known descriptions from the reported Philips-style study.
KNOWN_SERIES_CASES: list[tuple[str, str]] = [
    ("eT1W_SE", "t1"),
    ("T1W_SE", "t1"),
    ("T1W_IR", "t1"),
    ("eT2W_FLAIR SPIR CLEAR", "flair"),
    ("eT2W_TSE SENSE", "t2"),
    ("T2W_TSE", "t2"),
]


def test_known_study_series_descriptions() -> None:
    for description, expected in KNOWN_SERIES_CASES:
        assert classify_modality(description, None) == expected, (description, expected)


def test_flair_wins_over_embedded_t2() -> None:
    """FLAIR names often contain T2W; must not classify as t2."""
    assert classify_modality("eT2W_FLAIR SPIR CLEAR", None) == "flair"
    assert classify_modality("T2W_FLAIR", "brain") == "flair"


def test_t1_matches_vendor_t1w_variants() -> None:
    for desc in ("T1W_SE", "eT1W_SE", "T1W_IR", "t1w_ir", "T1_IR"):
        assert classify_modality(desc, None) == "t1", desc


def test_additional_modality_patterns() -> None:
    cases = [
        ("FLAIR", "flair"),
        ("T2 FLAIR", "flair"),
        ("t1_mprage", "t1"),
        ("MPRAGE", "t1"),
        ("SPGR", "t1"),
        ("BRAVO", "t1"),
        ("T1 CE", "t1c"),
        ("T1_post", "t1c"),
        ("t1 +c", "t1c"),
        ("Ax T1 post contrast", "t1c"),
        ("", "other"),
        (None, "other"),
    ]
    for description, expected in cases:
        assert classify_modality(description, None) == expected, (description, expected)


def test_contrast_bolus_promotes_t1_to_t1c() -> None:
    assert (
        classify_modality("T1W_SE", None, has_contrast_bolus=True, contrast_bolus_agent="Gd")
        == "t1c"
    )
    assert classify_modality("eT1W_SE", "HEAD", has_contrast_bolus=True) == "t1c"


def test_contrast_bolus_agent_present() -> None:
    ds_empty = Dataset()
    assert contrast_bolus_agent_present([ds_empty]) == (False, None)

    ds_none = Dataset()
    ds_none.ContrastBolusAgent = "None"
    assert contrast_bolus_agent_present([ds_none])[0] is False

    ds_gd = Dataset()
    ds_gd.ContrastBolusAgent = "Gadovist"
    ok, value = contrast_bolus_agent_present([ds_gd])
    assert ok is True
    assert value == "Gadovist"


def test_refine_t1c_warns_when_undetermined() -> None:
    series = [
        SeriesMeta(
            series_instance_uid="1",
            series_description="eT1W_SE",
            inferred_mri_contrast="t1",
        ),
        SeriesMeta(
            series_instance_uid="2",
            series_description="T1W_SE",
            inferred_mri_contrast="t1",
        ),
        SeriesMeta(
            series_instance_uid="3",
            series_description="T1W_IR",
            inferred_mri_contrast="t1",
        ),
    ]
    refine_t1c_across_study(series)
    assert all(s.inferred_mri_contrast == "t1" for s in series)
    assert any("T1c undetermined" in n for s in series for n in s.notes)


def test_refine_t1c_silent_when_t1c_present() -> None:
    series = [
        SeriesMeta(
            series_instance_uid="1",
            series_description="T1W_SE",
            inferred_mri_contrast="t1",
        ),
        SeriesMeta(
            series_instance_uid="2",
            series_description="T1W_SE post",
            inferred_mri_contrast="t1c",
            contrast_bolus_agent="Gd",
        ),
    ]
    refine_t1c_across_study(series)
    assert series[0].inferred_mri_contrast == "t1"
    assert series[1].inferred_mri_contrast == "t1c"


if __name__ == "__main__":
    failed = 0
    print("=== Known study series descriptions ===")
    for desc, expected in KNOWN_SERIES_CASES:
        got = classify_modality(desc, None)
        status = "OK" if got == expected else "FAIL"
        if got != expected:
            failed += 1
        print(f"  {status}: {desc!r} -> {got!r} (expected {expected!r})")

    print("=== Extra checks ===")
    for name, fn in [
        ("known_batch", test_known_study_series_descriptions),
        ("flair_over_t2", test_flair_wins_over_embedded_t2),
        ("t1_variants", test_t1_matches_vendor_t1w_variants),
        ("extra_patterns", test_additional_modality_patterns),
        ("bolus_promote", test_contrast_bolus_promotes_t1_to_t1c),
        ("bolus_tag", test_contrast_bolus_agent_present),
        ("t1c_warn", test_refine_t1c_warns_when_undetermined),
        ("t1c_ok", test_refine_t1c_silent_when_t1c_present),
    ]:
        try:
            fn()
            print(f"  OK: {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL: {name}: {exc}")

    if pytest is not None:
        print("(pytest is available; prefer: pytest tests/test_dicom_modality_classify.py)")
    raise SystemExit(failed)
