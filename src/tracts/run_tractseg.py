"""Wrap TractSeg for white-matter bundle segmentation.

Runs TractSeg (CLI or Python API) on preprocessed DTI / CSD peaks when
available. The structural MRI pipeline in this project does **not** produce
DTI; if no diffusion input is found, this module records a clear status note
and skips inference rather than inventing tracts.

Standard TractSeg bundles (CST, AF, OR, …) are written under
``bundle_segmentations/`` as binary NIfTI masks.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# Full TractSeg v1/v2 bundle list (72 tracts). Names match TractSeg outputs.
TRACTSEG_BUNDLES: tuple[str, ...] = (
    "AF_left",
    "AF_right",
    "ATR_left",
    "ATR_right",
    "CA",
    "CC_1",
    "CC_2",
    "CC_3",
    "CC_4",
    "CC_5",
    "CC_6",
    "CC_7",
    "CG_left",
    "CG_right",
    "CST_left",
    "CST_right",
    "MLF_left",
    "MLF_right",
    "FPT_left",
    "FPT_right",
    "FX_left",
    "FX_right",
    "ICP_left",
    "ICP_right",
    "IFO_left",
    "IFO_right",
    "ILF_left",
    "ILF_right",
    "MCP",
    "OR_left",
    "OR_right",
    "POPT_left",
    "POPT_right",
    "SCP_left",
    "SCP_right",
    "SLF_I_left",
    "SLF_I_right",
    "SLF_II_left",
    "SLF_II_right",
    "SLF_III_left",
    "SLF_III_right",
    "STR_left",
    "STR_right",
    "UF_left",
    "UF_right",
    "T_PREF_left",
    "T_PREF_right",
    "T_PREM_left",
    "T_PREM_right",
    "T_PREC_left",
    "T_PREC_right",
    "T_POSTC_left",
    "T_POSTC_right",
    "T_PAR_left",
    "T_PAR_right",
    "T_OCC_left",
    "T_OCC_right",
    "ST_FO_left",
    "ST_FO_right",
    "ST_PREF_left",
    "ST_PREF_right",
    "ST_PREM_left",
    "ST_PREM_right",
    "ST_PREC_left",
    "ST_PREC_right",
    "ST_POSTC_left",
    "ST_POSTC_right",
    "ST_PAR_left",
    "ST_PAR_right",
    "ST_OCC_left",
    "ST_OCC_right",
)

# Clinically highlighted subset (corticospinal, arcuate, optic radiation, …).
HIGHLIGHT_BUNDLES: tuple[str, ...] = (
    "CST_left",
    "CST_right",  # corticospinal tract
    "AF_left",
    "AF_right",  # arcuate fasciculus
    "OR_left",
    "OR_right",  # optic radiation
    "IFO_left",
    "IFO_right",  # inferior fronto-occipital
    "ILF_left",
    "ILF_right",  # inferior longitudinal
    "UF_left",
    "UF_right",  # uncinate
    "CG_left",
    "CG_right",  # cingulum
    "ATR_left",
    "ATR_right",  # anterior thalamic radiation
)

MISSING_DTI_NOTE = (
    "This module needs DTI (diffusion-weighted) input. The structural "
    "MRI preprocess pipeline (DICOM→NIfTI→N4→skull-strip→1 mm) does not "
    "produce DWI/DTI. Provide either (1) a raw DWI NIfTI + bvals/bvecs, or "
    "(2) MRtrix CSD peaks (peaks.nii.gz), then re-run TractSeg."
)

# Relative locations searched under a preprocessed case folder.
_DTI_DIR_CANDIDATES = (
    "dti",
    "DTI",
    "dwi",
    "DWI",
    "06_dti",
    "diffusion",
    "tractseg",
    "tractseg_input",
)

_PEAKS_NAMES = (
    "peaks.nii.gz",
    "peaks.nii",
    "Diffusion_mrtrix_peaks.nii.gz",
    "fod_peaks.nii.gz",
)

_DWI_NAMES = (
    "Diffusion.nii.gz",
    "diffusion.nii.gz",
    "dwi.nii.gz",
    "DWI.nii.gz",
    "data.nii.gz",
)


@dataclass
class DTIInputStatus:
    """Result of looking for diffusion data for TractSeg."""

    available: bool
    input_type: str | None = None  # "peaks" | "raw_dwi" | None
    input_path: str | None = None
    bvals: str | None = None
    bvecs: str | None = None
    case_dir: str | None = None
    message: str = ""
    searched_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TractSegResult:
    """Outcome of a TractSeg run (or a documented skip)."""

    status: str  # "ok" | "skipped_no_dti" | "failed"
    output_dir: str
    bundle_dir: str | None = None
    bundle_masks: dict[str, str] = field(default_factory=dict)
    highlight_masks: dict[str, str] = field(default_factory=dict)
    dti_status: dict[str, Any] = field(default_factory=dict)
    note: str = ""
    command: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _first_existing(paths: Sequence[Path]) -> Path | None:
    for p in paths:
        if p.is_file():
            return p
    return None


def find_dti_input(case_dir: str | Path) -> DTIInputStatus:
    """
    Locate TractSeg-compatible diffusion input under a preprocessed case.

    Preference order:
      1. CSD / MRtrix ``peaks.nii.gz`` (no ``--raw_diffusion_input``)
      2. Raw DWI + ``bvals`` / ``bvecs`` (``--raw_diffusion_input``)
    """
    case_dir = Path(case_dir)
    searched: list[str] = []
    search_roots = [case_dir] + [
        case_dir / sub for sub in _DTI_DIR_CANDIDATES if (case_dir / sub).is_dir()
    ]

    # Peaks first.
    peak_candidates: list[Path] = []
    for root in search_roots:
        for name in _PEAKS_NAMES:
            peak_candidates.append(root / name)
            searched.append(str(root / name))
    peaks = _first_existing(peak_candidates)
    if peaks is not None:
        return DTIInputStatus(
            available=True,
            input_type="peaks",
            input_path=str(peaks),
            case_dir=str(case_dir),
            message=f"Found CSD peaks: {peaks}",
            searched_paths=searched,
        )

    # Raw DWI + gradient tables.
    dwi_candidates: list[Path] = []
    for root in search_roots:
        for name in _DWI_NAMES:
            dwi_candidates.append(root / name)
            searched.append(str(root / name))
    dwi = _first_existing(dwi_candidates)
    if dwi is not None:
        parent = dwi.parent
        bvals = _first_existing(
            [
                parent / "bvals",
                parent / "bvals.txt",
                parent / f"{dwi.name.replace('.nii.gz', '').replace('.nii', '')}.bvals",
            ]
        )
        bvecs = _first_existing(
            [
                parent / "bvecs",
                parent / "bvecs.txt",
                parent / f"{dwi.name.replace('.nii.gz', '').replace('.nii', '')}.bvecs",
            ]
        )
        searched.extend(
            [str(parent / "bvals"), str(parent / "bvecs")]
        )
        if bvals is not None and bvecs is not None:
            return DTIInputStatus(
                available=True,
                input_type="raw_dwi",
                input_path=str(dwi),
                bvals=str(bvals),
                bvecs=str(bvecs),
                case_dir=str(case_dir),
                message=f"Found raw DWI + gradients: {dwi}",
                searched_paths=searched,
            )
        return DTIInputStatus(
            available=False,
            input_type=None,
            input_path=str(dwi),
            bvals=str(bvals) if bvals else None,
            bvecs=str(bvecs) if bvecs else None,
            case_dir=str(case_dir),
            message=(
                f"Found DWI at {dwi} but missing bvals/bvecs. "
                + MISSING_DTI_NOTE
            ),
            searched_paths=searched,
        )

    return DTIInputStatus(
        available=False,
        case_dir=str(case_dir),
        message=MISSING_DTI_NOTE,
        searched_paths=searched,
    )


def write_missing_dti_note(
    output_dir: str | Path,
    status: DTIInputStatus,
) -> Path:
    """Persist a README + JSON explaining that TractSeg needs DTI input."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "status": "skipped_no_dti",
        "module": "tracts.run_tractseg",
        "note": MISSING_DTI_NOTE,
        "dti_status": status.to_dict(),
        "expected_layout": {
            "peaks": "case_dir/dti/peaks.nii.gz  (preferred — MRtrix CSD peaks)",
            "raw_dwi": [
                "case_dir/dti/Diffusion.nii.gz",
                "case_dir/dti/bvals",
                "case_dir/dti/bvecs",
            ],
            "output": "case_dir/tractseg/bundle_segmentations/{BUNDLE}.nii.gz",
        },
        "standard_bundles": list(TRACTSEG_BUNDLES),
        "highlight_bundles": list(HIGHLIGHT_BUNDLES),
    }
    json_path = output_dir / "tractseg_status.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    readme = output_dir / "DTI_REQUIRED.md"
    readme.write_text(
        "\n".join(
            [
                "# TractSeg — DTI input required",
                "",
                MISSING_DTI_NOTE,
                "",
                "## Place diffusion data here",
                "",
                "```",
                "<case>/dti/peaks.nii.gz          # preferred (CSD peaks)",
                "# — or —",
                "<case>/dti/Diffusion.nii.gz",
                "<case>/dti/bvals",
                "<case>/dti/bvecs",
                "```",
                "",
                "## Then run",
                "",
                "```bash",
                "python -m tracts.run_tractseg --case-dir <case> -o <case>/tractseg",
                "```",
                "",
                "Output masks: `bundle_segmentations/CST_left.nii.gz`, "
                "`AF_right.nii.gz`, `OR_left.nii.gz`, …",
                "",
            ]
        ),
        encoding="utf-8",
    )
    logger.warning("%s", status.message)
    logger.info("Wrote missing-DTI documentation → %s", readme)
    return json_path


def _resolve_tractseg_command() -> list[str] | None:
    """Return argv prefix for TractSeg CLI, or None if not installed."""
    exe = shutil.which("TractSeg")
    if exe:
        return [exe]
    # Some installs expose ``python -m tractseg`` / ``TractSeg`` entry point.
    try:
        import tractseg  # noqa: F401

        return [sys.executable, "-m", "tractseg.commands.TractSeg"]
    except Exception:  # noqa: BLE001
        pass
    # Fall back to console script name via python -c is unreliable; try bare.
    return None


def run_tractseg_cli(
    input_nifti: str | Path,
    output_dir: str | Path,
    *,
    raw_diffusion_input: bool = False,
    output_type: str = "tract_segmentation",
    preprocess: bool = False,
    extra_args: Sequence[str] | None = None,
) -> tuple[Path, list[str]]:
    """
    Invoke the TractSeg command-line tool.

    Returns ``(output_dir, command_argv)``.
    """
    input_nifti = Path(input_nifti)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = _resolve_tractseg_command()
    if prefix is None:
        # Last resort: ``TractSeg`` on PATH via shell name (may still fail).
        prefix = ["TractSeg"]

    cmd = [
        *prefix,
        "-i",
        str(input_nifti),
        "-o",
        str(output_dir),
        "--output_type",
        output_type,
    ]
    if raw_diffusion_input:
        cmd.append("--raw_diffusion_input")
    if preprocess:
        cmd.append("--preprocess")
    if extra_args:
        cmd.extend(extra_args)

    logger.info("Running TractSeg: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "TractSeg executable not found. Install with "
            "`pip install TractSeg` (plus MRtrix3 / FSL for raw DWI). "
            f"Command attempted: {' '.join(cmd)}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"TractSeg failed (exit {exc.returncode}). Command: {' '.join(cmd)}"
        ) from exc
    return output_dir, cmd


def run_tractseg_python_api(
    peaks_nifti: str | Path,
    output_dir: str | Path,
) -> Path:
    """
    Call TractSeg's in-process Python API on a peaks volume.

    Saves a multi-label segmentation plus per-bundle binary masks.
    """
    import nibabel as nib
    import numpy as np

    peaks_nifti = Path(peaks_nifti)
    output_dir = Path(output_dir)
    bundle_dir = output_dir / "bundle_segmentations"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    img = nib.load(str(peaks_nifti))
    peaks = np.nan_to_num(np.asanyarray(img.dataobj)).astype(np.float32)

    # Prefer current API; fall back to older import path.
    try:
        from tractseg.python_api import run_tractseg as _ts_run
    except ImportError:
        try:
            from tractseg.TractSeg import run_tractseg as _ts_run
        except ImportError as exc:
            raise RuntimeError(
                "TractSeg Python API not importable. Install TractSeg or use the CLI."
            ) from exc

    logger.info("Running TractSeg Python API on %s", peaks_nifti)
    segmentation = _ts_run(peaks)
    segmentation = np.asarray(segmentation)

    # Multi-label (H, W, D[, C]) or (H, W, D) with integer labels.
    affine = img.affine
    if segmentation.ndim == 4:
        # One channel per bundle.
        n_bundles = min(segmentation.shape[-1], len(TRACTSEG_BUNDLES))
        for i in range(n_bundles):
            name = TRACTSEG_BUNDLES[i]
            mask = (segmentation[..., i] > 0.5).astype(np.uint8)
            nib.save(nib.Nifti1Image(mask, affine), str(bundle_dir / f"{name}.nii.gz"))
        multilabel = np.zeros(segmentation.shape[:3], dtype=np.uint8)
        for i in range(n_bundles):
            multilabel[segmentation[..., i] > 0.5] = i + 1
        nib.save(
            nib.Nifti1Image(multilabel, affine),
            str(output_dir / "bundle_segmentations_multilabel.nii.gz"),
        )
    else:
        nib.save(
            nib.Nifti1Image(segmentation.astype(np.uint8), affine),
            str(output_dir / "bundle_segmentations_multilabel.nii.gz"),
        )
        # Split integer labels 1..N into named masks when possible.
        for i, name in enumerate(TRACTSEG_BUNDLES, start=1):
            mask = (segmentation == i).astype(np.uint8)
            if np.any(mask):
                nib.save(nib.Nifti1Image(mask, affine), str(bundle_dir / f"{name}.nii.gz"))

    return output_dir


def collect_bundle_masks(
    output_dir: str | Path,
    *,
    bundles: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Map bundle name → NIfTI path for masks present under TractSeg output."""
    output_dir = Path(output_dir)
    bundle_dir = output_dir / "bundle_segmentations"
    if not bundle_dir.is_dir():
        # Some versions write directly under output_dir.
        bundle_dir = output_dir

    wanted = list(bundles) if bundles is not None else list(TRACTSEG_BUNDLES)
    found: dict[str, Path] = {}
    for name in wanted:
        for candidate in (
            bundle_dir / f"{name}.nii.gz",
            bundle_dir / f"{name}.nii",
            output_dir / f"{name}.nii.gz",
        ):
            if candidate.is_file():
                found[name] = candidate
                break
    return found


def run_tractseg_for_case(
    case_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    prefer_python_api: bool = False,
    preprocess: bool = False,
    bundles: Sequence[str] | None = None,
) -> TractSegResult:
    """
    Discover DTI for ``case_dir`` and run TractSeg tract segmentation.

    If DTI / peaks are missing, writes ``DTI_REQUIRED.md`` + status JSON and
    returns ``status="skipped_no_dti"`` without raising.
    """
    case_dir = Path(case_dir)
    if output_dir is None:
        output_dir = case_dir / "tractseg"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    status = find_dti_input(case_dir)
    if not status.available:
        write_missing_dti_note(output_dir, status)
        result = TractSegResult(
            status="skipped_no_dti",
            output_dir=str(output_dir),
            dti_status=status.to_dict(),
            note=MISSING_DTI_NOTE,
        )
        (output_dir / "tractseg_result.json").write_text(
            json.dumps(result.to_dict(), indent=2), encoding="utf-8"
        )
        return result

    assert status.input_path is not None
    cmd: list[str] | None = None
    try:
        if prefer_python_api and status.input_type == "peaks":
            run_tractseg_python_api(status.input_path, output_dir)
        else:
            raw = status.input_type == "raw_dwi"
            _, cmd = run_tractseg_cli(
                status.input_path,
                output_dir,
                raw_diffusion_input=raw,
                output_type="tract_segmentation",
                preprocess=preprocess,
            )
    except Exception as exc:  # noqa: BLE001
        # If CLI failed on peaks, try Python API once.
        if status.input_type == "peaks" and not prefer_python_api:
            logger.warning("TractSeg CLI failed (%s); trying Python API", exc)
            try:
                run_tractseg_python_api(status.input_path, output_dir)
            except Exception as exc2:  # noqa: BLE001
                result = TractSegResult(
                    status="failed",
                    output_dir=str(output_dir),
                    dti_status=status.to_dict(),
                    note=f"TractSeg failed: {exc2}",
                    command=cmd,
                )
                (output_dir / "tractseg_result.json").write_text(
                    json.dumps(result.to_dict(), indent=2), encoding="utf-8"
                )
                raise RuntimeError(result.note) from exc2
        else:
            result = TractSegResult(
                status="failed",
                output_dir=str(output_dir),
                dti_status=status.to_dict(),
                note=str(exc),
                command=cmd,
            )
            (output_dir / "tractseg_result.json").write_text(
                json.dumps(result.to_dict(), indent=2), encoding="utf-8"
            )
            raise

    wanted = list(bundles) if bundles is not None else list(TRACTSEG_BUNDLES)
    masks = collect_bundle_masks(output_dir, bundles=wanted)
    highlight = {k: str(v) for k, v in masks.items() if k in HIGHLIGHT_BUNDLES}
    bundle_dir = output_dir / "bundle_segmentations"
    result = TractSegResult(
        status="ok",
        output_dir=str(output_dir),
        bundle_dir=str(bundle_dir) if bundle_dir.is_dir() else str(output_dir),
        bundle_masks={k: str(v) for k, v in masks.items()},
        highlight_masks=highlight,
        dti_status=status.to_dict(),
        note=f"Segmented {len(masks)} bundles (highlights: {sorted(highlight)})",
        command=cmd,
    )
    (output_dir / "tractseg_result.json").write_text(
        json.dumps(result.to_dict(), indent=2), encoding="utf-8"
    )
    logger.info("%s", result.note)
    return result


# Backward-compatible name used by older imports / tractseg_run.py
def run_tractseg(
    peaks_or_fod_nifti: str | Path,
    output_dir: str | Path,
    *,
    bundles: str = "tractseg",
    raw_diffusion_input: bool = False,
) -> Path:
    """
    Thin entry: run TractSeg on an explicit peaks/DWI path.

    ``bundles`` is accepted for API compatibility with the previous stub
    (TractSeg always emits the standard bundle list; filter afterward).
    """
    _ = bundles
    out, _cmd = run_tractseg_cli(
        peaks_or_fod_nifti,
        output_dir,
        raw_diffusion_input=raw_diffusion_input,
        output_type="tract_segmentation",
    )
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description="Run TractSeg on preprocessed DTI/peaks (or document missing DTI)"
    )
    p.add_argument(
        "--case-dir",
        default=None,
        help="Preprocessed case folder (searches dti/, peaks, DWI)",
    )
    p.add_argument(
        "-i",
        "--input",
        default=None,
        help="Explicit peaks or DWI NIfTI (skips discovery)",
    )
    p.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output directory (default: <case>/tractseg)",
    )
    p.add_argument(
        "--raw-diffusion-input",
        action="store_true",
        help="Treat --input as raw DWI (passes --raw_diffusion_input to TractSeg)",
    )
    p.add_argument("--preprocess", action="store_true", help="TractSeg --preprocess")
    p.add_argument(
        "--python-api",
        action="store_true",
        help="Prefer in-process Python API (peaks only)",
    )
    args = p.parse_args(argv)

    if args.input is not None:
        out = Path(args.output_dir) if args.output_dir else Path("tractseg_output")
        run_tractseg(
            args.input,
            out,
            raw_diffusion_input=args.raw_diffusion_input,
        )
        masks = collect_bundle_masks(out)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "output_dir": str(out),
                    "n_bundles": len(masks),
                    "highlight_masks": {
                        k: str(v) for k, v in masks.items() if k in HIGHLIGHT_BUNDLES
                    },
                },
                indent=2,
            )
        )
        return 0

    if args.case_dir is None:
        p.error("Provide --case-dir or --input")

    result = run_tractseg_for_case(
        args.case_dir,
        args.output_dir,
        prefer_python_api=args.python_api,
        preprocess=args.preprocess,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.status in {"ok", "skipped_no_dti"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
