"""
End-to-end study pipeline: preprocess → segment → reconstruct → validate →
tract risk → HTML + JSON report.

Run from the project root::

    python -m src.pipeline --study_id XYZ
    python -m pipeline --study_id XYZ
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Support both ``python -m src.pipeline`` (project root on path) and
# ``python -m pipeline`` (src/ on path / editable install).
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_checkpoint(paths) -> Path | None:
    candidates = [
        paths.checkpoints / "real_finetune" / "best_model.pt",
        paths.checkpoints / "brats_pretrain" / "best_model.pt",
        paths.checkpoints / "brats_pretrain_smoke" / "best_model.pt",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def resolve_study_paths(
    study_id: str,
    *,
    config_path: str | Path | None = None,
) -> dict[str, Path]:
    """
    Map a study ID to raw DICOM (if any) and processed case directories.

    Searches discovered real-patient studies by exact folder name or substring
    match; processed output lives under ``data/processed/real_patients/<id>``.
    """
    from config import load_config
    from preprocessing.batch_pipeline import discover_real_patient_studies

    cfg = load_config(config_path)
    paths = cfg.paths
    processed_case = paths.processed_real_patients / study_id
    reports_dir = paths.processed / "reports" / study_id

    dicom_study: Path | None = None
    try:
        studies = discover_real_patient_studies(paths.raw_dicom, limit=None)
    except Exception:  # noqa: BLE001
        studies = []
    for s in studies:
        if s.name == study_id or study_id in s.name:
            dicom_study = s
            break
    if dicom_study is None:
        # Direct child under raw_dicom
        cand = paths.raw_dicom / study_id
        if cand.is_dir():
            dicom_study = cand

    return {
        "project_root": cfg.project_root,
        "raw_dicom_root": paths.raw_dicom,
        "dicom_study": dicom_study if dicom_study is not None else Path(""),
        "processed_case": processed_case,
        "reports_dir": reports_dir,
        "checkpoints": paths.checkpoints,
        "processed_root": paths.processed,
    }


def _stage_preprocess(
    study_id: str,
    study_paths: dict[str, Path],
    *,
    skip_preprocess: bool = False,
    skip_bias_correction: bool = False,
    skip_skull_strip: bool = False,
) -> dict[str, Any]:
    from preprocessing.registration import preprocess_patient_study

    out_dir = study_paths["processed_case"]
    iso = out_dir / "05_isotropic_1mm"
    if skip_preprocess or iso.is_dir() and any(iso.glob("*.nii*")):
        logger.info("Preprocess: using existing %s", out_dir)
        return {
            "status": "skipped_existing" if skip_preprocess or iso.is_dir() else "ok",
            "output_dir": str(out_dir),
        }

    dicom = study_paths["dicom_study"]
    if not dicom or not Path(dicom).is_dir():
        if out_dir.is_dir():
            logger.warning(
                "No DICOM for %s; continuing with existing processed case %s",
                study_id,
                out_dir,
            )
            return {"status": "skipped_no_dicom_using_processed", "output_dir": str(out_dir)}
        raise FileNotFoundError(
            f"No DICOM study found for study_id={study_id!r} under "
            f"{study_paths['raw_dicom_root']}, and no processed case at {out_dir}"
        )

    result = preprocess_patient_study(
        dicom,
        out_dir,
        study_id=study_id,
        skip_bias_correction=skip_bias_correction,
        skip_skull_strip=skip_skull_strip,
    )
    return {"status": "ok", "output_dir": str(out_dir), "manifest": result.to_dict()}


def _stage_segmentation(
    study_id: str,
    case_dir: Path,
    checkpoint: Path | None,
    *,
    skip_segmentation: bool = False,
) -> dict[str, Any]:
    from segmentation.pseudo_label import pseudo_label_study

    out = case_dir / "pseudo_labels"
    seg = out / "pseudo_seg.nii.gz"
    if skip_segmentation and seg.is_file():
        return {"status": "skipped_existing", "seg_path": str(seg), "output_dir": str(out)}
    if checkpoint is None:
        if seg.is_file():
            logger.warning("No checkpoint; reusing existing pseudo-label %s", seg)
            return {
                "status": "skipped_no_checkpoint_using_existing",
                "seg_path": str(seg),
                "output_dir": str(out),
            }
        raise FileNotFoundError(
            "No segmentation checkpoint found under models/checkpoints/ "
            "(expected real_finetune or brats_pretrain best_model.pt) and no "
            f"existing mask at {seg}"
        )

    result = pseudo_label_study(case_dir, checkpoint, overwrite=not seg.is_file())
    return {
        "status": result.status,
        "seg_path": result.seg_path,
        "output_dir": result.output_dir,
        "et_path": str(Path(result.output_dir) / "pseudo_et.nii.gz")
        if result.output_dir
        else "",
        "tc_path": str(Path(result.output_dir) / "pseudo_tc.nii.gz")
        if result.output_dir
        else "",
        "wt_path": str(Path(result.output_dir) / "pseudo_wt.nii.gz")
        if result.output_dir
        else "",
        "error": result.error,
    }


def _stage_reconstruction(
    study_id: str,
    seg_path: Path,
    reports_dir: Path,
    *,
    region_masks: dict[str, Path] | None = None,
) -> dict[str, Any]:
    from reconstruction.mesh import mask_to_mesh
    from reconstruction.measurements import measure_from_mask_nifti

    mesh_dir = reports_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    # Whole-tumor mesh (any nonzero) for primary morphometrics.
    wt_mesh_path = mesh_dir / f"{study_id}_wt.obj"
    try:
        wt_mesh = mask_to_mesh(seg_path, wt_mesh_path, label=None)
    except ValueError:
        # Fall back to WT region mask if exclusive label map is empty somehow.
        if region_masks and region_masks.get("wt") and region_masks["wt"].is_file():
            wt_mesh = mask_to_mesh(region_masks["wt"], wt_mesh_path, label=None)
        else:
            raise

    measurements = measure_from_mask_nifti(seg_path, label=None, build_surface=True)
    meas_dict = measurements.to_dict()

    region_mesh_paths: dict[str, str] = {"wt": str(wt_mesh_path)}
    for region, path in (region_masks or {}).items():
        if path is None or not Path(path).is_file():
            continue
        try:
            out_m = mesh_dir / f"{study_id}_{region}.obj"
            mask_to_mesh(path, out_m, label=None)
            region_mesh_paths[region] = str(out_m)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Mesh for region %s failed: %s", region, exc)

    # Optional brain surface from skull-strip mask.
    brain_mesh_path = None
    case_dir = seg_path.parent.parent if seg_path.parent.name == "pseudo_labels" else seg_path.parent
    brain_candidates = list((case_dir / "03_skull_stripped").glob("*_mask*.nii*")) + list(
        (case_dir / "03_skull_stripped").glob("*brain_mask*.nii*")
    )
    if not brain_candidates:
        brain_candidates = list(case_dir.rglob("*brain_mask*.nii*"))
    if brain_candidates:
        try:
            brain_mesh_path = mesh_dir / f"{study_id}_brain.obj"
            mask_to_mesh(brain_candidates[0], brain_mesh_path, label=None, step_size=2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Brain mesh skipped: %s", exc)
            brain_mesh_path = None

    return {
        "status": "ok",
        "measurements": meas_dict,
        "wt_mesh": str(wt_mesh_path),
        "region_meshes": region_mesh_paths,
        "brain_mesh": str(brain_mesh_path) if brain_mesh_path else None,
        "mesh_volume_note": "vertices in mm; volume_cm3 from voxel count + surface from mesh",
        "n_faces_wt": int(len(wt_mesh.faces)),
    }


def _stage_validation(seg_path: Path, measurements: dict[str, Any]) -> dict[str, Any]:
    from validation.ellipsoid import (
        compare_to_ellipsoid,
        diameters_from_segmentation,
    )

    est = diameters_from_segmentation(seg_path, label=None)
    segmented = float(measurements["volume_cm3"])
    cmp = compare_to_ellipsoid(
        segmented,
        est.a_mm,
        est.b_mm,
        est.c_mm,
        formula="abc_over_2",
    )
    pct_error = (
        (cmp["ellipsoid_cm3"] - segmented) / segmented * 100.0 if segmented > 0 else float("nan")
    )
    return {
        "status": "ok",
        "diameters_mm": {"a": est.a_mm, "b": est.b_mm, "c": est.c_mm},
        "diameter_source": est.diameter_source,
        "segmented_cm3": segmented,
        "abc_over_2_cm3": cmp["abc_over_2_cm3"],
        "true_ellipsoid_cm3": cmp["true_ellipsoid_cm3"],
        "abs_error_cm3": cmp["abs_error_cm3"],
        "rel_error": cmp["rel_error"],
        "pct_error": pct_error,
        "sphericity": measurements.get("sphericity"),
        "surface_area_mm2": measurements.get("surface_area_mm2"),
        "principal_axis_lengths_mm": measurements.get("principal_axis_lengths_mm"),
    }


def _stage_tract_risk(
    case_dir: Path,
    seg_path: Path,
    reports_dir: Path,
) -> dict[str, Any]:
    from tracts.distance import distances_to_tracts
    from tracts.risk_classifier import classify_distances, risk_summary
    from tracts.run_tractseg import HIGHLIGHT_BUNDLES, run_tractseg_for_case

    tract_out = case_dir / "tractseg"
    ts = run_tractseg_for_case(case_dir, tract_out)

    tract_payload: dict[str, Any] = {
        "tractseg_status": ts.status,
        "tractseg_note": ts.note,
        "dti_status": ts.dti_status,
        "distances_mm": {},
        "risk_table": [],
        "risk_summary": {},
        "tract_meshes": {},
    }

    if ts.status != "ok" or not ts.bundle_masks:
        tract_payload["status"] = ts.status
        return tract_payload

    # Prefer clinically highlighted bundles for distance / risk / viz.
    masks = {
        k: v
        for k, v in ts.bundle_masks.items()
        if k in HIGHLIGHT_BUNDLES or k in ts.highlight_masks
    }
    if not masks:
        masks = dict(ts.bundle_masks)

    dists = distances_to_tracts(seg_path, masks)
    risk_df = classify_distances(dists)
    tract_payload.update(
        {
            "status": "ok",
            "distances_mm": dists,
            "risk_table": risk_df.to_dict(orient="records"),
            "risk_summary": risk_summary(risk_df),
            "tract_meshes": {k: str(v) for k, v in masks.items()},
        }
    )
    risk_csv = reports_dir / "tract_risk_table.csv"
    risk_df.to_csv(risk_csv, index=False)
    tract_payload["risk_table_csv"] = str(risk_csv)
    return tract_payload


def _stage_visualization(
    study_id: str,
    reports_dir: Path,
    reconstruction: dict[str, Any],
    tract_risk: dict[str, Any],
) -> dict[str, Any]:
    from visualization.render import render_and_export

    tracts_for_viz: list[dict[str, Any]] = []
    risk_by_bundle = {
        r["bundle"]: r for r in tract_risk.get("risk_table", []) if isinstance(r, dict)
    }
    for name, mesh_path in (tract_risk.get("tract_meshes") or {}).items():
        row = risk_by_bundle.get(name, {})
        tracts_for_viz.append(
            {
                "name": name,
                "path": mesh_path,
                "distance_mm": row.get(
                    "min_distance_mm", tract_risk.get("distances_mm", {}).get(name)
                ),
                "risk": row.get("risk"),
                "function": row.get("function"),
                "possible_deficit": row.get("possible_deficit"),
            }
        )

    tumor_meshes = dict(reconstruction.get("region_meshes") or {})
    # Prefer et/tc/wt keys for coloring; ensure wt present.
    if "wt" not in tumor_meshes and reconstruction.get("wt_mesh"):
        tumor_meshes["wt"] = reconstruction["wt_mesh"]

    html_path = reports_dir / f"{study_id}_scene.html"
    try:
        render_and_export(
            html_path,
            brain_mesh=reconstruction.get("brain_mesh"),
            tumor_meshes=tumor_meshes or None,
            tumor_mesh=reconstruction.get("wt_mesh") if not tumor_meshes else None,
            tracts=tracts_for_viz,
            title=f"{study_id} — brain · tumor · tract risk",
        )
        return {"status": "ok", "html": str(html_path), "n_tracts_rendered": len(tracts_for_viz)}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Visualization failed")
        return {"status": "failed", "error": str(exc), "html": None}


def run_full_pipeline(
    study_id: str,
    *,
    config_path: str | Path | None = None,
    skip_preprocess: bool = False,
    skip_segmentation: bool = False,
    skip_tracts: bool = False,
    skip_bias_correction: bool = False,
    skip_skull_strip: bool = False,
) -> dict[str, Any]:
    """
    Chain preprocessing, segmentation, reconstruction, validation, tract risk,
    and visualization for one ``study_id``.

    Returns a JSON-serializable summary dict and writes::

        data/processed/reports/<study_id>/<study_id>_report.json
        data/processed/reports/<study_id>/<study_id>_scene.html
    """
    t0 = time.perf_counter()
    study_id = str(study_id).strip()
    if not study_id:
        raise ValueError("study_id must be a non-empty string")

    study_paths = resolve_study_paths(study_id, config_path=config_path)
    reports_dir = study_paths["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    case_dir = study_paths["processed_case"]

    report: dict[str, Any] = {
        "study_id": study_id,
        "started_at": _utc_now(),
        "paths": {k: str(v) for k, v in study_paths.items()},
        "stages": {},
        "status": "running",
    }

    try:
        # 1) Preprocess
        logger.info("[%s] Stage: preprocess", study_id)
        report["stages"]["preprocess"] = _stage_preprocess(
            study_id,
            study_paths,
            skip_preprocess=skip_preprocess,
            skip_bias_correction=skip_bias_correction,
            skip_skull_strip=skip_skull_strip,
        )

        # 2) Segmentation
        logger.info("[%s] Stage: segmentation", study_id)
        from config import load_config

        cfg = load_config(config_path)
        ckpt = _resolve_checkpoint(cfg.paths)
        report["checkpoint"] = str(ckpt) if ckpt else None
        seg_stage = _stage_segmentation(
            study_id, case_dir, ckpt, skip_segmentation=skip_segmentation
        )
        report["stages"]["segmentation"] = seg_stage
        if seg_stage.get("status") == "failed":
            raise RuntimeError(seg_stage.get("error") or "Segmentation failed")

        seg_path = Path(seg_stage["seg_path"])
        if not seg_path.is_file():
            raise FileNotFoundError(f"Segmentation mask missing: {seg_path}")

        region_masks = {}
        for key in ("et", "tc", "wt"):
            p = seg_stage.get(f"{key}_path") or str(
                Path(seg_stage.get("output_dir", "")) / f"pseudo_{key}.nii.gz"
            )
            if p and Path(p).is_file():
                region_masks[key] = Path(p)

        # 3) Reconstruction
        logger.info("[%s] Stage: reconstruction", study_id)
        recon = _stage_reconstruction(
            study_id, seg_path, reports_dir, region_masks=region_masks
        )
        report["stages"]["reconstruction"] = recon

        # 4) Validation (ellipsoid vs segmented)
        logger.info("[%s] Stage: validation", study_id)
        validation = _stage_validation(seg_path, recon["measurements"])
        report["stages"]["validation"] = validation

        # 5) Tract risk
        if skip_tracts:
            tract_risk = {"status": "skipped", "note": "skip_tracts=True"}
        else:
            logger.info("[%s] Stage: tract risk", study_id)
            tract_risk = _stage_tract_risk(case_dir, seg_path, reports_dir)
        report["stages"]["tract_risk"] = tract_risk

        # 6) Visualization
        logger.info("[%s] Stage: visualization", study_id)
        viz = _stage_visualization(study_id, reports_dir, recon, tract_risk)
        report["stages"]["visualization"] = viz

        # Flat summary block for the report consumers
        report["summary"] = {
            "volume_cm3": recon["measurements"].get("volume_cm3"),
            "surface_area_mm2": recon["measurements"].get("surface_area_mm2"),
            "sphericity": recon["measurements"].get("sphericity"),
            "principal_axis_lengths_mm": recon["measurements"].get(
                "principal_axis_lengths_mm"
            ),
            "ellipsoid_comparison": {
                "abc_over_2_cm3": validation.get("abc_over_2_cm3"),
                "segmented_cm3": validation.get("segmented_cm3"),
                "pct_error": validation.get("pct_error"),
                "diameters_mm": validation.get("diameters_mm"),
            },
            "tract_risk_table": tract_risk.get("risk_table", []),
            "tract_risk_counts": tract_risk.get("risk_summary", {}).get("counts"),
            "html_visualization": viz.get("html"),
        }
        report["status"] = "success"
    except Exception as exc:  # noqa: BLE001
        report["status"] = "failed"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        logger.error("[%s] Pipeline failed: %s", study_id, exc)
    finally:
        report["finished_at"] = _utc_now()
        report["elapsed_sec"] = round(time.perf_counter() - t0, 3)
        report_path = reports_dir / f"{study_id}_report.json"
        report["report_json"] = str(report_path)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info(
            "[%s] Wrote report → %s (status=%s, %.1fs)",
            study_id,
            report_path,
            report["status"],
            report["elapsed_sec"],
        )

    return report


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(
        description="Run the full brain-tumor + tract-risk pipeline for one study"
    )
    p.add_argument(
        "--study_id",
        "--study-id",
        dest="study_id",
        required=True,
        help="Study folder name / ID (matches DICOM study or processed case)",
    )
    p.add_argument("--config", default=None, help="Optional path to config.yaml")
    p.add_argument("--skip-preprocess", action="store_true")
    p.add_argument("--skip-segmentation", action="store_true")
    p.add_argument("--skip-tracts", action="store_true")
    p.add_argument("--skip-bias-correction", action="store_true")
    p.add_argument("--skip-skull-strip", action="store_true")
    args = p.parse_args(argv)

    report = run_full_pipeline(
        args.study_id,
        config_path=args.config,
        skip_preprocess=args.skip_preprocess,
        skip_segmentation=args.skip_segmentation,
        skip_tracts=args.skip_tracts,
        skip_bias_correction=args.skip_bias_correction,
        skip_skull_strip=args.skip_skull_strip,
    )

    # Concise CLI stdout
    summary = report.get("summary") or {}
    print(
        json.dumps(
            {
                "study_id": report["study_id"],
                "status": report["status"],
                "report_json": report.get("report_json"),
                "html": summary.get("html_visualization"),
                "volume_cm3": summary.get("volume_cm3"),
                "ellipsoid_pct_error": (summary.get("ellipsoid_comparison") or {}).get(
                    "pct_error"
                ),
                "n_tracts_risk": len(summary.get("tract_risk_table") or []),
                "error": report.get("error"),
            },
            indent=2,
        )
    )
    return 0 if report.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
