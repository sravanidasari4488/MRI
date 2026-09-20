"""Compare clinical ABC/2 ellipsoid volume to full segmented volume.

Runs across all cases with valid masks, reports percentage error and Pearson
correlation, writes a Bland–Altman plot, and breaks error down by tumor size
and sphericity — illustrating that larger / less spherical tumors incur
higher ellipsoid error.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .bland_altman import bland_altman_dataframe, bland_altman_stats
from .ellipsoid import abc_over_2_volume_cm3

logger = logging.getLogger(__name__)


def discover_mask_cases(data_root: str | Path) -> list[dict[str, str]]:
    """
    Find case folders that contain a usable segmentation mask.

    Supports:
      - BraTS-style ``<root>/<case_id>/mask.nii.gz``
      - Corrected real cases with ``mask.nii.gz`` / ``seg.nii.gz`` /
        ``pseudo_labels/pseudo_seg.nii.gz`` (same search as fine-tune)
    """
    from segmentation.finetune import _find_label_nifti

    root = Path(data_root)
    if not root.is_dir():
        raise NotADirectoryError(f"Data root not found: {root}")

    records: list[dict[str, str]] = []
    for case_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        # Prefer canonical BraTS mask name, then fine-tune label search.
        mask = case_dir / "mask.nii.gz"
        if not mask.is_file():
            found = _find_label_nifti(case_dir)
            if found is None:
                continue
            mask = found
        # Skip empty / all-zero masks later during measurement.
        records.append({"case_id": case_dir.name, "mask_path": str(mask)})

    logger.info("Found %d cases with masks under %s", len(records), root)
    return records


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    r = np.corrcoef(x, y)[0, 1]
    return float(r)


def measure_case_pair(
    mask_path: str | Path,
    *,
    case_id: str,
    label: int | None = None,
) -> dict[str, Any] | None:
    """
    Segmented voxel volume + ABC/2 from PCA diameters + sphericity for one mask.

    Returns ``None`` if the mask is empty / invalid.
    """
    from reconstruction.measurements import measure_from_mask_nifti

    try:
        m = measure_from_mask_nifti(mask_path, label=label, build_surface=True)
    except ValueError as exc:
        logger.warning("Skip %s: %s", case_id, exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skip %s (%s): %s", case_id, mask_path, exc)
        return None

    a, b, c = m.principal_axis_lengths_mm
    ell = abc_over_2_volume_cm3(a, b, c)
    seg = float(m.volume_cm3)
    if seg <= 0:
        logger.warning("Skip %s: non-positive segmented volume", case_id)
        return None

    pct_error = (ell - seg) / seg * 100.0
    abs_pct_error = abs(ell - seg) / seg * 100.0
    return {
        "case_id": case_id,
        "mask_path": str(mask_path),
        "segmented_cm3": seg,
        "ellipsoid_cm3": ell,
        "a_mm": a,
        "b_mm": b,
        "c_mm": c,
        "sphericity": float(m.sphericity),
        "pct_error": float(pct_error),
        "abs_pct_error": float(abs_pct_error),
        "abs_error_cm3": float(ell - seg),
        "n_voxels": m.n_voxels,
    }


def collect_comparison_dataframe(
    data_root: str | Path,
    *,
    label: int | None = None,
    max_cases: int | None = None,
) -> pd.DataFrame:
    """Run ellipsoid vs segmented volume on every valid mask under ``data_root``."""
    records = discover_mask_cases(data_root)
    if max_cases is not None:
        records = records[: max(0, max_cases)]

    rows: list[dict[str, Any]] = []
    for i, rec in enumerate(records, start=1):
        logger.info("[%d/%d] %s", i, len(records), rec["case_id"])
        row = measure_case_pair(rec["mask_path"], case_id=rec["case_id"], label=label)
        if row is not None:
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No valid masked cases under {data_root}")
    return df


def summary_statistics(df: pd.DataFrame) -> dict[str, Any]:
    """Percentage-error summaries + Pearson r (segmented vs ellipsoid)."""
    seg = df["segmented_cm3"].to_numpy(dtype=np.float64)
    ell = df["ellipsoid_cm3"].to_numpy(dtype=np.float64)
    ba = bland_altman_stats(seg, ell)
    return {
        "n_cases": int(len(df)),
        "pearson_r": _pearson_r(seg, ell),
        "mean_pct_error": float(df["pct_error"].mean()),
        "median_pct_error": float(df["pct_error"].median()),
        "mean_abs_pct_error": float(df["abs_pct_error"].mean()),
        "median_abs_pct_error": float(df["abs_pct_error"].median()),
        "mean_segmented_cm3": float(df["segmented_cm3"].mean()),
        "mean_ellipsoid_cm3": float(df["ellipsoid_cm3"].mean()),
        "mean_sphericity": float(df["sphericity"].mean()),
        "bland_altman": ba,
    }


def _bin_labels(series: pd.Series, *, n_bins: int, prefix: str) -> pd.Series:
    """Quantile bins with readable labels (handles tied edges)."""
    try:
        cats = pd.qcut(series, q=n_bins, duplicates="drop")
    except ValueError:
        cats = pd.cut(series, bins=n_bins, duplicates="drop")
    # Rename categories to size/sphericity tercile style.
    mapping = {}
    for i, cat in enumerate(cats.cat.categories):
        mapping[cat] = f"{prefix}_{i + 1}"
    return cats.cat.rename_categories(mapping)


def error_by_tumor_size(
    df: pd.DataFrame,
    *,
    n_bins: int = 3,
) -> pd.DataFrame:
    """
    Mean |% error| stratified by segmented volume quantile.

    Expectation: larger tumors → higher ellipsoid absolute % error.
    """
    work = df.copy()
    work["size_bin"] = _bin_labels(work["segmented_cm3"], n_bins=n_bins, prefix="size")
    grouped = (
        work.groupby("size_bin", observed=True)
        .agg(
            n=("case_id", "count"),
            mean_volume_cm3=("segmented_cm3", "mean"),
            mean_sphericity=("sphericity", "mean"),
            mean_abs_pct_error=("abs_pct_error", "mean"),
            median_abs_pct_error=("abs_pct_error", "median"),
            mean_pct_error=("pct_error", "mean"),
        )
        .reset_index()
    )
    return grouped


def error_by_sphericity(
    df: pd.DataFrame,
    *,
    n_bins: int = 3,
) -> pd.DataFrame:
    """
    Mean |% error| stratified by sphericity quantile.

    Expectation: less spherical (lower index) → higher ellipsoid error.
    """
    work = df.copy()
    work["sphericity_bin"] = _bin_labels(work["sphericity"], n_bins=n_bins, prefix="sph")
    grouped = (
        work.groupby("sphericity_bin", observed=True)
        .agg(
            n=("case_id", "count"),
            mean_sphericity=("sphericity", "mean"),
            mean_volume_cm3=("segmented_cm3", "mean"),
            mean_abs_pct_error=("abs_pct_error", "mean"),
            median_abs_pct_error=("abs_pct_error", "median"),
            mean_pct_error=("pct_error", "mean"),
        )
        .reset_index()
    )
    return grouped


def plot_bland_altman(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    title: str = "Bland–Altman: segmented vs ABC/2 ellipsoid",
) -> Path:
    """Difference (segmented − ellipsoid) vs mean volume."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ba_df = bland_altman_dataframe(
        df["segmented_cm3"],
        df["ellipsoid_cm3"],
        id_col=df["case_id"],
    )
    stats = bland_altman_stats(df["segmented_cm3"], df["ellipsoid_cm3"])

    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    ax.scatter(
        ba_df["mean"],
        ba_df["diff"],
        c="#4C78A8",
        alpha=0.75,
        edgecolors="none",
        s=36,
        label="cases",
    )
    ax.axhline(stats["bias"], color="#E45756", linewidth=1.5, label=f"bias={stats['bias']:.2f}")
    ax.axhline(
        stats["loa_lower"],
        color="#F58518",
        linestyle="--",
        linewidth=1.2,
        label=f"LoA [{stats['loa_lower']:.2f}, {stats['loa_upper']:.2f}]",
    )
    ax.axhline(stats["loa_upper"], color="#F58518", linestyle="--", linewidth=1.2)
    ax.axhline(0.0, color="#888888", linewidth=0.8)
    ax.set_xlabel("Mean volume (cm³)  [(segmented + ellipsoid) / 2]")
    ax.set_ylabel("Difference (cm³)  [segmented − ellipsoid]")
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    logger.info("Wrote Bland–Altman plot → %s", output_path)
    return output_path


def plot_error_breakdowns(
    df: pd.DataFrame,
    size_table: pd.DataFrame,
    sph_table: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """
    Scatter + stratified bar charts: |% error| vs size and vs sphericity.

    Visual evidence that larger / less spherical tumors have higher ABC/2 error.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), constrained_layout=True)

    ax = axes[0, 0]
    ax.scatter(df["segmented_cm3"], df["abs_pct_error"], c="#4C78A8", alpha=0.7, s=28)
    if len(df) >= 2:
        z = np.polyfit(df["segmented_cm3"], df["abs_pct_error"], 1)
        xs = np.linspace(df["segmented_cm3"].min(), df["segmented_cm3"].max(), 50)
        ax.plot(xs, np.poly1d(z)(xs), color="#E45756", linewidth=1.5, label="trend")
        ax.legend(frameon=False)
    ax.set_xlabel("Segmented volume (cm³)")
    ax.set_ylabel("|% error| of ABC/2")
    ax.set_title("Error vs tumor size")

    ax = axes[0, 1]
    ax.scatter(df["sphericity"], df["abs_pct_error"], c="#54A24B", alpha=0.7, s=28)
    if len(df) >= 2:
        z = np.polyfit(df["sphericity"], df["abs_pct_error"], 1)
        xs = np.linspace(df["sphericity"].min(), df["sphericity"].max(), 50)
        ax.plot(xs, np.poly1d(z)(xs), color="#E45756", linewidth=1.5, label="trend")
        ax.legend(frameon=False)
    ax.set_xlabel("Sphericity index")
    ax.set_ylabel("|% error| of ABC/2")
    ax.set_title("Error vs sphericity (↓ sph → ↑ error)")

    ax = axes[1, 0]
    ax.bar(
        size_table["size_bin"].astype(str),
        size_table["mean_abs_pct_error"],
        color="#4C78A8",
    )
    for i, row in size_table.iterrows():
        ax.annotate(
            f"V̄={row['mean_volume_cm3']:.1f}",
            xy=(i, row["mean_abs_pct_error"]),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylabel("Mean |% error|")
    ax.set_title("Error by volume tertile (larger → right)")

    ax = axes[1, 1]
    ax.bar(
        sph_table["sphericity_bin"].astype(str),
        sph_table["mean_abs_pct_error"],
        color="#54A24B",
    )
    for i, row in sph_table.iterrows():
        ax.annotate(
            f"Ψ̄={row['mean_sphericity']:.2f}",
            xy=(i, row["mean_abs_pct_error"]),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylabel("Mean |% error|")
    ax.set_title("Error by sphericity tertile (less spherical → left)")

    fig.suptitle(
        "Ellipsoid (ABC/2) error pattern: worse for larger / less spherical tumors",
        fontsize=12,
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    logger.info("Wrote error-breakdown figure → %s", output_path)
    return output_path


def plot_correlation(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    pearson_r: float,
) -> Path:
    """Scatter of segmented vs ellipsoid volumes with identity line."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.0, 5.5), constrained_layout=True)
    ax.scatter(df["segmented_cm3"], df["ellipsoid_cm3"], c="#4C78A8", alpha=0.75, s=32)
    lo = float(min(df["segmented_cm3"].min(), df["ellipsoid_cm3"].min()))
    hi = float(max(df["segmented_cm3"].max(), df["ellipsoid_cm3"].max()))
    ax.plot([lo, hi], [lo, hi], color="#888888", linestyle="--", linewidth=1.0, label="identity")
    ax.set_xlabel("Segmented volume (cm³)")
    ax.set_ylabel("ABC/2 ellipsoid volume (cm³)")
    ax.set_title(f"Volume correlation (Pearson r = {pearson_r:.3f})")
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def run_volume_comparison(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    label: int | None = None,
    max_cases: int | None = None,
    n_bins: int = 3,
) -> dict[str, Path]:
    """
    End-to-end report: per-case CSV, summary JSON, Bland–Altman + breakdown plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = collect_comparison_dataframe(data_root, label=label, max_cases=max_cases)
    summary = summary_statistics(df)
    size_table = error_by_tumor_size(df, n_bins=n_bins)
    sph_table = error_by_sphericity(df, n_bins=n_bins)

    # Attach bin labels to per-case table for downstream analysis.
    df = df.copy()
    df["size_bin"] = _bin_labels(df["segmented_cm3"], n_bins=n_bins, prefix="size")
    df["sphericity_bin"] = _bin_labels(df["sphericity"], n_bins=n_bins, prefix="sph")

    per_case_csv = output_dir / "ellipsoid_vs_segmented.csv"
    size_csv = output_dir / "error_by_size.csv"
    sph_csv = output_dir / "error_by_sphericity.csv"
    df.to_csv(per_case_csv, index=False)
    size_table.to_csv(size_csv, index=False)
    sph_table.to_csv(sph_csv, index=False)

    ba_png = plot_bland_altman(df, output_dir / "bland_altman.png")
    breakdown_png = plot_error_breakdowns(
        df, size_table, sph_table, output_dir / "error_by_size_sphericity.png"
    )
    corr_png = plot_correlation(
        df, output_dir / "volume_correlation.png", pearson_r=summary["pearson_r"]
    )

    report = {
        **summary,
        "data_root": str(data_root),
        "label": label,
        "error_by_size": size_table.to_dict(orient="records"),
        "error_by_sphericity": sph_table.to_dict(orient="records"),
        "interpretation": (
            "ABC/2 tends to err more on larger tumors and on less spherical "
            "(irregular) tumors; segmented voxel volume is the reference."
        ),
    }
    summary_json = output_dir / "comparison_summary.json"
    summary_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def _table_text(table: pd.DataFrame) -> str:
        try:
            return table.to_markdown(index=False)
        except Exception:  # noqa: BLE001
            return table.to_string(index=False)

    md_lines = [
        "# Ellipsoid (ABC/2) vs segmented volume",
        "",
        f"- Cases: **{summary['n_cases']}**",
        f"- Pearson r: **{summary['pearson_r']:.4f}**",
        f"- Mean % error (ell−seg)/seg: **{summary['mean_pct_error']:.2f}%**",
        f"- Mean |% error|: **{summary['mean_abs_pct_error']:.2f}%**",
        f"- Bland–Altman bias (seg−ell): **{summary['bland_altman']['bias']:.3f} cm³**",
        "",
        "## Error by tumor size",
        _table_text(size_table),
        "",
        "## Error by sphericity",
        _table_text(sph_table),
        "",
        report["interpretation"],
        "",
    ]
    md_path = output_dir / "comparison_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    logger.info(
        "Done: n=%d  r=%.3f  mean_|%%err|=%.1f%%  bias=%.2f cm³",
        summary["n_cases"],
        summary["pearson_r"],
        summary["mean_abs_pct_error"],
        summary["bland_altman"]["bias"],
    )
    return {
        "per_case_csv": per_case_csv,
        "error_by_size_csv": size_csv,
        "error_by_sphericity_csv": sph_csv,
        "summary_json": summary_json,
        "summary_md": md_path,
        "bland_altman_png": ba_png,
        "error_breakdown_png": breakdown_png,
        "correlation_png": corr_png,
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        from config import load_config

        cfg = load_config()
        default_data = str(cfg.paths.processed_brats_nifti)
        default_out = str(cfg.paths.processed / "metrics" / "ellipsoid_vs_segmented")
    except Exception:  # noqa: BLE001
        default_data = "data/processed/brats_nifti"
        default_out = "data/processed/metrics/ellipsoid_vs_segmented"

    p = argparse.ArgumentParser(
        description="ABC/2 vs segmented volume: % error, Pearson r, Bland–Altman"
    )
    p.add_argument("--data-root", default=default_data)
    p.add_argument("--output-dir", default=default_out)
    p.add_argument("--label", type=int, default=None, help="Exclusive label (default: any nonzero)")
    p.add_argument("--max-cases", type=int, default=None)
    p.add_argument("--n-bins", type=int, default=3)
    args = p.parse_args(argv)

    run_volume_comparison(
        args.data_root,
        args.output_dir,
        label=args.label,
        max_cases=args.max_cases,
        n_bins=args.n_bins,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
