"""Interactive Plotly 3D scene: brain + tumor + risk-colored tracts.

Renders with ``go.Mesh3d`` (and optional tract tubes / polylines), adds hover
text (name, distance, risk), and exports a standalone HTML file.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Risk → display color (cite alongside RISK_DISTANCE_THRESHOLDS_MM).
RISK_COLORS: dict[str, str] = {
    "Infiltrated": "#8B0000",  # dark red
    "High Risk": "#E45756",
    "Moderate Risk": "#F58518",
    "Low Risk": "#54A24B",
    "Unknown": "#9E9E9E",
}

# Tumor sub-region → solid Mesh3d color.
TUMOR_REGION_COLORS: dict[str, str] = {
    "et": "#D62728",  # enhancing tumor
    "tc": "#FF7F0E",  # tumor core
    "wt": "#9467BD",  # whole tumor / edema-inclusive shell
    "tumor": "#C44E52",
    "ncr": "#8C564B",
    "ed": "#17BECF",
}

BRAIN_COLOR = "#D0D7DE"
BRAIN_OPACITY = 0.18


def _load_trimesh(mesh_or_path: Any):
    import trimesh

    if mesh_or_path is None:
        return None
    if hasattr(mesh_or_path, "vertices") and hasattr(mesh_or_path, "faces"):
        return mesh_or_path
    return trimesh.load(str(mesh_or_path), force="mesh")


def _mesh3d_from_trimesh(
    mesh,
    *,
    name: str,
    color: str,
    opacity: float = 1.0,
    hovertext: str | None = None,
    showlegend: bool = True,
    flatshading: bool = True,
):
    """Build a ``go.Mesh3d`` trace from a trimesh (vertices in mm)."""
    import plotly.graph_objects as go

    v = np.asarray(mesh.vertices, dtype=np.float64)
    f = np.asarray(mesh.faces, dtype=np.int64)
    if v.size == 0 or f.size == 0:
        raise ValueError(f"Empty mesh for {name!r}")

    n_faces = len(f)
    if hovertext is None:
        hovertext = name
    # One hover label per face so the whole surface shows the same tooltip.
    hover = [hovertext] * n_faces

    return go.Mesh3d(
        x=v[:, 0],
        y=v[:, 1],
        z=v[:, 2],
        i=f[:, 0],
        j=f[:, 1],
        k=f[:, 2],
        color=color,
        opacity=opacity,
        name=name,
        flatshading=flatshading,
        hovertext=hover,
        hoverinfo="text",
        showlegend=showlegend,
        lighting=dict(ambient=0.45, diffuse=0.7, specular=0.15, roughness=0.6),
        lightposition=dict(x=1000, y=1000, z=2000),
    )


def _polyline_tube_mesh(
    points_mm: np.ndarray,
    *,
    radius_mm: float = 1.0,
    n_sides: int = 8,
):
    """
    Approximate a tube around a polyline (for tract centerlines without a mesh).

    Returns a ``trimesh.Trimesh`` or ``None`` if the path is too short.
    """
    import trimesh

    pts = np.asarray(points_mm, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 3:
        return None

    # Deduplicate consecutive points.
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-6
    pts = pts[keep]
    if len(pts) < 2:
        return None

    path = trimesh.load_path(pts)
    try:
        mesh = path.to_geometry(smooth=False)
        if mesh is not None and hasattr(mesh, "vertices") and len(mesh.vertices) > 0:
            return mesh
    except Exception:  # noqa: BLE001
        pass

    # Manual ring extrusion (tube along the polyline).
    verts: list[np.ndarray] = []
    faces: list[list[int]] = []
    tangents = np.diff(pts, axis=0)
    tangents = np.vstack([tangents, tangents[-1]])
    tangents = tangents / (np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-9)

    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(tangents[0], ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    normal = np.cross(tangents[0], ref)
    normal /= np.linalg.norm(normal) + 1e-9

    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    for i, (p, t) in enumerate(zip(pts, tangents)):
        binormal = np.cross(t, normal)
        bn = np.linalg.norm(binormal)
        if bn > 1e-8:
            binormal /= bn
            normal = np.cross(binormal, t)
            normal /= np.linalg.norm(normal) + 1e-9
        for a in angles:
            verts.append(p + radius_mm * (np.cos(a) * normal + np.sin(a) * binormal))
        if i > 0:
            base = (i - 1) * n_sides
            for s in range(n_sides):
                s2 = (s + 1) % n_sides
                faces.append([base + s, base + s2, base + n_sides + s2])
                faces.append([base + s, base + n_sides + s2, base + n_sides + s])

    return trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(faces), process=False)


def _scatter3d_tract_line(
    points_mm: np.ndarray,
    *,
    name: str,
    color: str,
    hovertext: str,
    width: float = 6.0,
):
    """Polyline tract when no mesh/tube is available."""
    import plotly.graph_objects as go

    pts = np.asarray(points_mm, dtype=np.float64)
    return go.Scatter3d(
        x=pts[:, 0],
        y=pts[:, 1],
        z=pts[:, 2],
        mode="lines",
        line=dict(color=color, width=width),
        name=name,
        hovertext=hovertext,
        hoverinfo="text",
        showlegend=True,
    )


def _tumor_hover(region: str, extra: Mapping[str, Any] | None = None) -> str:
    parts = [f"<b>Tumor — {region.upper()}</b>"]
    if extra:
        for k, v in extra.items():
            parts.append(f"{k}: {v}")
    return "<br>".join(parts)


def _tract_hover(
    name: str,
    *,
    distance_mm: float | None,
    risk: str | None,
    function: str | None = None,
    deficit: str | None = None,
) -> str:
    parts = [f"<b>{name}</b>"]
    if distance_mm is not None and np.isfinite(distance_mm):
        parts.append(f"Distance: {distance_mm:.2f} mm")
    elif distance_mm is not None:
        parts.append("Distance: n/a")
    if risk:
        parts.append(f"Risk: {risk}")
    if function:
        parts.append(f"Function: {function}")
    if deficit:
        parts.append(f"Possible deficit: {deficit}")
    return "<br>".join(parts)


def render_scene(
    *,
    brain_mesh: Any | None = None,
    tumor_meshes: Mapping[str, Any] | None = None,
    tumor_mesh: Any | None = None,
    tracts: Sequence[Mapping[str, Any]] | None = None,
    title: str = "Brain · tumor · tracts (risk)",
    brain_opacity: float = BRAIN_OPACITY,
) -> Any:
    """
    Build one interactive Plotly figure.

    Parameters
    ----------
    brain_mesh:
        Trimesh or ``.obj``/``.stl`` path — light, semi-transparent cortex.
    tumor_meshes:
        Mapping of sub-region name (``et`` / ``tc`` / ``wt`` / …) → mesh.
        Each region gets a solid color from :data:`TUMOR_REGION_COLORS`.
    tumor_mesh:
        Optional single tumor mesh if ``tumor_meshes`` is not given.
    tracts:
        Sequence of dicts with keys:
          - ``name`` (str)
          - ``mesh`` or ``path`` or ``points`` (N×3 mm polyline)
          - ``distance_mm`` (float, optional)
          - ``risk`` (str, optional — else classified from distance)
          - ``function`` / ``possible_deficit`` (optional)
    """
    import plotly.graph_objects as go

    from tracts.risk_classifier import (
        RISK_DISTANCE_THRESHOLDS_MM,
        classify_tract_risk,
        lookup_tract_function,
    )

    traces: list = []

    if brain_mesh is not None:
        brain = _load_trimesh(brain_mesh)
        traces.append(
            _mesh3d_from_trimesh(
                brain,
                name="Brain",
                color=BRAIN_COLOR,
                opacity=brain_opacity,
                hovertext="<b>Brain surface</b>",
                flatshading=False,
            )
        )

    region_map: dict[str, Any] = {}
    if tumor_meshes:
        region_map.update(tumor_meshes)
    elif tumor_mesh is not None:
        region_map["tumor"] = tumor_mesh

    for region, mesh_ref in region_map.items():
        mesh = _load_trimesh(mesh_ref)
        color = TUMOR_REGION_COLORS.get(region.lower(), TUMOR_REGION_COLORS["tumor"])
        traces.append(
            _mesh3d_from_trimesh(
                mesh,
                name=f"Tumor ({region.upper()})",
                color=color,
                opacity=1.0,
                hovertext=_tumor_hover(region),
            )
        )

    for tract in tracts or []:
        name = str(tract.get("name", "tract"))
        dist = tract.get("distance_mm")
        risk = tract.get("risk")
        if risk is None and dist is not None:
            risk = classify_tract_risk(float(dist))
        risk = risk or "Unknown"
        color = RISK_COLORS.get(risk, RISK_COLORS["Unknown"])

        meta = lookup_tract_function(name)
        function = tract.get("function", meta.get("function"))
        deficit = tract.get("possible_deficit", meta.get("possible_deficit"))
        hover = _tract_hover(
            tract.get("display_name", meta.get("name", name)),
            distance_mm=float(dist) if dist is not None else None,
            risk=risk,
            function=function,
            deficit=deficit,
        )

        mesh_ref = tract.get("mesh", tract.get("path"))
        points = tract.get("points")
        added = False
        if mesh_ref is not None:
            try:
                mesh = _load_trimesh(mesh_ref)
                traces.append(
                    _mesh3d_from_trimesh(
                        mesh,
                        name=f"{name} [{risk}]",
                        color=color,
                        opacity=0.85,
                        hovertext=hover,
                    )
                )
                added = True
            except Exception as exc:  # noqa: BLE001
                logger.warning("Tract mesh %s failed (%s); trying points/tube", name, exc)

        if not added and points is not None:
            pts = np.asarray(points, dtype=np.float64)
            tube = _polyline_tube_mesh(pts, radius_mm=float(tract.get("tube_radius_mm", 1.0)))
            if tube is not None and len(tube.faces) > 0:
                traces.append(
                    _mesh3d_from_trimesh(
                        tube,
                        name=f"{name} [{risk}]",
                        color=color,
                        opacity=0.9,
                        hovertext=hover,
                    )
                )
            else:
                traces.append(
                    _scatter3d_tract_line(
                        pts,
                        name=f"{name} [{risk}]",
                        color=color,
                        hovertext=hover,
                    )
                )
            added = True

        if not added:
            logger.warning("Skipping tract %s — no mesh or points", name)

    fig = go.Figure(data=traces)
    thr_note = (
        f"Risk thresholds (mm): infiltrated≤{RISK_DISTANCE_THRESHOLDS_MM['infiltrated']}, "
        f"high≤{RISK_DISTANCE_THRESHOLDS_MM['high']}, "
        f"moderate≤{RISK_DISTANCE_THRESHOLDS_MM['moderate']}"
    )
    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode="data",
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            bgcolor="rgb(248,249,250)",
            xaxis=dict(showbackground=True, backgroundcolor="rgb(248,249,250)"),
            yaxis=dict(showbackground=True, backgroundcolor="rgb(248,249,250)"),
            zaxis=dict(showbackground=True, backgroundcolor="rgb(248,249,250)"),
        ),
        legend=dict(title="Layers", itemsizing="constant"),
        margin=dict(l=0, r=0, t=70, b=0),
        paper_bgcolor="white",
        annotations=[
            dict(
                text=thr_note,
                xref="paper",
                yref="paper",
                x=0.0,
                y=1.08,
                showarrow=False,
                font=dict(size=11, color="#666666"),
                xanchor="left",
            )
        ],
    )
    return fig


def export_html(
    fig,
    output_html: str | Path,
    *,
    include_plotlyjs: str | bool = True,
) -> Path:
    """
    Write a **standalone** HTML file (Plotly.js embedded by default).

    Open the file in any browser — no Python required.
    """
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    # include_plotlyjs=True embeds the library → fully offline standalone.
    fig.write_html(
        str(output_html),
        include_plotlyjs=include_plotlyjs,
        full_html=True,
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": "brain_tumor_tracts",
                "scale": 2,
            },
        },
    )
    logger.info("Wrote standalone HTML → %s", output_html)
    return output_html


def render_and_export(
    output_html: str | Path,
    *,
    brain_mesh: Any | None = None,
    tumor_meshes: Mapping[str, Any] | None = None,
    tumor_mesh: Any | None = None,
    tracts: Sequence[Mapping[str, Any]] | None = None,
    title: str = "Brain · tumor · tracts (risk)",
) -> Path:
    """Convenience: build the scene and write standalone HTML."""
    fig = render_scene(
        brain_mesh=brain_mesh,
        tumor_meshes=tumor_meshes,
        tumor_mesh=tumor_mesh,
        tracts=tracts,
        title=title,
    )
    return export_html(fig, output_html)


def _load_tracts_from_json(path: Path) -> list[dict[str, Any]]:
    blob = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(blob, list):
        return blob
    if "tracts" in blob:
        return list(blob["tracts"])
    # Flat distances_mm → tracts without geometry (legend-only not useful);
    # expect mesh paths alongside.
    if "distances_mm" in blob:
        rows = []
        meshes = blob.get("tract_meshes", {})
        risks = blob.get("risks", {})
        for name, dist in blob["distances_mm"].items():
            rows.append(
                {
                    "name": name,
                    "distance_mm": dist,
                    "risk": risks.get(name),
                    "path": meshes.get(name),
                    "points": blob.get("tract_points", {}).get(name),
                }
            )
        return rows
    raise ValueError(f"Unrecognized tracts JSON schema in {path}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description="Render brain + tumor + risk-colored tracts to standalone HTML"
    )
    p.add_argument("--brain-mesh", default=None, help="Brain surface .obj/.stl")
    p.add_argument("--tumor-mesh", default=None, help="Single tumor mesh")
    p.add_argument("--tumor-et", default=None, help="Enhancing-tumor mesh")
    p.add_argument("--tumor-tc", default=None, help="Tumor-core mesh")
    p.add_argument("--tumor-wt", default=None, help="Whole-tumor mesh")
    p.add_argument(
        "--tracts-json",
        default=None,
        help="JSON list of tracts with name, path/points, distance_mm, risk",
    )
    p.add_argument(
        "-o",
        "--output-html",
        required=True,
        help="Standalone HTML output path",
    )
    p.add_argument("--title", default="Brain · tumor · tracts (risk)")
    args = p.parse_args(argv)

    tumor_meshes = {}
    if args.tumor_et:
        tumor_meshes["et"] = args.tumor_et
    if args.tumor_tc:
        tumor_meshes["tc"] = args.tumor_tc
    if args.tumor_wt:
        tumor_meshes["wt"] = args.tumor_wt

    tracts = _load_tracts_from_json(Path(args.tracts_json)) if args.tracts_json else []

    render_and_export(
        args.output_html,
        brain_mesh=args.brain_mesh,
        tumor_meshes=tumor_meshes or None,
        tumor_mesh=args.tumor_mesh,
        tracts=tracts,
        title=args.title,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
