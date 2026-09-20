"""Interactive Plotly 3D figures for tumor meshes and tract centerlines/masks."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def plot_tumor_mesh(
    mesh_or_path,
    *,
    title: str = "Tumor surface",
    color: str = "crimson",
    opacity: float = 0.75,
):
    """Render a single tumor mesh (trimesh or path to ``.stl`` / ``.ply``)."""
    import plotly.graph_objects as go
    import trimesh

    if isinstance(mesh_or_path, (str, Path)):
        mesh = trimesh.load(str(mesh_or_path), force="mesh")
    else:
        mesh = mesh_or_path

    v = mesh.vertices
    f = mesh.faces
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=v[:, 0],
                y=v[:, 1],
                z=v[:, 2],
                i=f[:, 0],
                j=f[:, 1],
                k=f[:, 2],
                color=color,
                opacity=opacity,
                name="tumor",
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data", xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def plot_tumor_and_tracts(
    tumor_mesh,
    tract_meshes: Sequence,
    *,
    tract_names: Sequence[str] | None = None,
    title: str = "Tumor + tracts",
):
    """
    Overlay tumor mesh with one or more tract meshes / point clouds.

    ``tract_meshes`` may be trimesh objects or file paths.
    """
    import plotly.graph_objects as go
    import plotly.express as px
    import trimesh

    fig = plot_tumor_mesh(tumor_mesh, title=title)
    palette = px.colors.qualitative.Dark24
    names = list(tract_names) if tract_names is not None else [f"tract_{i}" for i in range(len(tract_meshes))]

    for i, item in enumerate(tract_meshes):
        mesh = trimesh.load(str(item), force="mesh") if isinstance(item, (str, Path)) else item
        v = mesh.vertices
        f = mesh.faces
        fig.add_trace(
            go.Mesh3d(
                x=v[:, 0],
                y=v[:, 1],
                z=v[:, 2],
                i=f[:, 0],
                j=f[:, 1],
                k=f[:, 2],
                color=palette[i % len(palette)],
                opacity=0.35,
                name=names[i] if i < len(names) else f"tract_{i}",
            )
        )
    return fig
