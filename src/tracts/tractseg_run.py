"""Thin wrapper — prefer :mod:`tracts.run_tractseg`.

Kept so older imports of ``tracts.tractseg_run`` continue to work.
"""

from __future__ import annotations

from .run_tractseg import run_tractseg, run_tractseg_for_case

__all__ = ["run_tractseg", "run_tractseg_for_case"]
