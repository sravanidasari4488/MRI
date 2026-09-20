"""Classify tumor–tract proximity risk and look up tract function / deficits.

Risk tiers are driven by :data:`RISK_DISTANCE_THRESHOLDS_MM` — a single named
constant dict at module top so thresholds are easy to cite and adjust once
published neurosurgical cut-offs are confirmed.

No ML: functional consequences come from :data:`TRACT_FUNCTION_LOOKUP`.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Distance thresholds (millimeters)
# Provisional placeholders — replace with published cut-offs when confirmed.
# Classification rule (ascending distance):
#   distance <= infiltrated  →  "Infiltrated"   (overlap / involving)
#   distance <= high         →  "High Risk"
#   distance <= moderate     →  "Moderate Risk"
#   distance >  moderate     →  "Low Risk"
# ---------------------------------------------------------------------------
RISK_DISTANCE_THRESHOLDS_MM: dict[str, float] = {
    "infiltrated": 0.0,  # contact or overlap (min distance == 0 mm)
    "high": 2.0,  # TODO: cite published high-risk proximity (mm)
    "moderate": 5.0,  # TODO: cite published moderate-risk proximity (mm)
}

RISK_LABELS: tuple[str, ...] = (
    "Infiltrated",
    "High Risk",
    "Moderate Risk",
    "Low Risk",
)

# ---------------------------------------------------------------------------
# Tract → neurological function + possible deficit (lookup only; no ML).
# Keys match TractSeg bundle names. Values are plain dicts for easy editing.
# ---------------------------------------------------------------------------
TRACT_FUNCTION_LOOKUP: dict[str, dict[str, str]] = {
    # Corticospinal tract
    "CST_left": {
        "name": "Corticospinal tract (left)",
        "function": "Voluntary motor control of the right body",
        "possible_deficit": "Right hemiparesis / hemiplegia",
    },
    "CST_right": {
        "name": "Corticospinal tract (right)",
        "function": "Voluntary motor control of the left body",
        "possible_deficit": "Left hemiparesis / hemiplegia",
    },
    # Arcuate fasciculus
    "AF_left": {
        "name": "Arcuate fasciculus (left)",
        "function": "Language — dorsal stream (phonological / repetition)",
        "possible_deficit": "Conduction aphasia; impaired repetition",
    },
    "AF_right": {
        "name": "Arcuate fasciculus (right)",
        "function": "Prosody / alternative language network support",
        "possible_deficit": "Prosodic or mild language disruption",
    },
    # Optic radiation
    "OR_left": {
        "name": "Optic radiation (left)",
        "function": "Visual relay (left geniculocalcarine pathway)",
        "possible_deficit": "Right homonymous hemianopia / quadrantanopia",
    },
    "OR_right": {
        "name": "Optic radiation (right)",
        "function": "Visual relay (right geniculocalcarine pathway)",
        "possible_deficit": "Left homonymous hemianopia / quadrantanopia",
    },
    # Inferior fronto-occipital fasciculus
    "IFO_left": {
        "name": "Inferior fronto-occipital fasciculus (left)",
        "function": "Semantic language / ventral visual–frontal integration",
        "possible_deficit": "Semantic aphasia; visual–semantic deficits",
    },
    "IFO_right": {
        "name": "Inferior fronto-occipital fasciculus (right)",
        "function": "Visual–frontal integration; attention / semantics support",
        "possible_deficit": "Visuospatial or semantic support deficits",
    },
    # Inferior longitudinal fasciculus
    "ILF_left": {
        "name": "Inferior longitudinal fasciculus (left)",
        "function": "Ventral visual stream; object / word form processing",
        "possible_deficit": "Visual agnosia; reading difficulty (alexia risk)",
    },
    "ILF_right": {
        "name": "Inferior longitudinal fasciculus (right)",
        "function": "Ventral visual stream; face / object recognition support",
        "possible_deficit": "Prosopagnosia / visuospatial recognition deficits",
    },
    # Uncinate fasciculus
    "UF_left": {
        "name": "Uncinate fasciculus (left)",
        "function": "Orbitofrontal–temporal linking; semantic / emotion regulation",
        "possible_deficit": "Semantic impairment; behavioral / emotional change",
    },
    "UF_right": {
        "name": "Uncinate fasciculus (right)",
        "function": "Orbitofrontal–temporal linking; emotion / social cognition",
        "possible_deficit": "Emotional dysregulation; social cognition change",
    },
    # Cingulum
    "CG_left": {
        "name": "Cingulum (left)",
        "function": "Attention, memory, emotion (limbic)",
        "possible_deficit": "Executive / memory / affective disturbance",
    },
    "CG_right": {
        "name": "Cingulum (right)",
        "function": "Attention, memory, emotion (limbic)",
        "possible_deficit": "Executive / memory / affective disturbance",
    },
    # Anterior thalamic radiation
    "ATR_left": {
        "name": "Anterior thalamic radiation (left)",
        "function": "Prefrontal–thalamic executive connectivity",
        "possible_deficit": "Executive dysfunction; slowed processing",
    },
    "ATR_right": {
        "name": "Anterior thalamic radiation (right)",
        "function": "Prefrontal–thalamic executive connectivity",
        "possible_deficit": "Executive dysfunction; slowed processing",
    },
    # Superior longitudinal fasciculus
    "SLF_I_left": {
        "name": "Superior longitudinal fasciculus I (left)",
        "function": "Dorsal attention / visuomotor spatial processing",
        "possible_deficit": "Spatial attention / visuomotor deficits",
    },
    "SLF_I_right": {
        "name": "Superior longitudinal fasciculus I (right)",
        "function": "Dorsal attention / visuomotor spatial processing",
        "possible_deficit": "Hemispatial neglect risk; visuomotor deficits",
    },
    "SLF_II_left": {
        "name": "Superior longitudinal fasciculus II (left)",
        "function": "Attention and working-memory networks",
        "possible_deficit": "Attention / working-memory impairment",
    },
    "SLF_II_right": {
        "name": "Superior longitudinal fasciculus II (right)",
        "function": "Attention and working-memory networks",
        "possible_deficit": "Attention / neglect-related impairment",
    },
    "SLF_III_left": {
        "name": "Superior longitudinal fasciculus III (left)",
        "function": "Articulatory / phonological language support",
        "possible_deficit": "Speech / phonological deficits",
    },
    "SLF_III_right": {
        "name": "Superior longitudinal fasciculus III (right)",
        "function": "Attention / alternative language support",
        "possible_deficit": "Attention or mild communication deficits",
    },
    # Corpus callosum segments
    "CC_1": {
        "name": "Corpus callosum — rostrum",
        "function": "Interhemispheric prefrontal transfer",
        "possible_deficit": "Disconnection; executive / behavioral change",
    },
    "CC_2": {
        "name": "Corpus callosum — genu",
        "function": "Interhemispheric prefrontal transfer",
        "possible_deficit": "Alien-hand / callosal disconnection syndromes",
    },
    "CC_3": {
        "name": "Corpus callosum — rostral body",
        "function": "Premotor interhemispheric transfer",
        "possible_deficit": "Bimanual coordination impairment",
    },
    "CC_4": {
        "name": "Corpus callosum — anterior midbody",
        "function": "Primary motor interhemispheric transfer",
        "possible_deficit": "Bimanual motor coordination impairment",
    },
    "CC_5": {
        "name": "Corpus callosum — posterior midbody",
        "function": "Somatosensory interhemispheric transfer",
        "possible_deficit": "Tactile disconnection symptoms",
    },
    "CC_6": {
        "name": "Corpus callosum — isthmus",
        "function": "Temporoparietal interhemispheric transfer",
        "possible_deficit": "Auditory / language disconnection symptoms",
    },
    "CC_7": {
        "name": "Corpus callosum — splenium",
        "function": "Visual interhemispheric transfer",
        "possible_deficit": "Alexia without agraphia (classic callosal syndromes)",
    },
    # Cerebellar peduncles / commissures (shorter entries)
    "MCP": {
        "name": "Middle cerebellar peduncle",
        "function": "Cortico-ponto-cerebellar motor coordination",
        "possible_deficit": "Ataxia; dysmetria; dysarthria",
    },
    "CA": {
        "name": "Anterior commissure",
        "function": "Interhemispheric temporal / olfactory linking",
        "possible_deficit": "Mild disconnection; olfactory / memory nuance",
    },
    "FX_left": {
        "name": "Fornix (left)",
        "function": "Episodic memory (hippocampal output)",
        "possible_deficit": "Episodic memory impairment",
    },
    "FX_right": {
        "name": "Fornix (right)",
        "function": "Episodic memory (hippocampal output)",
        "possible_deficit": "Episodic memory impairment",
    },
}


def classify_tract_risk(
    distance_mm: float,
    *,
    thresholds: Mapping[str, float] | None = None,
) -> str:
    """
    Map a tumor–tract min distance (mm) to a risk label.

    Uses :data:`RISK_DISTANCE_THRESHOLDS_MM` unless ``thresholds`` is passed.
    Non-finite distances (empty mask) → ``"Unknown"``.
    """
    if not math.isfinite(distance_mm):
        return "Unknown"

    thr = dict(RISK_DISTANCE_THRESHOLDS_MM if thresholds is None else thresholds)
    infiltrated = float(thr["infiltrated"])
    high = float(thr["high"])
    moderate = float(thr["moderate"])

    if distance_mm <= infiltrated:
        return "Infiltrated"
    if distance_mm <= high:
        return "High Risk"
    if distance_mm <= moderate:
        return "Moderate Risk"
    return "Low Risk"


def lookup_tract_function(tract_name: str) -> dict[str, str]:
    """
    Return ``{name, function, possible_deficit}`` for a TractSeg bundle.

    Unknown bundles get a generic placeholder (still a plain dict).
    """
    if tract_name in TRACT_FUNCTION_LOOKUP:
        return dict(TRACT_FUNCTION_LOOKUP[tract_name])
    # Strip laterality for a soft match (e.g. custom names).
    base = tract_name.replace("_left", "").replace("_right", "")
    for key, meta in TRACT_FUNCTION_LOOKUP.items():
        if key.startswith(base):
            return dict(meta)
    return {
        "name": tract_name,
        "function": "Not listed in TRACT_FUNCTION_LOOKUP",
        "possible_deficit": "Unknown — add an entry to TRACT_FUNCTION_LOOKUP",
    }


def classify_distances(
    bundle_distances_mm: Mapping[str, float],
    *,
    thresholds: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """
    Build a per-tract table: distance, risk class, function, possible deficit.

    Parameters
    ----------
    bundle_distances_mm:
        Mapping of TractSeg bundle name → min distance in millimeters
        (e.g. from :func:`tracts.distance.distances_to_tracts`).
    """
    rows: list[dict[str, Any]] = []
    thr = dict(RISK_DISTANCE_THRESHOLDS_MM if thresholds is None else thresholds)

    for bundle, dist in bundle_distances_mm.items():
        dist_f = float(dist)
        risk = classify_tract_risk(dist_f, thresholds=thr)
        meta = lookup_tract_function(bundle)
        rows.append(
            {
                "bundle": bundle,
                "min_distance_mm": dist_f,
                "risk": risk,
                "tract_name": meta["name"],
                "function": meta["function"],
                "possible_deficit": meta["possible_deficit"],
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    risk_order = {label: i for i, label in enumerate(RISK_LABELS)}
    risk_order["Unknown"] = len(RISK_LABELS)
    df["_sort"] = df["risk"].map(risk_order)
    df = df.sort_values(["_sort", "min_distance_mm"], ascending=[True, True]).drop(
        columns=["_sort"]
    )
    return df.reset_index(drop=True)


def risk_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Counts per risk class + thresholds used (for reports)."""
    counts = {label: 0 for label in RISK_LABELS}
    counts["Unknown"] = 0
    if "risk" in df.columns:
        for label, n in df["risk"].value_counts().items():
            counts[str(label)] = int(n)
    return {
        "n_tracts": int(len(df)),
        "counts": counts,
        "thresholds_mm": dict(RISK_DISTANCE_THRESHOLDS_MM),
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(
        description="Classify tumor–tract distances into Infiltrated/High/Moderate/Low Risk"
    )
    p.add_argument(
        "--distances-json",
        required=True,
        help='JSON with {"distances_mm": {"CST_left": 1.2, ...}} or a flat dict',
    )
    p.add_argument("-o", "--output-csv", default=None)
    p.add_argument("--output-json", default=None)
    args = p.parse_args(argv)

    blob = json.loads(Path(args.distances_json).read_text(encoding="utf-8"))
    if "distances_mm" in blob:
        distances = blob["distances_mm"]
    elif "distances" in blob:
        distances = blob["distances"]
    else:
        distances = blob

    df = classify_distances(distances)
    summary = risk_summary(df)
    logger.info(
        "Classified %d tracts | %s",
        summary["n_tracts"],
        summary["counts"],
    )
    print(df.to_string(index=False))
    print(json.dumps(summary, indent=2))

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
    if args.output_json:
        out_j = Path(args.output_json)
        out_j.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "thresholds_mm": dict(RISK_DISTANCE_THRESHOLDS_MM),
            "summary": summary,
            "tracts": df.to_dict(orient="records"),
        }
        out_j.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
