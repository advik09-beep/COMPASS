#!/usr/bin/env python3
"""
Stage 08.5: Feature audit, deduplication, and Feature Set v2 freeze.

This script:
- verifies exact duplicate numeric features;
- identifies near-perfect Spearman correlations;
- applies a transparent, predeclared pruning policy;
- preserves biologically distinct correlated features unless they are exact or
  near-deterministic duplicates;
- exports final sequence, structural, and combined feature sets;
- freezes Feature Set v2 with an audit table and checksums.

Example:
    python3 scripts/08_5_audit_and_freeze_features.py \
      --benchmark data/frozen/benchmark_v1_0/benchmark_v1.0.csv \
      --output-dir data/frozen/feature_set_v2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


IDENTIFIER_COLUMNS = [
    "current_uniprot_accession",
    "gene_minami",
    "jwid",
    "binary_label",
    "cohort_label",
    "label_source",
    "label_evidence_type",
    "label_confidence",
    "source_class_summary",
    "label_notes",
    "sccs",
]

# Exact/semantic sequence duplicates known from feature construction.
PREDECLARED_REMOVALS = {
    "seq_fraction_G": (
        "remove",
        "exact_duplicate",
        "Exact duplicate of seq_glycine_fraction; retain descriptive grouped name.",
        "seq_glycine_fraction",
    ),
    "seq_fraction_P": (
        "remove",
        "exact_duplicate",
        "Exact duplicate of seq_proline_fraction; retain descriptive grouped name.",
        "seq_proline_fraction",
    ),
    "seq_fraction_C": (
        "remove",
        "exact_duplicate",
        "Exact duplicate of seq_cysteine_fraction; retain descriptive grouped name.",
        "seq_cysteine_fraction",
    ),
    "seq_length": (
        "remove",
        "deterministic_transform",
        "Duplicate biological information with structure_ca_length and seq_log_length; retain seq_log_length in sequence models and structure_ca_length in structural models.",
        "seq_log_length",
    ),
    "seq_estimated_molecular_weight": (
        "remove",
        "near_deterministic_transform",
        "Almost entirely determined by sequence length; remove to avoid size duplication.",
        "seq_log_length",
    ),
}

# Structural sets frozen in Stage 06B. These remain interpretable and restrained.
STRUCTURAL_MINIMAL = [
    "structure_ca_length",
    "relative_contact_order",
    "long_range_contact_fraction",
    "ca_contact_density_per_residue",
    "radius_of_gyration_normalized_n_cuberoot",
    "asphericity",
    "ss_helix_fraction",
    "ss_sheet_fraction",
    "minimum_cut_cross_contact_density",
]

STRUCTURAL_CORE = [
    "structure_ca_length",
    "relative_contact_order",
    "mean_contact_sequence_separation",
    "median_contact_sequence_separation",
    "long_range_contact_fraction",
    "medium_range_contact_fraction",
    "local_contact_fraction",
    "ca_contact_density_per_residue",
    "ca_contact_pair_fraction",
    "zero_contact_residue_fraction",
    "radius_of_gyration_normalized_n_cuberoot",
    "axis_ratio_1_to_2",
    "axis_ratio_1_to_3",
    "asphericity",
    "ss_helix_fraction",
    "ss_sheet_fraction",
    "ss_coil_fraction",
    "minimum_cut_cross_contact_density",
    "best_cut_fraction",
]

# Keep biologically distinct but correlated features. Remove only clearly
# redundant structural identities/complements.
STRUCTURAL_CORE_REMOVALS = {
    "mean_contact_sequence_separation": (
        "remove",
        "redundant_summary",
        "Highly overlapping contact-separation summary; retain relative_contact_order and median_contact_sequence_separation.",
        "relative_contact_order",
    ),
    "local_contact_fraction": (
        "remove",
        "compositional_redundancy",
        "Local, medium, and long-range fractions sum to one; retain medium and long-range fractions.",
        "medium_range_contact_fraction;long_range_contact_fraction",
    ),
    "ss_coil_fraction": (
        "remove",
        "compositional_redundancy",
        "Helix, sheet, and coil fractions sum to one; retain helix and sheet fractions.",
        "ss_helix_fraction;ss_sheet_fraction",
    ),
    "axis_ratio_1_to_3": (
        "remove",
        "high_shape_redundancy",
        "Strongly overlapping whole-shape descriptor; retain axis_ratio_1_to_2 and asphericity.",
        "axis_ratio_1_to_2;asphericity",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and freeze deduplicated Feature Set v2."
    )
    parser.add_argument("--benchmark", required=True)
    parser.add_argument(
        "--output-dir",
        default="data/frozen/feature_set_v2",
    )
    parser.add_argument(
        "--exact-tolerance",
        type=float,
        default=1e-12,
    )
    parser.add_argument(
        "--near-perfect-correlation",
        type=float,
        default=0.995,
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column not in IDENTIFIER_COLUMNS
        and pd.api.types.is_numeric_dtype(df[column])
        and column != "binary_label"
        and df[column].nunique(dropna=True) > 1
    ]


def feature_group(feature: str) -> str:
    if feature.startswith("seq_"):
        return "sequence"
    return "structure"


def exact_duplicate_pairs(
    df: pd.DataFrame,
    features: list[str],
    tolerance: float,
) -> pd.DataFrame:
    records = []
    for i, first in enumerate(features):
        a = pd.to_numeric(df[first], errors="coerce").to_numpy(dtype=float)
        for second in features[i + 1:]:
            b = pd.to_numeric(df[second], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() != len(df):
                continue
            difference = np.max(np.abs(a[mask] - b[mask]))
            if difference <= tolerance:
                records.append({
                    "feature_1": first,
                    "feature_2": second,
                    "maximum_absolute_difference": float(difference),
                    "exact_duplicate": True,
                })
    return pd.DataFrame(
        records,
        columns=[
            "feature_1",
            "feature_2",
            "maximum_absolute_difference",
            "exact_duplicate",
        ],
    )


def correlation_pairs(
    df: pd.DataFrame,
    features: list[str],
    threshold: float,
) -> pd.DataFrame:
    corr = df[features].corr(method="spearman")
    records = []
    for i, first in enumerate(features):
        for second in features[i + 1:]:
            value = corr.loc[first, second]
            if pd.notna(value) and abs(value) >= threshold:
                records.append({
                    "feature_1": first,
                    "feature_2": second,
                    "spearman_rho": float(value),
                    "absolute_rho": float(abs(value)),
                    "same_feature_group": feature_group(first) == feature_group(second),
                })
    result = pd.DataFrame(
        records,
        columns=[
            "feature_1",
            "feature_2",
            "spearman_rho",
            "absolute_rho",
            "same_feature_group",
        ],
    )
    if not result.empty:
        result = result.sort_values("absolute_rho", ascending=False)
    return result


def build_audit(
    all_features: list[str],
    sequence_features: list[str],
    structural_features: list[str],
) -> pd.DataFrame:
    rows = []
    final_sequence = set(sequence_features)
    final_structure = set(structural_features)

    for feature in all_features:
        group = feature_group(feature)
        if feature in PREDECLARED_REMOVALS:
            status, category, reason, retained = PREDECLARED_REMOVALS[feature]
        elif feature in STRUCTURAL_CORE_REMOVALS:
            status, category, reason, retained = STRUCTURAL_CORE_REMOVALS[feature]
        elif feature in final_sequence or feature in final_structure:
            status = "keep"
            category = "retained"
            reason = "Retained as a distinct, interpretable predictor in Feature Set v2."
            retained = feature
        else:
            status = "exclude"
            category = "outside_frozen_final_sets"
            reason = "Not selected for the final restrained feature sets."
            retained = ""

        rows.append({
            "feature": feature,
            "feature_group": group,
            "status": status,
            "audit_category": category,
            "reason": reason,
            "retained_representative": retained,
            "in_sequence_final": feature in final_sequence,
            "in_structure_minimal_final": feature in STRUCTURAL_MINIMAL,
            "in_structure_core_final": feature in final_structure,
        })
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    benchmark_path = Path(args.benchmark)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not benchmark_path.exists():
        print(f"ERROR: benchmark not found: {benchmark_path}", file=sys.stderr)
        return 2

    df = pd.read_csv(benchmark_path)
    if len(df) != 181:
        print(f"ERROR: expected 181 rows, observed {len(df)}", file=sys.stderr)
        return 2

    missing_metadata = [c for c in IDENTIFIER_COLUMNS if c not in df.columns]
    if missing_metadata:
        print(
            "ERROR: missing metadata columns: " + ", ".join(missing_metadata),
            file=sys.stderr,
        )
        return 2

    all_features = numeric_feature_columns(df)
    sequence_candidates = [f for f in all_features if f.startswith("seq_")]
    structural_candidates = [f for f in all_features if not f.startswith("seq_")]

    exact_pairs = exact_duplicate_pairs(
        df,
        all_features,
        args.exact_tolerance,
    )
    exact_pairs.to_csv(
        output_dir / "exact_duplicate_feature_pairs.csv",
        index=False,
    )

    near_pairs = correlation_pairs(
        df,
        all_features,
        args.near_perfect_correlation,
    )
    near_pairs.to_csv(
        output_dir / "near_perfect_correlation_pairs.csv",
        index=False,
    )

    sequence_final = [
        feature
        for feature in sequence_candidates
        if feature not in PREDECLARED_REMOVALS
    ]

    structural_minimal_final = [
        feature for feature in STRUCTURAL_MINIMAL if feature in df.columns
    ]
    structural_core_final = [
        feature
        for feature in STRUCTURAL_CORE
        if feature in df.columns
        and feature not in STRUCTURAL_CORE_REMOVALS
    ]

    if not sequence_final or not structural_minimal_final or not structural_core_final:
        raise RuntimeError("One or more final feature sets are empty")

    metadata = df[IDENTIFIER_COLUMNS].copy()

    exports = {
        "feature_set_v2_sequence_final.csv": sequence_final,
        "feature_set_v2_structure_minimal_final.csv": structural_minimal_final,
        "feature_set_v2_structure_core_final.csv": structural_core_final,
        "feature_set_v2_sequence_plus_structure_minimal.csv": (
            sequence_final + structural_minimal_final
        ),
        "feature_set_v2_sequence_plus_structure_core.csv": (
            sequence_final + structural_core_final
        ),
    }

    for filename, features in exports.items():
        df[IDENTIFIER_COLUMNS + features].to_csv(
            output_dir / filename,
            index=False,
        )

    audit = build_audit(
        all_features,
        sequence_final,
        structural_core_final,
    )
    audit.to_csv(output_dir / "feature_audit_v2.csv", index=False)

    manifest_rows = []
    for filename, features in exports.items():
        set_name = filename.removesuffix(".csv")
        for position, feature in enumerate(features, start=1):
            manifest_rows.append({
                "feature_set": set_name,
                "feature_order": position,
                "feature": feature,
                "feature_group": feature_group(feature),
            })
    pd.DataFrame(manifest_rows).to_csv(
        output_dir / "feature_set_v2_manifest.csv",
        index=False,
    )

    summary = {
        "version": "2.0",
        "protein_count": len(df),
        "candidate_sequence_features": len(sequence_candidates),
        "candidate_structural_features": len(structural_candidates),
        "final_sequence_features": len(sequence_final),
        "final_structural_minimal_features": len(structural_minimal_final),
        "final_structural_core_features": len(structural_core_final),
        "exact_duplicate_pair_count": len(exact_pairs),
        "near_perfect_correlation_pair_count": len(near_pairs),
        "predeclared_sequence_removals": PREDECLARED_REMOVALS,
        "predeclared_structural_removals": STRUCTURAL_CORE_REMOVALS,
        "policy": (
            "Remove exact duplicates, deterministic transforms, and compositional "
            "identities. Retain correlated features when they represent distinct "
            "biological concepts. Feature decisions are frozen before Random Forest."
        ),
    }
    (output_dir / "feature_set_v2_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    methods = f"""# Feature Set v2 Methods

## Purpose
Feature Set v2 was frozen before Random Forest development to prevent feature
selection from being influenced by downstream model performance.

## Sequence-feature policy
The initial sequence set contained {len(sequence_candidates)} numeric predictors.
Exact duplicates and near-deterministic size transforms were removed. Amino-acid
composition variables were otherwise retained because each represents distinct
sequence chemistry.

## Structural-feature policy
The minimal structural set contains {len(structural_minimal_final)} predictors.
The final core structural set contains {len(structural_core_final)} predictors.
Compositional identities were removed:
- coil fraction when helix and sheet fractions are present;
- local contact fraction when medium- and long-range fractions are present.

Highly correlated descriptors were retained only when they represented distinct
biological concepts. No label-based feature selection was performed.

## Downstream rule
All Random Forest and subsequent logistic-regression comparisons must use these
frozen v2 files. Any later change requires a documented feature-set version
increment.
"""
    (output_dir / "feature_set_v2_methods.md").write_text(methods)

    checksums = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "feature_set_v2_sha256.txt":
            checksums.append(f"{sha256_file(path)}  {path.name}")
    (output_dir / "feature_set_v2_sha256.txt").write_text(
        "\n".join(checksums) + "\n"
    )

    print("Stage 08.5 complete")
    print(f"  Sequence candidates:       {len(sequence_candidates)}")
    print(f"  Sequence final:            {len(sequence_final)}")
    print(f"  Structural minimal final:  {len(structural_minimal_final)}")
    print(f"  Structural core final:     {len(structural_core_final)}")
    print(f"  Exact duplicate pairs:     {len(exact_pairs)}")
    print(f"  Near-perfect correlations: {len(near_pairs)}")
    print(f"  Output directory:          {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
