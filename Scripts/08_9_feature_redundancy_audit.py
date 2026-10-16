#!/usr/bin/env python3
"""
Stage 08.9: Feature redundancy audit and Feature Set v3 freeze.

Purpose
-------
Create a decorrelated, label-independent Feature Set v3 from the corrected
Benchmark v1.1 Feature Set v2 tables.

Principles
----------
- No target labels are used to choose or remove features.
- Correlation is computed only from predictor values.
- Exact/near-exact redundancy is removed automatically.
- Very high correlations are auto-pruned only when both variables belong to
  the same predefined biological/measurement family.
- High correlations across distinct biological families are flagged for review
  but retained.
- All decisions are written to an explicit audit trail.
- Benchmark v1.1 labels and frozen Stage 07 folds are not modified.

Example
-------
python3 scripts/08_9_feature_redundancy_audit.py \
  --feature-dir data/frozen/benchmark_v1_1/feature_set_v2_labels_v1.1 \
  --output-dir data/frozen/feature_set_v3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ACCESSION = "current_uniprot_accession"
TARGET = "binary_label"

METADATA_COLUMNS = {
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
    "benchmark_version",
    "previous_binary_label",
    "label_updated_from_minami_validation",
    "minami_2023_table1_groe_dependency",
    "minami_2023_validation_status",
    "minami_2023_validation_source",
}

FEATURE_FILES = {
    "sequence": "feature_set_v2_sequence_final.csv",
    "structure_minimal": "feature_set_v2_structure_minimal_final.csv",
    "structure_core": "feature_set_v2_structure_core_final.csv",
    "combined_minimal": "feature_set_v2_sequence_plus_structure_minimal.csv",
    "combined_core": "feature_set_v2_sequence_plus_structure_core.csv",
}

# A feature appearing earlier in a family has priority when two same-family
# variables are near-redundant. This is a methods choice, not a performance choice.
PRIORITY = {
    "size": [
        "structure_ca_length",
        "seq_log_length",
        "seq_length",
        "seq_estimated_molecular_weight",
    ],
    "hydrophobicity": [
        "seq_mean_kd_hydrophobicity",
        "seq_gravy",
        "seq_hydrophobic_fraction",
        "seq_max_kd_window_19",
    ],
    "charge": [
        "seq_charge_density",
        "seq_net_charge_proxy",
        "seq_positive_fraction",
        "seq_negative_fraction",
    ],
    "secondary_structure": [
        "ss_helix_fraction",
        "ss_sheet_fraction",
        "ss_coil_fraction",
    ],
    "contact_topology": [
        "relative_contact_order",
        "long_range_contact_fraction",
        "minimum_cut_cross_contact_density",
        "ca_contact_density_per_residue",
        "mean_contact_sequence_separation",
        "median_contact_sequence_separation",
    ],
    "compactness_shape": [
        "radius_of_gyration_normalized_n_cuberoot",
        "asphericity",
        "axis_ratio_1_to_2",
        "axis_ratio_1_to_3",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit highly correlated predictors and freeze Feature Set v3."
    )
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument(
        "--output-dir",
        default="data/frozen/feature_set_v3",
    )
    parser.add_argument(
        "--pearson-threshold",
        type=float,
        default=0.95,
        help="Absolute Pearson correlation threshold for redundancy flagging.",
    )
    parser.add_argument(
        "--spearman-threshold",
        type=float,
        default=0.95,
        help="Absolute Spearman correlation threshold for redundancy flagging.",
    )
    parser.add_argument(
        "--exact-threshold",
        type=float,
        default=0.999999,
        help="Absolute correlation threshold treated as effectively exact.",
    )
    parser.add_argument(
        "--auto-prune-threshold",
        type=float,
        default=0.97,
        help=(
            "Same-family pairs at or above this absolute Spearman correlation "
            "are automatically reduced to one representative."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_family(feature: str) -> str:
    f = feature.lower()

    if any(token in f for token in [
        "length", "molecular_weight", "molecularweight", "log_length"
    ]):
        return "size"

    if any(token in f for token in [
        "hydrophobic", "gravy", "kd_", "aliphatic"
    ]):
        return "hydrophobicity"

    if any(token in f for token in [
        "charge", "positive_fraction", "negative_fraction",
        "acidic_fraction", "basic_fraction", "pi"
    ]):
        return "charge"

    if f.startswith("seq_fraction_") or f.endswith("_fraction") and any(
        aa in f for aa in [
            "_a", "_c", "_d", "_e", "_f", "_g", "_h", "_i", "_k", "_l",
            "_m", "_n", "_p", "_q", "_r", "_s", "_t", "_v", "_w", "_y"
        ]
    ):
        return "aa_composition"

    if any(token in f for token in [
        "helix", "sheet", "coil", "strand", "secondary_structure", "ss_"
    ]):
        return "secondary_structure"

    if any(token in f for token in [
        "contact_order", "contact_density", "contact_fraction",
        "contact_sequence_separation", "cut_cross", "long_range",
        "medium_range", "local_contact", "zero_contact"
    ]):
        return "contact_topology"

    if any(token in f for token in [
        "radius_of_gyration", "asphericity", "axis_ratio", "pair_fraction"
    ]):
        return "compactness_shape"

    if any(token in f for token in [
        "entropy", "complexity", "instability"
    ]):
        return "sequence_complexity"

    if any(token in f for token in [
        "glycine", "proline", "cysteine", "aromatic"
    ]):
        return "special_residue_composition"

    return "other"


def predictor_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in METADATA_COLUMNS
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].nunique(dropna=True) > 1
    ]


def rank_in_family(feature: str, family: str) -> tuple[int, str]:
    priorities = PRIORITY.get(family, [])
    try:
        rank = priorities.index(feature)
    except ValueError:
        rank = len(priorities) + 100
    return rank, feature


def choose_representative(a: str, b: str, family: str) -> tuple[str, str]:
    """
    Deterministic representative selection:
    1. predefined family priority;
    2. shorter feature name as interpretability proxy;
    3. lexical tie-break.
    """
    ra = rank_in_family(a, family)
    rb = rank_in_family(b, family)

    if ra < rb:
        return a, b
    if rb < ra:
        return b, a

    # fallback: concise name, then lexical
    key_a = (len(a), a)
    key_b = (len(b), b)
    if key_a <= key_b:
        return a, b
    return b, a


def build_pair_table(
    df: pd.DataFrame,
    features: list[str],
    pearson_threshold: float,
    spearman_threshold: float,
    exact_threshold: float,
) -> pd.DataFrame:
    pearson = df[features].corr(method="pearson")
    spearman = df[features].corr(method="spearman")

    rows = []
    for i, a in enumerate(features):
        for b in features[i + 1:]:
            p = pearson.loc[a, b]
            s = spearman.loc[a, b]
            if pd.isna(p) or pd.isna(s):
                continue

            abs_p = abs(float(p))
            abs_s = abs(float(s))

            if abs_p < pearson_threshold and abs_s < spearman_threshold:
                continue

            family_a = feature_family(a)
            family_b = feature_family(b)
            same_family = family_a == family_b

            if abs_p >= exact_threshold and abs_s >= exact_threshold:
                relation = "effectively_exact"
            elif same_family:
                relation = "high_correlation_same_family"
            else:
                relation = "high_correlation_cross_family"

            rows.append(
                {
                    "feature_1": a,
                    "feature_2": b,
                    "family_1": family_a,
                    "family_2": family_b,
                    "same_family": same_family,
                    "pearson_r": float(p),
                    "abs_pearson_r": abs_p,
                    "spearman_rho": float(s),
                    "abs_spearman_rho": abs_s,
                    "relation": relation,
                }
            )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["abs_spearman_rho", "abs_pearson_r"],
            ascending=False,
        )
    return result


def determine_removals(
    pairs: pd.DataFrame,
    auto_prune_threshold: float,
    exact_threshold: float,
) -> tuple[set[str], pd.DataFrame]:
    """
    Greedy, deterministic pruning.

    Cross-family correlations are never auto-pruned.
    Same-family pairs are auto-pruned only when Spearman correlation is very high.
    Effectively exact pairs are always eligible for pruning.
    """
    removed: set[str] = set()
    decision_rows = []

    if pairs.empty:
        return removed, pd.DataFrame()

    for _, row in pairs.iterrows():
        a = row["feature_1"]
        b = row["feature_2"]

        if a in removed or b in removed:
            decision_rows.append(
                {
                    **row.to_dict(),
                    "decision": "already_resolved_by_prior_pair",
                    "kept_feature": "",
                    "removed_feature": "",
                    "decision_reason": "",
                }
            )
            continue

        same_family = bool(row["same_family"])
        abs_s = float(row["abs_spearman_rho"])
        abs_p = float(row["abs_pearson_r"])

        eligible = (
            (abs_s >= exact_threshold and abs_p >= exact_threshold)
            or (same_family and abs_s >= auto_prune_threshold)
        )

        if not eligible:
            decision_rows.append(
                {
                    **row.to_dict(),
                    "decision": "retain_both_review_only",
                    "kept_feature": "",
                    "removed_feature": "",
                    "decision_reason": (
                        "Cross-family or below auto-prune threshold; "
                        "correlation alone does not justify removal."
                    ),
                }
            )
            continue

        family = row["family_1"] if same_family else row["family_1"]
        keep, remove = choose_representative(a, b, family)
        removed.add(remove)

        decision_rows.append(
            {
                **row.to_dict(),
                "decision": "auto_prune",
                "kept_feature": keep,
                "removed_feature": remove,
                "decision_reason": (
                    "Very high within-family redundancy; representative chosen "
                    "using predefined label-independent interpretability priority."
                ),
            }
        )

    return removed, pd.DataFrame(decision_rows)


def connected_components_from_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame(
            columns=["group_id", "feature", "family"]
        )

    graph: dict[str, set[str]] = defaultdict(set)
    for _, row in pairs.iterrows():
        a, b = row["feature_1"], row["feature_2"]
        graph[a].add(b)
        graph[b].add(a)

    visited = set()
    rows = []
    group_id = 0

    for node in sorted(graph):
        if node in visited:
            continue
        group_id += 1
        stack = [node]
        component = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(sorted(graph[current] - visited))

        for feature in sorted(component):
            rows.append(
                {
                    "group_id": group_id,
                    "feature": feature,
                    "family": feature_family(feature),
                }
            )

    return pd.DataFrame(rows)


def correlation_heatmap(
    df: pd.DataFrame,
    features: list[str],
    method: str,
    output_path: Path,
) -> None:
    corr = df[features].corr(method=method)
    n = len(features)
    figsize = max(10, min(24, n * 0.45))

    fig, axis = plt.subplots(figsize=(figsize, figsize))
    image = axis.imshow(corr.to_numpy(), vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(np.arange(n))
    axis.set_yticks(np.arange(n))
    axis.set_xticklabels(features, rotation=90, fontsize=7)
    axis.set_yticklabels(features, fontsize=7)
    axis.set_title(f"{method.capitalize()} feature correlation")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def export_v3_dataset(
    source: pd.DataFrame,
    features: list[str],
    output_path: Path,
) -> None:
    metadata = [c for c in source.columns if c not in predictor_columns(source)]
    # Preserve original column order for metadata + retained predictors.
    output_cols = [c for c in source.columns if c in metadata or c in features]
    source[output_cols].to_csv(output_path, index=False)


def main() -> int:
    args = parse_args()

    feature_dir = Path(args.feature_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not feature_dir.exists():
        print(f"ERROR: feature directory not found: {feature_dir}", file=sys.stderr)
        return 2

    loaded: dict[str, pd.DataFrame] = {}
    feature_lists: dict[str, list[str]] = {}

    for name, filename in FEATURE_FILES.items():
        path = feature_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)

        if len(df) != 181:
            raise RuntimeError(
                f"{filename}: expected 181 proteins, got {len(df)}"
            )
        if int(df[TARGET].sum()) != 73:
            raise RuntimeError(
                f"{filename}: expected 73 positives for Benchmark v1.1"
            )

        loaded[name] = df
        feature_lists[name] = predictor_columns(df)

    # Audit redundancy on the largest combined-core table so every candidate
    # predictor is considered exactly once.
    master = loaded["combined_core"]
    all_features = feature_lists["combined_core"]

    pearson_matrix = master[all_features].corr(method="pearson")
    spearman_matrix = master[all_features].corr(method="spearman")
    pearson_matrix.to_csv(output_dir / "pearson_correlation_matrix.csv")
    spearman_matrix.to_csv(output_dir / "spearman_correlation_matrix.csv")

    pair_table = build_pair_table(
        master,
        all_features,
        pearson_threshold=args.pearson_threshold,
        spearman_threshold=args.spearman_threshold,
        exact_threshold=args.exact_threshold,
    )
    pair_table.to_csv(
        output_dir / "high_correlation_pairs.csv",
        index=False,
    )

    groups = connected_components_from_pairs(pair_table)
    groups.to_csv(
        output_dir / "feature_redundancy_groups.csv",
        index=False,
    )

    removed, decisions = determine_removals(
        pair_table,
        auto_prune_threshold=args.auto_prune_threshold,
        exact_threshold=args.exact_threshold,
    )
    decisions.to_csv(
        output_dir / "feature_pruning_decisions.csv",
        index=False,
    )

    # Produce a feature-level audit.
    audit_rows = []
    for feature in all_features:
        family = feature_family(feature)
        removed_rows = (
            decisions[
                decisions.get("removed_feature", pd.Series(dtype=str)) == feature
            ]
            if not decisions.empty and "removed_feature" in decisions.columns
            else pd.DataFrame()
        )

        if feature in removed:
            decision = "remove"
            reason = (
                removed_rows.iloc[0]["decision_reason"]
                if not removed_rows.empty
                else "Highly correlated within-family redundancy."
            )
            representative = (
                removed_rows.iloc[0]["kept_feature"]
                if not removed_rows.empty
                else ""
            )
        else:
            decision = "keep"
            reason = (
                "Retained in Feature Set v3. Cross-family correlation alone "
                "was not used as a removal criterion."
            )
            representative = feature

        audit_rows.append(
            {
                "feature": feature,
                "family": family,
                "decision": decision,
                "retained_representative": representative,
                "reason": reason,
            }
        )

    feature_audit = pd.DataFrame(audit_rows)
    feature_audit.to_csv(
        output_dir / "feature_set_v3_audit.csv",
        index=False,
    )

    # Create v3 versions of each original feature set by applying only removal
    # decisions to variables actually present in that set.
    export_records = []
    v3_manifest_rows = []

    output_names = {
        "sequence": "feature_set_v3_sequence_final.csv",
        "structure_minimal": "feature_set_v3_structure_minimal_final.csv",
        "structure_core": "feature_set_v3_structure_core_final.csv",
        "combined_minimal": "feature_set_v3_sequence_plus_structure_minimal.csv",
        "combined_core": "feature_set_v3_sequence_plus_structure_core.csv",
    }

    for name, df in loaded.items():
        original_features = feature_lists[name]
        retained = [f for f in original_features if f not in removed]

        if not retained:
            raise RuntimeError(f"{name}: pruning removed every predictor")

        output_path = output_dir / output_names[name]
        export_v3_dataset(df, retained, output_path)

        for position, feature in enumerate(retained, start=1):
            v3_manifest_rows.append(
                {
                    "feature_set": name,
                    "feature_order": position,
                    "feature": feature,
                    "family": feature_family(feature),
                }
            )

        export_records.append(
            {
                "feature_set": name,
                "source_feature_count": len(original_features),
                "v3_feature_count": len(retained),
                "removed_feature_count": len(original_features) - len(retained),
                "removed_features": ";".join(
                    [f for f in original_features if f in removed]
                ),
                "output_file": output_names[name],
                "sha256": sha256_file(output_path),
            }
        )

    pd.DataFrame(v3_manifest_rows).to_csv(
        output_dir / "feature_set_v3_manifest.csv",
        index=False,
    )
    pd.DataFrame(export_records).to_csv(
        output_dir / "feature_set_v3_export_summary.csv",
        index=False,
    )

    correlation_heatmap(
        master,
        all_features,
        "pearson",
        figures_dir / "pearson_correlation_heatmap",
    )
    correlation_heatmap(
        master,
        all_features,
        "spearman",
        figures_dir / "spearman_correlation_heatmap",
    )

    summary = {
        "stage": "08.9",
        "benchmark_version": "1.1",
        "feature_set_input_version": "2.0",
        "feature_set_output_version": "3.0",
        "protein_count": 181,
        "positive_count": 73,
        "negative_count": 108,
        "master_candidate_feature_count": len(all_features),
        "flagged_high_correlation_pair_count": len(pair_table),
        "auto_pruned_feature_count": len(removed),
        "auto_pruned_features": sorted(removed),
        "pearson_threshold": args.pearson_threshold,
        "spearman_threshold": args.spearman_threshold,
        "auto_prune_threshold": args.auto_prune_threshold,
        "exact_threshold": args.exact_threshold,
        "selection_policy": (
            "No labels were used. Exact/near-exact redundancy and very high "
            "within-family Spearman correlation were eligible for automatic "
            "pruning. Cross-family correlations were retained and flagged."
        ),
        "feature_set_exports": export_records,
    }

    (output_dir / "stage08_9_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    methods = """# Feature Set v3 methods

Feature Set v3 was created from Benchmark v1.1 / Feature Set v2 before final
Random Forest development.

No target labels, cross-validation scores, model coefficients, feature
importance values, or downstream performance statistics were used to select
features.

Pearson and Spearman correlations were calculated among all predictors in the
combined-core Feature Set v2 table. High-correlation pairs were flagged when
either absolute Pearson or Spearman correlation exceeded the predefined
threshold.

Automatic pruning was deliberately conservative:
- effectively exact duplicates were eligible for removal;
- very highly correlated variables were eligible for removal only when they
  belonged to the same predefined biological/measurement family;
- highly correlated variables from different biological families were retained
  and flagged for review rather than automatically removed.

Representatives within a redundant family were chosen using a predefined,
label-independent interpretability priority. All decisions are preserved in
`feature_pruning_decisions.csv` and `feature_set_v3_audit.csv`.

Benchmark labels, protein membership, and frozen homology-grouped folds were not
changed.
"""
    (output_dir / "feature_set_v3_methods.md").write_text(methods)

    checksums = []
    for path in sorted(output_dir.rglob("*")):
        if (
            path.is_file()
            and path.name != "feature_set_v3_sha256.txt"
        ):
            checksums.append(
                f"{sha256_file(path)}  {path.relative_to(output_dir)}"
            )

    (output_dir / "feature_set_v3_sha256.txt").write_text(
        "\n".join(checksums) + "\n"
    )

    print("Stage 08.9 complete")
    print(f"  Proteins:                    181")
    print(f"  Benchmark positives:         73")
    print(f"  Candidate predictors:        {len(all_features)}")
    print(f"  High-correlation pairs:      {len(pair_table)}")
    print(f"  Automatically pruned:        {len(removed)}")
    print(f"  Output directory:            {output_dir}")

    if removed:
        print("  Removed features:")
        for feature in sorted(removed):
            print(f"    - {feature}")
    else:
        print("  No additional predictors met the conservative auto-prune rule.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
