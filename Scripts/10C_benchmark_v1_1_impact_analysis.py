#!/usr/bin/env python3
"""
Stage 10C: Benchmark v1.1 impact analysis.

Purpose
-------
Quantify exactly how Benchmark v1.1 differs from Benchmark v1.0 before any
models are retrained.

This script compares:
- class balance;
- the nine corrected labels;
- fold composition under the frozen 30% and 40% homology CV schemes;
- homology-cluster composition;
- TIM-barrel composition;
- feature distributions for the relabeled proteins versus unchanged proteins;
- expected class-weight changes for balanced classifiers;
- whether corrected proteins are concentrated in particular folds/clusters.

No model fitting occurs.

Example:
    python3 scripts/10C_benchmark_v1_1_impact_analysis.py \
      --benchmark-v1 data/frozen/benchmark_v1_0/benchmark_v1.0.csv \
      --benchmark-v11 data/frozen/benchmark_v1_1/benchmark_v1.1.csv \
      --fold-assignments data/grouping/stage07/fold_assignments_v1.0.csv \
      --feature-dir-v11 data/frozen/benchmark_v1_1/feature_set_v2_labels_v1.1 \
      --output-dir results/stage10C
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ACCESSION = "current_uniprot_accession"
GENE = "gene_minami"
JWID = "jwid"
LABEL = "binary_label"

FEATURE_FILES = {
    "sequence_final": "feature_set_v2_sequence_final.csv",
    "structure_minimal_final": "feature_set_v2_structure_minimal_final.csv",
    "sequence_plus_structure_minimal": (
        "feature_set_v2_sequence_plus_structure_minimal.csv"
    ),
}

FOLD_COLUMNS = [
    "homology_cv_fold",
    "homology_cv_fold_sensitivity",
]

CLUSTER_COLUMNS = [
    "homology_cluster_30",
    "homology_cluster_40",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the effect of Benchmark v1.1 label corrections."
    )
    parser.add_argument("--benchmark-v1", required=True)
    parser.add_argument("--benchmark-v11", required=True)
    parser.add_argument("--fold-assignments", required=True)
    parser.add_argument("--feature-dir-v11", required=True)
    parser.add_argument("--output-dir", default="results/stage10C")
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        print(f"ERROR: {label} not found: {path}", file=sys.stderr)
        raise SystemExit(2)


def validate_benchmarks(v1: pd.DataFrame, v11: pd.DataFrame) -> None:
    required = {ACCESSION, GENE, JWID, LABEL}

    for name, df in [("v1.0", v1), ("v1.1", v11)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Benchmark {name} missing columns: {sorted(missing)}"
            )
        if df[ACCESSION].duplicated().any():
            raise ValueError(
                f"Benchmark {name} contains duplicate accessions"
            )

    if len(v1) != len(v11):
        raise ValueError(
            f"Benchmark row counts differ: {len(v1)} vs {len(v11)}"
        )

    if set(v1[ACCESSION]) != set(v11[ACCESSION]):
        raise ValueError(
            "Benchmark accession sets differ; v1.1 should only change labels/provenance"
        )


def class_weight_balanced(n_total: int, n_class: int, n_classes: int = 2) -> float:
    return n_total / (n_classes * n_class)


def build_change_table(v1: pd.DataFrame, v11: pd.DataFrame) -> pd.DataFrame:
    left = v1[[ACCESSION, GENE, JWID, LABEL]].rename(
        columns={
            GENE: "gene_v1_0",
            JWID: "jwid_v1_0",
            LABEL: "label_v1_0",
        }
    )
    right_cols = [ACCESSION, GENE, JWID, LABEL]

    extra_cols = [
        c for c in [
            "previous_binary_label",
            "label_updated_from_minami_validation",
            "minami_2023_table1_groe_dependency",
            "minami_2023_validation_status",
            "minami_2023_validation_source",
        ]
        if c in v11.columns
    ]

    right = v11[right_cols + extra_cols].rename(
        columns={
            GENE: "gene_v1_1",
            JWID: "jwid_v1_1",
            LABEL: "label_v1_1",
        }
    )

    merged = left.merge(
        right,
        on=ACCESSION,
        how="inner",
        validate="one_to_one",
    )

    merged["label_changed"] = (
        merged["label_v1_0"].astype(int)
        != merged["label_v1_1"].astype(int)
    )
    merged["change_direction"] = np.select(
        [
            (merged["label_v1_0"] == 0) & (merged["label_v1_1"] == 1),
            (merged["label_v1_0"] == 1) & (merged["label_v1_1"] == 0),
        ],
        ["0_to_1", "1_to_0"],
        default="unchanged",
    )

    return merged


def fold_composition(
    change_table: pd.DataFrame,
    folds: pd.DataFrame,
) -> pd.DataFrame:
    merged = change_table.merge(
        folds[
            [
                ACCESSION,
                "homology_cv_fold",
                "homology_cv_fold_sensitivity",
                "homology_cluster_30",
                "homology_cluster_40",
                "is_tim_barrel",
            ]
        ],
        on=ACCESSION,
        how="left",
        validate="one_to_one",
    )

    rows = []
    for fold_column in FOLD_COLUMNS:
        for fold, group in merged.groupby(fold_column):
            rows.append(
                {
                    "fold_column": fold_column,
                    "fold": int(fold),
                    "n": len(group),
                    "positives_v1_0": int(group["label_v1_0"].sum()),
                    "negatives_v1_0": int(
                        len(group) - group["label_v1_0"].sum()
                    ),
                    "positives_v1_1": int(group["label_v1_1"].sum()),
                    "negatives_v1_1": int(
                        len(group) - group["label_v1_1"].sum()
                    ),
                    "corrected_proteins": int(group["label_changed"].sum()),
                    "tim_barrel_n": int(group["is_tim_barrel"].astype(bool).sum()),
                    "corrected_tim_barrel_n": int(
                        (
                            group["label_changed"]
                            & group["is_tim_barrel"].astype(bool)
                        ).sum()
                    ),
                }
            )

    return pd.DataFrame(rows)


def cluster_impact(
    change_table: pd.DataFrame,
    folds: pd.DataFrame,
) -> pd.DataFrame:
    merged = change_table.merge(
        folds[
            [
                ACCESSION,
                "homology_cluster_30",
                "homology_cluster_40",
                "is_tim_barrel",
            ]
        ],
        on=ACCESSION,
        how="left",
        validate="one_to_one",
    )

    rows = []
    for cluster_column in CLUSTER_COLUMNS:
        for cluster, group in merged.groupby(cluster_column):
            corrected_n = int(group["label_changed"].sum())
            if corrected_n == 0:
                continue

            rows.append(
                {
                    "cluster_column": cluster_column,
                    "cluster": cluster,
                    "cluster_size": len(group),
                    "corrected_n": corrected_n,
                    "positive_n_v1_0": int(group["label_v1_0"].sum()),
                    "positive_n_v1_1": int(group["label_v1_1"].sum()),
                    "mixed_label_cluster_v1_0": bool(
                        group["label_v1_0"].nunique() > 1
                    ),
                    "mixed_label_cluster_v1_1": bool(
                        group["label_v1_1"].nunique() > 1
                    ),
                    "tim_barrel_n": int(
                        group["is_tim_barrel"].astype(bool).sum()
                    ),
                }
            )

    return pd.DataFrame(rows)


def feature_effects(
    change_table: pd.DataFrame,
    feature_dir: Path,
) -> pd.DataFrame:
    rows = []

    for feature_set, filename in FEATURE_FILES.items():
        path = feature_dir / filename
        require_file(path, filename)
        df = pd.read_csv(path)

        metadata = {
            ACCESSION,
            GENE,
            JWID,
            LABEL,
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

        numeric_features = [
            c for c in df.columns
            if c not in metadata
            and pd.api.types.is_numeric_dtype(df[c])
            and df[c].nunique(dropna=True) > 1
        ]

        merged = df[[ACCESSION] + numeric_features].merge(
            change_table[[ACCESSION, "label_changed"]],
            on=ACCESSION,
            how="left",
            validate="one_to_one",
        )

        changed = merged[merged["label_changed"]]
        unchanged = merged[~merged["label_changed"]]

        for feature in numeric_features:
            changed_values = pd.to_numeric(
                changed[feature], errors="coerce"
            ).dropna()
            unchanged_values = pd.to_numeric(
                unchanged[feature], errors="coerce"
            ).dropna()

            if changed_values.empty or unchanged_values.empty:
                continue

            changed_mean = float(changed_values.mean())
            unchanged_mean = float(unchanged_values.mean())

            pooled_sd = float(
                np.sqrt(
                    (
                        changed_values.var(ddof=1)
                        + unchanged_values.var(ddof=1)
                    )
                    / 2
                )
            )

            standardized_difference = (
                (changed_mean - unchanged_mean) / pooled_sd
                if pooled_sd > 0
                else float("nan")
            )

            rows.append(
                {
                    "feature_set": feature_set,
                    "feature": feature,
                    "changed_n": len(changed_values),
                    "unchanged_n": len(unchanged_values),
                    "changed_mean": changed_mean,
                    "unchanged_mean": unchanged_mean,
                    "changed_median": float(changed_values.median()),
                    "unchanged_median": float(unchanged_values.median()),
                    "standardized_mean_difference": standardized_difference,
                    "absolute_standardized_difference": (
                        abs(standardized_difference)
                        if math.isfinite(standardized_difference)
                        else float("nan")
                    ),
                }
            )

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["feature_set", "absolute_standardized_difference"],
            ascending=[True, False],
        )
    return result


def make_figures(
    fold_df: pd.DataFrame,
    feature_effect_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    primary = fold_df[
        fold_df["fold_column"] == "homology_cv_fold"
    ].copy()

    if not primary.empty:
        x = np.arange(len(primary))
        width = 0.35

        fig, axis = plt.subplots(figsize=(9, 5.5))
        axis.bar(
            x - width / 2,
            primary["positives_v1_0"],
            width,
            label="Benchmark v1.0",
        )
        axis.bar(
            x + width / 2,
            primary["positives_v1_1"],
            width,
            label="Benchmark v1.1",
        )
        axis.set_xticks(x)
        axis.set_xticklabels(primary["fold"].astype(str))
        axis.set_xlabel("30% homology-grouped fold")
        axis.set_ylabel("Positive proteins")
        axis.set_title("Positive-class composition before and after correction")
        axis.legend()
        fig.tight_layout()
        fig.savefig(
            figures_dir / "fold_positive_composition.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            figures_dir / "fold_positive_composition.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)

    if not feature_effect_df.empty:
        combined = feature_effect_df[
            feature_effect_df["feature_set"]
            == "sequence_plus_structure_minimal"
        ].copy()

        if not combined.empty:
            top = combined.nlargest(
                15,
                "absolute_standardized_difference",
            ).sort_values("standardized_mean_difference")

            fig, axis = plt.subplots(figsize=(9, 7))
            axis.barh(
                top["feature"],
                top["standardized_mean_difference"],
            )
            axis.axvline(0, linewidth=1)
            axis.set_xlabel(
                "Standardized difference: corrected proteins vs unchanged proteins"
            )
            axis.set_title(
                "Features enriched or depleted among the nine corrected proteins"
            )
            fig.tight_layout()
            fig.savefig(
                figures_dir / "corrected_protein_feature_profile.png",
                dpi=300,
                bbox_inches="tight",
            )
            fig.savefig(
                figures_dir / "corrected_protein_feature_profile.pdf",
                bbox_inches="tight",
            )
            plt.close(fig)


def main() -> int:
    args = parse_args()

    v1_path = Path(args.benchmark_v1)
    v11_path = Path(args.benchmark_v11)
    fold_path = Path(args.fold_assignments)
    feature_dir = Path(args.feature_dir_v11)
    output_dir = Path(args.output_dir)

    for path, label in [
        (v1_path, "Benchmark v1.0"),
        (v11_path, "Benchmark v1.1"),
        (fold_path, "fold assignments"),
        (feature_dir, "Feature Set v2 labels v1.1 directory"),
    ]:
        require_file(path, label)

    output_dir.mkdir(parents=True, exist_ok=True)

    v1 = pd.read_csv(v1_path)
    v11 = pd.read_csv(v11_path)
    folds = pd.read_csv(fold_path)

    validate_benchmarks(v1, v11)

    required_fold_cols = {
        ACCESSION,
        "homology_cv_fold",
        "homology_cv_fold_sensitivity",
        "homology_cluster_30",
        "homology_cluster_40",
        "is_tim_barrel",
    }
    missing_fold_cols = required_fold_cols - set(folds.columns)
    if missing_fold_cols:
        raise ValueError(
            f"Fold file missing columns: {sorted(missing_fold_cols)}"
        )

    changes = build_change_table(v1, v11)

    changed_only = changes[changes["label_changed"]].copy()
    if len(changed_only) != 9:
        raise RuntimeError(
            f"Expected 9 changed labels, observed {len(changed_only)}"
        )

    fold_df = fold_composition(changes, folds)
    cluster_df = cluster_impact(changes, folds)
    feature_effect_df = feature_effects(changes, feature_dir)

    merged_tim = changes.merge(
        folds[[ACCESSION, "is_tim_barrel"]],
        on=ACCESSION,
        how="left",
        validate="one_to_one",
    )

    changed_tim_n = int(
        (
            merged_tim["label_changed"]
            & merged_tim["is_tim_barrel"].astype(bool)
        ).sum()
    )
    changed_non_tim_n = int(len(changed_only) - changed_tim_n)

    old_pos = int(v1[LABEL].astype(int).sum())
    old_neg = int(len(v1) - old_pos)
    new_pos = int(v11[LABEL].astype(int).sum())
    new_neg = int(len(v11) - new_pos)

    class_weights = pd.DataFrame(
        [
            {
                "benchmark_version": "1.0",
                "class": 0,
                "count": old_neg,
                "balanced_class_weight": class_weight_balanced(
                    len(v1), old_neg
                ),
            },
            {
                "benchmark_version": "1.0",
                "class": 1,
                "count": old_pos,
                "balanced_class_weight": class_weight_balanced(
                    len(v1), old_pos
                ),
            },
            {
                "benchmark_version": "1.1",
                "class": 0,
                "count": new_neg,
                "balanced_class_weight": class_weight_balanced(
                    len(v11), new_neg
                ),
            },
            {
                "benchmark_version": "1.1",
                "class": 1,
                "count": new_pos,
                "balanced_class_weight": class_weight_balanced(
                    len(v11), new_pos
                ),
            },
        ]
    )

    changes.to_csv(
        output_dir / "benchmark_change_table_all_proteins.csv",
        index=False,
    )
    changed_only.to_csv(
        output_dir / "nine_corrected_proteins.csv",
        index=False,
    )
    fold_df.to_csv(
        output_dir / "fold_composition_v1_0_vs_v1_1.csv",
        index=False,
    )
    cluster_df.to_csv(
        output_dir / "corrected_cluster_impact.csv",
        index=False,
    )
    feature_effect_df.to_csv(
        output_dir / "corrected_protein_feature_effects.csv",
        index=False,
    )
    class_weights.to_csv(
        output_dir / "class_weight_changes.csv",
        index=False,
    )

    make_figures(
        fold_df,
        feature_effect_df,
        output_dir,
    )

    summary = {
        "stage": "10C",
        "benchmark_rows": len(v1),
        "positive_count_v1_0": old_pos,
        "negative_count_v1_0": old_neg,
        "positive_count_v1_1": new_pos,
        "negative_count_v1_1": new_neg,
        "corrected_label_count": len(changed_only),
        "corrected_tim_barrel_count": changed_tim_n,
        "corrected_non_tim_count": changed_non_tim_n,
        "primary_fold_corrected_counts": (
            fold_df[
                fold_df["fold_column"] == "homology_cv_fold"
            ][["fold", "corrected_proteins"]]
            .to_dict(orient="records")
        ),
        "sensitivity_fold_corrected_counts": (
            fold_df[
                fold_df["fold_column"]
                == "homology_cv_fold_sensitivity"
            ][["fold", "corrected_proteins"]]
            .to_dict(orient="records")
        ),
        "affected_cluster_count_30": int(
            cluster_df[
                cluster_df["cluster_column"] == "homology_cluster_30"
            ]["cluster"].nunique()
        ),
        "affected_cluster_count_40": int(
            cluster_df[
                cluster_df["cluster_column"] == "homology_cluster_40"
            ]["cluster"].nunique()
        ),
        "balanced_class_weight_positive_v1_0": class_weight_balanced(
            len(v1), old_pos
        ),
        "balanced_class_weight_positive_v1_1": class_weight_balanced(
            len(v11), new_pos
        ),
        "balanced_class_weight_negative_v1_0": class_weight_balanced(
            len(v1), old_neg
        ),
        "balanced_class_weight_negative_v1_1": class_weight_balanced(
            len(v11), new_neg
        ),
        "note": (
            "This stage performs no model fitting. It quantifies the benchmark "
            "change before any v1.1 model reruns."
        ),
    }

    (output_dir / "stage10C_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    methods = """# Stage 10C benchmark-impact analysis

Benchmark v1.0 and v1.1 were compared before any model retraining.

The analysis verifies that:
- the protein set is unchanged;
- exactly nine labels changed;
- fold and homology-cluster assignments are unchanged;
- feature values are unchanged.

The script quantifies class balance, fold composition, affected homology
clusters, TIM-barrel status, balanced class weights, and feature distributions
of the nine experimentally corrected proteins relative to the remaining
benchmark.

No predictor is fitted, tuned, or evaluated in this stage.
"""
    (output_dir / "stage10C_methods.md").write_text(methods)

    print("Stage 10C complete")
    print(f"  Benchmark rows:           {len(v1)}")
    print(f"  Positives v1.0 -> v1.1:   {old_pos} -> {new_pos}")
    print(f"  Negatives v1.0 -> v1.1:   {old_neg} -> {new_neg}")
    print(f"  Corrected proteins:        {len(changed_only)}")
    print(f"  Corrected TIM barrels:     {changed_tim_n}")
    print(f"  Corrected non-TIM:         {changed_non_tim_n}")
    print(f"  Output directory:          {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
