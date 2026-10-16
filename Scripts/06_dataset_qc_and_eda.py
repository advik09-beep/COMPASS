#!/usr/bin/env python3
"""
Stage 06: Dataset quality control, leakage audit, and exploratory analysis.

This script is designed for:
    minami_binary_benchmark_features_181.csv

It does NOT train a predictive model.

Main outputs:
- a PDF QC report
- leakage/feature-role manifest
- missingness and variance reports
- univariate positive-vs-negative statistics with FDR correction
- highly correlated feature pairs
- fold-composition tables
- outlier and integrity audits
- two model-ready structure datasets:
    1) a broad eligible structural feature set
    2) a restrained core structural/topology feature set

Example:
    python3 scripts/06_dataset_qc_and_eda.py \
      --input data/modeling/minami_binary_benchmark_features_181.csv \
      --output-dir results/qc
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats


EXPECTED_ROWS = 181
EXPECTED_POSITIVES = 64
EXPECTED_NEGATIVES = 117

IDENTIFIER_COLUMNS = {
    "jwid",
    "minami_uniprot_accession",
    "current_uniprot_accession",
    "gene_minami",
    "accession",
    "gene",
    "b_number",
    "pdb_path",
    "fasta_path",
}

TARGET_COLUMNS = {
    "cohort_label",
    "binary_label",
}

PRIOR_STUDY_COLUMNS = {
    "minami_length",
    "minami_structure_source",
    "minami_relative_contact_order",
    "minami_rw_prediction",
    "minami_rwrr_prediction",
}

GROUPING_ONLY_COLUMNS = {
    "sccs",
}

QC_ONLY_COLUMNS = {
    "fasta_structure_length_match",
    "chain_count",
    "chain_break_count",
    "mean_consecutive_ca_distance",
    "max_consecutive_ca_distance",
    "ss_assignment_count",
    "ss_note",
    "mean_plddt",
    "median_plddt",
    "minimum_plddt",
    "maximum_plddt",
    "plddt_std",
    "fraction_plddt_below_50",
    "fraction_plddt_below_70",
    "fraction_plddt_at_least_90",
}

CONFIGURATION_COLUMNS = {
    "contact_cutoff_angstrom",
    "minimum_contact_sequence_separation",
    "long_range_sequence_separation",
}

# These are derived count/size variables that are mathematically close to protein
# length or to one another. They remain available in the broad feature table,
# but the restrained core table avoids most of them.
CORE_FEATURE_CANDIDATES = [
    "structure_ca_length",
    "radius_of_gyration_normalized_n_cuberoot",
    "compactness_n_over_rg_cubed",
    "axis_ratio_1_to_2",
    "axis_ratio_1_to_3",
    "asphericity",
    "ss_helix_fraction",
    "ss_sheet_fraction",
    "ca_contact_density_per_residue",
    "ca_contact_pair_fraction",
    "mean_contact_degree",
    "zero_contact_residue_fraction",
    "relative_contact_order",
    "long_range_contact_fraction",
    "medium_range_contact_fraction",
    "minimum_cut_cross_contact_density",
    "best_cut_fraction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and explore the Minami binary benchmark feature table."
    )
    parser.add_argument("--input", required=True, help="181-protein feature CSV.")
    parser.add_argument(
        "--output-dir",
        default="results/qc",
        help="Directory for reports and cleaned datasets.",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.95,
        help="Absolute Spearman correlation used to flag feature pairs.",
    )
    parser.add_argument(
        "--maximum-missing-fraction",
        type=float,
        default=0.10,
        help="Features above this missingness are excluded from ML exports.",
    )
    parser.add_argument(
        "--top-feature-count",
        type=int,
        default=20,
        help="Number of ranked features shown in report tables/plots.",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg FDR-adjusted p-values."""
    p = pd.to_numeric(p_values, errors="coerce")
    valid = p.notna()
    adjusted = pd.Series(np.nan, index=p.index, dtype=float)

    if valid.sum() == 0:
        return adjusted

    values = p[valid].to_numpy()
    order = np.argsort(values)
    ranked = values[order]
    n = len(ranked)

    corrected = ranked * n / np.arange(1, n + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    corrected = np.clip(corrected, 0.0, 1.0)

    restored = np.empty(n, dtype=float)
    restored[order] = corrected
    adjusted.loc[valid] = restored
    return adjusted


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cliff's delta, positive when values in x tend to exceed values in y.
    Uses pairwise comparison; the dataset is small enough for this exact form.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return float("nan")

    comparisons = np.subtract.outer(x, y)
    greater = np.count_nonzero(comparisons > 0)
    less = np.count_nonzero(comparisons < 0)
    return float((greater - less) / comparisons.size)


def standardized_mean_difference(x: np.ndarray, y: np.ndarray) -> float:
    """Hedges-corrected standardized mean difference (positive: x > y)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return float("nan")

    nx, ny = len(x), len(y)
    pooled_numerator = (nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)
    pooled_denominator = nx + ny - 2
    if pooled_denominator <= 0:
        return float("nan")

    pooled_sd = math.sqrt(pooled_numerator / pooled_denominator)
    if pooled_sd == 0:
        return 0.0

    d = (np.mean(x) - np.mean(y)) / pooled_sd
    correction = 1.0 - (3.0 / (4.0 * (nx + ny) - 9.0))
    return float(d * correction)


def classify_column(column: str, series: pd.Series) -> tuple[str, str]:
    if column in IDENTIFIER_COLUMNS:
        return "identifier", "Identifier/path; prohibited from model input"
    if column in TARGET_COLUMNS:
        return "target", "Outcome label; prohibited from model input"
    if column in PRIOR_STUDY_COLUMNS:
        return "prior_study", "Published Minami metadata/prediction; prohibited from model input"
    if column in GROUPING_ONLY_COLUMNS:
        return "grouping_only", "Retained for fold-aware grouping, not prediction"
    if column in QC_ONLY_COLUMNS:
        return "qc_only", "Structure quality or pipeline QC; excluded from primary biology model"
    if column in CONFIGURATION_COLUMNS:
        return "configuration", "Constant analysis setting; not a protein feature"
    if not pd.api.types.is_numeric_dtype(series):
        return "non_numeric", "Non-numeric annotation; not used as a predictor"
    return "candidate_feature", "Eligible computed numeric feature pending QC"


def create_feature_manifest(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for column in df.columns:
        role, reason = classify_column(column, df[column])
        records.append(
            {
                "column": column,
                "role": role,
                "reason": reason,
                "dtype": str(df[column].dtype),
                "missing_count": int(df[column].isna().sum()),
                "missing_fraction": float(df[column].isna().mean()),
                "unique_nonmissing": int(df[column].nunique(dropna=True)),
            }
        )
    return pd.DataFrame(records)


def integrity_audit(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []

    def add_check(name: str, passed: bool, observed: Any, expected: Any, severity: str):
        checks.append(
            {
                "check": name,
                "passed": bool(passed),
                "observed": observed,
                "expected": expected,
                "severity": severity,
            }
        )
        if not passed and severity == "blocker":
            blockers.append(name)

    add_check("row_count", len(df) == EXPECTED_ROWS, len(df), EXPECTED_ROWS, "blocker")
    add_check(
        "positive_count",
        int((df["binary_label"] == 1).sum()) == EXPECTED_POSITIVES,
        int((df["binary_label"] == 1).sum()),
        EXPECTED_POSITIVES,
        "blocker",
    )
    add_check(
        "negative_count",
        int((df["binary_label"] == 0).sum()) == EXPECTED_NEGATIVES,
        int((df["binary_label"] == 0).sum()),
        EXPECTED_NEGATIVES,
        "blocker",
    )
    add_check(
        "binary_labels_only",
        set(df["binary_label"].dropna().unique()).issubset({0, 1}),
        sorted(df["binary_label"].dropna().unique().tolist()),
        [0, 1],
        "blocker",
    )

    for column in ("jwid", "current_uniprot_accession", "accession"):
        if column in df.columns:
            duplicate_count = int(df[column].duplicated(keep=False).sum())
            add_check(
                f"duplicate_{column}",
                duplicate_count == 0,
                duplicate_count,
                0,
                "blocker",
            )

    if "fasta_structure_length_match" in df.columns:
        mismatches = int((~df["fasta_structure_length_match"].fillna(False).astype(bool)).sum())
        add_check(
            "sequence_structure_length_mismatches",
            mismatches == 0,
            mismatches,
            0,
            "warning",
        )

    if "chain_break_count" in df.columns:
        chain_breaks = int((pd.to_numeric(df["chain_break_count"], errors="coerce") > 0).sum())
        add_check("proteins_with_chain_breaks", chain_breaks == 0, chain_breaks, 0, "warning")

    if {"accession", "current_uniprot_accession"}.issubset(df.columns):
        mismatch = int(
            (
                df["accession"].astype(str).str.strip()
                != df["current_uniprot_accession"].astype(str).str.strip()
            ).sum()
        )
        add_check("accession_alignment", mismatch == 0, mismatch, 0, "blocker")

    return pd.DataFrame(checks), blockers


def numeric_candidate_columns(
    df: pd.DataFrame,
    manifest: pd.DataFrame,
    maximum_missing_fraction: float,
) -> tuple[list[str], list[str], list[str]]:
    candidates = manifest.loc[
        manifest["role"] == "candidate_feature", "column"
    ].tolist()

    constant = [
        column
        for column in candidates
        if df[column].nunique(dropna=True) <= 1
    ]
    high_missing = [
        column
        for column in candidates
        if df[column].isna().mean() > maximum_missing_fraction
    ]

    eligible = [
        column
        for column in candidates
        if column not in constant and column not in high_missing
    ]
    return eligible, constant, high_missing


def univariate_statistics(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    records = []
    positives = df["binary_label"] == 1
    negatives = df["binary_label"] == 0

    for feature in feature_columns:
        positive = pd.to_numeric(df.loc[positives, feature], errors="coerce").dropna().to_numpy()
        negative = pd.to_numeric(df.loc[negatives, feature], errors="coerce").dropna().to_numpy()

        if len(positive) == 0 or len(negative) == 0:
            continue

        try:
            mann_whitney = stats.mannwhitneyu(
                positive,
                negative,
                alternative="two-sided",
                method="auto",
            )
            u_statistic = float(mann_whitney.statistic)
            p_value = float(mann_whitney.pvalue)
        except ValueError:
            u_statistic = float("nan")
            p_value = float("nan")

        records.append(
            {
                "feature": feature,
                "positive_n": len(positive),
                "negative_n": len(negative),
                "positive_mean": float(np.mean(positive)),
                "negative_mean": float(np.mean(negative)),
                "positive_median": float(np.median(positive)),
                "negative_median": float(np.median(negative)),
                "median_difference_positive_minus_negative": float(
                    np.median(positive) - np.median(negative)
                ),
                "mann_whitney_u": u_statistic,
                "p_value": p_value,
                "cliffs_delta": cliffs_delta(positive, negative),
                "hedges_g": standardized_mean_difference(positive, negative),
            }
        )

    result = pd.DataFrame(records)
    if result.empty:
        return result

    result["fdr_q_value"] = benjamini_hochberg(result["p_value"])
    result["absolute_cliffs_delta"] = result["cliffs_delta"].abs()
    result["absolute_hedges_g"] = result["hedges_g"].abs()
    result["direction"] = np.where(
        result["cliffs_delta"] > 0,
        "higher_in_positive",
        np.where(result["cliffs_delta"] < 0, "lower_in_positive", "no_direction"),
    )

    return result.sort_values(
        ["absolute_cliffs_delta", "fdr_q_value"],
        ascending=[False, True],
    ).reset_index(drop=True)


def high_correlation_pairs(
    df: pd.DataFrame,
    features: list[str],
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    correlation = df[features].corr(method="spearman")
    pairs = []

    for i, first in enumerate(features):
        for second in features[i + 1 :]:
            value = correlation.loc[first, second]
            if pd.notna(value) and abs(value) >= threshold:
                pairs.append(
                    {
                        "feature_1": first,
                        "feature_2": second,
                        "spearman_rho": float(value),
                        "absolute_rho": float(abs(value)),
                    }
                )

    pairs_df = pd.DataFrame(
        pairs,
        columns=["feature_1", "feature_2", "spearman_rho", "absolute_rho"],
    )
    if not pairs_df.empty:
        pairs_df = pairs_df.sort_values("absolute_rho", ascending=False)
    return correlation, pairs_df


def fold_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    fold_df = df[["jwid", "binary_label", "cohort_label", "sccs"]].copy()
    fold_df["sccs"] = fold_df["sccs"].fillna("").astype(str)
    fold_df["primary_sccs"] = fold_df["sccs"].str.split(";").str[0].str.strip()
    fold_df["scop_class"] = fold_df["primary_sccs"].str.extract(r"^([a-z])", expand=False)
    fold_df["contains_tim_barrel_c1"] = fold_df["sccs"].apply(
        lambda value: any(
            token.strip() == "c.1" or token.strip().startswith("c.1.")
            for token in value.split(";")
        )
    )

    class_counts = (
        fold_df.groupby(["cohort_label", "scop_class"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    class_totals = class_counts.groupby("cohort_label")["count"].transform("sum")
    class_counts["fraction_within_label"] = class_counts["count"] / class_totals

    primary_counts = (
        fold_df.groupby(["cohort_label", "primary_sccs"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["cohort_label", "count"], ascending=[True, False])
    )

    contingency = pd.crosstab(
        fold_df["binary_label"],
        fold_df["contains_tim_barrel_c1"],
    ).reindex(index=[0, 1], columns=[False, True], fill_value=0)

    odds_ratio, fisher_p = stats.fisher_exact(contingency.to_numpy())
    tim_summary = {
        "negative_non_tim": int(contingency.loc[0, False]),
        "negative_tim": int(contingency.loc[0, True]),
        "positive_non_tim": int(contingency.loc[1, False]),
        "positive_tim": int(contingency.loc[1, True]),
        "odds_ratio": float(odds_ratio),
        "fisher_exact_p_value": float(fisher_p),
    }

    return class_counts, primary_counts, tim_summary


def write_text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    figure = plt.figure(figsize=(8.5, 11))
    figure.suptitle(title, fontsize=16, y=0.96)
    wrapped_lines = []
    for line in lines:
        if not line:
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(textwrap.wrap(str(line), width=100) or [""])

    y = 0.91
    for line in wrapped_lines:
        figure.text(0.07, y, line, fontsize=9.5, va="top")
        y -= 0.024
        if y < 0.06:
            break

    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def plot_dataset_summary(pdf: PdfPages, df: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.8))

    counts = df["cohort_label"].value_counts().reindex(["positive", "negative"])
    axes[0].bar(counts.index, counts.values)
    axes[0].set_title("Benchmark class balance")
    axes[0].set_ylabel("Protein count")

    if "minami_structure_source" in df.columns:
        source_counts = pd.crosstab(
            df["minami_structure_source"],
            df["cohort_label"],
        )
        source_counts.plot(kind="bar", ax=axes[1])
        axes[1].set_title("Original Minami structure source")
        axes[1].set_xlabel("Structure source")
        axes[1].set_ylabel("Protein count")
        axes[1].tick_params(axis="x", rotation=0)
    else:
        axes[1].axis("off")

    figure.tight_layout()
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def plot_missingness(pdf: PdfPages, missingness: pd.DataFrame) -> None:
    nonzero = missingness[missingness["missing_count"] > 0].copy()
    if nonzero.empty:
        write_text_page(
            pdf,
            "Missingness",
            ["No missing values were detected in the audited columns."],
        )
        return

    nonzero = nonzero.sort_values("missing_fraction").tail(25)
    figure, axis = plt.subplots(figsize=(10, 7))
    axis.barh(nonzero["column"], nonzero["missing_fraction"])
    axis.set_xlabel("Missing fraction")
    axis.set_title("Features with missing values")
    figure.tight_layout()
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def plot_top_effects(
    pdf: PdfPages,
    statistics_df: pd.DataFrame,
    top_n: int,
) -> None:
    top = statistics_df.head(top_n).sort_values("cliffs_delta")
    figure, axis = plt.subplots(figsize=(10, 8))
    axis.barh(top["feature"], top["cliffs_delta"])
    axis.axvline(0)
    axis.set_xlabel("Cliff's delta (positive class minus negative class)")
    axis.set_title(f"Top {len(top)} univariate structural differences")
    figure.tight_layout()
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def plot_correlation_matrix(
    pdf: PdfPages,
    correlation: pd.DataFrame,
    statistics_df: pd.DataFrame,
    maximum_features: int = 25,
) -> None:
    ranked = statistics_df["feature"].tolist()
    selected = [feature for feature in ranked if feature in correlation.columns][
        :maximum_features
    ]
    if len(selected) < 2:
        return

    matrix = correlation.loc[selected, selected]
    figure, axis = plt.subplots(figsize=(11, 10))
    image = axis.imshow(matrix.to_numpy(), vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(np.arange(len(selected)))
    axis.set_yticks(np.arange(len(selected)))
    axis.set_xticklabels(selected, rotation=90, fontsize=7)
    axis.set_yticklabels(selected, fontsize=7)
    axis.set_title("Spearman correlation among top-ranked features")
    figure.colorbar(image, ax=axis, label="Spearman rho")
    figure.tight_layout()
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def plot_fold_composition(
    pdf: PdfPages,
    class_counts: pd.DataFrame,
) -> None:
    pivot = class_counts.pivot(
        index="scop_class",
        columns="cohort_label",
        values="fraction_within_label",
    ).fillna(0)
    if pivot.empty:
        return

    pivot.plot(kind="bar", figsize=(10, 6))
    figure = plt.gcf()
    axis = plt.gca()
    axis.set_title("SCOP primary-class composition by benchmark label")
    axis.set_xlabel("SCOP class")
    axis.set_ylabel("Fraction within label")
    axis.tick_params(axis="x", rotation=0)
    figure.tight_layout()
    pdf.savefig(figure, bbox_inches="tight")
    plt.close(figure)


def save_feature_distribution_plots(
    df: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    positive_mask = df["binary_label"] == 1
    negative_mask = df["binary_label"] == 0

    for feature in feature_columns:
        positive = pd.to_numeric(df.loc[positive_mask, feature], errors="coerce").dropna()
        negative = pd.to_numeric(df.loc[negative_mask, feature], errors="coerce").dropna()
        if positive.empty or negative.empty:
            continue

        figure, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(
            [negative.to_numpy(), positive.to_numpy()],
            bins="auto",
            label=["negative", "positive"],
            alpha=0.7,
        )
        axes[0].set_title(feature)
        axes[0].set_xlabel("Feature value")
        axes[0].set_ylabel("Count")
        axes[0].legend()

        axes[1].boxplot(
            [negative.to_numpy(), positive.to_numpy()],
            tick_labels=["negative", "positive"],
            showfliers=True,
        )
        axes[1].set_title("Class comparison")
        axes[1].set_ylabel("Feature value")

        figure.tight_layout()
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", feature)
        figure.savefig(output_dir / f"{safe_name}.png", dpi=180, bbox_inches="tight")
        plt.close(figure)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    distribution_dir = figures_dir / "feature_distributions"

    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"ERROR: Input not found: {input_path}", file=sys.stderr)
        return 2

    df = pd.read_csv(input_path)

    required = {"binary_label", "cohort_label", "sccs"}
    missing_required = required - set(df.columns)
    if missing_required:
        print(
            "ERROR: Missing required columns: "
            + ", ".join(sorted(missing_required)),
            file=sys.stderr,
        )
        return 2

    df["binary_label"] = pd.to_numeric(df["binary_label"], errors="coerce")

    # 1. Integrity
    integrity_df, blockers = integrity_audit(df)
    integrity_df.to_csv(tables_dir / "integrity_audit.csv", index=False)

    # 2. Column roles / leakage
    manifest = create_feature_manifest(df)
    manifest.to_csv(tables_dir / "feature_role_manifest.csv", index=False)

    # 3. Candidate features and basic QC
    eligible_features, constant_features, high_missing_features = numeric_candidate_columns(
        df,
        manifest,
        args.maximum_missing_fraction,
    )

    missingness = manifest[
        ["column", "role", "missing_count", "missing_fraction", "unique_nonmissing"]
    ].sort_values(["missing_fraction", "column"], ascending=[False, True])
    missingness.to_csv(tables_dir / "missingness_report.csv", index=False)

    pd.DataFrame({"feature": constant_features}).to_csv(
        tables_dir / "constant_features.csv",
        index=False,
    )
    pd.DataFrame({"feature": high_missing_features}).to_csv(
        tables_dir / "high_missing_features.csv",
        index=False,
    )

    # 4. Univariate analysis
    statistics_df = univariate_statistics(df, eligible_features)
    statistics_df.to_csv(tables_dir / "univariate_feature_statistics.csv", index=False)

    # 5. Correlations
    correlation, correlation_pairs = high_correlation_pairs(
        df,
        eligible_features,
        args.correlation_threshold,
    )
    correlation.to_csv(tables_dir / "spearman_correlation_matrix.csv")
    correlation_pairs.to_csv(tables_dir / "high_correlation_pairs.csv", index=False)

    # 6. Fold bias
    class_counts, primary_counts, tim_summary = fold_tables(df)
    class_counts.to_csv(tables_dir / "scop_class_composition.csv", index=False)
    primary_counts.to_csv(tables_dir / "primary_sccs_composition.csv", index=False)
    with (tables_dir / "tim_barrel_enrichment.json").open("w") as handle:
        json.dump(tim_summary, handle, indent=2)

    # 7. Model-ready exports
    broad_output_columns = [
        "current_uniprot_accession",
        "gene_minami",
        "jwid",
        "binary_label",
        "cohort_label",
        "sccs",
    ] + eligible_features

    broad_df = df[broad_output_columns].copy()
    broad_df.to_csv(output_dir / "ml_dataset_structure_broad.csv", index=False)

    core_features = [
        feature for feature in CORE_FEATURE_CANDIDATES if feature in eligible_features
    ]
    core_output_columns = [
        "current_uniprot_accession",
        "gene_minami",
        "jwid",
        "binary_label",
        "cohort_label",
        "sccs",
    ] + core_features

    core_df = df[core_output_columns].copy()
    core_df.to_csv(output_dir / "ml_dataset_structure_core.csv", index=False)
    pd.DataFrame({"feature": core_features}).to_csv(
        tables_dir / "core_feature_list.csv",
        index=False,
    )

    # 8. Distribution figures for every eligible feature
    save_feature_distribution_plots(df, eligible_features, distribution_dir)

    # 9. Readiness assessment
    readiness_reasons = []
    if blockers:
        readiness_reasons.extend(f"Integrity blocker: {item}" for item in blockers)
    if len(eligible_features) == 0:
        readiness_reasons.append("No eligible numeric structural features remain")
    if df["binary_label"].isna().any():
        readiness_reasons.append("Missing binary labels remain")
    if not core_features:
        readiness_reasons.append("No predefined core features were found")

    ready = len(readiness_reasons) == 0

    summary = {
        "input_file": str(input_path),
        "row_count": int(len(df)),
        "positive_count": int((df["binary_label"] == 1).sum()),
        "negative_count": int((df["binary_label"] == 0).sum()),
        "total_columns": int(df.shape[1]),
        "candidate_numeric_features_before_qc": int(
            (manifest["role"] == "candidate_feature").sum()
        ),
        "eligible_broad_features": int(len(eligible_features)),
        "core_features": int(len(core_features)),
        "constant_features_removed": constant_features,
        "high_missing_features_removed": high_missing_features,
        "high_correlation_pair_count": int(len(correlation_pairs)),
        "tim_barrel_enrichment": tim_summary,
        "ready_for_next_stage": bool(ready),
        "readiness_reasons": readiness_reasons,
        "important_method_note": (
            "SCCS is retained only for fold-aware grouping and bias analysis. "
            "Minami predictions, published RCO, structure source, identifiers, "
            "and labels are excluded from predictor columns."
        ),
    }
    with (output_dir / "qc_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)

    # 10. PDF report
    pdf_path = output_dir / "QC_Report.pdf"
    with PdfPages(pdf_path) as pdf:
        status_text = "READY FOR HOMOLOGY/FOLD GROUPING" if ready else "NOT READY"
        write_text_page(
            pdf,
            "GroEL/GroES Benchmark Dataset Audit",
            [
                f"Dataset status: {status_text}",
                "",
                f"Input: {input_path}",
                f"Proteins: {len(df)}",
                f"Positive obligate substrates: {(df['binary_label'] == 1).sum()}",
                f"Experimentally defined negatives: {(df['binary_label'] == 0).sum()}",
                f"Input columns: {df.shape[1]}",
                f"Broad eligible structural features: {len(eligible_features)}",
                f"Restrained core structural features: {len(core_features)}",
                "",
                "This report audits the dataset before any model training. "
                "The next valid stage is sequence-homology and fold-aware grouping, "
                "not ordinary random cross-validation.",
            ],
        )

        plot_dataset_summary(pdf, df)

        integrity_lines = []
        for _, row in integrity_df.iterrows():
            marker = "PASS" if row["passed"] else "FAIL"
            integrity_lines.append(
                f"{marker}: {row['check']} - observed={row['observed']}; "
                f"expected={row['expected']}; severity={row['severity']}"
            )
        write_text_page(pdf, "Dataset Integrity Audit", integrity_lines)

        role_counts = manifest["role"].value_counts()
        leakage_lines = [
            f"{role}: {count} columns"
            for role, count in role_counts.items()
        ]
        leakage_lines.extend(
            [
                "",
                "Prohibited predictors include identifiers, labels, Minami's "
                "published RCO, Minami's RW/RWRR predictions, and original "
                "structure-source metadata.",
                "",
                "SCCS is grouping-only. It will be used to test fold generalization "
                "but will not be supplied to the classifier as a feature.",
                "",
                "AlphaFold pLDDT and pipeline QC variables are excluded from the "
                "primary biological feature set. They may be used in sensitivity "
                "analyses to check whether model performance depends on structure quality.",
            ]
        )
        write_text_page(pdf, "Leakage and Feature-Role Audit", leakage_lines)

        plot_missingness(pdf, missingness)

        quality_lines = [
            f"Constant candidate features removed: {len(constant_features)}",
            *[f"  - {feature}" for feature in constant_features],
            "",
            f"Features above missingness threshold removed: {len(high_missing_features)}",
            *[f"  - {feature}" for feature in high_missing_features],
            "",
            f"Highly correlated pairs at |rho| >= {args.correlation_threshold}: "
            f"{len(correlation_pairs)}",
            "",
            "High correlation is reported rather than automatically deleted. "
            "Feature reduction must occur inside model-training folds or be fixed "
            "before evaluation.",
        ]
        write_text_page(pdf, "Feature Quality", quality_lines)

        if not statistics_df.empty:
            plot_top_effects(pdf, statistics_df, args.top_feature_count)
            top_lines = []
            for _, row in statistics_df.head(args.top_feature_count).iterrows():
                top_lines.append(
                    f"{row['feature']}: Cliff's delta={row['cliffs_delta']:.3f}, "
                    f"Hedges g={row['hedges_g']:.3f}, "
                    f"FDR q={row['fdr_q_value']:.3g}, "
                    f"{row['direction']}"
                )
            write_text_page(pdf, "Top Univariate Feature Statistics", top_lines)
            plot_correlation_matrix(pdf, correlation, statistics_df)

        plot_fold_composition(pdf, class_counts)
        write_text_page(
            pdf,
            "TIM-Barrel Enrichment Audit",
            [
                f"Positive TIM-barrel proteins: {tim_summary['positive_tim']}",
                f"Positive non-TIM proteins: {tim_summary['positive_non_tim']}",
                f"Negative TIM-barrel proteins: {tim_summary['negative_tim']}",
                f"Negative non-TIM proteins: {tim_summary['negative_non_tim']}",
                f"Fisher exact odds ratio: {tim_summary['odds_ratio']:.3f}",
                f"Fisher exact p-value: {tim_summary['fisher_exact_p_value']:.3g}",
                "",
                "A strong fold-label association is a benchmark bias, not proof that "
                "fold identity causes GroEL dependence. Subsequent evaluation must "
                "include fold-aware and homology-aware splits.",
            ],
        )

        final_lines = [
            f"Status: {status_text}",
            "",
            f"Broad ML dataset: {output_dir / 'ml_dataset_structure_broad.csv'}",
            f"Core ML dataset: {output_dir / 'ml_dataset_structure_core.csv'}",
            "",
            "Next stage:",
            "1. Download canonical sequences for all 181 proteins if not already centralized.",
            "2. Cluster sequences by homology.",
            "3. Construct grouped cross-validation folds that keep homologs together.",
            "4. Add a stricter fold-held-out evaluation using SCCS.",
            "5. Only then fit baselines and Random Forest models.",
        ]
        if readiness_reasons:
            final_lines.extend(["", "Blocking issues:", *readiness_reasons])
        write_text_page(pdf, "Readiness Decision and Next Steps", final_lines)

    print("Stage 06 complete")
    print(f"  Proteins:             {len(df)}")
    print(f"  Positive / negative:  {(df['binary_label'] == 1).sum()} / {(df['binary_label'] == 0).sum()}")
    print(f"  Broad features:       {len(eligible_features)}")
    print(f"  Core features:        {len(core_features)}")
    print(f"  Constant removed:     {len(constant_features)}")
    print(f"  Correlated pairs:     {len(correlation_pairs)}")
    print(f"  Status:               {'READY' if ready else 'NOT READY'}")
    print(f"  PDF report:           {pdf_path}")
    print(f"  Broad dataset:        {output_dir / 'ml_dataset_structure_broad.csv'}")
    print(f"  Core dataset:         {output_dir / 'ml_dataset_structure_core.csv'}")

    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
