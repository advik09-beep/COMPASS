#!/usr/bin/env python3
"""
Stage 10A: Held-out permutation importance and error analysis.

Inputs:
- Frozen Feature Set v2 datasets
- Frozen Stage 07 fold assignments
- Stage 09 out-of-fold predictions
- Stage 09 selected hyperparameters

Analyses:
1. Refit each selected Random Forest on its outer-training partition.
2. Calculate permutation importance on the untouched outer-test partition.
3. Aggregate feature importance across folds.
4. Generate pooled out-of-fold error analyses.
5. Identify high-confidence false positives and false negatives.
6. Identify disagreement cases among sequence, structure, and combined models.
7. Extract the SufS/CsdA case study.

No feature engineering or retuning occurs.

Example:
    python3 scripts/10A_permutation_importance_and_error_analysis.py \
      --feature-dir data/frozen/feature_set_v2 \
      --fold-assignments data/grouping/stage07/fold_assignments_v1.0.csv \
      --oof-predictions results/stage09/stage09_oof_predictions.csv \
      --selected-hyperparameters results/stage09/stage09_selected_hyperparameters.csv \
      --output-dir results/stage10A
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)


IDENTIFIER = "current_uniprot_accession"
TARGET = "binary_label"

FEATURE_FILES = {
    "sequence_final": "feature_set_v2_sequence_final.csv",
    "structure_minimal_final": "feature_set_v2_structure_minimal_final.csv",
    "sequence_plus_structure_minimal": (
        "feature_set_v2_sequence_plus_structure_minimal.csv"
    ),
}

OUTER_SCHEMES = {
    "homology_cv_fold": "homology_cluster_30",
    "homology_cv_fold_sensitivity": "homology_cluster_40",
}

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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Held-out permutation importance and error analysis."
    )
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--fold-assignments", required=True)
    parser.add_argument("--oof-predictions", required=True)
    parser.add_argument("--selected-hyperparameters", required=True)
    parser.add_argument("--output-dir", default="results/stage10A")
    parser.add_argument("--permutation-repeats", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--importance-scoring",
        choices=["roc_auc", "average_precision", "balanced_accuracy", "mcc"],
        default="roc_auc",
    )
    parser.add_argument(
        "--high-confidence-positive-threshold",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--high-confidence-negative-threshold",
        type=float,
        default=0.25,
    )
    parser.add_argument("--top-features", type=int, default=20)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def scoring_callable(name: str):
    if name == "roc_auc":
        def scorer(estimator, X, y):
            if len(np.unique(y)) < 2:
                return float("nan")
            return roc_auc_score(y, estimator.predict_proba(X)[:, 1])
        return scorer

    if name == "average_precision":
        def scorer(estimator, X, y):
            if len(np.unique(y)) < 2:
                return float("nan")
            return average_precision_score(y, estimator.predict_proba(X)[:, 1])
        return scorer

    if name == "balanced_accuracy":
        def scorer(estimator, X, y):
            return balanced_accuracy_score(y, estimator.predict(X))
        return scorer

    if name == "mcc":
        def scorer(estimator, X, y):
            return matthews_corrcoef(y, estimator.predict(X))
        return scorer

    raise ValueError(name)


def normalize_param_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def extract_rf_params(row: pd.Series) -> dict[str, Any]:
    permitted = {
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "class_weight",
        "max_samples",
        "bootstrap",
    }

    params = {}
    for name in permitted:
        if name not in row.index:
            continue
        value = normalize_param_value(row[name])

        if name in {"n_estimators", "min_samples_split", "min_samples_leaf"}:
            if value is not None:
                value = int(value)
        elif name == "max_depth":
            if value is not None:
                value = int(value)
        elif name in {"max_features", "max_samples"}:
            if isinstance(value, str):
                if value.lower() in {"none", "nan", ""}:
                    value = None
                elif value not in {"sqrt", "log2"}:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
        elif name == "bootstrap":
            if isinstance(value, str):
                value = value.lower() == "true"

        params[name] = value

    return params


def load_feature_dataset(path: Path, feature_set: str) -> tuple[pd.DataFrame, list[str]]:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    features = [
        column
        for column in df.columns
        if column not in METADATA_COLUMNS
        and pd.api.types.is_numeric_dtype(df[column])
        and df[column].nunique(dropna=True) > 1
    ]

    if not features:
        raise RuntimeError(f"{feature_set} contains no usable predictors")

    return df, features


def subset_frame(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    if subset == "full":
        return df.copy()
    if subset == "non_tim":
        return df[~df["is_tim_barrel"].astype(bool)].copy()
    raise ValueError(subset)


def run_permutation_importance(
    feature_dir: Path,
    folds: pd.DataFrame,
    selected_params: pd.DataFrame,
    repeats: int,
    scoring_name: str,
    seed: int,
    n_jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scorer = scoring_callable(scoring_name)
    importance_rows = []
    fold_summary_rows = []

    for feature_set, filename in FEATURE_FILES.items():
        dataset, features = load_feature_dataset(
            feature_dir / filename,
            feature_set,
        )

        data = dataset.merge(
            folds[
                [
                    IDENTIFIER,
                    "homology_cv_fold",
                    "homology_cv_fold_sensitivity",
                    "homology_cluster_30",
                    "homology_cluster_40",
                    "is_tim_barrel",
                ]
            ],
            on=IDENTIFIER,
            how="left",
            validate="one_to_one",
        )

        parameter_rows = selected_params[
            selected_params["feature_set"] == feature_set
        ].copy()

        for _, param_row in parameter_rows.iterrows():
            subset = str(param_row["subset"])
            fold_column = str(param_row["outer_fold_column"])
            outer_fold = int(param_row["outer_fold"])

            if fold_column not in OUTER_SCHEMES:
                continue

            subset_data = subset_frame(data, subset)
            train = subset_data[subset_data[fold_column] != outer_fold].copy()
            test = subset_data[subset_data[fold_column] == outer_fold].copy()

            if train.empty or test.empty:
                raise RuntimeError(
                    f"Empty split for {feature_set}/{subset}/{fold_column}/{outer_fold}"
                )
            if train[TARGET].nunique() < 2:
                raise RuntimeError("Outer training set lacks both classes")

            params = extract_rf_params(param_row)

            model = RandomForestClassifier(
                criterion="gini",
                random_state=seed + outer_fold,
                n_jobs=n_jobs,
                **params,
            )
            model.fit(train[features], train[TARGET].astype(int))

            baseline_score = scorer(
                model,
                test[features],
                test[TARGET].astype(int),
            )

            if not math.isfinite(float(baseline_score)):
                fold_summary_rows.append(
                    {
                        "subset": subset,
                        "feature_set": feature_set,
                        "outer_fold_column": fold_column,
                        "outer_fold": outer_fold,
                        "test_n": len(test),
                        "test_positive_n": int(test[TARGET].sum()),
                        "test_negative_n": int((test[TARGET] == 0).sum()),
                        "importance_scoring": scoring_name,
                        "baseline_score": baseline_score,
                        "importance_status": "skipped_single_class_test_fold",
                    }
                )
                continue

            result = permutation_importance(
                model,
                test[features],
                test[TARGET].astype(int),
                scoring=scorer,
                n_repeats=repeats,
                random_state=seed + outer_fold * 1000 + len(feature_set),
                n_jobs=n_jobs,
            )

            fold_summary_rows.append(
                {
                    "subset": subset,
                    "feature_set": feature_set,
                    "outer_fold_column": fold_column,
                    "outer_fold": outer_fold,
                    "test_n": len(test),
                    "test_positive_n": int(test[TARGET].sum()),
                    "test_negative_n": int((test[TARGET] == 0).sum()),
                    "importance_scoring": scoring_name,
                    "baseline_score": float(baseline_score),
                    "importance_status": "complete",
                }
            )

            for index, feature in enumerate(features):
                repeated_values = result.importances[index]
                importance_rows.append(
                    {
                        "subset": subset,
                        "feature_set": feature_set,
                        "outer_fold_column": fold_column,
                        "outer_fold": outer_fold,
                        "feature": feature,
                        "test_n": len(test),
                        "baseline_score": float(baseline_score),
                        "importance_mean": float(np.mean(repeated_values)),
                        "importance_std": float(np.std(repeated_values, ddof=1)),
                        "importance_median": float(np.median(repeated_values)),
                        "importance_min": float(np.min(repeated_values)),
                        "importance_max": float(np.max(repeated_values)),
                        "positive_importance_fraction": float(
                            np.mean(repeated_values > 0)
                        ),
                        "permutation_repeats": repeats,
                        "importance_scoring": scoring_name,
                    }
                )

            print(
                f"Permutation importance complete: {feature_set} | {subset} | "
                f"{fold_column} | fold {outer_fold}"
            )

    return pd.DataFrame(importance_rows), pd.DataFrame(fold_summary_rows)


def summarize_importance(by_fold: pd.DataFrame) -> pd.DataFrame:
    if by_fold.empty:
        return pd.DataFrame()

    summary = (
        by_fold.groupby(
            [
                "subset",
                "feature_set",
                "outer_fold_column",
                "feature",
                "importance_scoring",
            ]
        )
        .agg(
            fold_count=("outer_fold", "nunique"),
            mean_importance=("importance_mean", "mean"),
            sd_across_folds=("importance_mean", "std"),
            median_importance=("importance_mean", "median"),
            minimum_fold_importance=("importance_mean", "min"),
            maximum_fold_importance=("importance_mean", "max"),
            mean_positive_importance_fraction=(
                "positive_importance_fraction",
                "mean",
            ),
        )
        .reset_index()
    )

    summary["sem_across_folds"] = (
        summary["sd_across_folds"]
        / np.sqrt(summary["fold_count"].clip(lower=1))
    )
    summary["importance_rank"] = (
        summary.groupby(
            ["subset", "feature_set", "outer_fold_column"]
        )["mean_importance"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    return summary.sort_values(
        [
            "subset",
            "outer_fold_column",
            "feature_set",
            "importance_rank",
        ]
    )


def prepare_error_analysis(
    oof: pd.DataFrame,
    high_positive: float,
    high_negative: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = oof[
        (oof["subset"] == "full")
        & (oof["outer_fold_column"] == "homology_cv_fold")
    ].copy()

    primary["error_type_0_5"] = np.select(
        [
            (primary[TARGET] == 1) & (primary["predicted_label_0_5"] == 1),
            (primary[TARGET] == 0) & (primary["predicted_label_0_5"] == 0),
            (primary[TARGET] == 0) & (primary["predicted_label_0_5"] == 1),
            (primary[TARGET] == 1) & (primary["predicted_label_0_5"] == 0),
        ],
        [
            "true_positive",
            "true_negative",
            "false_positive",
            "false_negative",
        ],
        default="unknown",
    )

    primary["probability_margin_from_0_5"] = (
        primary["predicted_probability"] - 0.5
    ).abs()

    high_fp = primary[
        (primary[TARGET] == 0)
        & (primary["predicted_probability"] >= high_positive)
    ].copy().sort_values("predicted_probability", ascending=False)

    high_fn = primary[
        (primary[TARGET] == 1)
        & (primary["predicted_probability"] <= high_negative)
    ].copy().sort_values("predicted_probability", ascending=True)

    pivot_prob = primary.pivot_table(
        index=[
            IDENTIFIER,
            "gene_minami",
            "jwid",
            TARGET,
            "is_tim_barrel",
        ],
        columns="feature_set",
        values="predicted_probability",
        aggfunc="first",
    ).reset_index()

    required_models = {
        "sequence_final",
        "structure_minimal_final",
        "sequence_plus_structure_minimal",
    }
    missing_models = required_models - set(pivot_prob.columns)
    if missing_models:
        raise RuntimeError(
            f"Missing OOF feature-set probabilities: {sorted(missing_models)}"
        )

    pivot_prob["sequence_label"] = (
        pivot_prob["sequence_final"] >= 0.5
    ).astype(int)
    pivot_prob["structure_label"] = (
        pivot_prob["structure_minimal_final"] >= 0.5
    ).astype(int)
    pivot_prob["combined_label"] = (
        pivot_prob["sequence_plus_structure_minimal"] >= 0.5
    ).astype(int)

    pivot_prob["sequence_structure_disagree"] = (
        pivot_prob["sequence_label"] != pivot_prob["structure_label"]
    )
    pivot_prob["combined_correct"] = (
        pivot_prob["combined_label"] == pivot_prob[TARGET]
    )
    pivot_prob["sequence_correct"] = (
        pivot_prob["sequence_label"] == pivot_prob[TARGET]
    )
    pivot_prob["structure_correct"] = (
        pivot_prob["structure_label"] == pivot_prob[TARGET]
    )

    disagreement = pivot_prob[
        pivot_prob["sequence_structure_disagree"]
    ].copy()

    disagreement["resolution_type"] = np.select(
        [
            disagreement["combined_correct"]
            & ~disagreement["sequence_correct"]
            & disagreement["structure_correct"],
            disagreement["combined_correct"]
            & disagreement["sequence_correct"]
            & ~disagreement["structure_correct"],
            disagreement["combined_correct"]
            & ~disagreement["sequence_correct"]
            & ~disagreement["structure_correct"],
            ~disagreement["combined_correct"],
        ],
        [
            "combined_follows_structure_correctly",
            "combined_follows_sequence_correctly",
            "combined_rescues_both",
            "combined_still_incorrect",
        ],
        default="other",
    )

    gene_lower = pivot_prob["gene_minami"].fillna("").astype(str).str.lower()
    sufs_csda = pivot_prob[
        gene_lower.isin({"sufs", "csda"})
    ].copy()

    return primary, high_fp, high_fn, disagreement, sufs_csda


def plot_top_importance(
    summary: pd.DataFrame,
    output_dir: Path,
    top_n: int,
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    filtered = summary[
        (summary["subset"] == "full")
        & (summary["outer_fold_column"] == "homology_cv_fold")
    ].copy()

    for feature_set, group in filtered.groupby("feature_set"):
        top = group.nsmallest(top_n, "importance_rank").copy()
        top = top.sort_values("mean_importance")

        fig, axis = plt.subplots(figsize=(9, 7))
        axis.barh(
            top["feature"],
            top["mean_importance"],
            xerr=top["sem_across_folds"].fillna(0),
        )
        axis.axvline(0, linewidth=1)
        axis.set_xlabel("Mean held-out permutation importance")
        axis.set_title(
            f"Permutation importance: {feature_set.replace('_', ' ')}"
        )
        fig.tight_layout()
        fig.savefig(
            figures_dir / f"permutation_importance_{feature_set}.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            figures_dir / f"permutation_importance_{feature_set}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_probability_distributions(
    error_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for feature_set, group in error_table.groupby("feature_set"):
        positive = group.loc[
            group[TARGET] == 1,
            "predicted_probability",
        ]
        negative = group.loc[
            group[TARGET] == 0,
            "predicted_probability",
        ]

        fig, axis = plt.subplots(figsize=(8, 5.5))
        axis.hist(
            [negative.to_numpy(), positive.to_numpy()],
            bins=np.linspace(0, 1, 21),
            label=["Negative", "Positive"],
            alpha=0.7,
        )
        axis.axvline(0.5, linestyle="--", linewidth=1)
        axis.set_xlabel("Out-of-fold predicted probability")
        axis.set_ylabel("Protein count")
        axis.set_title(
            f"Probability distributions: {feature_set.replace('_', ' ')}"
        )
        axis.legend()
        fig.tight_layout()
        fig.savefig(
            figures_dir / f"probability_distribution_{feature_set}.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            figures_dir / f"probability_distribution_{feature_set}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


def main() -> int:
    args = parse_args()

    feature_dir = Path(args.feature_dir)
    fold_path = Path(args.fold_assignments)
    oof_path = Path(args.oof_predictions)
    params_path = Path(args.selected_hyperparameters)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for path, label in [
        (feature_dir, "feature directory"),
        (fold_path, "fold assignments"),
        (oof_path, "OOF predictions"),
        (params_path, "selected hyperparameters"),
    ]:
        if not path.exists():
            print(f"ERROR: {label} not found: {path}", file=sys.stderr)
            return 2

    folds = pd.read_csv(fold_path)
    oof = pd.read_csv(oof_path)
    selected_params = pd.read_csv(params_path)

    by_fold, fold_summary = run_permutation_importance(
        feature_dir=feature_dir,
        folds=folds,
        selected_params=selected_params,
        repeats=args.permutation_repeats,
        scoring_name=args.importance_scoring,
        seed=args.random_seed,
        n_jobs=args.n_jobs,
    )

    importance_summary = summarize_importance(by_fold)

    (
        error_table,
        high_fp,
        high_fn,
        disagreement,
        sufs_csda,
    ) = prepare_error_analysis(
        oof,
        high_positive=args.high_confidence_positive_threshold,
        high_negative=args.high_confidence_negative_threshold,
    )

    by_fold.to_csv(
        output_dir / "permutation_importance_by_fold.csv",
        index=False,
    )
    importance_summary.to_csv(
        output_dir / "permutation_importance_summary.csv",
        index=False,
    )
    fold_summary.to_csv(
        output_dir / "permutation_importance_fold_audit.csv",
        index=False,
    )
    error_table.to_csv(
        output_dir / "error_analysis_all_proteins.csv",
        index=False,
    )
    high_fp.to_csv(
        output_dir / "high_confidence_false_positives.csv",
        index=False,
    )
    high_fn.to_csv(
        output_dir / "high_confidence_false_negatives.csv",
        index=False,
    )
    disagreement.to_csv(
        output_dir / "model_disagreement_cases.csv",
        index=False,
    )
    sufs_csda.to_csv(
        output_dir / "sufS_csdA_case.csv",
        index=False,
    )

    plot_top_importance(
        importance_summary,
        output_dir,
        top_n=args.top_features,
    )
    plot_probability_distributions(error_table, output_dir)

    primary_importance = importance_summary[
        (importance_summary["subset"] == "full")
        & (
            importance_summary["outer_fold_column"]
            == "homology_cv_fold"
        )
    ].copy()

    top_features = {}
    for feature_set, group in primary_importance.groupby("feature_set"):
        top_features[feature_set] = (
            group.sort_values("importance_rank")
            .head(10)[
                [
                    "feature",
                    "mean_importance",
                    "sd_across_folds",
                    "mean_positive_importance_fraction",
                ]
            ]
            .to_dict(orient="records")
        )

    summary = {
        "stage": "10A",
        "importance_scoring": args.importance_scoring,
        "permutation_repeats": args.permutation_repeats,
        "importance_rows": len(by_fold),
        "importance_fold_runs": len(fold_summary),
        "high_confidence_positive_threshold": (
            args.high_confidence_positive_threshold
        ),
        "high_confidence_negative_threshold": (
            args.high_confidence_negative_threshold
        ),
        "high_confidence_false_positive_rows": len(high_fp),
        "high_confidence_false_negative_rows": len(high_fn),
        "disagreement_case_count": len(disagreement),
        "sufS_csdA_case_rows": len(sufs_csda),
        "top_primary_features": top_features,
        "note": (
            "Permutation importance was calculated only on untouched outer "
            "test folds using the exact hyperparameters selected in Stage 09."
        ),
    }
    (output_dir / "stage10A_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    methods = {
        "model_source": "Stage 09 selected Random Forest hyperparameters",
        "feature_set_version": "2.0",
        "importance_method": "held-out permutation importance",
        "importance_scoring": args.importance_scoring,
        "permutation_repeats": args.permutation_repeats,
        "primary_grouping": "30% MMseqs2 homology-grouped outer folds",
        "sensitivity_grouping": "40% MMseqs2 homology-grouped outer folds",
        "subsets": ["full", "non_tim"],
        "error_analysis_source": "Stage 09 pooled outer-fold OOF predictions",
        "note": (
            "No hyperparameter tuning, threshold optimization, or feature "
            "selection was performed in Stage 10A."
        ),
    }
    (output_dir / "stage10A_methods.json").write_text(
        json.dumps(methods, indent=2)
    )

    print("Stage 10A complete")
    print(f"  Importance fold runs:      {len(fold_summary)}")
    print(f"  Importance rows:           {len(by_fold)}")
    print(f"  High-confidence FPs:       {len(high_fp)}")
    print(f"  High-confidence FNs:       {len(high_fn)}")
    print(f"  Model disagreement cases: {len(disagreement)}")
    print(f"  SufS/CsdA rows:            {len(sufs_csda)}")
    print(f"  Output directory:          {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
