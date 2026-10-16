#!/usr/bin/env python3
"""
Stage 09 v2: Final nested grouped Random Forest pipeline.

Inputs
------
- Corrected Benchmark v1.1 labels embedded in frozen Feature Set v3.
- Frozen Stage 07 homology-grouped folds.

Primary models
--------------
1. sequence_final
2. structure_minimal_final
3. sequence_plus_structure_minimal

Secondary models
----------------
4. structure_core_final
5. sequence_plus_structure_core

Evaluation
----------
- Primary outer CV: frozen 30% MMseqs2 homology-grouped folds.
- Sensitivity outer CV: frozen 40% MMseqs2 homology-grouped folds.
- Inner CV: grouped 3-fold StratifiedGroupKFold within each outer-training set.
- Hyperparameter selection: mean inner-fold MCC at threshold 0.5.
- Threshold reporting:
    a) fixed 0.5
    b) inner-CV MCC-maximizing threshold, chosen without outer-test labels.
- Subsets:
    a) full benchmark
    b) non-TIM benchmark
- Permutation importance:
    calculated only on untouched outer-test folds.
- Saved models:
    every selected outer-fold model is serialized with joblib.
- Calibration diagnostics:
    Brier score and reliability-bin tables from pooled OOF probabilities.
- No feature engineering or relabeling occurs here.

Example
-------
python3 scripts/09_v2_final_random_forest.py \
  --feature-dir data/frozen/feature_set_v3 \
  --fold-assignments data/grouping/stage07/fold_assignments_v1.0.csv \
  --output-dir results/stage09_v2 \
  --include-non-tim
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import ParameterSampler, StratifiedGroupKFold


IDENTIFIER = "current_uniprot_accession"
TARGET = "binary_label"

FEATURE_FILES = {
    "sequence_final": "feature_set_v3_sequence_final.csv",
    "structure_minimal_final": "feature_set_v3_structure_minimal_final.csv",
    "structure_core_final": "feature_set_v3_structure_core_final.csv",
    "sequence_plus_structure_minimal": (
        "feature_set_v3_sequence_plus_structure_minimal.csv"
    ),
    "sequence_plus_structure_core": (
        "feature_set_v3_sequence_plus_structure_core.csv"
    ),
}

PRIMARY_FEATURE_SETS = {
    "sequence_final",
    "structure_minimal_final",
    "sequence_plus_structure_minimal",
}

OUTER_SCHEMES = {
    "homology_cv_fold": {
        "group_column": "homology_cluster_30",
        "scheme_label": "30_percent_primary",
    },
    "homology_cv_fold_sensitivity": {
        "group_column": "homology_cluster_40",
        "scheme_label": "40_percent_sensitivity",
    },
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
    "benchmark_version",
    "previous_binary_label",
    "label_updated_from_minami_validation",
    "minami_2023_table1_groe_dependency",
    "minami_2023_validation_status",
    "minami_2023_validation_source",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final nested grouped Random Forest pipeline."
    )
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--fold-assignments", required=True)
    parser.add_argument("--output-dir", default="results/stage09_v2")
    parser.add_argument("--include-non-tim", action="store_true")
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--n-iter", type=int, default=80)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--permutation-repeats", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Run only sequence, structural-minimal, and combined-minimal models.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    result = {
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "specificity": (
            float(tn / (tn + fp)) if (tn + fp) else float("nan")
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "roc_auc": (
            float(roc_auc_score(y_true, y_prob))
            if len(np.unique(y_true)) == 2
            else float("nan")
        ),
        "pr_auc": (
            float(average_precision_score(y_true, y_prob))
            if len(np.unique(y_true)) == 2
            else float("nan")
        ),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
    }
    return result


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    metric_name: str,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = []
    n = len(y_true)

    for _ in range(iterations):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        yp = y_pred[idx]
        pr = y_prob[idx]

        if metric_name in {"roc_auc", "pr_auc"} and len(np.unique(yt)) < 2:
            continue

        value = metric_dict(yt, yp, pr)[metric_name]
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))

    if not values:
        return float("nan"), float("nan")

    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def threshold_grid() -> np.ndarray:
    return np.round(np.linspace(0.10, 0.90, 81), 3)


def choose_threshold_by_mcc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, float]:
    best_threshold = 0.5
    best_mcc = -np.inf

    for threshold in threshold_grid():
        pred = (y_prob >= threshold).astype(int)
        score = matthews_corrcoef(y_true, pred)

        if score > best_mcc + 1e-12:
            best_threshold = float(threshold)
            best_mcc = float(score)
        elif abs(score - best_mcc) <= 1e-12:
            if abs(threshold - 0.5) < abs(best_threshold - 0.5):
                best_threshold = float(threshold)

    return best_threshold, best_mcc


def parameter_space() -> dict[str, Any]:
    return {
        "n_estimators": [300, 500, 750, 1000],
        "max_depth": [None, 4, 6, 8, 12, 16],
        "min_samples_leaf": [1, 2, 4, 6, 8],
        "min_samples_split": [2, 5, 10, 15],
        "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
        "class_weight": ["balanced", "balanced_subsample"],
        "bootstrap": [True],
        "max_samples": [None, 0.7, 0.85, 1.0],
    }


def build_rf(params: dict[str, Any], seed: int, n_jobs: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        criterion="gini",
        random_state=seed,
        n_jobs=n_jobs,
        **params,
    )


def construct_inner_splitter(
    y: np.ndarray,
    groups: np.ndarray,
    requested_folds: int,
    seed: int,
) -> tuple[StratifiedGroupKFold, int]:
    max_splits = min(requested_folds, len(np.unique(groups)))

    for n_splits in range(max_splits, 1, -1):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        valid = True
        try:
            for train_idx, valid_idx in splitter.split(
                np.zeros(len(y)), y, groups
            ):
                if len(np.unique(y[train_idx])) < 2:
                    valid = False
                    break
                if len(valid_idx) == 0:
                    valid = False
                    break
        except ValueError:
            valid = False

        if valid:
            return splitter, n_splits

    raise RuntimeError("Unable to construct at least two valid grouped inner folds")


def evaluate_candidate(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    splitter: StratifiedGroupKFold,
    params: dict[str, Any],
    seed: int,
    n_jobs: int,
) -> dict[str, Any]:
    oof_prob = np.full(len(y), np.nan, dtype=float)
    fold_mccs = []
    fold_balanced = []
    fold_rocs = []
    fold_prs = []

    for inner_fold, (train_idx, valid_idx) in enumerate(
        splitter.split(X, y, groups)
    ):
        model = build_rf(
            params,
            seed=seed + inner_fold * 100,
            n_jobs=n_jobs,
        )
        model.fit(X.iloc[train_idx], y[train_idx])

        prob = model.predict_proba(X.iloc[valid_idx])[:, 1]
        pred = (prob >= 0.5).astype(int)
        oof_prob[valid_idx] = prob

        fold_mccs.append(
            matthews_corrcoef(y[valid_idx], pred)
        )
        fold_balanced.append(
            balanced_accuracy_score(y[valid_idx], pred)
        )
        if len(np.unique(y[valid_idx])) == 2:
            fold_rocs.append(
                roc_auc_score(y[valid_idx], prob)
            )
            fold_prs.append(
                average_precision_score(y[valid_idx], prob)
            )

    if np.isnan(oof_prob).any():
        raise RuntimeError("Inner OOF probabilities incomplete")

    threshold, threshold_mcc = choose_threshold_by_mcc(y, oof_prob)

    return {
        "mean_inner_mcc_0_5": float(np.mean(fold_mccs)),
        "std_inner_mcc_0_5": (
            float(np.std(fold_mccs, ddof=1))
            if len(fold_mccs) > 1 else 0.0
        ),
        "mean_inner_balanced_accuracy_0_5": float(
            np.mean(fold_balanced)
        ),
        "mean_inner_roc_auc": (
            float(np.mean(fold_rocs)) if fold_rocs else float("nan")
        ),
        "mean_inner_pr_auc": (
            float(np.mean(fold_prs)) if fold_prs else float("nan")
        ),
        "inner_selected_threshold": threshold,
        "inner_oof_mcc_selected_threshold": threshold_mcc,
    }


def candidate_sort_key(row: dict[str, Any]) -> tuple:
    depth = row["params"]["max_depth"]
    depth_rank = 10_000 if depth is None else int(depth)

    return (
        -row["mean_inner_mcc_0_5"],
        depth_rank,
        -int(row["params"]["min_samples_leaf"]),
        int(row["params"]["n_estimators"]),
    )


def tune_outer_training(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    inner_folds: int,
    n_iter: int,
    seed: int,
    n_jobs: int,
) -> tuple[
    dict[str, Any],
    float,
    pd.DataFrame,
    pd.DataFrame,
    int,
]:
    splitter, actual_inner_folds = construct_inner_splitter(
        y_train,
        groups_train,
        requested_folds=inner_folds,
        seed=seed,
    )

    sampled = list(
        ParameterSampler(
            parameter_space(),
            n_iter=n_iter,
            random_state=seed,
        )
    )

    records = []
    for candidate_index, params in enumerate(sampled):
        metrics = evaluate_candidate(
            X_train,
            y_train,
            groups_train,
            splitter,
            params,
            seed=seed + candidate_index * 1000,
            n_jobs=n_jobs,
        )
        records.append(
            {
                "candidate_index": candidate_index,
                "params": params,
                **metrics,
            }
        )

    ranked = sorted(records, key=candidate_sort_key)
    best = ranked[0]

    flattened = []
    for rank, row in enumerate(ranked, start=1):
        rec = {
            "rank": rank,
            "candidate_index": row["candidate_index"],
            "selected_candidate": rank == 1,
            "mean_inner_mcc_0_5": row["mean_inner_mcc_0_5"],
            "std_inner_mcc_0_5": row["std_inner_mcc_0_5"],
            "mean_inner_balanced_accuracy_0_5": (
                row["mean_inner_balanced_accuracy_0_5"]
            ),
            "mean_inner_roc_auc": row["mean_inner_roc_auc"],
            "mean_inner_pr_auc": row["mean_inner_pr_auc"],
            "inner_selected_threshold": row["inner_selected_threshold"],
            "inner_oof_mcc_selected_threshold": (
                row["inner_oof_mcc_selected_threshold"]
            ),
        }
        rec.update({f"param_{k}": v for k, v in row["params"].items()})
        flattened.append(rec)

    search_df = pd.DataFrame(flattened)
    top5_df = search_df.head(5).copy()

    return (
        best["params"],
        float(best["inner_selected_threshold"]),
        search_df,
        top5_df,
        actual_inner_folds,
    )


def audit_outer_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    group_column: str,
) -> dict[str, Any]:
    train_groups = set(train[group_column].astype(str))
    test_groups = set(test[group_column].astype(str))
    overlap = sorted(train_groups & test_groups)

    train_ids = set(train[IDENTIFIER].astype(str))
    test_ids = set(test[IDENTIFIER].astype(str))
    id_overlap = sorted(train_ids & test_ids)

    return {
        "group_overlap_count": len(overlap),
        "protein_overlap_count": len(id_overlap),
        "passed": len(overlap) == 0 and len(id_overlap) == 0,
        "group_overlap_values": ";".join(overlap),
        "protein_overlap_values": ";".join(id_overlap),
    }


def subset_frame(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    if subset == "full":
        return df.copy()
    if subset == "non_tim":
        return df[~df["is_tim_barrel"].astype(bool)].copy()
    raise ValueError(subset)


def predictor_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in METADATA_COLUMNS
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].nunique(dropna=True) > 1
    ]


def calibration_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    bins: int = 10,
) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    rows = []

    for i in range(bins):
        low = edges[i]
        high = edges[i + 1]
        if i == bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)

        if not mask.any():
            continue

        rows.append(
            {
                "bin": i + 1,
                "bin_low": low,
                "bin_high": high,
                "n": int(mask.sum()),
                "mean_predicted_probability": float(y_prob[mask].mean()),
                "observed_positive_fraction": float(y_true[mask].mean()),
            }
        )

    return pd.DataFrame(rows)


def pooled_summary(
    predictions: pd.DataFrame,
    bootstrap_iterations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    group_cols = [
        "subset",
        "outer_fold_column",
        "feature_set",
        "threshold_type",
    ]

    for keys, group in predictions.groupby(group_cols):
        subset, fold_column, feature_set, threshold_type = keys

        y_true = group[TARGET].to_numpy(dtype=int)
        y_prob = group["predicted_probability"].to_numpy(dtype=float)
        label_col = (
            "predicted_label_0_5"
            if threshold_type == "fixed_0_5"
            else "predicted_label_inner_selected"
        )
        y_pred = group[label_col].to_numpy(dtype=int)

        record = {
            "benchmark_version": "1.1",
            "feature_set_version": "3.0",
            "subset": subset,
            "outer_fold_column": fold_column,
            "feature_set": feature_set,
            "threshold_type": threshold_type,
            "n": len(group),
            "positive_n": int(y_true.sum()),
            "negative_n": int((y_true == 0).sum()),
        }

        metrics = metric_dict(y_true, y_pred, y_prob)
        for metric_name, value in metrics.items():
            record[metric_name] = value

            if metric_name in {
                "mcc",
                "balanced_accuracy",
                "recall",
                "precision",
                "specificity",
                "roc_auc",
                "pr_auc",
                "brier_score",
            }:
                low, high = bootstrap_ci(
                    y_true,
                    y_pred,
                    y_prob,
                    metric_name,
                    bootstrap_iterations,
                    rng,
                )
                record[f"{metric_name}_ci_low"] = low
                record[f"{metric_name}_ci_high"] = high

        rows.append(record)

    return pd.DataFrame(rows).sort_values(group_cols)


def summarize_permutation_importance(by_fold: pd.DataFrame) -> pd.DataFrame:
    if by_fold.empty:
        return pd.DataFrame()

    summary = (
        by_fold.groupby(
            [
                "subset",
                "outer_fold_column",
                "feature_set",
                "feature",
            ]
        )
        .agg(
            fold_count=("outer_fold", "nunique"),
            mean_importance=("importance_mean", "mean"),
            sd_across_folds=("importance_mean", "std"),
            median_importance=("importance_mean", "median"),
            min_fold_importance=("importance_mean", "min"),
            max_fold_importance=("importance_mean", "max"),
            positive_fold_fraction=("importance_mean", lambda x: float(np.mean(np.asarray(x) > 0))),
            top10_fold_count=("top10_in_fold", "sum"),
        )
        .reset_index()
    )

    summary["importance_cv_abs"] = np.where(
        summary["mean_importance"].abs() > 1e-12,
        summary["sd_across_folds"] / summary["mean_importance"].abs(),
        np.nan,
    )

    summary["importance_rank"] = (
        summary.groupby(
            ["subset", "outer_fold_column", "feature_set"]
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


def make_figures(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    importance_summary: pd.DataFrame,
    selected_params: pd.DataFrame,
    output_dir: Path,
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    primary = predictions[
        (predictions["subset"] == "full")
        & (predictions["outer_fold_column"] == "homology_cv_fold")
    ].copy()

    if not primary.empty:
        # ROC
        fig, axis = plt.subplots(figsize=(7, 6))
        for feature_set, group in primary.groupby("feature_set"):
            y = group[TARGET].to_numpy(dtype=int)
            p = group["predicted_probability"].to_numpy(dtype=float)
            fpr, tpr, _ = roc_curve(y, p)
            axis.plot(fpr, tpr, label=feature_set)
        axis.plot([0, 1], [0, 1], linestyle="--")
        axis.set_xlabel("False-positive rate")
        axis.set_ylabel("True-positive rate")
        axis.set_title("Stage 09 v2 pooled OOF ROC curves")
        axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / "roc_curves.png", dpi=300, bbox_inches="tight")
        fig.savefig(figures_dir / "roc_curves.pdf", bbox_inches="tight")
        plt.close(fig)

        # PR
        fig, axis = plt.subplots(figsize=(7, 6))
        prevalence = float(primary[TARGET].mean())
        for feature_set, group in primary.groupby("feature_set"):
            y = group[TARGET].to_numpy(dtype=int)
            p = group["predicted_probability"].to_numpy(dtype=float)
            precision, recall, _ = precision_recall_curve(y, p)
            axis.plot(recall, precision, label=feature_set)
        axis.axhline(prevalence, linestyle="--")
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_title("Stage 09 v2 pooled OOF precision-recall curves")
        axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(
            figures_dir / "precision_recall_curves.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            figures_dir / "precision_recall_curves.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)

        # Calibration
        fig, axis = plt.subplots(figsize=(7, 6))
        for feature_set, group in primary.groupby("feature_set"):
            y = group[TARGET].to_numpy(dtype=int)
            p = group["predicted_probability"].to_numpy(dtype=float)
            frac_pos, mean_pred = calibration_curve(
                y, p, n_bins=10, strategy="quantile"
            )
            axis.plot(mean_pred, frac_pos, marker="o", label=feature_set)
        axis.plot([0, 1], [0, 1], linestyle="--")
        axis.set_xlabel("Mean predicted probability")
        axis.set_ylabel("Observed positive fraction")
        axis.set_title("OOF calibration")
        axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(
            figures_dir / "calibration_curves.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            figures_dir / "calibration_curves.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)

    # Permutation importance for primary combined-minimal
    target_imp = importance_summary[
        (importance_summary["subset"] == "full")
        & (importance_summary["outer_fold_column"] == "homology_cv_fold")
        & (
            importance_summary["feature_set"]
            == "sequence_plus_structure_minimal"
        )
    ].copy()

    if not target_imp.empty:
        top = target_imp.nsmallest(20, "importance_rank").sort_values(
            "mean_importance"
        )
        fig, axis = plt.subplots(figsize=(9, 8))
        axis.barh(
            top["feature"],
            top["mean_importance"],
            xerr=top["sd_across_folds"].fillna(0),
        )
        axis.axvline(0, linewidth=1)
        axis.set_xlabel("Mean held-out permutation importance ± fold SD")
        axis.set_title("Combined-minimal RF permutation importance")
        fig.tight_layout()
        fig.savefig(
            figures_dir / "permutation_importance_combined_minimal.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            figures_dir / "permutation_importance_combined_minimal.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)

    # Hyperparameter frequency
    primary_params = selected_params[
        (selected_params["subset"] == "full")
        & (selected_params["outer_fold_column"] == "homology_cv_fold")
        & (
            selected_params["feature_set"]
            == "sequence_plus_structure_minimal"
        )
    ].copy()

    if not primary_params.empty:
        for param_name in [
            "max_depth",
            "min_samples_leaf",
            "max_features",
            "class_weight",
        ]:
            counts = (
                primary_params[param_name]
                .astype(str)
                .value_counts()
                .sort_index()
            )
            fig, axis = plt.subplots(figsize=(7, 5))
            axis.bar(counts.index.astype(str), counts.values)
            axis.set_xlabel(param_name)
            axis.set_ylabel("Outer folds selecting value")
            axis.set_title(f"Selected hyperparameter frequency: {param_name}")
            axis.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            fig.savefig(
                figures_dir / f"hyperparameter_{param_name}.png",
                dpi=300,
                bbox_inches="tight",
            )
            fig.savefig(
                figures_dir / f"hyperparameter_{param_name}.pdf",
                bbox_inches="tight",
            )
            plt.close(fig)

    # Performance summary
    perf = summary[
        (summary["subset"] == "full")
        & (summary["outer_fold_column"] == "homology_cv_fold")
        & (summary["threshold_type"] == "fixed_0_5")
    ].copy()

    if not perf.empty:
        fig, axis = plt.subplots(figsize=(10, 5.5))
        perf = perf.sort_values("mcc")
        axis.barh(perf["feature_set"], perf["mcc"])
        axis.set_xlabel("Pooled OOF MCC")
        axis.set_title("Stage 09 v2 primary model comparison")
        fig.tight_layout()
        fig.savefig(
            figures_dir / "primary_model_mcc.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            figures_dir / "primary_model_mcc.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)


def main() -> int:
    args = parse_args()

    feature_dir = Path(args.feature_dir)
    fold_path = Path(args.fold_assignments)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models_dir = output_dir / "saved_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if not feature_dir.exists():
        print(f"ERROR: feature directory not found: {feature_dir}", file=sys.stderr)
        return 2
    if not fold_path.exists():
        print(f"ERROR: fold file not found: {fold_path}", file=sys.stderr)
        return 2

    folds = pd.read_csv(fold_path)

    required_fold_columns = {
        IDENTIFIER,
        "homology_cv_fold",
        "homology_cv_fold_sensitivity",
        "homology_cluster_30",
        "homology_cluster_40",
        "is_tim_barrel",
    }
    missing = required_fold_columns - set(folds.columns)
    if missing:
        raise RuntimeError(
            f"Fold assignments missing columns: {sorted(missing)}"
        )

    subsets = ["full"]
    if args.include_non_tim:
        subsets.append("non_tim")

    feature_files = FEATURE_FILES.copy()
    if args.primary_only:
        feature_files = {
            k: v
            for k, v in feature_files.items()
            if k in PRIMARY_FEATURE_SETS
        }

    all_predictions = []
    all_fold_metrics = []
    all_selected_params = []
    all_top5 = []
    all_search_results = []
    all_thresholds = []
    all_leakage_audits = []
    all_permutation_rows = []
    all_calibration_rows = []
    feature_manifest_rows = []
    model_manifest_rows = []

    total_start = time.time()

    for feature_set, filename in feature_files.items():
        dataset_path = feature_dir / filename
        if not dataset_path.exists():
            raise FileNotFoundError(dataset_path)

        dataset = pd.read_csv(dataset_path)

        if len(dataset) != 181:
            raise RuntimeError(
                f"{feature_set}: expected 181 proteins, got {len(dataset)}"
            )
        if int(dataset[TARGET].sum()) != 73:
            raise RuntimeError(
                f"{feature_set}: expected 73 positives, got {int(dataset[TARGET].sum())}"
            )
        if dataset[IDENTIFIER].duplicated().any():
            raise RuntimeError(f"{feature_set}: duplicate accessions")

        features = predictor_columns(dataset)

        if len(features) != len(set(features)):
            raise RuntimeError(f"{feature_set}: duplicate predictor names")

        feature_manifest_rows.extend(
            {
                "benchmark_version": "1.1",
                "feature_set_version": "3.0",
                "feature_set": feature_set,
                "feature_order": i,
                "feature": feature,
            }
            for i, feature in enumerate(features, start=1)
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

        if data[
            [
                "homology_cv_fold",
                "homology_cv_fold_sensitivity",
                "homology_cluster_30",
                "homology_cluster_40",
            ]
        ].isna().any().any():
            raise RuntimeError(
                f"{feature_set}: missing fold or homology assignments"
            )

        for subset_name in subsets:
            subset = subset_frame(data, subset_name)
            if subset[TARGET].nunique() < 2:
                continue

            for outer_fold_column, scheme in OUTER_SCHEMES.items():
                group_column = scheme["group_column"]

                print(
                    f"\nRunning {feature_set} | {subset_name} | {outer_fold_column}"
                )

                for outer_fold in sorted(
                    subset[outer_fold_column].dropna().unique()
                ):
                    fold_start = time.time()

                    train = subset[
                        subset[outer_fold_column] != outer_fold
                    ].copy()
                    test = subset[
                        subset[outer_fold_column] == outer_fold
                    ].copy()

                    if train.empty or test.empty:
                        raise RuntimeError("Empty outer train/test split")
                    if train[TARGET].nunique() < 2:
                        raise RuntimeError(
                            f"Outer training fold {outer_fold} lacks both classes"
                        )

                    audit = audit_outer_split(
                        train,
                        test,
                        group_column,
                    )
                    audit.update(
                        {
                            "subset": subset_name,
                            "feature_set": feature_set,
                            "outer_fold_column": outer_fold_column,
                            "outer_fold": int(outer_fold),
                            "group_column": group_column,
                        }
                    )
                    all_leakage_audits.append(audit)

                    if not audit["passed"]:
                        raise RuntimeError(
                            f"Leakage detected in {feature_set}/{subset_name}/"
                            f"{outer_fold_column}/fold {outer_fold}"
                        )

                    X_train = train[features]
                    y_train = train[TARGET].to_numpy(dtype=int)
                    groups_train = train[group_column].astype(str).to_numpy()

                    (
                        best_params,
                        selected_threshold,
                        search_df,
                        top5_df,
                        actual_inner_folds,
                    ) = tune_outer_training(
                        X_train,
                        y_train,
                        groups_train,
                        inner_folds=args.inner_folds,
                        n_iter=args.n_iter,
                        seed=(
                            args.random_seed
                            + int(outer_fold) * 100_000
                            + len(feature_set) * 1000
                            + len(subset_name) * 100
                        ),
                        n_jobs=args.n_jobs,
                    )

                    for frame, collection in [
                        (search_df, all_search_results),
                        (top5_df, all_top5),
                    ]:
                        frame = frame.copy()
                        frame["subset"] = subset_name
                        frame["feature_set"] = feature_set
                        frame["outer_fold_column"] = outer_fold_column
                        frame["outer_fold"] = int(outer_fold)
                        frame["actual_inner_folds"] = actual_inner_folds
                        collection.append(frame)

                    selected_record = {
                        "subset": subset_name,
                        "feature_set": feature_set,
                        "outer_fold_column": outer_fold_column,
                        "outer_fold": int(outer_fold),
                        "actual_inner_folds": actual_inner_folds,
                        "selected_threshold": selected_threshold,
                        **best_params,
                    }
                    all_selected_params.append(selected_record)

                    all_thresholds.append(
                        {
                            "subset": subset_name,
                            "feature_set": feature_set,
                            "outer_fold_column": outer_fold_column,
                            "outer_fold": int(outer_fold),
                            "selected_threshold": selected_threshold,
                        }
                    )

                    model = build_rf(
                        best_params,
                        seed=args.random_seed + int(outer_fold),
                        n_jobs=args.n_jobs,
                    )
                    model.fit(X_train, y_train)

                    y_test = test[TARGET].to_numpy(dtype=int)
                    y_prob = model.predict_proba(test[features])[:, 1]
                    y_pred_05 = (y_prob >= 0.5).astype(int)
                    y_pred_selected = (
                        y_prob >= selected_threshold
                    ).astype(int)

                    metrics_05 = metric_dict(
                        y_test,
                        y_pred_05,
                        y_prob,
                    )
                    metrics_selected = metric_dict(
                        y_test,
                        y_pred_selected,
                        y_prob,
                    )

                    for threshold_type, threshold_value, metrics in [
                        ("fixed_0_5", 0.5, metrics_05),
                        ("inner_selected", selected_threshold, metrics_selected),
                    ]:
                        all_fold_metrics.append(
                            {
                                "benchmark_version": "1.1",
                                "feature_set_version": "3.0",
                                "subset": subset_name,
                                "feature_set": feature_set,
                                "outer_fold_column": outer_fold_column,
                                "outer_fold": int(outer_fold),
                                "threshold_type": threshold_type,
                                "threshold_value": threshold_value,
                                "train_n": len(train),
                                "test_n": len(test),
                                "test_positive_n": int(y_test.sum()),
                                "test_negative_n": int((y_test == 0).sum()),
                                **metrics,
                            }
                        )

                    # Held-out permutation importance on untouched outer test.
                    if len(np.unique(y_test)) == 2:
                        perm = permutation_importance(
                            model,
                            test[features],
                            y_test,
                            scoring="roc_auc",
                            n_repeats=args.permutation_repeats,
                            random_state=(
                                args.random_seed
                                + int(outer_fold) * 10_000
                                + len(feature_set)
                            ),
                            n_jobs=args.n_jobs,
                        )

                        fold_imp = []
                        for i, feature in enumerate(features):
                            values = perm.importances[i]
                            fold_imp.append(
                                {
                                    "feature": feature,
                                    "importance_mean": float(values.mean()),
                                    "importance_std": float(values.std(ddof=1)),
                                }
                            )

                        top10_features = {
                            r["feature"]
                            for r in sorted(
                                fold_imp,
                                key=lambda r: r["importance_mean"],
                                reverse=True,
                            )[:10]
                        }

                        for row in fold_imp:
                            all_permutation_rows.append(
                                {
                                    "benchmark_version": "1.1",
                                    "feature_set_version": "3.0",
                                    "subset": subset_name,
                                    "feature_set": feature_set,
                                    "outer_fold_column": outer_fold_column,
                                    "outer_fold": int(outer_fold),
                                    "feature": row["feature"],
                                    "importance_mean": row["importance_mean"],
                                    "importance_std": row["importance_std"],
                                    "top10_in_fold": (
                                        row["feature"] in top10_features
                                    ),
                                    "permutation_repeats": args.permutation_repeats,
                                    "scoring": "roc_auc",
                                }
                            )

                    # OOF predictions.
                    best_params_json = json.dumps(
                        best_params,
                        sort_keys=True,
                    )

                    for position, (_, row) in enumerate(test.iterrows()):
                        all_predictions.append(
                            {
                                "benchmark_version": "1.1",
                                "feature_set_version": "3.0",
                                IDENTIFIER: row[IDENTIFIER],
                                "gene_minami": row.get("gene_minami", ""),
                                "jwid": row.get("jwid", ""),
                                TARGET: int(row[TARGET]),
                                "subset": subset_name,
                                "feature_set": feature_set,
                                "outer_fold_column": outer_fold_column,
                                "outer_fold": int(outer_fold),
                                "homology_cluster_30": row[
                                    "homology_cluster_30"
                                ],
                                "homology_cluster_40": row[
                                    "homology_cluster_40"
                                ],
                                "is_tim_barrel": bool(row["is_tim_barrel"]),
                                "predicted_probability": float(
                                    y_prob[position]
                                ),
                                "predicted_label_0_5": int(
                                    y_pred_05[position]
                                ),
                                "selected_threshold": float(
                                    selected_threshold
                                ),
                                "predicted_label_inner_selected": int(
                                    y_pred_selected[position]
                                ),
                                "correct_0_5": bool(
                                    y_pred_05[position] == y_test[position]
                                ),
                                "correct_inner_selected": bool(
                                    y_pred_selected[position]
                                    == y_test[position]
                                ),
                                "confidence_distance_from_0_5": float(
                                    abs(y_prob[position] - 0.5)
                                ),
                                "selected_hyperparameters": best_params_json,
                            }
                        )

                    # Calibration bins for this outer fold.
                    fold_cal = calibration_table(
                        y_test,
                        y_prob,
                        bins=5,
                    )
                    if not fold_cal.empty:
                        fold_cal["subset"] = subset_name
                        fold_cal["feature_set"] = feature_set
                        fold_cal["outer_fold_column"] = outer_fold_column
                        fold_cal["outer_fold"] = int(outer_fold)
                        all_calibration_rows.append(fold_cal)

                    # Save model.
                    safe_subset = subset_name.replace("/", "_")
                    safe_foldcol = outer_fold_column.replace("/", "_")
                    model_name = (
                        f"{feature_set}__{safe_subset}__"
                        f"{safe_foldcol}__fold{int(outer_fold)}.joblib"
                    )
                    model_path = models_dir / model_name
                    joblib.dump(
                        {
                            "model": model,
                            "features": features,
                            "feature_set": feature_set,
                            "subset": subset_name,
                            "outer_fold_column": outer_fold_column,
                            "outer_fold": int(outer_fold),
                            "selected_threshold": selected_threshold,
                            "selected_hyperparameters": best_params,
                            "benchmark_version": "1.1",
                            "feature_set_version": "3.0",
                        },
                        model_path,
                        compress=3,
                    )

                    model_manifest_rows.append(
                        {
                            "model_file": model_name,
                            "feature_set": feature_set,
                            "subset": subset_name,
                            "outer_fold_column": outer_fold_column,
                            "outer_fold": int(outer_fold),
                            "sha256": sha256_file(model_path),
                        }
                    )

                    elapsed = time.time() - fold_start
                    print(
                        f"  fold {outer_fold}: "
                        f"train={len(train)}, test={len(test)}, "
                        f"threshold={selected_threshold:.2f}, "
                        f"{elapsed/60:.1f} min"
                    )

    predictions = pd.DataFrame(all_predictions)
    fold_metrics = pd.DataFrame(all_fold_metrics)
    selected_params = pd.DataFrame(all_selected_params)
    thresholds = pd.DataFrame(all_thresholds)
    leakage_audit = pd.DataFrame(all_leakage_audits)
    permutation_by_fold = pd.DataFrame(all_permutation_rows)
    permutation_summary = summarize_permutation_importance(
        permutation_by_fold
    )
    feature_manifest = pd.DataFrame(feature_manifest_rows)
    model_manifest = pd.DataFrame(model_manifest_rows)

    search_results = pd.concat(
        all_search_results,
        ignore_index=True,
    )
    top5_results = pd.concat(
        all_top5,
        ignore_index=True,
    )
    calibration_by_fold = (
        pd.concat(all_calibration_rows, ignore_index=True)
        if all_calibration_rows
        else pd.DataFrame()
    )

    # Every protein should receive one OOF prediction per experiment.
    prediction_counts = (
        predictions.groupby(
            [
                "subset",
                "feature_set",
                "outer_fold_column",
                IDENTIFIER,
            ]
        )
        .size()
        .reset_index(name="prediction_count")
    )
    bad_counts = prediction_counts[
        prediction_counts["prediction_count"] != 1
    ]
    if not bad_counts.empty:
        raise RuntimeError(
            "OOF prediction audit failed: some proteins do not have exactly "
            "one prediction per experiment."
        )

    if (~leakage_audit["passed"]).any():
        raise RuntimeError("Leakage audit contains failures")

    summary_frames = []
    for threshold_type in ["fixed_0_5", "inner_selected"]:
        temp = predictions.copy()
        temp["threshold_type"] = threshold_type
        summary_frames.append(temp)

    summary_input = pd.concat(summary_frames, ignore_index=True)
    summary = pooled_summary(
        summary_input,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.random_seed,
    )

    # Hyperparameter stability summary.
    hyperparameter_rows = []
    for keys, group in selected_params.groupby(
        ["subset", "outer_fold_column", "feature_set"]
    ):
        subset, fold_column, feature_set = keys

        for param in [
            "n_estimators",
            "max_depth",
            "min_samples_leaf",
            "min_samples_split",
            "max_features",
            "class_weight",
            "max_samples",
        ]:
            counter = Counter(group[param].astype(str))
            for value, count in sorted(counter.items()):
                hyperparameter_rows.append(
                    {
                        "subset": subset,
                        "outer_fold_column": fold_column,
                        "feature_set": feature_set,
                        "hyperparameter": param,
                        "value": value,
                        "outer_fold_count": count,
                    }
                )

    hyperparameter_stability = pd.DataFrame(hyperparameter_rows)

    # Write files.
    predictions.to_csv(
        output_dir / "rf_oof_predictions.csv",
        index=False,
    )
    fold_metrics.to_csv(
        output_dir / "rf_fold_metrics.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "rf_summary_metrics.csv",
        index=False,
    )
    selected_params.to_csv(
        output_dir / "rf_best_hyperparameters.csv",
        index=False,
    )
    top5_results.to_csv(
        output_dir / "rf_top5_hyperparameters_by_outer_fold.csv",
        index=False,
    )
    search_results.to_csv(
        output_dir / "rf_inner_search_results.csv",
        index=False,
    )
    thresholds.to_csv(
        output_dir / "rf_thresholds.csv",
        index=False,
    )
    leakage_audit.to_csv(
        output_dir / "rf_leakage_audit.csv",
        index=False,
    )
    permutation_by_fold.to_csv(
        output_dir / "permutation_importance_by_fold.csv",
        index=False,
    )
    permutation_summary.to_csv(
        output_dir / "permutation_importance.csv",
        index=False,
    )
    feature_manifest.to_csv(
        output_dir / "rf_feature_manifest.csv",
        index=False,
    )
    model_manifest.to_csv(
        output_dir / "saved_model_manifest.csv",
        index=False,
    )
    hyperparameter_stability.to_csv(
        output_dir / "hyperparameter_stability.csv",
        index=False,
    )
    prediction_counts.to_csv(
        output_dir / "oof_prediction_audit.csv",
        index=False,
    )
    if not calibration_by_fold.empty:
        calibration_by_fold.to_csv(
            output_dir / "calibration_by_outer_fold.csv",
            index=False,
        )

    # Pooled calibration tables.
    pooled_cal_rows = []
    for keys, group in predictions.groupby(
        ["subset", "outer_fold_column", "feature_set"]
    ):
        subset, fold_column, feature_set = keys
        cal = calibration_table(
            group[TARGET].to_numpy(dtype=int),
            group["predicted_probability"].to_numpy(dtype=float),
            bins=10,
        )
        if cal.empty:
            continue
        cal["subset"] = subset
        cal["outer_fold_column"] = fold_column
        cal["feature_set"] = feature_set
        pooled_cal_rows.append(cal)

    pooled_calibration = (
        pd.concat(pooled_cal_rows, ignore_index=True)
        if pooled_cal_rows
        else pd.DataFrame()
    )
    if not pooled_calibration.empty:
        pooled_calibration.to_csv(
            output_dir / "calibration_pooled_oof.csv",
            index=False,
        )

    make_figures(
        predictions,
        summary,
        permutation_summary,
        selected_params,
        output_dir,
    )

    methods = {
        "stage": "09_v2",
        "benchmark_version": "1.1",
        "feature_set_version": "3.0",
        "feature_files": feature_files,
        "outer_schemes": OUTER_SCHEMES,
        "inner_folds_requested": args.inner_folds,
        "randomized_candidates_per_outer_fold": args.n_iter,
        "selection_metric": "mean inner-fold MCC at threshold 0.5",
        "threshold_selection": (
            "MCC-maximizing threshold from pooled inner OOF predictions, "
            "applied only to the corresponding outer test fold"
        ),
        "permutation_importance": {
            "scoring": "roc_auc",
            "repeats": args.permutation_repeats,
            "evaluation_data": "untouched outer test fold only",
        },
        "bootstrap_iterations": args.bootstrap_iterations,
        "random_seed": args.random_seed,
        "hyperparameter_space": parameter_space(),
        "subsets": subsets,
        "saved_outer_fold_models": True,
        "note": (
            "No target-aware feature engineering, relabeling, or outer-test "
            "tuning occurred in Stage 09 v2."
        ),
    }
    (output_dir / "methods.json").write_text(
        json.dumps(methods, indent=2)
    )

    runtime_minutes = (time.time() - total_start) / 60
    run_summary = {
        "stage": "09_v2",
        "status": "COMPLETE",
        "benchmark_version": "1.1",
        "feature_set_version": "3.0",
        "protein_count": 181,
        "positive_count": 73,
        "negative_count": 108,
        "feature_sets": list(feature_files),
        "subsets": subsets,
        "outer_schemes": list(OUTER_SCHEMES),
        "n_iter": args.n_iter,
        "inner_folds_requested": args.inner_folds,
        "permutation_repeats": args.permutation_repeats,
        "prediction_rows": len(predictions),
        "saved_models": len(model_manifest),
        "inner_search_rows": len(search_results),
        "top5_rows": len(top5_results),
        "leakage_failures": int((~leakage_audit["passed"]).sum()),
        "oof_prediction_audit_failures": int(len(bad_counts)),
        "runtime_minutes": runtime_minutes,
    }
    (output_dir / "reproducibility_report.json").write_text(
        json.dumps(run_summary, indent=2)
    )

    # Directory checksum manifest.
    checksum_rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.txt":
            checksum_rows.append(
                f"{sha256_file(path)}  {path.relative_to(output_dir)}"
            )
    (output_dir / "sha256_manifest.txt").write_text(
        "\n".join(checksum_rows) + "\n"
    )

    print("\nStage 09 v2 complete")
    print(f"Runtime: {runtime_minutes:.1f} minutes")
    print(f"Saved models: {len(model_manifest)}")
    print(f"Leakage failures: {int((~leakage_audit['passed']).sum())}")
    print(f"OOF audit failures: {len(bad_counts)}")
    print(f"Output directory: {output_dir}")
    print()
    print(
        summary[
            [
                "subset",
                "outer_fold_column",
                "feature_set",
                "threshold_type",
                "mcc",
                "balanced_accuracy",
                "roc_auc",
                "pr_auc",
                "brier_score",
            ]
        ].to_string(index=False)
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
