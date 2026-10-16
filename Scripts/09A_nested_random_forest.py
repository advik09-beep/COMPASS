#!/usr/bin/env python3
"""
Stage 09A: Nested, grouped Random Forest evaluation.

Primary feature sets:
1. sequence_final
2. structure_minimal_final
3. sequence_plus_structure_minimal

Secondary feature sets can be added later after the primary experiment is complete.

Evaluation:
- Primary outer folds: frozen 30% MMseqs2 homology-grouped folds
- Sensitivity outer folds: frozen 40% MMseqs2 homology-grouped folds
- Inner tuning: grouped 3-fold CV using the corresponding homology clusters
- Hyperparameter selection metric: MCC
- Thresholds:
    a) fixed 0.5
    b) inner-CV threshold maximizing MCC, selected using training data only
- Subsets:
    a) full benchmark
    b) non-TIM benchmark
- All predictions are outer-fold out-of-fold predictions.

No feature engineering occurs here. This script consumes frozen Feature Set v2.

Example:
    python3 scripts/09A_nested_random_forest.py \
      --feature-dir data/frozen/feature_set_v2 \
      --fold-assignments data/grouping/stage07/fold_assignments_v1.0.csv \
      --output-dir results/stage09 \
      --include-non-tim
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    ParameterSampler,
    StratifiedGroupKFold,
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
    "homology_cv_fold": {
        "group_column": "homology_cluster_30",
        "label": "30_percent_primary",
    },
    "homology_cv_fold_sensitivity": {
        "group_column": "homology_cluster_40",
        "label": "40_percent_sensitivity",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nested grouped Random Forest evaluation."
    )
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--fold-assignments", required=True)
    parser.add_argument("--output-dir", default="results/stage09")
    parser.add_argument("--include-non-tim", action="store_true")
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--n-iter", type=int, default=50)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


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
    }

    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        result["pr_auc"] = float(
            average_precision_score(y_true, y_prob)
        )
    else:
        result["roc_auc"] = float("nan")
        result["pr_auc"] = float("nan")

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


def best_threshold_by_mcc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, float]:
    best_threshold = 0.5
    best_mcc = -np.inf

    for threshold in threshold_grid():
        y_pred = (y_prob >= threshold).astype(int)
        score = matthews_corrcoef(y_true, y_pred)

        if score > best_mcc + 1e-12:
            best_mcc = float(score)
            best_threshold = float(threshold)
        elif abs(score - best_mcc) <= 1e-12:
            # Predeclared tie-breaker: threshold closest to 0.5.
            if abs(threshold - 0.5) < abs(best_threshold - 0.5):
                best_threshold = float(threshold)

    return best_threshold, best_mcc


def make_inner_splitter(
    y: np.ndarray,
    groups: np.ndarray,
    requested_folds: int,
    seed: int,
) -> tuple[Any, int]:
    unique_groups = len(np.unique(groups))
    maximum = min(requested_folds, unique_groups)

    for n_splits in range(maximum, 1, -1):
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

    raise RuntimeError(
        "Could not construct at least two valid inner grouped folds"
    )


def parameter_space() -> dict[str, Any]:
    return {
        "n_estimators": [500, 750, 1000],
        "max_depth": [3, 5, 8, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [2, 4, 6, 8],
        "max_features": ["sqrt", 0.5, 1.0],
        "class_weight": ["balanced", "balanced_subsample"],
        "max_samples": [0.7, 0.9, None],
        "bootstrap": [True],
    }


def fit_model(
    params: dict[str, Any],
    seed: int,
    n_jobs: int,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        random_state=seed,
        n_jobs=n_jobs,
        criterion="gini",
        **params,
    )
    return model


def evaluate_parameter_candidate(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    splitter: Any,
    params: dict[str, Any],
    seed: int,
    n_jobs: int,
) -> tuple[dict[str, float], np.ndarray]:
    oof_prob = np.full(len(y), np.nan, dtype=float)
    fold_mccs = []
    fold_balanced = []
    fold_roc = []

    for inner_fold, (train_idx, valid_idx) in enumerate(
        splitter.split(X, y, groups)
    ):
        model = fit_model(
            params,
            seed=seed + inner_fold,
            n_jobs=n_jobs,
        )
        model.fit(X.iloc[train_idx], y[train_idx])
        probability = model.predict_proba(X.iloc[valid_idx])[:, 1]
        prediction = (probability >= 0.5).astype(int)

        oof_prob[valid_idx] = probability
        fold_mccs.append(
            matthews_corrcoef(y[valid_idx], prediction)
        )
        fold_balanced.append(
            balanced_accuracy_score(y[valid_idx], prediction)
        )

        if len(np.unique(y[valid_idx])) == 2:
            fold_roc.append(
                roc_auc_score(y[valid_idx], probability)
            )

    if np.isnan(oof_prob).any():
        raise RuntimeError(
            "Inner CV failed to assign probabilities to every training row"
        )

    selected_threshold, threshold_mcc = best_threshold_by_mcc(
        y, oof_prob
    )

    metrics = {
        "mean_inner_mcc_0_5": float(np.mean(fold_mccs)),
        "std_inner_mcc_0_5": float(np.std(fold_mccs, ddof=1))
        if len(fold_mccs) > 1 else 0.0,
        "mean_inner_balanced_accuracy_0_5": float(
            np.mean(fold_balanced)
        ),
        "mean_inner_roc_auc": float(np.mean(fold_roc))
        if fold_roc else float("nan"),
        "inner_selected_threshold": selected_threshold,
        "inner_oof_mcc_selected_threshold": threshold_mcc,
    }
    return metrics, oof_prob


def hyperparameter_sort_key(row: dict[str, Any]) -> tuple:
    max_depth = row["params"]["max_depth"]
    depth_rank = 10_000 if max_depth is None else int(max_depth)

    return (
        -row["mean_inner_mcc_0_5"],
        depth_rank,
        -int(row["params"]["min_samples_leaf"]),
        int(row["params"]["n_estimators"]),
    )


def tune_outer_training_set(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    inner_folds: int,
    n_iter: int,
    seed: int,
    n_jobs: int,
) -> tuple[dict[str, Any], float, pd.DataFrame, int]:
    splitter, actual_inner_folds = make_inner_splitter(
        y_train,
        groups_train,
        requested_folds=inner_folds,
        seed=seed,
    )

    sampled_params = list(
        ParameterSampler(
            parameter_space(),
            n_iter=n_iter,
            random_state=seed,
        )
    )

    rows = []
    candidate_probabilities = []

    for candidate_index, params in enumerate(sampled_params):
        metrics, oof_prob = evaluate_parameter_candidate(
            X_train,
            y_train,
            groups_train,
            splitter,
            params,
            seed=seed + candidate_index * 100,
            n_jobs=n_jobs,
        )

        record = {
            "candidate_index": candidate_index,
            "params": params,
            **metrics,
        }
        rows.append(record)
        candidate_probabilities.append(oof_prob)

    rows_sorted = sorted(rows, key=hyperparameter_sort_key)
    best = rows_sorted[0]
    best_index = int(best["candidate_index"])
    best_params = best["params"]
    best_threshold = float(best["inner_selected_threshold"])

    search_rows = []
    for row in rows:
        flat = {
            "candidate_index": row["candidate_index"],
            "mean_inner_mcc_0_5": row["mean_inner_mcc_0_5"],
            "std_inner_mcc_0_5": row["std_inner_mcc_0_5"],
            "mean_inner_balanced_accuracy_0_5": (
                row["mean_inner_balanced_accuracy_0_5"]
            ),
            "mean_inner_roc_auc": row["mean_inner_roc_auc"],
            "inner_selected_threshold": row[
                "inner_selected_threshold"
            ],
            "inner_oof_mcc_selected_threshold": row[
                "inner_oof_mcc_selected_threshold"
            ],
            "selected_candidate": row["candidate_index"] == best_index,
        }
        flat.update({f"param_{k}": v for k, v in row["params"].items()})
        search_rows.append(flat)

    search_df = pd.DataFrame(search_rows).sort_values(
        [
            "selected_candidate",
            "mean_inner_mcc_0_5",
        ],
        ascending=[False, False],
    )

    return (
        best_params,
        best_threshold,
        search_df,
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
        "group_overlap_values": ";".join(overlap),
        "protein_overlap_count": len(id_overlap),
        "protein_overlap_values": ";".join(id_overlap),
        "passed": len(overlap) == 0 and len(id_overlap) == 0,
    }


def subset_frame(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    if subset == "full":
        return df.copy()
    if subset == "non_tim":
        return df[~df["is_tim_barrel"].astype(bool)].copy()
    raise ValueError(subset)


def pooled_summary(
    predictions: pd.DataFrame,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []

    group_columns = [
        "subset",
        "outer_fold_column",
        "feature_set",
        "threshold_type",
    ]

    for keys, group in predictions.groupby(group_columns):
        subset, outer_fold_column, feature_set, threshold_type = keys

        y_true = group[TARGET].to_numpy(dtype=int)
        y_prob = group["predicted_probability"].to_numpy(dtype=float)
        label_column = (
            "predicted_label_0_5"
            if threshold_type == "fixed_0_5"
            else "predicted_label_inner_selected"
        )
        y_pred = group[label_column].to_numpy(dtype=int)

        record = {
            "subset": subset,
            "outer_fold_column": outer_fold_column,
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
            }:
                low, high = bootstrap_ci(
                    y_true,
                    y_pred,
                    y_prob,
                    metric_name,
                    iterations,
                    rng,
                )
                record[f"{metric_name}_ci_low"] = low
                record[f"{metric_name}_ci_high"] = high

        records.append(record)

    return pd.DataFrame(records).sort_values(group_columns)


def main() -> int:
    args = parse_args()
    feature_dir = Path(args.feature_dir)
    fold_path = Path(args.fold_assignments)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not feature_dir.exists():
        print(f"ERROR: feature directory not found: {feature_dir}", file=sys.stderr)
        return 2
    if not fold_path.exists():
        print(f"ERROR: fold assignments not found: {fold_path}", file=sys.stderr)
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
        print(
            "ERROR: fold assignments missing columns: "
            + ", ".join(sorted(missing)),
            file=sys.stderr,
        )
        return 2

    subsets = ["full"]
    if args.include_non_tim:
        subsets.append("non_tim")

    all_predictions = []
    all_outer_metrics = []
    all_selected_params = []
    all_search_results = []
    all_thresholds = []
    all_leakage_audits = []
    feature_manifest = []

    run_start = time.time()

    for feature_set, filename in FEATURE_FILES.items():
        dataset_path = feature_dir / filename
        if not dataset_path.exists():
            raise FileNotFoundError(dataset_path)

        feature_df = pd.read_csv(dataset_path)

        metadata_columns = {
            IDENTIFIER,
            "gene_minami",
            "jwid",
            TARGET,
            "cohort_label",
            "label_source",
            "label_evidence_type",
            "label_confidence",
            "source_class_summary",
            "label_notes",
            "sccs",
        }

        features = [
            column
            for column in feature_df.columns
            if column not in metadata_columns
            and pd.api.types.is_numeric_dtype(feature_df[column])
            and feature_df[column].nunique(dropna=True) > 1
        ]

        if not features:
            raise RuntimeError(f"{feature_set} has no predictors")

        feature_manifest.extend(
            {
                "feature_set": feature_set,
                "feature_order": position,
                "feature": feature,
            }
            for position, feature in enumerate(features, start=1)
        )

        data = feature_df.merge(
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

        if data[list(OUTER_SCHEMES)].isna().any().any():
            raise RuntimeError(
                f"{feature_set}: missing outer fold assignments"
            )

        for subset_name in subsets:
            subset = subset_frame(data, subset_name)

            if subset[TARGET].nunique() < 2:
                print(
                    f"Skipping {feature_set}/{subset_name}: "
                    "insufficient class diversity"
                )
                continue

            for outer_fold_column, scheme in OUTER_SCHEMES.items():
                group_column = scheme["group_column"]
                outer_folds = sorted(
                    subset[outer_fold_column].dropna().unique()
                )

                print(
                    f"\nRunning {feature_set} | {subset_name} | "
                    f"{outer_fold_column}"
                )

                for outer_fold in outer_folds:
                    fold_start = time.time()
                    train = subset[
                        subset[outer_fold_column] != outer_fold
                    ].copy()
                    test = subset[
                        subset[outer_fold_column] == outer_fold
                    ].copy()

                    if train.empty or test.empty:
                        raise RuntimeError(
                            f"Empty outer train/test split: fold {outer_fold}"
                        )
                    if train[TARGET].nunique() < 2:
                        raise RuntimeError(
                            f"Outer training set lacks both classes: "
                            f"fold {outer_fold}"
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
                            "Outer split leakage detected"
                        )

                    X_train = train[features]
                    y_train = train[TARGET].to_numpy(dtype=int)
                    groups_train = train[group_column].astype(str).to_numpy()

                    (
                        best_params,
                        selected_threshold,
                        search_df,
                        actual_inner_folds,
                    ) = tune_outer_training_set(
                        X_train,
                        y_train,
                        groups_train,
                        inner_folds=args.inner_folds,
                        n_iter=args.n_iter,
                        seed=(
                            args.random_seed
                            + int(outer_fold) * 10_000
                            + len(feature_set) * 100
                        ),
                        n_jobs=args.n_jobs,
                    )

                    search_df["subset"] = subset_name
                    search_df["feature_set"] = feature_set
                    search_df["outer_fold_column"] = outer_fold_column
                    search_df["outer_fold"] = int(outer_fold)
                    search_df["actual_inner_folds"] = actual_inner_folds
                    all_search_results.append(search_df)

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

                    model = fit_model(
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

                    for threshold_type, threshold_value, metrics in (
                        ("fixed_0_5", 0.5, metrics_05),
                        (
                            "inner_selected",
                            selected_threshold,
                            metrics_selected,
                        ),
                    ):
                        all_outer_metrics.append(
                            {
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

                    for position, (_, row) in enumerate(test.iterrows()):
                        all_predictions.append(
                            {
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
                                "is_tim_barrel": bool(
                                    row["is_tim_barrel"]
                                ),
                                "predicted_probability": float(
                                    y_prob[position]
                                ),
                                "selected_threshold": float(
                                    selected_threshold
                                ),
                                "predicted_label_0_5": int(
                                    y_pred_05[position]
                                ),
                                "predicted_label_inner_selected": int(
                                    y_pred_selected[position]
                                ),
                                "selected_hyperparameters": json.dumps(
                                    best_params,
                                    sort_keys=True,
                                ),
                            }
                        )

                    elapsed = time.time() - fold_start
                    print(
                        f"  outer fold {outer_fold}: "
                        f"train={len(train)}, test={len(test)}, "
                        f"threshold={selected_threshold:.2f}, "
                        f"{elapsed/60:.1f} min"
                    )

    predictions = pd.DataFrame(all_predictions)
    outer_metrics = pd.DataFrame(all_outer_metrics)
    selected_params = pd.DataFrame(all_selected_params)
    search_results = pd.concat(all_search_results, ignore_index=True)
    thresholds = pd.DataFrame(all_thresholds)
    leakage_audit = pd.DataFrame(all_leakage_audits)
    feature_manifest_df = pd.DataFrame(feature_manifest)

    # Verify each protein receives exactly one OOF prediction per experiment.
    duplicate_check = (
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
    bad = duplicate_check[duplicate_check["prediction_count"] != 1]
    if not bad.empty:
        raise RuntimeError(
            "At least one protein does not have exactly one OOF prediction"
        )

    summary_frames = []
    for threshold_type in ("fixed_0_5", "inner_selected"):
        copy = predictions.copy()
        copy["threshold_type"] = threshold_type
        summary_frames.append(copy)

    summary_input = pd.concat(summary_frames, ignore_index=True)
    summary = pooled_summary(
        summary_input,
        iterations=args.bootstrap_iterations,
        seed=args.random_seed,
    )

    predictions.to_csv(
        output_dir / "stage09_oof_predictions.csv",
        index=False,
    )
    outer_metrics.to_csv(
        output_dir / "stage09_outer_fold_metrics.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "stage09_summary_metrics.csv",
        index=False,
    )
    selected_params.to_csv(
        output_dir / "stage09_selected_hyperparameters.csv",
        index=False,
    )
    search_results.to_csv(
        output_dir / "stage09_inner_search_results.csv",
        index=False,
    )
    thresholds.to_csv(
        output_dir / "stage09_thresholds.csv",
        index=False,
    )
    feature_manifest_df.to_csv(
        output_dir / "stage09_feature_manifest.csv",
        index=False,
    )
    leakage_audit.to_csv(
        output_dir / "stage09_leakage_audit.csv",
        index=False,
    )

    methods = {
        "stage": "09A",
        "model": "RandomForestClassifier",
        "feature_set_version": "2.0",
        "feature_sets": FEATURE_FILES,
        "outer_schemes": OUTER_SCHEMES,
        "inner_folds_requested": args.inner_folds,
        "randomized_parameter_candidates_per_outer_fold": args.n_iter,
        "selection_metric": "mean inner-fold MCC at fixed threshold 0.5",
        "threshold_selection": (
            "MCC-maximizing threshold from pooled inner out-of-fold "
            "probabilities; applied only to the corresponding outer test fold"
        ),
        "primary_threshold": 0.5,
        "bootstrap_iterations": args.bootstrap_iterations,
        "random_seed": args.random_seed,
        "hyperparameter_space": parameter_space(),
        "subsets": subsets,
        "note": (
            "No outer-test observations were used for tuning, threshold "
            "selection, or model fitting."
        ),
    }
    (output_dir / "stage09_methods.json").write_text(
        json.dumps(methods, indent=2)
    )

    total_elapsed = time.time() - run_start
    run_summary = {
        "protein_count": 181,
        "feature_sets": list(FEATURE_FILES),
        "subsets": subsets,
        "outer_grouping_schemes": list(OUTER_SCHEMES),
        "prediction_rows": len(predictions),
        "outer_metric_rows": len(outer_metrics),
        "selected_hyperparameter_rows": len(selected_params),
        "inner_search_rows": len(search_results),
        "leakage_failures": int((~leakage_audit["passed"]).sum()),
        "runtime_minutes": total_elapsed / 60,
        "status": "COMPLETE",
    }
    (output_dir / "stage09_run_summary.json").write_text(
        json.dumps(run_summary, indent=2)
    )

    print("\nStage 09A complete")
    print(f"Runtime: {total_elapsed/60:.1f} minutes")
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
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
