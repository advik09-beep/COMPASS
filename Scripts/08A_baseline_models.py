#!/usr/bin/env python3
"""
Stage 08A: fixed-fold baseline models for GroEL/GroES dependence.

Models:
1. Majority-class baseline
2. Length-only logistic regression
3. Contact-order-only logistic regression

Primary evaluation:
- frozen 30%-identity homology-grouped folds from Stage 07

Sensitivity evaluation:
- frozen 40%-identity homology-grouped folds from Stage 07

No hyperparameter tuning is performed. All preprocessing is fitted within each
training fold. Predictions are strictly out-of-fold.

Example:
    python3 scripts/08A_baseline_models.py \
      --benchmark data/frozen/benchmark_v1_0/benchmark_v1.0.csv \
      --fold-assignments data/grouping/stage07/fold_assignments_v1.0.csv \
      --output-dir results/stage08A
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


IDENTIFIER = "current_uniprot_accession"
TARGET = "binary_label"

MODEL_SPECS = {
    "majority": [],
    "length_only": ["seq_length"],
    "contact_order_only": ["relative_contact_order"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 08A fixed-fold baseline models."
    )
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--fold-assignments", required=True)
    parser.add_argument(
        "--output-dir",
        default="results/stage08A",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=5000,
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--include-subset-analyses",
        action="store_true",
        help="Also run non-TIM and TIM-only exploratory analyses.",
    )
    return parser.parse_args()


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()
    return float(tn / (tn + fp)) if (tn + fp) else float("nan")


def metric_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    result = {
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "specificity": specificity_score(y_true, y_pred),
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
    values: list[float] = []
    n = len(y_true)

    for _ in range(iterations):
        index = rng.integers(0, n, n)
        yt = y_true[index]
        yp = y_pred[index]
        pr = y_prob[index]

        if metric_name in {"roc_auc", "pr_auc"} and len(np.unique(yt)) < 2:
            continue

        value = metric_dict(yt, yp, pr)[metric_name]
        if math.isfinite(value):
            values.append(value)

    if not values:
        return float("nan"), float("nan")

    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def logistic_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def generate_oof_predictions(
    data: pd.DataFrame,
    fold_column: str,
    model_name: str,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = sorted(data[fold_column].dropna().unique().tolist())
    predictions = []
    fold_metrics = []

    for fold in folds:
        train = data[data[fold_column] != fold].copy()
        test = data[data[fold_column] == fold].copy()

        if train.empty or test.empty:
            raise RuntimeError(f"Empty train/test set for fold {fold}")
        if train[TARGET].nunique() < 2:
            raise RuntimeError(
                f"Training fold complement for {fold} lacks both classes"
            )

        y_train = train[TARGET].to_numpy(dtype=int)
        y_test = test[TARGET].to_numpy(dtype=int)

        if model_name == "majority":
            training_prevalence = float(y_train.mean())
            majority_class = int(training_prevalence >= 0.5)
            y_prob = np.full(len(test), training_prevalence, dtype=float)
            y_pred = np.full(len(test), majority_class, dtype=int)
            coefficient = float("nan")
            intercept = float("nan")
        else:
            pipeline = logistic_pipeline()
            pipeline.fit(train[features], y_train)
            y_prob = pipeline.predict_proba(test[features])[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            model = pipeline.named_steps["model"]
            coefficient = float(model.coef_[0][0])
            intercept = float(model.intercept_[0])

        fold_result = metric_dict(y_test, y_pred, y_prob)
        fold_result.update(
            {
                "model": model_name,
                "fold_column": fold_column,
                "fold": int(fold),
                "train_n": int(len(train)),
                "test_n": int(len(test)),
                "test_positive_n": int(y_test.sum()),
                "test_negative_n": int((y_test == 0).sum()),
                "standardized_coefficient": coefficient,
                "intercept": intercept,
            }
        )
        fold_metrics.append(fold_result)

        for row_position, (_, row) in enumerate(test.iterrows()):
            predictions.append(
                {
                    IDENTIFIER: row[IDENTIFIER],
                    "gene_minami": row.get("gene_minami", ""),
                    "jwid": row.get("jwid", ""),
                    "sccs": row.get("sccs", ""),
                    "is_tim_barrel": row.get("is_tim_barrel", np.nan),
                    TARGET: int(row[TARGET]),
                    "model": model_name,
                    "fold_column": fold_column,
                    "fold": int(fold),
                    "predicted_probability": float(y_prob[row_position]),
                    "predicted_label_0_5": int(y_pred[row_position]),
                }
            )

    prediction_df = pd.DataFrame(predictions)
    fold_metric_df = pd.DataFrame(fold_metrics)

    if prediction_df[IDENTIFIER].duplicated().any():
        raise RuntimeError(
            f"Duplicate OOF predictions for {model_name}, {fold_column}"
        )
    if len(prediction_df) != len(data):
        raise RuntimeError(
            f"Expected {len(data)} OOF predictions; got {len(prediction_df)}"
        )

    return prediction_df, fold_metric_df


def summarize_oof(
    predictions: pd.DataFrame,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    records = []
    rng = np.random.default_rng(seed)

    for (fold_column, model), group in predictions.groupby(
        ["fold_column", "model"]
    ):
        y_true = group[TARGET].to_numpy(dtype=int)
        y_pred = group["predicted_label_0_5"].to_numpy(dtype=int)
        y_prob = group["predicted_probability"].to_numpy(dtype=float)

        metrics = metric_dict(y_true, y_pred, y_prob)
        record = {
            "fold_column": fold_column,
            "model": model,
            "n": len(group),
            "positive_n": int(y_true.sum()),
            "negative_n": int((y_true == 0).sum()),
        }

        for metric_name, value in metrics.items():
            low, high = bootstrap_ci(
                y_true,
                y_pred,
                y_prob,
                metric_name,
                iterations,
                rng,
            )
            record[metric_name] = value
            record[f"{metric_name}_ci_low"] = low
            record[f"{metric_name}_ci_high"] = high

        records.append(record)

    return pd.DataFrame(records).sort_values(
        ["fold_column", "model"]
    )


def subset_frame(data: pd.DataFrame, subset_name: str) -> pd.DataFrame:
    if subset_name == "full":
        return data.copy()
    if subset_name == "non_tim":
        return data[~data["is_tim_barrel"].astype(bool)].copy()
    if subset_name == "tim_only":
        return data[data["is_tim_barrel"].astype(bool)].copy()
    raise ValueError(subset_name)


def main() -> int:
    args = parse_args()
    benchmark_path = Path(args.benchmark)
    fold_path = Path(args.fold_assignments)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not benchmark_path.exists():
        print(f"ERROR: benchmark not found: {benchmark_path}", file=sys.stderr)
        return 2
    if not fold_path.exists():
        print(f"ERROR: fold assignments not found: {fold_path}", file=sys.stderr)
        return 2

    benchmark = pd.read_csv(benchmark_path)
    folds = pd.read_csv(fold_path)

    required_benchmark = {
        IDENTIFIER,
        TARGET,
        "seq_length",
        "relative_contact_order",
    }
    missing = required_benchmark - set(benchmark.columns)
    if missing:
        print(
            "ERROR: benchmark missing: " + ", ".join(sorted(missing)),
            file=sys.stderr,
        )
        return 2

    fold_columns = [
        "homology_cv_fold",
        "homology_cv_fold_sensitivity",
    ]
    missing_fold_columns = [
        column for column in fold_columns if column not in folds.columns
    ]
    if missing_fold_columns:
        print(
            "ERROR: fold file missing: "
            + ", ".join(missing_fold_columns),
            file=sys.stderr,
        )
        return 2

    fold_keep = [
        IDENTIFIER,
        *fold_columns,
        "homology_cluster_30",
        "homology_cluster_40",
        "is_tim_barrel",
    ]
    fold_keep = [column for column in fold_keep if column in folds.columns]

    data = benchmark.merge(
        folds[fold_keep],
        on=IDENTIFIER,
        how="left",
        validate="one_to_one",
    )

    if data[fold_columns].isna().any().any():
        raise RuntimeError("At least one protein lacks a CV fold assignment")

    subsets = ["full"]
    if args.include_subset_analyses:
        subsets.extend(["non_tim", "tim_only"])

    all_predictions = []
    all_fold_metrics = []
    all_summaries = []

    for subset_name in subsets:
        subset = subset_frame(data, subset_name)
        if subset.empty or subset[TARGET].nunique() < 2:
            print(f"Skipping subset {subset_name}: insufficient classes")
            continue

        for fold_column in fold_columns:
            # Remove fold IDs that lose all samples after subset filtering.
            active = subset[subset[fold_column].notna()].copy()

            for model_name, features in MODEL_SPECS.items():
                prediction_df, fold_metric_df = generate_oof_predictions(
                    active,
                    fold_column,
                    model_name,
                    features,
                )
                prediction_df["subset"] = subset_name
                fold_metric_df["subset"] = subset_name
                all_predictions.append(prediction_df)
                all_fold_metrics.append(fold_metric_df)

        subset_predictions = pd.concat(
            [
                frame
                for frame in all_predictions
                if frame["subset"].iloc[0] == subset_name
            ],
            ignore_index=True,
        )
        subset_summary = summarize_oof(
            subset_predictions,
            args.bootstrap_iterations,
            args.random_seed,
        )
        subset_summary["subset"] = subset_name
        all_summaries.append(subset_summary)

    predictions = pd.concat(all_predictions, ignore_index=True)
    fold_metrics = pd.concat(all_fold_metrics, ignore_index=True)
    summary = pd.concat(all_summaries, ignore_index=True)

    predictions.to_csv(
        output_dir / "stage08A_oof_predictions.csv",
        index=False,
    )
    fold_metrics.to_csv(
        output_dir / "stage08A_fold_metrics.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "stage08A_summary_metrics.csv",
        index=False,
    )

    coefficient_summary = (
        fold_metrics[
            fold_metrics["model"].isin(["length_only", "contact_order_only"])
        ]
        .groupby(["subset", "fold_column", "model"])[
            "standardized_coefficient"
        ]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    coefficient_summary.to_csv(
        output_dir / "stage08A_logistic_coefficients.csv",
        index=False,
    )

    methods = {
        "primary_grouping": "homology_cv_fold (30% MMseqs2 identity)",
        "sensitivity_grouping": (
            "homology_cv_fold_sensitivity (40% MMseqs2 identity)"
        ),
        "models": MODEL_SPECS,
        "logistic_regression": {
            "penalty": "L2",
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "liblinear",
            "threshold": 0.5,
        },
        "primary_metric": "MCC",
        "bootstrap_iterations": args.bootstrap_iterations,
        "note": (
            "No hyperparameter tuning. Preprocessing is fitted only inside "
            "each training fold. All reported predictions are out-of-fold."
        ),
    }
    (output_dir / "stage08A_methods.json").write_text(
        json.dumps(methods, indent=2)
    )

    print("Stage 08A complete")
    print(f"  Proteins: {len(data)}")
    print(f"  Models: {', '.join(MODEL_SPECS)}")
    print(f"  Grouping schemes: {', '.join(fold_columns)}")
    print(f"  Subsets: {', '.join(subsets)}")
    print(f"  Output: {output_dir}")
    print()
    print(
        summary[
            [
                "subset",
                "fold_column",
                "model",
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
