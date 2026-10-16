#!/usr/bin/env python3
"""
Stage 08.6: Re-run multivariable logistic baselines using frozen Feature Set v2.

Feature sets:
1. sequence_final
2. structure_minimal_final
3. structure_core_final
4. sequence_plus_structure_minimal
5. sequence_plus_structure_core

Evaluation:
- frozen 30%-identity homology-grouped folds (primary)
- frozen 40%-identity homology-grouped folds (sensitivity)
- optional full, non-TIM, and TIM-only subsets
- strictly out-of-fold predictions
- scaling and imputation fitted only within training folds
- no hyperparameter tuning

Example:
    python3 scripts/08_6_rerun_logistic_feature_set_v2.py \
      --feature-dir data/frozen/feature_set_v2 \
      --fold-assignments data/grouping/stage07/fold_assignments_v1.0.csv \
      --output-dir results/stage08_6 \
      --include-subset-analyses
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

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

FEATURE_FILES = {
    "sequence_final": "feature_set_v2_sequence_final.csv",
    "structure_minimal_final": "feature_set_v2_structure_minimal_final.csv",
    "structure_core_final": "feature_set_v2_structure_core_final.csv",
    "sequence_plus_structure_minimal": (
        "feature_set_v2_sequence_plus_structure_minimal.csv"
    ),
    "sequence_plus_structure_core": (
        "feature_set_v2_sequence_plus_structure_core.csv"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run logistic baselines on frozen Feature Set v2."
    )
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--fold-assignments", required=True)
    parser.add_argument("--output-dir", default="results/stage08_6")
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--include-subset-analyses",
        action="store_true",
    )
    return parser.parse_args()


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
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
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


def bootstrap_metric(
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


def load_dataset(path: Path, feature_set_name: str) -> tuple[pd.DataFrame, list[str]]:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    required = {IDENTIFIER, TARGET}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{feature_set_name} missing required columns: {sorted(missing)}"
        )

    features = [
        column
        for column in df.columns
        if column not in METADATA_COLUMNS
        and pd.api.types.is_numeric_dtype(df[column])
        and df[column].nunique(dropna=True) > 1
    ]

    if not features:
        raise ValueError(f"{feature_set_name} has no usable predictors")

    return df, features


def subset_frame(df: pd.DataFrame, subset_name: str) -> pd.DataFrame:
    if subset_name == "full":
        return df.copy()
    if subset_name == "non_tim":
        return df[~df["is_tim_barrel"].astype(bool)].copy()
    if subset_name == "tim_only":
        return df[df["is_tim_barrel"].astype(bool)].copy()
    raise ValueError(subset_name)


def run_oof(
    data: pd.DataFrame,
    features: list[str],
    feature_set_name: str,
    fold_column: str,
    subset_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = []
    fold_metrics = []
    coefficients = []

    for fold in sorted(data[fold_column].dropna().unique()):
        train = data[data[fold_column] != fold].copy()
        test = data[data[fold_column] == fold].copy()

        if train.empty or test.empty:
            continue
        if train[TARGET].nunique() < 2:
            raise RuntimeError(
                f"{feature_set_name}/{subset_name}/{fold_column}/fold {fold}: "
                "training set lacks both classes"
            )

        pipeline = logistic_pipeline()
        pipeline.fit(train[features], train[TARGET].astype(int))

        y_true = test[TARGET].to_numpy(dtype=int)
        y_prob = pipeline.predict_proba(test[features])[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = metric_dict(y_true, y_pred, y_prob)
        metrics.update(
            {
                "subset": subset_name,
                "fold_column": fold_column,
                "feature_set": feature_set_name,
                "fold": int(fold),
                "train_n": int(len(train)),
                "test_n": int(len(test)),
                "test_positive_n": int(y_true.sum()),
                "test_negative_n": int((y_true == 0).sum()),
                "test_has_both_classes": bool(len(np.unique(y_true)) == 2),
            }
        )
        fold_metrics.append(metrics)

        model = pipeline.named_steps["model"]
        for feature, coefficient in zip(features, model.coef_[0]):
            coefficients.append(
                {
                    "subset": subset_name,
                    "fold_column": fold_column,
                    "feature_set": feature_set_name,
                    "fold": int(fold),
                    "feature": feature,
                    "standardized_coefficient": float(coefficient),
                }
            )

        for position, (_, row) in enumerate(test.iterrows()):
            predictions.append(
                {
                    IDENTIFIER: row[IDENTIFIER],
                    "gene_minami": row.get("gene_minami", ""),
                    "jwid": row.get("jwid", ""),
                    TARGET: int(row[TARGET]),
                    "subset": subset_name,
                    "fold_column": fold_column,
                    "feature_set": feature_set_name,
                    "fold": int(fold),
                    "predicted_probability": float(y_prob[position]),
                    "predicted_label_0_5": int(y_pred[position]),
                }
            )

    prediction_df = pd.DataFrame(predictions)
    fold_metric_df = pd.DataFrame(fold_metrics)
    coefficient_df = pd.DataFrame(coefficients)

    if len(prediction_df) != len(data):
        raise RuntimeError(
            f"{feature_set_name}/{subset_name}/{fold_column}: "
            f"expected {len(data)} OOF predictions, got {len(prediction_df)}"
        )

    if prediction_df[IDENTIFIER].duplicated().any():
        raise RuntimeError(
            f"Duplicate OOF predictions for {feature_set_name}"
        )

    return prediction_df, fold_metric_df, coefficient_df


def summarize_predictions(
    predictions: pd.DataFrame,
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []

    group_columns = ["subset", "fold_column", "feature_set"]

    for keys, group in predictions.groupby(group_columns):
        subset, fold_column, feature_set = keys
        y_true = group[TARGET].to_numpy(dtype=int)
        y_pred = group["predicted_label_0_5"].to_numpy(dtype=int)
        y_prob = group["predicted_probability"].to_numpy(dtype=float)

        record = {
            "subset": subset,
            "fold_column": fold_column,
            "feature_set": feature_set,
            "n": int(len(group)),
            "positive_n": int(y_true.sum()),
            "negative_n": int((y_true == 0).sum()),
        }

        metrics = metric_dict(y_true, y_pred, y_prob)
        for metric_name, value in metrics.items():
            low, high = bootstrap_metric(
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
        print(f"ERROR: fold file not found: {fold_path}", file=sys.stderr)
        return 2

    folds = pd.read_csv(fold_path)
    required_fold_columns = {
        IDENTIFIER,
        "homology_cv_fold",
        "homology_cv_fold_sensitivity",
        "is_tim_barrel",
    }
    missing_fold_columns = required_fold_columns - set(folds.columns)
    if missing_fold_columns:
        print(
            "ERROR: missing fold columns: "
            + ", ".join(sorted(missing_fold_columns)),
            file=sys.stderr,
        )
        return 2

    subsets = ["full"]
    if args.include_subset_analyses:
        subsets.extend(["non_tim", "tim_only"])

    fold_columns = [
        "homology_cv_fold",
        "homology_cv_fold_sensitivity",
    ]

    all_predictions = []
    all_fold_metrics = []
    all_coefficients = []
    feature_manifest = []

    for feature_set_name, filename in FEATURE_FILES.items():
        dataset_path = feature_dir / filename
        dataset, features = load_dataset(
            dataset_path,
            feature_set_name,
        )

        feature_manifest.extend(
            {
                "feature_set": feature_set_name,
                "feature_order": position,
                "feature": feature,
            }
            for position, feature in enumerate(features, start=1)
        )

        data = dataset.merge(
            folds[
                [
                    IDENTIFIER,
                    "homology_cv_fold",
                    "homology_cv_fold_sensitivity",
                    "is_tim_barrel",
                ]
            ],
            on=IDENTIFIER,
            how="left",
            validate="one_to_one",
        )

        if data[
            ["homology_cv_fold", "homology_cv_fold_sensitivity"]
        ].isna().any().any():
            raise RuntimeError(
                f"{feature_set_name}: missing fold assignments"
            )

        for subset_name in subsets:
            subset = subset_frame(data, subset_name)
            if subset.empty or subset[TARGET].nunique() < 2:
                print(
                    f"Skipping {feature_set_name}/{subset_name}: "
                    "insufficient class diversity"
                )
                continue

            for fold_column in fold_columns:
                prediction_df, fold_metric_df, coefficient_df = run_oof(
                    subset,
                    features,
                    feature_set_name,
                    fold_column,
                    subset_name,
                )
                all_predictions.append(prediction_df)
                all_fold_metrics.append(fold_metric_df)
                all_coefficients.append(coefficient_df)

    predictions = pd.concat(all_predictions, ignore_index=True)
    fold_metrics = pd.concat(all_fold_metrics, ignore_index=True)
    coefficients = pd.concat(all_coefficients, ignore_index=True)

    summary = summarize_predictions(
        predictions,
        args.bootstrap_iterations,
        args.random_seed,
    )

    coefficient_summary = (
        coefficients.groupby(
            ["subset", "fold_column", "feature_set", "feature"]
        )["standardized_coefficient"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .sort_values(
            ["subset", "fold_column", "feature_set", "mean"],
            ascending=[True, True, True, False],
        )
    )

    predictions.to_csv(
        output_dir / "stage08_6_oof_predictions.csv",
        index=False,
    )
    fold_metrics.to_csv(
        output_dir / "stage08_6_fold_metrics.csv",
        index=False,
    )
    summary.to_csv(
        output_dir / "stage08_6_summary_metrics.csv",
        index=False,
    )
    coefficients.to_csv(
        output_dir / "stage08_6_fold_coefficients.csv",
        index=False,
    )
    coefficient_summary.to_csv(
        output_dir / "stage08_6_coefficient_summary.csv",
        index=False,
    )
    pd.DataFrame(feature_manifest).to_csv(
        output_dir / "stage08_6_feature_manifest.csv",
        index=False,
    )

    methods = {
        "feature_set_version": "2.0",
        "model": "L2-regularized class-weighted logistic regression",
        "C": 1.0,
        "solver": "liblinear",
        "threshold": 0.5,
        "primary_grouping": "30% MMseqs2 homology-grouped folds",
        "sensitivity_grouping": "40% MMseqs2 homology-grouped folds",
        "bootstrap_iterations": args.bootstrap_iterations,
        "feature_files": FEATURE_FILES,
        "note": (
            "No hyperparameter tuning. Imputation and scaling were fitted only "
            "inside each training fold. All reported predictions are out-of-fold."
        ),
    }
    (output_dir / "stage08_6_methods.json").write_text(
        json.dumps(methods, indent=2)
    )

    print("Stage 08.6 complete")
    print(f"Output directory: {output_dir}")
    print()
    print(
        summary[
            [
                "subset",
                "fold_column",
                "feature_set",
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
