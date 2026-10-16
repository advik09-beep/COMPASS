#!/usr/bin/env python3
"""
Stage 09.3A — SHAP + Leave-One-Feature-Out (LOFO) analysis

Goal:
Understand why Structural Core outperforms Combined Core in Stage 09 v2.

Default scope:
- Full 181-protein Benchmark v1.1
- Primary 30% homology-grouped outer folds
- Structural Core
- Sequence + Structural Core

Methods:
1. TreeSHAP on untouched outer-test folds using frozen Stage 09 v2 models.
2. Leave-one-feature-out retraining with the exact Stage 09 v2
   fold-specific RF hyperparameters held fixed.
3. Paired bootstrap estimates for delta MCC, ROC-AUC, and PR-AUC.

No relabeling, feature engineering, fold changes, or retuning occur here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
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

IDENTIFIER = "current_uniprot_accession"
TARGET = "binary_label"

FEATURE_FILES = {
    "structure_core_final": "feature_set_v3_structure_core_final.csv",
    "sequence_plus_structure_core": "feature_set_v3_sequence_plus_structure_core.csv",
}

METADATA_COLUMNS = {
    "current_uniprot_accession","gene_minami","jwid","binary_label","cohort_label",
    "label_source","label_evidence_type","label_confidence","source_class_summary",
    "label_notes","sccs","benchmark_version","previous_binary_label",
    "label_updated_from_minami_validation","minami_2023_table1_groe_dependency",
    "minami_2023_validation_status","minami_2023_validation_source",
}

SCHEMES = {
    "homology_cv_fold": "homology_cluster_30",
    "homology_cv_fold_sensitivity": "homology_cluster_40",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feature-dir", required=True)
    p.add_argument("--fold-assignments", required=True)
    p.add_argument("--stage09-v2-dir", required=True)
    p.add_argument("--output-dir", default="results/stage09_3A")
    p.add_argument("--include-sensitivity", action="store_true")
    p.add_argument("--bootstrap-iterations", type=int, default=3000)
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--top-n-plot", type=int, default=20)
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def predictor_columns(df):
    return [
        c for c in df.columns
        if c not in METADATA_COLUMNS
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].nunique(dropna=True) > 1
    ]


def metric_dict(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()
    return {
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "roc_auc": (
            float(roc_auc_score(y_true, y_prob))
            if len(np.unique(y_true)) == 2 else np.nan
        ),
        "pr_auc": (
            float(average_precision_score(y_true, y_prob))
            if len(np.unique(y_true)) == 2 else np.nan
        ),
    }


def build_rf(params: dict[str, Any], seed: int, n_jobs: int):
    normalized = {}
    for k, v in params.items():
        if isinstance(v, np.generic):
            v = v.item()
        normalized[k] = v

    return RandomForestClassifier(
        criterion="gini",
        random_state=seed,
        n_jobs=n_jobs,
        **normalized,
    )


def extract_positive_class_shap(model, X):
    explainer = shap.TreeExplainer(model)
    try:
        values = explainer.shap_values(X, check_additivity=False)
    except TypeError:
        values = explainer.shap_values(X)

    if isinstance(values, list):
        arr = np.asarray(values[1] if len(values) == 2 else values[-1])
    else:
        arr = np.asarray(values)
        if arr.ndim == 3:
            if arr.shape[-1] == 2:
                arr = arr[:, :, 1]
            elif arr.shape[0] == 2:
                arr = arr[1, :, :]
            else:
                raise RuntimeError(f"Unexpected SHAP shape {arr.shape}")

    if arr.ndim != 2 or arr.shape != X.shape:
        raise RuntimeError(
            f"SHAP shape {arr.shape} does not match X shape {X.shape}"
        )
    return arr


def locate_model_bundle(models_dir, feature_set, fold_column, fold):
    matches = []
    for path in sorted(models_dir.glob("*.joblib")):
        if feature_set not in path.name or f"fold{fold}" not in path.name:
            continue
        bundle = joblib.load(path)
        if (
            bundle.get("feature_set") == feature_set
            and bundle.get("subset") == "full"
            and bundle.get("outer_fold_column") == fold_column
            and int(bundle.get("outer_fold")) == int(fold)
        ):
            matches.append((path, bundle))

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one model for "
            f"{feature_set}/full/{fold_column}/fold{fold}, found {len(matches)}"
        )
    return matches[0]


def paired_bootstrap_delta(
    y,
    base_prob,
    alt_prob,
    metric,
    iterations,
    rng,
):
    deltas = []
    n = len(y)

    for _ in range(iterations):
        idx = rng.integers(0, n, n)
        yt = y[idx]
        bp = base_prob[idx]
        ap = alt_prob[idx]

        if metric in {"roc_auc", "pr_auc"} and len(np.unique(yt)) < 2:
            continue

        base_pred = (bp >= 0.5).astype(int)
        alt_pred = (ap >= 0.5).astype(int)

        b = metric_dict(yt, base_pred, bp)[metric]
        a = metric_dict(yt, alt_pred, ap)[metric]

        if math.isfinite(float(a)) and math.isfinite(float(b)):
            deltas.append(float(a - b))

    if not deltas:
        return np.nan, np.nan, np.nan

    mean_delta = float(np.mean(deltas))
    low, high = np.percentile(deltas, [2.5, 97.5])
    return mean_delta, float(low), float(high)


def make_shap_summary(shap_long):
    rows = []
    for keys, group in shap_long.groupby(
        ["feature_set", "outer_fold_column", "feature"]
    ):
        feature_set, fold_column, feature = keys
        rho, p_value = spearmanr(
            group["feature_value"],
            group["shap_value"],
            nan_policy="omit",
        )
        rows.append(
            {
                "feature_set": feature_set,
                "outer_fold_column": fold_column,
                "feature": feature,
                "feature_origin": group["feature_origin"].iloc[0],
                "mean_abs_shap": float(group["shap_value"].abs().mean()),
                "median_abs_shap": float(group["shap_value"].abs().median()),
                "mean_signed_shap": float(group["shap_value"].mean()),
                "mean_shap_true_positive_class": float(
                    group.loc[group[TARGET] == 1, "shap_value"].mean()
                ),
                "mean_shap_true_negative_class": float(
                    group.loc[group[TARGET] == 0, "shap_value"].mean()
                ),
                "feature_value_shap_spearman_rho": float(rho),
                "feature_value_shap_spearman_p": float(p_value),
                "protein_count": int(len(group)),
                "top10_fold_count": int(
                    group.groupby("outer_fold")["top10_in_fold"].first().sum()
                ),
            }
        )

    out = pd.DataFrame(rows)
    out["shap_rank"] = (
        out.groupby(["feature_set", "outer_fold_column"])["mean_abs_shap"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    return out.sort_values(
        ["feature_set", "outer_fold_column", "shap_rank"]
    )


def make_plots(shap_summary, lofo_summary, output_dir, top_n):
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    primary = "homology_cv_fold"

    for feature_set in FEATURE_FILES:
        s = shap_summary[
            (shap_summary["feature_set"] == feature_set)
            & (shap_summary["outer_fold_column"] == primary)
        ].copy()
        if not s.empty:
            top = s.nsmallest(top_n, "shap_rank").sort_values("mean_abs_shap")
            fig, ax = plt.subplots(figsize=(9, 8))
            ax.barh(top["feature"], top["mean_abs_shap"])
            ax.set_xlabel("Mean absolute SHAP value")
            ax.set_ylabel("Feature")
            ax.set_title(f"SHAP importance — {feature_set.replace('_',' ')}")
            fig.tight_layout()
            fig.savefig(
                fig_dir / f"shap_importance_{feature_set}.png",
                dpi=300,
                bbox_inches="tight",
            )
            fig.savefig(
                fig_dir / f"shap_importance_{feature_set}.pdf",
                bbox_inches="tight",
            )
            plt.close(fig)

        l = lofo_summary[
            (lofo_summary["feature_set"] == feature_set)
            & (lofo_summary["outer_fold_column"] == primary)
        ].copy()
        if not l.empty:
            l["lofo_mcc_importance"] = -l["delta_mcc"]
            top = l.sort_values(
                "lofo_mcc_importance",
                ascending=False,
            ).head(top_n).sort_values("lofo_mcc_importance")
            fig, ax = plt.subplots(figsize=(9, 8))
            ax.barh(top["feature_removed"], top["lofo_mcc_importance"])
            ax.axvline(0, linewidth=1)
            ax.set_xlabel("Decrease in pooled OOF MCC when feature is removed")
            ax.set_ylabel("Feature")
            ax.set_title(
                f"Leave-one-feature-out importance — "
                f"{feature_set.replace('_',' ')}"
            )
            fig.tight_layout()
            fig.savefig(
                fig_dir / f"lofo_mcc_{feature_set}.png",
                dpi=300,
                bbox_inches="tight",
            )
            fig.savefig(
                fig_dir / f"lofo_mcc_{feature_set}.pdf",
                bbox_inches="tight",
            )
            plt.close(fig)


def main():
    args = parse_args()

    feature_dir = Path(args.feature_dir)
    folds_path = Path(args.fold_assignments)
    stage09_dir = Path(args.stage09_v2_dir)
    models_dir = stage09_dir / "saved_models"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for p, label in [
        (feature_dir, "Feature Set v3 directory"),
        (folds_path, "fold assignments"),
        (models_dir, "Stage 09 v2 saved_models"),
    ]:
        if not p.exists():
            print(f"ERROR: {label} not found: {p}", file=sys.stderr)
            return 2

    folds = pd.read_csv(folds_path)
    required = {
        IDENTIFIER,
        "homology_cv_fold",
        "homology_cv_fold_sensitivity",
        "homology_cluster_30",
        "homology_cluster_40",
        "is_tim_barrel",
    }
    missing = required - set(folds.columns)
    if missing:
        raise RuntimeError(f"Fold assignments missing {sorted(missing)}")

    schemes = ["homology_cv_fold"]
    if args.include_sensitivity:
        schemes.append("homology_cv_fold_sensitivity")

    datasets = {}
    feature_lists = {}

    for feature_set, filename in FEATURE_FILES.items():
        path = feature_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)
        if len(df) != 181 or int(df[TARGET].sum()) != 73:
            raise RuntimeError(
                f"{feature_set}: expected 181 proteins / 73 positives"
            )

        features = predictor_columns(df)
        df = df.merge(
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
        datasets[feature_set] = df
        feature_lists[feature_set] = features

    structural_features = set(feature_lists["structure_core_final"])

    shap_rows = []
    lofo_prediction_rows = []
    baseline_prediction_rows = []
    model_audit_rows = []
    total_start = time.time()

    for feature_set, data in datasets.items():
        features = feature_lists[feature_set]

        for fold_column in schemes:
            group_column = SCHEMES[fold_column]
            print(f"\n=== {feature_set} | {fold_column} ===")

            for fold in sorted(data[fold_column].unique()):
                fold = int(fold)
                fold_start = time.time()

                train = data[data[fold_column] != fold].copy()
                test = data[data[fold_column] == fold].copy()

                overlap = (
                    set(train[group_column].astype(str))
                    & set(test[group_column].astype(str))
                )
                if overlap:
                    raise RuntimeError(f"Homology leakage in fold {fold}")

                model_path, bundle = locate_model_bundle(
                    models_dir,
                    feature_set,
                    fold_column,
                    fold,
                )
                model = bundle["model"]
                bundle_features = list(bundle["features"])

                if bundle_features != features:
                    raise RuntimeError(
                        f"Saved model features differ from current v3 for "
                        f"{feature_set}/fold{fold}"
                    )

                X_train = train[features]
                X_test = test[features]
                y_train = train[TARGET].to_numpy(dtype=int)
                y_test = test[TARGET].to_numpy(dtype=int)

                baseline_prob = model.predict_proba(X_test)[:, 1]

                for i, (_, row) in enumerate(test.iterrows()):
                    baseline_prediction_rows.append(
                        {
                            IDENTIFIER: row[IDENTIFIER],
                            "gene_minami": row.get("gene_minami", ""),
                            TARGET: int(row[TARGET]),
                            "feature_set": feature_set,
                            "outer_fold_column": fold_column,
                            "outer_fold": fold,
                            "predicted_probability": float(baseline_prob[i]),
                        }
                    )

                shap_values = extract_positive_class_shap(model, X_test)
                fold_mean_abs = np.mean(np.abs(shap_values), axis=0)
                top10_indices = set(
                    np.argsort(fold_mean_abs)[::-1][:10]
                )

                for sample_i, (_, row) in enumerate(test.iterrows()):
                    for feature_i, feature in enumerate(features):
                        shap_rows.append(
                            {
                                IDENTIFIER: row[IDENTIFIER],
                                "gene_minami": row.get("gene_minami", ""),
                                TARGET: int(row[TARGET]),
                                "feature_set": feature_set,
                                "outer_fold_column": fold_column,
                                "outer_fold": fold,
                                "feature": feature,
                                "feature_origin": (
                                    "structure"
                                    if feature in structural_features
                                    else "sequence"
                                ),
                                "feature_value": float(
                                    X_test.iloc[sample_i, feature_i]
                                ),
                                "shap_value": float(
                                    shap_values[sample_i, feature_i]
                                ),
                                "top10_in_fold": feature_i in top10_indices,
                            }
                        )

                params = bundle["selected_hyperparameters"]

                for removed_feature in features:
                    retained = [f for f in features if f != removed_feature]
                    lofo_model = build_rf(
                        params,
                        seed=args.random_seed + fold,
                        n_jobs=args.n_jobs,
                    )
                    lofo_model.fit(train[retained], y_train)
                    lofo_prob = lofo_model.predict_proba(
                        test[retained]
                    )[:, 1]

                    for i, (_, row) in enumerate(test.iterrows()):
                        lofo_prediction_rows.append(
                            {
                                IDENTIFIER: row[IDENTIFIER],
                                "gene_minami": row.get("gene_minami", ""),
                                TARGET: int(row[TARGET]),
                                "feature_set": feature_set,
                                "outer_fold_column": fold_column,
                                "outer_fold": fold,
                                "feature_removed": removed_feature,
                                "feature_origin": (
                                    "structure"
                                    if removed_feature in structural_features
                                    else "sequence"
                                ),
                                "predicted_probability": float(lofo_prob[i]),
                            }
                        )

                model_audit_rows.append(
                    {
                        "feature_set": feature_set,
                        "outer_fold_column": fold_column,
                        "outer_fold": fold,
                        "model_file": model_path.name,
                        "feature_count": len(features),
                        "train_n": len(train),
                        "test_n": len(test),
                        "group_overlap_count": len(overlap),
                        "model_sha256": sha256_file(model_path),
                    }
                )

                print(
                    f"fold {fold}: SHAP + {len(features)} LOFO fits complete "
                    f"in {(time.time()-fold_start)/60:.1f} min"
                )

    shap_long = pd.DataFrame(shap_rows)
    baseline_predictions = pd.DataFrame(baseline_prediction_rows)
    lofo_predictions = pd.DataFrame(lofo_prediction_rows)
    model_audit = pd.DataFrame(model_audit_rows)

    shap_summary = make_shap_summary(shap_long)

    lofo_summary_rows = []
    rng = np.random.default_rng(args.random_seed)

    for keys, base in baseline_predictions.groupby(
        ["feature_set", "outer_fold_column"]
    ):
        feature_set, fold_column = keys
        base = base.sort_values(IDENTIFIER)
        y = base[TARGET].to_numpy(dtype=int)
        base_prob = base["predicted_probability"].to_numpy(dtype=float)
        base_pred = (base_prob >= 0.5).astype(int)
        base_metrics = metric_dict(y, base_pred, base_prob)

        subset_lofo = lofo_predictions[
            (lofo_predictions["feature_set"] == feature_set)
            & (lofo_predictions["outer_fold_column"] == fold_column)
        ]

        for removed_feature, alt in subset_lofo.groupby("feature_removed"):
            alt = alt.sort_values(IDENTIFIER)

            if list(base[IDENTIFIER]) != list(alt[IDENTIFIER]):
                raise RuntimeError(
                    f"Protein-order mismatch for {removed_feature}"
                )

            alt_prob = alt["predicted_probability"].to_numpy(dtype=float)
            alt_pred = (alt_prob >= 0.5).astype(int)
            alt_metrics = metric_dict(y, alt_pred, alt_prob)

            record = {
                "feature_set": feature_set,
                "outer_fold_column": fold_column,
                "feature_removed": removed_feature,
                "feature_origin": alt["feature_origin"].iloc[0],
            }

            for metric in [
                "mcc",
                "balanced_accuracy",
                "roc_auc",
                "pr_auc",
            ]:
                record[f"baseline_{metric}"] = base_metrics[metric]
                record[f"lofo_{metric}"] = alt_metrics[metric]
                record[f"delta_{metric}"] = (
                    alt_metrics[metric] - base_metrics[metric]
                )

            for metric in ["mcc", "roc_auc", "pr_auc"]:
                mean_delta, low, high = paired_bootstrap_delta(
                    y,
                    base_prob,
                    alt_prob,
                    metric,
                    args.bootstrap_iterations,
                    rng,
                )
                record[f"bootstrap_mean_delta_{metric}"] = mean_delta
                record[f"delta_{metric}_ci_low"] = low
                record[f"delta_{metric}_ci_high"] = high

            record["lofo_interpretation_mcc"] = (
                "important_positive_contributor"
                if record["delta_mcc"] < -0.01
                else (
                    "candidate_harmful_or_noisy"
                    if record["delta_mcc"] > 0.01
                    else "small_or_neutral_effect"
                )
            )
            lofo_summary_rows.append(record)

    lofo_summary = pd.DataFrame(lofo_summary_rows)

    combined = shap_summary.merge(
        lofo_summary,
        left_on=["feature_set", "outer_fold_column", "feature"],
        right_on=[
            "feature_set",
            "outer_fold_column",
            "feature_removed",
        ],
        how="inner",
        validate="one_to_one",
    )
    combined["lofo_mcc_importance"] = -combined["delta_mcc"]
    combined["lofo_roc_auc_importance"] = -combined["delta_roc_auc"]
    combined["lofo_pr_auc_importance"] = -combined["delta_pr_auc"]
    combined["lofo_mcc_rank"] = (
        combined.groupby(
            ["feature_set", "outer_fold_column"]
        )["lofo_mcc_importance"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    shap_long.to_csv(output_dir / "shap_values_oof_long.csv", index=False)
    shap_summary.to_csv(output_dir / "shap_feature_summary.csv", index=False)
    baseline_predictions.to_csv(
        output_dir / "baseline_stage09_v2_predictions_reconstructed.csv",
        index=False,
    )
    lofo_predictions.to_csv(
        output_dir / "lofo_oof_predictions.csv",
        index=False,
    )
    lofo_summary.to_csv(
        output_dir / "lofo_feature_summary.csv",
        index=False,
    )
    combined.to_csv(
        output_dir / "shap_lofo_consensus.csv",
        index=False,
    )
    model_audit.to_csv(output_dir / "model_audit.csv", index=False)

    make_plots(
        shap_summary,
        lofo_summary,
        output_dir,
        args.top_n_plot,
    )

    headline = {}
    primary = "homology_cv_fold"

    for feature_set in FEATURE_FILES:
        c = combined[
            (combined["feature_set"] == feature_set)
            & (combined["outer_fold_column"] == primary)
        ].copy()

        if c.empty:
            continue

        headline[feature_set] = {
            "top_by_shap": (
                c.nsmallest(10, "shap_rank")[
                    [
                        "feature",
                        "feature_origin",
                        "mean_abs_shap",
                        "shap_rank",
                        "delta_mcc",
                    ]
                ].to_dict(orient="records")
            ),
            "top_by_lofo_mcc": (
                c.sort_values(
                    "lofo_mcc_importance",
                    ascending=False,
                ).head(10)[
                    [
                        "feature",
                        "feature_origin",
                        "lofo_mcc_importance",
                        "delta_mcc",
                        "delta_mcc_ci_low",
                        "delta_mcc_ci_high",
                    ]
                ].to_dict(orient="records")
            ),
            "features_whose_removal_most_improves_mcc": (
                c.sort_values(
                    "delta_mcc",
                    ascending=False,
                ).head(10)[
                    [
                        "feature",
                        "feature_origin",
                        "mean_abs_shap",
                        "delta_mcc",
                        "delta_mcc_ci_low",
                        "delta_mcc_ci_high",
                    ]
                ].to_dict(orient="records")
            ),
        }

    summary = {
        "stage": "09.3A",
        "status": "COMPLETE",
        "benchmark_version": "1.1",
        "feature_set_version": "3.0",
        "analysis_scope": (
            "full benchmark; primary 30% grouping"
            + (" plus 40% sensitivity" if args.include_sensitivity else "")
        ),
        "methods": {
            "shap": (
                "TreeSHAP on each untouched outer test fold using the "
                "corresponding frozen Stage 09 v2 model."
            ),
            "lofo": (
                "One feature removed at a time; RF retrained using the exact "
                "Stage 09 v2 fold-specific hyperparameters with no retuning."
            ),
            "lofo_delta_definition": (
                "LOFO metric minus baseline metric; positive delta means "
                "removing the feature improved performance."
            ),
            "bootstrap_iterations": args.bootstrap_iterations,
        },
        "headline_primary_results": headline,
        "runtime_minutes": (time.time() - total_start) / 60,
    }

    (output_dir / "stage09_3A_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    methods_text = """# Stage 09.3A Methods

TreeSHAP is calculated on each untouched outer test fold using the matching
saved Stage 09 v2 Random Forest.

Leave-one-feature-out (LOFO) ablation removes one feature at a time and retrains
the Random Forest on the corresponding outer-training fold using the exact
hyperparameters selected in Stage 09 v2. No additional tuning is performed.

LOFO delta = metric(feature removed) - metric(full model).

Therefore:
- negative delta: removing the feature hurt performance;
- near-zero delta: little detectable contribution;
- positive delta: removing the feature improved performance.

Paired bootstrap intervals resample the same proteins for the baseline and
LOFO predictions.
"""
    (output_dir / "stage09_3A_methods.md").write_text(methods_text)

    print("\nStage 09.3A complete")
    print(f"Runtime: {(time.time()-total_start)/60:.1f} minutes")
    print(f"Output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
