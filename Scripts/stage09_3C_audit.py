#!/usr/bin/env python3
"""
Stage 09.3C — Evaluation Reconciliation + Fold Difficulty Audit

Questions
---------
1. Why did Structural Core move from Stage 09 v2 (~0.537 MCC) to
   Stage 09.3B (~0.559 MCC) despite the same benchmark and 30% folds?

2. Why is outer fold 3 consistently worse across nearly every 09.3B feature set?

This is an AUDIT stage, not a model-improvement stage.

It:
- reconstructs Stage 09 v2 Structural Core OOF predictions from saved models;
- compares them protein-by-protein with Stage 09.3B Structural Core OOF;
- compares fold metrics and selected RF hyperparameters;
- separates prediction changes caused by different fitted models from changes
  caused by data/fold composition;
- quantifies fold difficulty across every 09.3B candidate model;
- profiles fold 3 for class balance, TIM-barrel enrichment, homology-cluster
  structure, feature-distribution shift, and repeatedly misclassified proteins.

No labels, features, folds, thresholds, or models are modified.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
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

STRUCTURE_FILE = "feature_set_v3_structure_core_final.csv"

METADATA_COLUMNS = {
    "current_uniprot_accession","gene_minami","jwid","binary_label","cohort_label",
    "label_source","label_evidence_type","label_confidence","source_class_summary",
    "label_notes","sccs","benchmark_version","previous_binary_label",
    "label_updated_from_minami_validation","minami_2023_table1_groe_dependency",
    "minami_2023_validation_status","minami_2023_validation_source",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feature-dir", required=True)
    p.add_argument("--fold-assignments", required=True)
    p.add_argument("--stage09-v2-dir", required=True)
    p.add_argument("--stage09-3b-dir", required=True)
    p.add_argument("--output-dir", default="results/stage09_3C")
    p.add_argument("--focus-fold", type=int, default=3)
    return p.parse_args()


def predictor_columns(df):
    return [
        c for c in df.columns
        if c not in METADATA_COLUMNS
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].nunique(dropna=True) > 1
    ]


def metrics(y, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    return {
        "n": int(len(y)),
        "positives": int(np.sum(y == 1)),
        "negatives": int(np.sum(y == 0)),
        "positive_fraction": float(np.mean(y)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "roc_auc": (
            float(roc_auc_score(y, prob))
            if len(np.unique(y)) == 2 else np.nan
        ),
        "pr_auc": (
            float(average_precision_score(y, prob))
            if len(np.unique(y)) == 2 else np.nan
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def locate_v2_bundle(models_dir, fold):
    """
    Find full-benchmark / 30%-grouping / Structural Core v2 model
    without assuming an exact filename.
    """
    candidates = []

    for path in sorted(models_dir.glob("*.joblib")):
        try:
            bundle = joblib.load(path)
        except Exception:
            continue

        if (
            bundle.get("feature_set") == "structure_core_final"
            and bundle.get("subset") == "full"
            and bundle.get("outer_fold_column") == "homology_cv_fold"
            and int(bundle.get("outer_fold", -999)) == int(fold)
        ):
            candidates.append((path, bundle))

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one Stage 09 v2 Structural Core model "
            f"for fold {fold}; found {len(candidates)}"
        )

    return candidates[0]


def reconstruct_v2_oof(data, features, stage09_v2_dir):
    models_dir = stage09_v2_dir / "saved_models"
    rows = []
    hp_rows = []

    for fold in sorted(data["homology_cv_fold"].unique()):
        fold = int(fold)
        test = data[data["homology_cv_fold"] == fold].copy()

        path, bundle = locate_v2_bundle(models_dir, fold)

        model_features = list(bundle["features"])
        if model_features != features:
            raise RuntimeError(
                f"Stage 09 v2 feature order mismatch in fold {fold}"
            )

        prob = bundle["model"].predict_proba(test[features])[:, 1]

        for i, (_, row) in enumerate(test.iterrows()):
            rows.append({
                IDENTIFIER: row[IDENTIFIER],
                "gene_minami": row.get("gene_minami", ""),
                TARGET: int(row[TARGET]),
                "outer_fold": fold,
                "predicted_probability": float(prob[i]),
            })

        params = bundle.get("selected_hyperparameters", {})
        hp_rows.append({
            "source": "stage09_v2",
            "outer_fold": fold,
            "model_file": path.name,
            "selected_threshold": bundle.get("selected_threshold", np.nan),
            **{f"hp_{k}": v for k, v in params.items()},
        })

    return pd.DataFrame(rows), pd.DataFrame(hp_rows)


def read_3b_oof(stage09_3b_dir):
    path = stage09_3b_dir / "stage09_3B_oof_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. This file was generated by Stage 09.3B."
        )
    return pd.read_csv(path)


def read_3b_fold_metrics(stage09_3b_dir):
    path = stage09_3b_dir / "stage09_3B_fold_metrics.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def read_3b_models(stage09_3b_dir):
    model_dir = stage09_3b_dir / "saved_models"
    rows = []

    if not model_dir.exists():
        return pd.DataFrame()

    for path in sorted(model_dir.glob("structural_core__fold*.joblib")):
        bundle = joblib.load(path)
        params = bundle.get("selected_hyperparameters", {})
        rows.append({
            "source": "stage09_3B",
            "outer_fold": int(bundle["outer_fold"]),
            "model_file": path.name,
            "selected_threshold": bundle.get("selected_threshold", np.nan),
            **{f"hp_{k}": v for k, v in params.items()},
        })

    return pd.DataFrame(rows)


def pooled_and_fold_metrics(oof, feature_set_col=None):
    rows = []

    if feature_set_col is None:
        groups = [("model", oof)]
    else:
        groups = list(oof.groupby(feature_set_col))

    for model_name, group in groups:
        y = group[TARGET].to_numpy(dtype=int)
        p = group["predicted_probability"].to_numpy(dtype=float)
        rows.append({
            "feature_set": model_name,
            "outer_fold": "POOLED",
            **metrics(y, p, 0.5),
        })

        for fold, fg in group.groupby("outer_fold"):
            y_f = fg[TARGET].to_numpy(dtype=int)
            p_f = fg["predicted_probability"].to_numpy(dtype=float)
            rows.append({
                "feature_set": model_name,
                "outer_fold": int(fold),
                **metrics(y_f, p_f, 0.5),
            })

    return pd.DataFrame(rows)


def feature_shift_profile(data, features, focus_fold):
    """
    Compare focus fold with the other 4 folds.

    Measures:
    - standardized mean difference (focus - others)
    - Wasserstein distance after global z-scoring
    - focus and other means/SDs
    """
    focus = data[data["homology_cv_fold"] == focus_fold]
    other = data[data["homology_cv_fold"] != focus_fold]

    rows = []

    for f in features:
        x = focus[f].astype(float).to_numpy()
        y = other[f].astype(float).to_numpy()

        pooled_sd = np.std(data[f].astype(float).to_numpy(), ddof=1)
        if pooled_sd == 0 or not np.isfinite(pooled_sd):
            smd = 0.0
            wd = 0.0
        else:
            smd = (np.mean(x) - np.mean(y)) / pooled_sd
            xz = (x - np.mean(data[f])) / pooled_sd
            yz = (y - np.mean(data[f])) / pooled_sd
            wd = wasserstein_distance(xz, yz)

        rows.append({
            "feature": f,
            "focus_fold": focus_fold,
            "focus_mean": float(np.mean(x)),
            "focus_sd": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
            "other_folds_mean": float(np.mean(y)),
            "other_folds_sd": float(np.std(y, ddof=1)) if len(y) > 1 else np.nan,
            "standardized_mean_difference": float(smd),
            "abs_standardized_mean_difference": float(abs(smd)),
            "wasserstein_z": float(wd),
        })

    return pd.DataFrame(rows).sort_values(
        ["abs_standardized_mean_difference", "wasserstein_z"],
        ascending=False,
    )


def fold_composition(data):
    rows = []

    for fold, g in data.groupby("homology_cv_fold"):
        cluster_sizes = g["homology_cluster_30"].value_counts()
        row = {
            "outer_fold": int(fold),
            "n": int(len(g)),
            "positives": int(g[TARGET].sum()),
            "negatives": int((1 - g[TARGET]).sum()),
            "positive_fraction": float(g[TARGET].mean()),
            "unique_homology_clusters": int(g["homology_cluster_30"].nunique()),
            "largest_cluster_size": int(cluster_sizes.max()),
            "mean_cluster_size": float(cluster_sizes.mean()),
        }

        if "is_tim_barrel" in g.columns:
            row["tim_barrel_n"] = int(g["is_tim_barrel"].astype(bool).sum())
            row["tim_barrel_fraction"] = float(
                g["is_tim_barrel"].astype(bool).mean()
            )

            pos = g[g[TARGET] == 1]
            neg = g[g[TARGET] == 0]
            row["tim_fraction_positive"] = (
                float(pos["is_tim_barrel"].astype(bool).mean())
                if len(pos) else np.nan
            )
            row["tim_fraction_negative"] = (
                float(neg["is_tim_barrel"].astype(bool).mean())
                if len(neg) else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows).sort_values("outer_fold")


def repeated_error_profile(oof_3b):
    """
    Count how often each protein is misclassified across all Stage 09.3B
    feature-set trials at the fixed 0.5 threshold.
    """
    x = oof_3b.copy()
    x["predicted_label"] = (
        x["predicted_probability"] >= 0.5
    ).astype(int)
    x["is_error"] = x["predicted_label"] != x[TARGET]

    agg = (
        x.groupby(
            [IDENTIFIER, "gene_minami", TARGET, "outer_fold"],
            dropna=False,
        )
        .agg(
            models_evaluated=("feature_set", "nunique"),
            error_count=("is_error", "sum"),
            mean_probability=("predicted_probability", "mean"),
            min_probability=("predicted_probability", "min"),
            max_probability=("predicted_probability", "max"),
        )
        .reset_index()
    )
    agg["error_fraction"] = (
        agg["error_count"] / agg["models_evaluated"]
    )
    return agg.sort_values(
        ["error_fraction", "outer_fold"],
        ascending=[False, True],
    )


def fold_difficulty_across_trials(oof_3b):
    """
    For each fold, summarize how it performs across all feature-set trials.
    """
    per_model_fold = pooled_and_fold_metrics(
        oof_3b,
        feature_set_col="feature_set",
    )
    numeric = per_model_fold[
        per_model_fold["outer_fold"] != "POOLED"
    ].copy()
    numeric["outer_fold"] = numeric["outer_fold"].astype(int)

    rows = []
    for fold, g in numeric.groupby("outer_fold"):
        rows.append({
            "outer_fold": int(fold),
            "trials": int(len(g)),
            "mean_mcc": float(g["mcc"].mean()),
            "median_mcc": float(g["mcc"].median()),
            "min_mcc": float(g["mcc"].min()),
            "max_mcc": float(g["mcc"].max()),
            "mean_roc_auc": float(g["roc_auc"].mean()),
            "mean_pr_auc": float(g["pr_auc"].mean()),
            "mean_balanced_accuracy": float(
                g["balanced_accuracy"].mean()
            ),
            "mean_recall": float(g["recall"].mean()),
            "mean_specificity": float(g["specificity"].mean()),
            "times_worst_mcc": 0,
        })

    out = pd.DataFrame(rows)

    for feature_set, g in numeric.groupby("feature_set"):
        worst = g.loc[g["mcc"].idxmin(), "outer_fold"]
        out.loc[
            out["outer_fold"] == int(worst),
            "times_worst_mcc",
        ] += 1

    return out.sort_values("mean_mcc")


def compare_predictions(v2, b3):
    a = v2.rename(
        columns={"predicted_probability": "prob_stage09_v2"}
    )
    b = b3.rename(
        columns={"predicted_probability": "prob_stage09_3B"}
    )

    cols = [
        IDENTIFIER,
        TARGET,
        "outer_fold",
        "prob_stage09_3B",
    ]
    merged = a.merge(
        b[cols],
        on=[IDENTIFIER, TARGET, "outer_fold"],
        how="inner",
        validate="one_to_one",
    )

    merged["probability_delta_3B_minus_v2"] = (
        merged["prob_stage09_3B"] - merged["prob_stage09_v2"]
    )
    merged["abs_probability_delta"] = (
        merged["probability_delta_3B_minus_v2"].abs()
    )
    merged["pred_v2"] = (
        merged["prob_stage09_v2"] >= 0.5
    ).astype(int)
    merged["pred_3B"] = (
        merged["prob_stage09_3B"] >= 0.5
    ).astype(int)
    merged["classification_changed"] = (
        merged["pred_v2"] != merged["pred_3B"]
    )
    merged["v2_correct"] = merged["pred_v2"] == merged[TARGET]
    merged["3B_correct"] = merged["pred_3B"] == merged[TARGET]

    return merged.sort_values(
        "abs_probability_delta",
        ascending=False,
    )


def main():
    args = parse_args()

    feature_dir = Path(args.feature_dir)
    stage09_v2_dir = Path(args.stage09_v2_dir)
    stage09_3b_dir = Path(args.stage09_3b_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    structure = pd.read_csv(feature_dir / STRUCTURE_FILE)
    features = predictor_columns(structure)
    folds = pd.read_csv(args.fold_assignments)

    required_fold_cols = [
        IDENTIFIER,
        "homology_cv_fold",
        "homology_cluster_30",
    ]
    if "is_tim_barrel" in folds.columns:
        required_fold_cols.append("is_tim_barrel")

    data = structure.merge(
        folds[required_fold_cols],
        on=IDENTIFIER,
        how="left",
        validate="one_to_one",
    )

    if len(data) != 181 or int(data[TARGET].sum()) != 73:
        raise RuntimeError(
            "Expected Benchmark v1.1: 181 proteins / 73 positives"
        )

    # ------------------------------------------------------------------
    # 1. Exact Stage 09 v2 reconstruction
    # ------------------------------------------------------------------
    print("Reconstructing Stage 09 v2 Structural Core OOF predictions...")
    v2_oof, v2_hp = reconstruct_v2_oof(
        data,
        features,
        stage09_v2_dir,
    )

    # ------------------------------------------------------------------
    # 2. Stage 09.3B outputs
    # ------------------------------------------------------------------
    print("Reading Stage 09.3B OOF predictions...")
    b3_all = read_3b_oof(stage09_3b_dir)
    b3_struct = b3_all[
        b3_all["feature_set"] == "structural_core"
    ].copy()

    if len(b3_struct) != 181:
        raise RuntimeError(
            f"Expected 181 Structural Core 09.3B predictions; "
            f"found {len(b3_struct)}"
        )

    b3_hp = read_3b_models(stage09_3b_dir)

    # ------------------------------------------------------------------
    # 3. Prediction-level reconciliation
    # ------------------------------------------------------------------
    comparison = compare_predictions(v2_oof, b3_struct)

    v2_metrics = metrics(
        v2_oof[TARGET].to_numpy(dtype=int),
        v2_oof["predicted_probability"].to_numpy(dtype=float),
        0.5,
    )
    b3_metrics = metrics(
        b3_struct[TARGET].to_numpy(dtype=int),
        b3_struct["predicted_probability"].to_numpy(dtype=float),
        0.5,
    )

    reconciliation_summary = {
        "stage09_v2_structural_core": v2_metrics,
        "stage09_3B_structural_core": b3_metrics,
        "delta_3B_minus_v2": {
            k: float(b3_metrics[k] - v2_metrics[k])
            for k in (
                "mcc",
                "balanced_accuracy",
                "roc_auc",
                "pr_auc",
                "recall",
                "specificity",
            )
        },
        "proteins_with_changed_fixed_0_5_classification": int(
            comparison["classification_changed"].sum()
        ),
        "mean_abs_probability_change": float(
            comparison["abs_probability_delta"].mean()
        ),
        "max_abs_probability_change": float(
            comparison["abs_probability_delta"].max()
        ),
        "v2_wrong_to_3B_correct": int(
            ((~comparison["v2_correct"]) & comparison["3B_correct"]).sum()
        ),
        "v2_correct_to_3B_wrong": int(
            (comparison["v2_correct"] & (~comparison["3B_correct"])).sum()
        ),
    }

    # Fold-specific v2 vs 3B.
    recon_fold_rows = []
    for fold in sorted(data["homology_cv_fold"].unique()):
        fold = int(fold)
        a = v2_oof[v2_oof["outer_fold"] == fold]
        b = b3_struct[b3_struct["outer_fold"] == fold]

        ma = metrics(
            a[TARGET].to_numpy(dtype=int),
            a["predicted_probability"].to_numpy(dtype=float),
            0.5,
        )
        mb = metrics(
            b[TARGET].to_numpy(dtype=int),
            b["predicted_probability"].to_numpy(dtype=float),
            0.5,
        )

        recon_fold_rows.append({
            "outer_fold": fold,
            **{f"v2_{k}": v for k, v in ma.items()},
            **{f"3B_{k}": v for k, v in mb.items()},
            "delta_mcc": mb["mcc"] - ma["mcc"],
            "delta_roc_auc": mb["roc_auc"] - ma["roc_auc"],
            "delta_pr_auc": mb["pr_auc"] - ma["pr_auc"],
        })

    recon_fold = pd.DataFrame(recon_fold_rows)

    # Hyperparameter comparison.
    hp_compare = pd.concat(
        [v2_hp, b3_hp],
        ignore_index=True,
        sort=False,
    )

    # ------------------------------------------------------------------
    # 4. Fold difficulty across all 09.3B trials
    # ------------------------------------------------------------------
    print("Profiling fold difficulty across Stage 09.3B trials...")
    fold_difficulty = fold_difficulty_across_trials(b3_all)
    all_trial_fold_metrics = pooled_and_fold_metrics(
        b3_all,
        feature_set_col="feature_set",
    )

    # ------------------------------------------------------------------
    # 5. Fold composition + feature distribution shift
    # ------------------------------------------------------------------
    composition = fold_composition(data)
    feature_shift = feature_shift_profile(
        data,
        features,
        args.focus_fold,
    )

    # ------------------------------------------------------------------
    # 6. Repeatedly difficult proteins across models
    # ------------------------------------------------------------------
    errors = repeated_error_profile(b3_all)
    focus_errors = errors[
        errors["outer_fold"] == args.focus_fold
    ].copy()

    # Join structural metadata for focused error interpretation.
    extra_cols = [
        IDENTIFIER,
        "sccs",
    ]
    for c in ["is_tim_barrel"]:
        if c in data.columns:
            extra_cols.append(c)

    focus_errors = focus_errors.merge(
        data[extra_cols],
        on=IDENTIFIER,
        how="left",
        validate="one_to_one",
    )

    # ------------------------------------------------------------------
    # 7. Write outputs
    # ------------------------------------------------------------------
    v2_oof.to_csv(
        outdir / "stage09_v2_structural_core_oof_reconstructed.csv",
        index=False,
    )
    comparison.to_csv(
        outdir / "v2_vs_3B_structural_core_prediction_comparison.csv",
        index=False,
    )
    recon_fold.to_csv(
        outdir / "v2_vs_3B_fold_metrics.csv",
        index=False,
    )
    hp_compare.to_csv(
        outdir / "v2_vs_3B_hyperparameters.csv",
        index=False,
    )
    all_trial_fold_metrics.to_csv(
        outdir / "all_09_3B_feature_set_fold_metrics.csv",
        index=False,
    )
    fold_difficulty.to_csv(
        outdir / "fold_difficulty_summary.csv",
        index=False,
    )
    composition.to_csv(
        outdir / "fold_composition_summary.csv",
        index=False,
    )
    feature_shift.to_csv(
        outdir / f"fold{args.focus_fold}_feature_shift_vs_others.csv",
        index=False,
    )
    errors.to_csv(
        outdir / "protein_error_frequency_across_09_3B_models.csv",
        index=False,
    )
    focus_errors.to_csv(
        outdir / f"fold{args.focus_fold}_repeated_errors.csv",
        index=False,
    )

    # Human-readable summary.
    focus_diff = fold_difficulty[
        fold_difficulty["outer_fold"] == args.focus_fold
    ].iloc[0].to_dict()
    focus_comp = composition[
        composition["outer_fold"] == args.focus_fold
    ].iloc[0].to_dict()

    top_shift = feature_shift.head(10)[
        [
            "feature",
            "standardized_mean_difference",
            "wasserstein_z",
        ]
    ].to_dict(orient="records")

    hard_focus = focus_errors.head(15)[
        [
            c for c in [
                IDENTIFIER,
                "gene_minami",
                TARGET,
                "models_evaluated",
                "error_count",
                "error_fraction",
                "mean_probability",
                "sccs",
                "is_tim_barrel",
            ]
            if c in focus_errors.columns
        ]
    ].to_dict(orient="records")

    summary = {
        "stage": "09.3C",
        "status": "COMPLETE",
        "audit_only": True,
        "focus_fold": args.focus_fold,
        "reconciliation": reconciliation_summary,
        "focus_fold_difficulty": focus_diff,
        "focus_fold_composition": focus_comp,
        "top_focus_fold_feature_shifts": top_shift,
        "most_repeatedly_misclassified_focus_fold_proteins": hard_focus,
    }

    (outdir / "stage09_3C_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    # Console interpretation aid.
    print("\n=== STRUCTURAL CORE RECONCILIATION ===")
    print(
        f"Stage 09 v2: MCC={v2_metrics['mcc']:.4f}, "
        f"ROC={v2_metrics['roc_auc']:.4f}, "
        f"PR={v2_metrics['pr_auc']:.4f}"
    )
    print(
        f"Stage 09.3B: MCC={b3_metrics['mcc']:.4f}, "
        f"ROC={b3_metrics['roc_auc']:.4f}, "
        f"PR={b3_metrics['pr_auc']:.4f}"
    )
    print(
        "Changed fixed-threshold classifications: "
        f"{reconciliation_summary['proteins_with_changed_fixed_0_5_classification']}"
    )
    print(
        "v2 wrong -> 3B correct: "
        f"{reconciliation_summary['v2_wrong_to_3B_correct']}"
    )
    print(
        "v2 correct -> 3B wrong: "
        f"{reconciliation_summary['v2_correct_to_3B_wrong']}"
    )

    print(f"\n=== FOLD {args.focus_fold} DIFFICULTY ===")
    print(pd.DataFrame([focus_diff]).to_string(index=False))

    print(f"\n=== FOLD {args.focus_fold} COMPOSITION ===")
    print(pd.DataFrame([focus_comp]).to_string(index=False))

    print(f"\n=== TOP FOLD {args.focus_fold} FEATURE SHIFTS ===")
    print(feature_shift.head(10).to_string(index=False))

    print(
        f"\nOutputs written to {outdir}\n"
        "Upload stage09_3C_summary.json, fold_difficulty_summary.csv, "
        "fold_composition_summary.csv, v2_vs_3B_fold_metrics.csv, "
        "v2_vs_3B_hyperparameters.csv, and the fold3 error/shift files."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
