#!/usr/bin/env python3
"""
Stage 09.3B — Sequence-family add-back + structural ablation experiment

Purpose
-------
Explain why Sequence + Structural Core underperformed Structural Core in Stage 09 v2.

This stage starts from the frozen Structural Core feature set and tests
predefined, biologically coherent sequence families rather than arbitrary
feature combinations. It also tests the specific structural concern identified
in Stage 09.3A: asphericity / global shape descriptors.

Design
------
Primary analysis:
- Benchmark v1.1
- Full 181-protein benchmark
- Frozen 30% homology-grouped folds
- Random Forest
- Nested grouped CV
- Same Stage 09 v2 RF search space
- Inner objective: MCC
- Fixed 0.5 threshold for primary model comparison
- Optional inner-selected threshold retained as secondary output

Candidate feature sets:
1. Structural Core
2. Structural Core - asphericity
3. Structural Core - shape family
4. Structural Core + amino-acid composition
5. Structural Core + charge/electrostatics
6. Structural Core + hydrophobicity
7. Structural Core + sequence complexity
8. Structural Core + other sequence physicochemical features
9. Structural Core + all sequence features (Combined Core control)

IMPORTANT:
The script does NOT automatically build arbitrary combinations of "winning"
families. Those should be chosen only after this predefined screen is reviewed.

Example
-------
python3 scripts/09_3B_sequence_family_addback.py \
  --feature-dir data/frozen/feature_set_v3 \
  --fold-assignments data/grouping/stage07/fold_assignments_v1.0.csv \
  --output-dir results/stage09_3B
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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
from sklearn.model_selection import ParameterSampler

IDENTIFIER = "current_uniprot_accession"
TARGET = "binary_label"

STRUCTURE_FILE = "feature_set_v3_structure_core_final.csv"
COMBINED_FILE = "feature_set_v3_sequence_plus_structure_core.csv"

METADATA_COLUMNS = {
    "current_uniprot_accession","gene_minami","jwid","binary_label","cohort_label",
    "label_source","label_evidence_type","label_confidence","source_class_summary",
    "label_notes","sccs","benchmark_version","previous_binary_label",
    "label_updated_from_minami_validation","minami_2023_table1_groe_dependency",
    "minami_2023_validation_status","minami_2023_validation_source",
}

# Predefined biological families. Classification is intentionally explicit
# and exported for audit before modeling.
AA_FRACTION_PREFIX = "seq_fraction_"

CHARGE_EXACT = {
    "seq_net_charge_proxy_ph7",
    "seq_charge_density_proxy_ph7",
    "seq_positive_fraction",
    "seq_negative_fraction",
    "seq_basic_fraction",
    "seq_acidic_fraction",
    "seq_fraction_K",
    "seq_fraction_R",
    "seq_fraction_H",
    "seq_fraction_D",
    "seq_fraction_E",
}

HYDROPHOBIC_KEYWORDS = (
    "hydro",
    "kd_",
    "gravy",
)

COMPLEXITY_KEYWORDS = (
    "entropy",
    "complex",
)

# Structural "shape family" is deliberately narrow: anisotropy / global shape.
# Radius of gyration is NOT removed here because 09.3A showed it was useful.
SHAPE_FEATURES = {
    "asphericity",
    "axis_ratio_1_to_2",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--feature-dir", required=True)
    p.add_argument("--fold-assignments", required=True)
    p.add_argument("--output-dir", default="results/stage09_3B")
    p.add_argument("--random-seed", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--n-candidates", type=int, default=80)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--bootstrap-iterations", type=int, default=5000)
    return p.parse_args()


def predictor_columns(df):
    return [
        c for c in df.columns
        if c not in METADATA_COLUMNS
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].nunique(dropna=True) > 1
    ]


def classify_sequence_features(sequence_features):
    families = {
        "amino_acid_composition": [],
        "charge_electrostatics": [],
        "hydrophobicity": [],
        "sequence_complexity": [],
        "other_sequence_physicochemical": [],
    }

    for f in sequence_features:
        # Charge-specific residue fractions are assigned to charge rather than
        # general composition so families are non-overlapping.
        if f in CHARGE_EXACT or any(k in f.lower() for k in ("charge", "acidic", "basic")):
            families["charge_electrostatics"].append(f)
        elif any(k in f.lower() for k in HYDROPHOBIC_KEYWORDS):
            families["hydrophobicity"].append(f)
        elif any(k in f.lower() for k in COMPLEXITY_KEYWORDS):
            families["sequence_complexity"].append(f)
        elif f.startswith(AA_FRACTION_PREFIX):
            families["amino_acid_composition"].append(f)
        else:
            families["other_sequence_physicochemical"].append(f)

    return families


def metrics(y, prob, threshold=0.5):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "mcc": float(matthews_corrcoef(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
    }


def make_rf(params, seed, n_jobs):
    return RandomForestClassifier(
        random_state=seed,
        n_jobs=n_jobs,
        criterion="gini",
        **params,
    )


def search_space():
    # Same general restrained RF space used for Stage 09-style nested tuning.
    return {
        "n_estimators": [300, 500, 800, 1200],
        "max_depth": [None, 4, 6, 8, 12],
        "min_samples_split": [2, 4, 6, 10],
        "min_samples_leaf": [1, 2, 3, 5],
        "max_features": ["sqrt", "log2", 0.5, 0.75, None],
        "class_weight": [None, "balanced", "balanced_subsample"],
        "bootstrap": [True],
    }


def stratified_group_inner_folds(y, groups, n_splits, seed):
    from sklearn.model_selection import StratifiedGroupKFold
    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    return list(cv.split(np.zeros(len(y)), y, groups))


def choose_threshold(y, prob):
    # Inner-OOF threshold only. Candidate grid is deterministic.
    thresholds = np.linspace(0.20, 0.80, 121)
    scored = []
    for t in thresholds:
        scored.append((matthews_corrcoef(y, (prob >= t).astype(int)), t))
    scored.sort(key=lambda x: (x[0], -abs(x[1] - 0.5)), reverse=True)
    return float(scored[0][1])


def tune_on_outer_train(
    X,
    y,
    groups,
    candidate_params,
    inner_folds,
    seed,
    n_jobs,
):
    splits = stratified_group_inner_folds(
        y, groups, inner_folds, seed
    )
    rows = []

    for candidate_id, params in enumerate(candidate_params):
        oof = np.full(len(y), np.nan)

        for inner_fold, (tr, va) in enumerate(splits):
            model = make_rf(
                params,
                seed + candidate_id * 101 + inner_fold,
                n_jobs,
            )
            model.fit(X.iloc[tr], y[tr])
            oof[va] = model.predict_proba(X.iloc[va])[:, 1]

        if np.isnan(oof).any():
            raise RuntimeError("Incomplete inner OOF predictions")

        m = metrics(y, oof, 0.5)
        rows.append({
            "candidate_id": candidate_id,
            **params,
            "inner_mcc": m["mcc"],
            "inner_roc_auc": m["roc_auc"],
            "inner_pr_auc": m["pr_auc"],
        })

    results = pd.DataFrame(rows).sort_values(
        ["inner_mcc", "inner_roc_auc", "inner_pr_auc"],
        ascending=False,
    )

    best_id = int(results.iloc[0]["candidate_id"])
    best_params = candidate_params[best_id]

    # Reconstruct inner OOF for selected candidate to select threshold.
    selected_oof = np.full(len(y), np.nan)
    for inner_fold, (tr, va) in enumerate(splits):
        model = make_rf(
            best_params,
            seed + best_id * 101 + inner_fold,
            n_jobs,
        )
        model.fit(X.iloc[tr], y[tr])
        selected_oof[va] = model.predict_proba(X.iloc[va])[:, 1]

    threshold = choose_threshold(y, selected_oof)
    return best_params, threshold, results


def paired_bootstrap(y, base_prob, alt_prob, iterations, seed):
    rng = np.random.default_rng(seed)
    rows = []
    n = len(y)

    for metric_name in ("mcc", "roc_auc", "pr_auc"):
        deltas = []
        for _ in range(iterations):
            idx = rng.integers(0, n, n)
            yt = y[idx]
            if len(np.unique(yt)) < 2:
                continue

            b = metrics(yt, base_prob[idx], 0.5)[metric_name]
            a = metrics(yt, alt_prob[idx], 0.5)[metric_name]
            deltas.append(a - b)

        low, high = np.percentile(deltas, [2.5, 97.5])
        rows.append({
            "metric": metric_name,
            "mean_delta": float(np.mean(deltas)),
            "ci_low": float(low),
            "ci_high": float(high),
        })

    return rows


def main():
    args = parse_args()
    start = time.time()

    feature_dir = Path(args.feature_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "saved_models"
    model_dir.mkdir(exist_ok=True)

    structure = pd.read_csv(feature_dir / STRUCTURE_FILE)
    combined = pd.read_csv(feature_dir / COMBINED_FILE)
    folds = pd.read_csv(args.fold_assignments)

    if len(structure) != 181 or int(structure[TARGET].sum()) != 73:
        raise RuntimeError("Expected Benchmark v1.1: 181 proteins / 73 positives")

    structure_features = predictor_columns(structure)
    combined_features = predictor_columns(combined)
    sequence_features = [f for f in combined_features if f not in structure_features]

    families = classify_sequence_features(sequence_features)

    # Hard audit: every sequence feature must belong to exactly one family.
    assigned = [f for vals in families.values() for f in vals]
    if sorted(assigned) != sorted(sequence_features):
        raise RuntimeError("Sequence-family classification is incomplete or overlapping")

    feature_sets = {
        "structural_core": structure_features,
        "structural_core_minus_asphericity": [
            f for f in structure_features if f != "asphericity"
        ],
        "structural_core_minus_shape_family": [
            f for f in structure_features if f not in SHAPE_FEATURES
        ],
    }

    for family_name, family_features in families.items():
        feature_sets[f"structural_core_plus_{family_name}"] = (
            structure_features + family_features
        )

    feature_sets["combined_core_control"] = combined_features

    # Merge all required columns from combined + frozen folds.
    data = combined.merge(
        folds[
            [
                IDENTIFIER,
                "homology_cv_fold",
                "homology_cluster_30",
            ]
        ],
        on=IDENTIFIER,
        how="left",
        validate="one_to_one",
    )

    if data["homology_cv_fold"].isna().any():
        raise RuntimeError("Missing frozen fold assignments")

    # Export family manifest BEFORE modeling.
    manifest_rows = []
    for feature in structure_features:
        manifest_rows.append({
            "feature": feature,
            "origin": "structure",
            "family": (
                "global_shape"
                if feature in SHAPE_FEATURES
                else "structural_core_other"
            ),
        })
    for family_name, vals in families.items():
        for feature in vals:
            manifest_rows.append({
                "feature": feature,
                "origin": "sequence",
                "family": family_name,
            })

    pd.DataFrame(manifest_rows).to_csv(
        output_dir / "feature_family_manifest.csv",
        index=False,
    )

    feature_set_rows = []
    for name, feats in feature_sets.items():
        for f in feats:
            feature_set_rows.append({
                "feature_set": name,
                "feature": f,
                "origin": "structure" if f in structure_features else "sequence",
            })
    pd.DataFrame(feature_set_rows).to_csv(
        output_dir / "candidate_feature_sets.csv",
        index=False,
    )

    candidate_params = list(ParameterSampler(
        search_space(),
        n_iter=args.n_candidates,
        random_state=args.random_seed,
    ))

    oof_rows = []
    fold_rows = []
    tuning_rows = []

    for set_index, (set_name, features) in enumerate(feature_sets.items()):
        print(f"\n=== {set_name} ({len(features)} features) ===")
        set_start = time.time()

        for outer_fold in sorted(data["homology_cv_fold"].unique()):
            outer_fold = int(outer_fold)
            train = data[data["homology_cv_fold"] != outer_fold].copy()
            test = data[data["homology_cv_fold"] == outer_fold].copy()

            overlap = (
                set(train["homology_cluster_30"].astype(str))
                & set(test["homology_cluster_30"].astype(str))
            )
            if overlap:
                raise RuntimeError(
                    f"Homology leakage: {set_name}, fold {outer_fold}"
                )

            X_train = train[features]
            X_test = test[features]
            y_train = train[TARGET].to_numpy(dtype=int)
            y_test = test[TARGET].to_numpy(dtype=int)
            groups_train = train["homology_cluster_30"].astype(str).to_numpy()

            best_params, selected_threshold, tuning = tune_on_outer_train(
                X_train,
                y_train,
                groups_train,
                candidate_params,
                args.inner_folds,
                args.random_seed + set_index * 1000 + outer_fold,
                args.n_jobs,
            )

            tuning["feature_set"] = set_name
            tuning["outer_fold"] = outer_fold
            tuning_rows.append(tuning)

            model = make_rf(
                best_params,
                args.random_seed + set_index * 1000 + outer_fold,
                args.n_jobs,
            )
            model.fit(X_train, y_train)
            prob = model.predict_proba(X_test)[:, 1]

            fixed = metrics(y_test, prob, 0.5)
            selected = metrics(y_test, prob, selected_threshold)

            fold_rows.append({
                "feature_set": set_name,
                "outer_fold": outer_fold,
                "feature_count": len(features),
                "train_n": len(train),
                "test_n": len(test),
                "selected_threshold": selected_threshold,
                **{f"fixed_{k}": v for k, v in fixed.items()},
                **{f"selected_{k}": v for k, v in selected.items()},
                "selected_hyperparameters": json.dumps(best_params),
            })

            for i, (_, row) in enumerate(test.iterrows()):
                oof_rows.append({
                    IDENTIFIER: row[IDENTIFIER],
                    "gene_minami": row.get("gene_minami", ""),
                    TARGET: int(row[TARGET]),
                    "feature_set": set_name,
                    "outer_fold": outer_fold,
                    "predicted_probability": float(prob[i]),
                    "selected_threshold": selected_threshold,
                })

            joblib.dump(
                {
                    "stage": "09.3B",
                    "feature_set": set_name,
                    "outer_fold": outer_fold,
                    "features": features,
                    "selected_hyperparameters": best_params,
                    "selected_threshold": selected_threshold,
                    "model": model,
                },
                model_dir / f"{set_name}__fold{outer_fold}.joblib",
            )

            print(
                f"fold {outer_fold}: MCC={fixed['mcc']:.3f}, "
                f"ROC={fixed['roc_auc']:.3f}, PR={fixed['pr_auc']:.3f}"
            )

        print(
            f"{set_name} complete in {(time.time()-set_start)/60:.1f} min"
        )

    oof = pd.DataFrame(oof_rows)
    folds_out = pd.DataFrame(fold_rows)
    tuning_out = pd.concat(tuning_rows, ignore_index=True)

    summary_rows = []
    for set_name, group in oof.groupby("feature_set"):
        group = group.sort_values(IDENTIFIER)
        y = group[TARGET].to_numpy(dtype=int)
        prob = group["predicted_probability"].to_numpy(dtype=float)
        fixed = metrics(y, prob, 0.5)

        # Each protein uses its fold's inner-selected threshold.
        pred_selected = (
            prob >= group["selected_threshold"].to_numpy(dtype=float)
        ).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred_selected, labels=[0,1]).ravel()
        selected = {
            "mcc": float(matthews_corrcoef(y, pred_selected)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred_selected)),
            "precision": float(precision_score(y, pred_selected, zero_division=0)),
            "recall": float(recall_score(y, pred_selected, zero_division=0)),
            "specificity": float(tn/(tn+fp)) if (tn+fp) else np.nan,
            "roc_auc": fixed["roc_auc"],
            "pr_auc": fixed["pr_auc"],
        }

        summary_rows.append({
            "feature_set": set_name,
            "feature_count": len(feature_sets[set_name]),
            **{f"fixed_{k}": v for k, v in fixed.items()},
            **{f"selected_{k}": v for k, v in selected.items()},
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(
        "fixed_mcc", ascending=False
    )

    # Paired bootstrap versus Structural Core.
    base = oof[oof["feature_set"] == "structural_core"].sort_values(IDENTIFIER)
    y = base[TARGET].to_numpy(dtype=int)
    base_prob = base["predicted_probability"].to_numpy(dtype=float)

    bootstrap_rows = []
    for set_name in feature_sets:
        if set_name == "structural_core":
            continue
        alt = oof[oof["feature_set"] == set_name].sort_values(IDENTIFIER)
        if list(base[IDENTIFIER]) != list(alt[IDENTIFIER]):
            raise RuntimeError(f"OOF protein mismatch for {set_name}")
        alt_prob = alt["predicted_probability"].to_numpy(dtype=float)

        for row in paired_bootstrap(
            y,
            base_prob,
            alt_prob,
            args.bootstrap_iterations,
            args.random_seed + sum(map(ord, set_name)),
        ):
            bootstrap_rows.append({
                "comparison": f"{set_name} minus structural_core",
                "candidate_feature_set": set_name,
                **row,
            })

    bootstrap_df = pd.DataFrame(bootstrap_rows)

    oof.to_csv(output_dir / "stage09_3B_oof_predictions.csv", index=False)
    folds_out.to_csv(output_dir / "stage09_3B_fold_metrics.csv", index=False)
    summary_df.to_csv(output_dir / "stage09_3B_summary_metrics.csv", index=False)
    tuning_out.to_csv(output_dir / "stage09_3B_inner_search_results.csv", index=False)
    bootstrap_df.to_csv(
        output_dir / "stage09_3B_paired_bootstrap_vs_structural_core.csv",
        index=False,
    )

    summary = {
        "stage": "09.3B",
        "status": "COMPLETE",
        "benchmark_version": "1.1",
        "input_feature_set_version": "3.0",
        "primary_grouping": "30% homology clusters",
        "purpose": (
            "Determine which predefined sequence feature families add value "
            "to Structural Core and test Stage 09.3A structural ablation findings."
        ),
        "sequence_families": families,
        "shape_features_tested": sorted(SHAPE_FEATURES),
        "candidate_feature_sets": {
            k: len(v) for k, v in feature_sets.items()
        },
        "best_fixed_mcc_feature_set": str(summary_df.iloc[0]["feature_set"]),
        "best_fixed_mcc": float(summary_df.iloc[0]["fixed_mcc"]),
        "runtime_minutes": (time.time() - start) / 60,
    }

    (output_dir / "stage09_3B_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print("\n=== FINAL PRIMARY RANKING ===")
    print(
        summary_df[
            [
                "feature_set",
                "feature_count",
                "fixed_mcc",
                "fixed_roc_auc",
                "fixed_pr_auc",
            ]
        ].to_string(index=False)
    )
    print(f"\nOutputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
