#!/usr/bin/env python3
"""Held-out permutation importance for Stage 09.4F group ablations."""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import matthews_corrcoef

STRUCTURAL = [
    "structure_ca_length",
    "relative_contact_order",
    "median_contact_sequence_separation",
    "long_range_contact_fraction",
    "medium_range_contact_fraction",
    "ca_contact_density_per_residue",
    "ca_contact_pair_fraction",
    "zero_contact_residue_fraction",
    "radius_of_gyration_normalized_n_cuberoot",
    "axis_ratio_1_to_2",
    "asphericity",
    "ss_helix_fraction",
    "ss_sheet_fraction",
    "minimum_cut_cross_contact_density",
    "best_cut_fraction",
]

MUTATIONAL = [
    "mut_mean",
    "mut_std",
    "mut_high_fraction",
    "mut_minimal_fraction",
    "mut_high_residue_fraction",
    "mut_long_range_high_fraction",
]

FEATURES = STRUCTURAL + MUTATIONAL
MODEL_NAME = "structural_plus_mutational"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    structural = pd.read_csv(
        root / "data/frozen/feature_set_v3/"
        "feature_set_v3_structure_core_final.csv"
    )

    frustration = pd.read_csv(
        root / "results/stage09_4F/frustration_features.csv"
    ).rename(columns={"accession": "current_uniprot_accession"})

    folds = pd.read_csv(
        root / "data/grouping/stage07/fold_assignments_v1.0.csv"
    )

    selected = pd.read_csv(
        root / "results/stage09_4F/group_ablation/"
        "selected_hyperparameters.csv"
    )

    original_path = (
        root / "Scripts/09_v2_final_random_forest_FIXED.py"
    )

    spec = importlib.util.spec_from_file_location(
        "original_stage09", original_path
    )
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)

    ID = "current_uniprot_accession"

    d = (
        structural[[ID, "binary_label"] + STRUCTURAL]
        .merge(
            frustration[[ID] + MUTATIONAL],
            on=ID,
            validate="one_to_one",
        )
        .merge(
            folds[[ID, "homology_cv_fold", "homology_cluster_30"]],
            on=ID,
            validate="one_to_one",
        )
        .reset_index(drop=True)
    )

    assert len(d) == 181
    assert int(d.binary_label.sum()) == 73
    assert not d[FEATURES].isna().any().any()
    assert np.isfinite(d[FEATURES].to_numpy(dtype=float)).all()

    chosen = selected[selected.model == MODEL_NAME].copy()

    assert len(chosen) == 5
    assert chosen.outer_fold.nunique() == 5

    output = root / "results/stage09_4F/permutation_importance"
    output.mkdir(parents=True, exist_ok=True)

    results = []

    for fold in sorted(d.homology_cv_fold.unique()):
        train = d[d.homology_cv_fold != fold]
        test = d[d.homology_cv_fold == fold]

        overlap = (
            set(train.homology_cluster_30.astype(str))
            & set(test.homology_cluster_30.astype(str))
        )
        assert not overlap, f"Group leakage in fold {fold}"

        row = chosen[chosen.outer_fold == fold]
        assert len(row) == 1

        params = json.loads(row.iloc[0]["params"])

        model = original.build_rf(
            params,
            seed=42 + int(fold),
            n_jobs=args.n_jobs,
        )

        model.fit(
            train[FEATURES],
            train.binary_label.astype(int),
        )

        x_test = test[FEATURES]
        y_test = test.binary_label.astype(int)

        baseline_probability = model.predict_proba(x_test)[:, 1]
        baseline_prediction = (baseline_probability >= 0.5).astype(int)
        baseline_mcc = matthews_corrcoef(
            y_test, baseline_prediction
        )

        def fixed_threshold_mcc(estimator, X, y):
            probability = estimator.predict_proba(X)[:, 1]
            prediction = (probability >= 0.5).astype(int)
            return matthews_corrcoef(y, prediction)

        importance = permutation_importance(
            model,
            x_test,
            y_test,
            scoring=fixed_threshold_mcc,
            n_repeats=args.repeats,
            random_state=42 + int(fold),
            n_jobs=args.n_jobs,
        )

        for i, feature in enumerate(FEATURES):
            for repeat, value in enumerate(
                importance.importances[i]
            ):
                results.append({
                    "outer_fold": int(fold),
                    "feature": feature,
                    "feature_group": (
                        "mutational"
                        if feature in MUTATIONAL
                        else "structural"
                    ),
                    "repeat": repeat,
                    "baseline_mcc": baseline_mcc,
                    "mcc_decrease": float(value),
                })

        print(
            f"Fold {fold}: baseline MCC={baseline_mcc:.4f}",
            flush=True,
        )

    raw = pd.DataFrame(results)
    raw.to_csv(output / "permutation_importance_raw.csv", index=False)

    # Average repeats within folds, then average across folds.
    per_fold = (
        raw.groupby(
            ["outer_fold", "feature", "feature_group"],
            as_index=False,
        )
        .agg(
            mean_mcc_decrease=("mcc_decrease", "mean"),
            std_mcc_decrease=("mcc_decrease", "std"),
        )
    )

    per_fold.to_csv(
        output / "permutation_importance_by_fold.csv",
        index=False,
    )

    summary = (
        per_fold.groupby(
            ["feature", "feature_group"],
            as_index=False,
        )
        .agg(
            mean_mcc_decrease=("mean_mcc_decrease", "mean"),
            std_across_folds=("mean_mcc_decrease", "std"),
            folds_positive=("mean_mcc_decrease", lambda x: int((x > 0).sum())),
        )
        .sort_values("mean_mcc_decrease", ascending=False)
    )

    summary.to_csv(
        output / "permutation_importance_summary.csv",
        index=False,
    )

    print("\nPERMUTATION IMPORTANCE — ALL FEATURES")
    print(summary.to_string(index=False, float_format="%.4f"))

    print("\nMUTATIONAL FEATURES ONLY")
    print(
        summary[summary.feature_group == "mutational"]
        .to_string(index=False, float_format="%.4f")
    )

    print("\nResults saved to:", output)


if __name__ == "__main__":
    main()
