#!/usr/bin/env python3
"""Audit reconstruction of Stage 09.4F Structural + Mutational models."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

ROOT = Path(__file__).resolve().parents[1]
ID = "current_uniprot_accession"
MODEL = "structural_plus_mutational"

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

structural = pd.read_csv(
    ROOT / "data/frozen/feature_set_v3/"
    "feature_set_v3_structure_core_final.csv"
)

frustration = pd.read_csv(
    ROOT / "results/stage09_4F/frustration_features.csv"
).rename(columns={"accession": ID})

folds = pd.read_csv(
    ROOT / "data/grouping/stage07/fold_assignments_v1.0.csv"
)

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

selected = pd.read_csv(
    ROOT / "results/stage09_4F/group_ablation/"
    "selected_hyperparameters.csv"
)

original_oof = pd.read_csv(
    ROOT / "results/stage09_4F/group_ablation/"
    "oof_predictions.csv"
)

selected = selected[selected.model == MODEL]
original_oof = original_oof[original_oof.model == MODEL]

assert len(d) == 181
assert d.binary_label.sum() == 73
assert d[ID].is_unique
assert len(selected) == 5
assert selected.outer_fold.is_unique
assert len(original_oof) == 181
assert original_oof.accession.is_unique
assert set(original_oof.accession) == set(d[ID])
assert not d[FEATURES].isna().any().any()
assert np.isfinite(d[FEATURES].to_numpy(dtype=float)).all()

spec = importlib.util.spec_from_file_location(
    "stage09_original",
    ROOT / "Scripts/09_v2_final_random_forest_FIXED.py",
)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)

records = []
fold_records = []

for fold in sorted(d.homology_cv_fold.unique()):
    train = d[d.homology_cv_fold != fold]
    test = d[d.homology_cv_fold == fold]

    overlap = (
        set(train.homology_cluster_30.astype(str))
        & set(test.homology_cluster_30.astype(str))
    )
    assert not overlap, f"Group leakage in fold {fold}"

    row = selected[selected.outer_fold == fold]
    assert len(row) == 1

    params = json.loads(row.iloc[0]["params"])

    model = original.build_rf(
        params,
        seed=42 + int(fold),
        n_jobs=4,
    )

    model.fit(
        train[FEATURES],
        train.binary_label.astype(int),
    )

    probabilities = model.predict_proba(test[FEATURES])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    reconstructed = pd.DataFrame({
        "accession": test[ID].to_numpy(),
        "outer_fold": int(fold),
        "binary_label": test.binary_label.to_numpy(dtype=int),
        "reconstructed_probability": probabilities,
        "reconstructed_prediction": predictions,
    })

    reference = original_oof[
        original_oof.outer_fold == fold
    ][[
        "accession",
        "outer_fold",
        "binary_label",
        "probability",
        "prediction",
    ]]

    comparison = reconstructed.merge(
        reference,
        on=["accession", "outer_fold", "binary_label"],
        validate="one_to_one",
    )

    assert len(comparison) == len(test)

    comparison["absolute_probability_difference"] = (
        comparison.reconstructed_probability
        - comparison.probability
    ).abs()

    comparison["prediction_changed"] = (
        comparison.reconstructed_prediction
        != comparison.prediction
    )

    records.append(comparison)

    original_mcc = matthews_corrcoef(
        comparison.binary_label,
        comparison.prediction,
    )

    reconstructed_mcc = matthews_corrcoef(
        comparison.binary_label,
        comparison.reconstructed_prediction,
    )

    fold_records.append({
        "outer_fold": int(fold),
        "n": len(comparison),
        "original_mcc": original_mcc,
        "reconstructed_mcc": reconstructed_mcc,
        "changed_predictions": int(
            comparison.prediction_changed.sum()
        ),
        "max_probability_difference": float(
            comparison.absolute_probability_difference.max()
        ),
        "mean_probability_difference": float(
            comparison.absolute_probability_difference.mean()
        ),
    })

comparison = pd.concat(records, ignore_index=True)
summary = pd.DataFrame(fold_records)

output = ROOT / "results/stage09_4F/reconstruction_audit"
output.mkdir(parents=True, exist_ok=True)

comparison.to_csv(
    output / "protein_level_comparison.csv",
    index=False,
)

summary.to_csv(
    output / "fold_summary.csv",
    index=False,
)

print("\nMODEL RECONSTRUCTION AUDIT")
print("=" * 85)
print(summary.to_string(index=False, float_format="%.8f"))

print("\nOVERALL")
print("Proteins:", len(comparison))
print(
    "Changed classifications:",
    int(comparison.prediction_changed.sum()),
)
print(
    "Maximum probability difference:",
    comparison.absolute_probability_difference.max(),
)
print(
    "Mean probability difference:",
    comparison.absolute_probability_difference.mean(),
)

print(
    "Original pooled MCC:",
    matthews_corrcoef(
        comparison.binary_label,
        comparison.prediction,
    ),
)

print(
    "Reconstructed pooled MCC:",
    matthews_corrcoef(
        comparison.binary_label,
        comparison.reconstructed_prediction,
    ),
)

print("\nLargest probability differences:")
print(
    comparison.nlargest(
        10, "absolute_probability_difference"
    )[[
        "accession",
        "outer_fold",
        "binary_label",
        "probability",
        "reconstructed_probability",
        "absolute_probability_difference",
        "prediction_changed",
    ]].to_string(index=False)
)

print("\nSaved to:", output)
