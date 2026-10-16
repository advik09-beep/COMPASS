#!/usr/bin/env python3
"""Feature correlation analysis for Structural + Mutational model."""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ID = "current_uniprot_accession"

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

structural = pd.read_csv(
    ROOT / "data/frozen/feature_set_v3/"
    "feature_set_v3_structure_core_final.csv"
)

frustration = pd.read_csv(
    ROOT / "results/stage09_4F/frustration_features.csv"
).rename(columns={"accession": ID})

d = structural[[ID] + STRUCTURAL].merge(
    frustration[[ID] + MUTATIONAL],
    on=ID,
    validate="one_to_one",
)

assert len(d) == 181
assert d[ID].is_unique
assert not d[STRUCTURAL + MUTATIONAL].isna().any().any()
assert np.isfinite(
    d[STRUCTURAL + MUTATIONAL].to_numpy(dtype=float)
).all()

features = STRUCTURAL + MUTATIONAL

corr = d[features].corr(method="spearman")

output = ROOT / "results/stage09_4F/feature_correlations"
output.mkdir(parents=True, exist_ok=True)

corr.to_csv(output / "spearman_correlation_matrix.csv")

# Mutational versus structural correlations.
cross = []

for mut in MUTATIONAL:
    for structural_feature in STRUCTURAL:
        rho = corr.loc[mut, structural_feature]

        cross.append({
            "mutational_feature": mut,
            "structural_feature": structural_feature,
            "spearman_rho": rho,
            "absolute_rho": abs(rho),
        })

cross = pd.DataFrame(cross).sort_values(
    "absolute_rho",
    ascending=False,
)

cross.to_csv(
    output / "mutational_structural_correlations.csv",
    index=False,
)

# Relationships within mutational frustration.
within = []

for i, first in enumerate(MUTATIONAL):
    for second in MUTATIONAL[i + 1:]:
        rho = corr.loc[first, second]

        within.append({
            "feature_1": first,
            "feature_2": second,
            "spearman_rho": rho,
            "absolute_rho": abs(rho),
        })

within = pd.DataFrame(within).sort_values(
    "absolute_rho",
    ascending=False,
)

within.to_csv(
    output / "within_mutational_correlations.csv",
    index=False,
)

print("\nTOP 20 MUTATIONAL-STRUCTURAL CORRELATIONS")
print("=" * 90)

print(
    cross.head(20).to_string(
        index=False,
        float_format="%.3f",
    )
)

print("\nWITHIN-MUTATIONAL CORRELATIONS")
print("=" * 90)

print(
    within.to_string(
        index=False,
        float_format="%.3f",
    )
)

print("\nMUTATIONAL FEATURES VS MINIMUM CROSS-CUT DENSITY")
print("=" * 90)

target = "minimum_cut_cross_contact_density"

for feature in MUTATIONAL:
    print(
        f"{feature:35s} "
        f"rho={corr.loc[feature, target]:+.3f}"
    )

print("\nSTRONG CROSS-GROUP CORRELATIONS")
print(
    "Absolute rho >= 0.70:",
    int((cross.absolute_rho >= 0.70).sum()),
    "of",
    len(cross),
)

print("\nSaved to:", output)
