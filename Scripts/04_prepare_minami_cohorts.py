#!/usr/bin/env python3
"""
Prepare the Minami et al. Table S1 benchmark cohorts.

IMPORTANT:
- Positive (P) and Negative (N) form the primary binary benchmark.
- Neutral (-) proteins are NOT treated as negatives.
- Neutral proteins are exported separately for later exploratory/holdout analyses.

Usage:
    python3 04_prepare_minami_cohorts.py \
      --supplement FEB4-13-779-s002.xlsx \
      --existing-features data/features/class_iv_structural_features.csv \
      --output-dir data/cohorts
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--supplement", required=True)
    p.add_argument("--existing-features", required=True)
    p.add_argument("--output-dir", default="data/cohorts")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_excel(args.supplement, sheet_name="Table S1", header=1)
    raw = raw[raw["JWID"].astype(str).str.startswith("JW", na=False)].copy()

    label_map = {"P": "positive", "N": "negative", "-": "neutral"}
    raw["cohort_label"] = (
        raw["Positive/Negative/Neutral"].astype(str).str.strip().map(label_map)
    )
    raw = raw[raw["cohort_label"].notna()].copy()
    raw["binary_label"] = raw["cohort_label"].map(
        {"positive": 1, "negative": 0}
    )

    features = pd.read_csv(args.existing_features)
    accession_column = (
        "accession" if "accession" in features.columns else "uniprot_accession"
    )
    existing = set(features[accession_column].dropna().astype(str).str.strip())
    raw["already_has_structural_features"] = (
        raw["UniProt AC"].astype(str).str.strip().isin(existing)
    )

    keep = [
        "JWID", "UniProt AC", "GeneName", "Synonyms", "length",
        "Sol", "Sol(GroE)", "DSol", "Class(Kerner)",
        "Class(Fujiwara, Niwa)", "Positive/Negative/Neutral",
        "cohort_label", "binary_label", "Structure",
        "Relative Contact Order", "SCCS",
        "RW", "RWRR", "already_has_structural_features",
    ]
    clean = raw[keep].copy()

    clean.to_csv(output_dir / "minami_table_s1_clean.csv", index=False)
    clean[clean["cohort_label"].isin(["positive", "negative"])].to_csv(
        output_dir / "minami_binary_benchmark.csv", index=False
    )
    clean[clean["cohort_label"] == "neutral"].to_csv(
        output_dir / "minami_neutral_holdout.csv", index=False
    )
    clean[~clean["already_has_structural_features"]].to_csv(
        output_dir / "minami_missing_structural_features.csv", index=False
    )

    print("Cohort counts:")
    print(clean["cohort_label"].value_counts().to_string())
    print()
    print("Primary binary benchmark:",
          len(clean[clean["cohort_label"].isin(["positive", "negative"])]))
    print("Already featured:",
          int(clean["already_has_structural_features"].sum()))
    print("Missing features:",
          int((~clean["already_has_structural_features"]).sum()))


if __name__ == "__main__":
    main()
