#!/usr/bin/env python3
"""
Merge the original Class IV feature table with the 134 newly extracted Minami
features and attach Minami benchmark labels.

Expected benchmark:
- 64 positives
- 117 usable negatives
- 181 total
- 2 documented exclusions: P76000/ycgI and JW5457/pbl

Usage:
    python3 05_merge_minami_features.py \
      --class-iv-features data/features/class_iv_structural_features.csv \
      --new-features data/minami_binary/features/minami_missing_structural_features.csv \
      --benchmark minami_binary_benchmark.csv \
      --output data/modeling/minami_binary_benchmark_features_181.csv \
      --audit data/modeling/minami_binary_benchmark_merge_audit.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


LEGACY_TO_CURRENT = {"Q8RNI8": "Q47155"}


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--class-iv-features", required=True)
    p.add_argument("--new-features", required=True)
    p.add_argument("--benchmark", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--audit", required=True)
    args = p.parse_args()

    old_rows = read_csv(Path(args.class_iv_features))
    new_rows = read_csv(Path(args.new_features))
    benchmark_rows = read_csv(Path(args.benchmark))

    feature_by_accession = {}
    for row in old_rows + new_rows:
        accession = (row.get("accession") or row.get("uniprot_accession") or "").strip()
        if accession:
            feature_by_accession[accession] = row

    merged_rows = []
    audit_rows = []

    for bench in benchmark_rows:
        legacy_accession = bench["UniProt AC"].strip()
        current_accession = LEGACY_TO_CURRENT.get(legacy_accession, legacy_accession)
        feature = feature_by_accession.get(current_accession)

        audit = {
            "jwid": bench["JWID"],
            "gene": bench["GeneName"],
            "minami_accession": legacy_accession,
            "current_accession": current_accession,
            "cohort_label": bench["cohort_label"],
            "feature_status": "found" if feature else "missing",
            "exclusion_reason": "",
        }

        if feature is None:
            if legacy_accession == "P76000":
                audit["exclusion_reason"] = (
                    "Historical 73-aa homology model unavailable; current sequence differs"
                )
            elif legacy_accession == "-":
                audit["exclusion_reason"] = (
                    "No UniProt accession and original homology model unavailable"
                )
            else:
                audit["exclusion_reason"] = "No extracted structural features found"
            audit_rows.append(audit)
            continue

        merged = dict(feature)
        merged.update({
            "jwid": bench["JWID"],
            "minami_uniprot_accession": legacy_accession,
            "current_uniprot_accession": current_accession,
            "gene_minami": bench["GeneName"],
            "cohort_label": bench["cohort_label"],
            "binary_label": bench["binary_label"],
            "minami_length": bench["length"],
            "minami_structure_source": bench["Structure"],
            "minami_relative_contact_order": bench["Relative Contact Order"],
            "sccs": bench["SCCS"],
            "minami_rw_prediction": bench["RW"],
            "minami_rwrr_prediction": bench["RWRR"],
        })
        merged_rows.append(merged)
        audit_rows.append(audit)

    output_path = Path(args.output)
    audit_path = Path(args.audit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    metadata_headers = [
        "jwid", "minami_uniprot_accession", "current_uniprot_accession",
        "gene_minami", "cohort_label", "binary_label", "minami_length",
        "minami_structure_source", "minami_relative_contact_order", "sccs",
        "minami_rw_prediction", "minami_rwrr_prediction",
    ]
    feature_headers = list(old_rows[0].keys())
    headers = metadata_headers + [h for h in feature_headers if h not in metadata_headers]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged_rows)

    audit_headers = [
        "jwid", "gene", "minami_accession", "current_accession",
        "cohort_label", "feature_status", "exclusion_reason",
    ]
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=audit_headers)
        writer.writeheader()
        writer.writerows(audit_rows)

    positives = sum(r["cohort_label"] == "positive" for r in merged_rows)
    negatives = sum(r["cohort_label"] == "negative" for r in merged_rows)
    missing = sum(r["feature_status"] == "missing" for r in audit_rows)

    print(f"Positive rows: {positives}")
    print(f"Negative rows: {negatives}")
    print(f"Total usable:  {len(merged_rows)}")
    print(f"Excluded:      {missing}")


if __name__ == "__main__":
    main()
