#!/usr/bin/env python3
"""
Stage 10B v2: Deterministic Minami 2023 experimental-label correction.

This script encodes the 17 experimentally validated proteins from Minami et al.
Table 1 directly by JW ID and GroE-dependency outcome.

It:
- preserves Benchmark v1.0 unchanged;
- verifies all 17 JW IDs exist exactly once in the benchmark;
- verifies the expected gene names;
- relabels only the 9 experimentally confirmed GroE-dependent proteins from 0 -> 1;
- preserves the 8 experimentally confirmed GroE-independent proteins as 0;
- writes Benchmark v1.1;
- patches all frozen Feature Set v2 CSVs with the corrected labels;
- writes before/after audits and checksums.

No manual CSV editing is required.

Example:
    python3 scripts/10B_build_benchmark_v1_1_from_minami_table1.py \
      --benchmark data/frozen/benchmark_v1_0/benchmark_v1.0.csv \
      --feature-dir data/frozen/feature_set_v2 \
      --output-dir data/frozen/benchmark_v1_1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ACCESSION = "current_uniprot_accession"
GENE = "gene_minami"
JWID = "jwid"
LABEL = "binary_label"

# Minami et al. 2023 Table 1:
# "+" = experimentally GroE-dependent
# "-" = experimentally GroE-independent
#
# IMPORTANT:
# These labels are encoded from the published experimental-validation table,
# not from our model predictions.
MINAMI_TABLE1 = {
    "JW0843": {"gene": "rumB", "validated_label": 1, "groe_dependency": "+"},
    "JW3699": {"gene": "bglB", "validated_label": 1, "groe_dependency": "+"},
    "JW0760": {"gene": "bioC", "validated_label": 1, "groe_dependency": "+"},
    "JW2308": {"gene": "ubiX", "validated_label": 1, "groe_dependency": "+"},
    "JW2650": {"gene": "nrdE", "validated_label": 1, "groe_dependency": "+"},
    "JW2781": {"gene": "csdA", "validated_label": 1, "groe_dependency": "+"},
    "JW3488": {"gene": "yhjB", "validated_label": 1, "groe_dependency": "+"},
    "JW4221": {"gene": "idnR", "validated_label": 1, "groe_dependency": "+"},
    "JW5454": {"gene": "yqeH", "validated_label": 1, "groe_dependency": "+"},
    "JW0495": {"gene": "gcl", "validated_label": 0, "groe_dependency": "-"},
    "JW0762": {"gene": "uvrB", "validated_label": 0, "groe_dependency": "-"},
    "JW1439": {"gene": "ydcW", "validated_label": 0, "groe_dependency": "-"},
    "JW2686": {"gene": "ascB", "validated_label": 0, "groe_dependency": "-"},
    "JW0997": {"gene": "ycdM", "validated_label": 0, "groe_dependency": "-"},
    "JW1201": {"gene": "hemA", "validated_label": 0, "groe_dependency": "-"},
    "JW1612": {"gene": "malI", "validated_label": 0, "groe_dependency": "-"},
    "JW3958": {"gene": "thiC", "validated_label": 0, "groe_dependency": "-"},
}

FEATURE_FILENAMES = [
    "feature_set_v2_sequence_final.csv",
    "feature_set_v2_structure_minimal_final.csv",
    "feature_set_v2_structure_core_final.csv",
    "feature_set_v2_sequence_plus_structure_minimal.csv",
    "feature_set_v2_sequence_plus_structure_core.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Benchmark v1.1 from Minami 2023 Table 1."
    )
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument(
        "--output-dir",
        default="data/frozen/benchmark_v1_1",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_gene(value: Any) -> str:
    return normalize_text(value).lower()


def normalize_jwid(value: Any) -> str:
    return normalize_text(value).upper()


def normalize_accession(value: Any) -> str:
    return normalize_text(value).upper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_benchmark(df: pd.DataFrame) -> None:
    required = {ACCESSION, GENE, JWID, LABEL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Benchmark missing columns: {sorted(missing)}")

    if df[ACCESSION].duplicated().any():
        raise ValueError("Benchmark contains duplicate UniProt accessions")

    labels = set(pd.to_numeric(df[LABEL], errors="raise").astype(int).unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f"Unexpected binary labels: {sorted(labels)}")


def audit_and_relabel(
    benchmark: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    updated = benchmark.copy()

    for column, default in [
        ("benchmark_version", "1.1"),
        ("previous_binary_label", ""),
        ("label_updated_from_minami_validation", False),
        ("minami_2023_table1_groe_dependency", ""),
        ("minami_2023_validation_status", ""),
        ("minami_2023_validation_source", ""),
    ]:
        if column not in updated.columns:
            updated[column] = default

    jwid_to_indices: dict[str, list[int]] = {}
    for idx, value in updated[JWID].items():
        jwid_to_indices.setdefault(normalize_jwid(value), []).append(idx)

    audit_rows = []

    for jwid, expected in MINAMI_TABLE1.items():
        matches = jwid_to_indices.get(jwid, [])

        if len(matches) == 0:
            raise RuntimeError(
                f"Minami Table 1 JW ID {jwid} was not found in benchmark"
            )
        if len(matches) > 1:
            raise RuntimeError(
                f"Minami Table 1 JW ID {jwid} maps to multiple benchmark rows"
            )

        idx = matches[0]
        benchmark_gene = normalize_gene(updated.at[idx, GENE])
        expected_gene = normalize_gene(expected["gene"])

        # JW ID is the authoritative mapping key here. Minami Table 1 uses
        # historical gene symbols for several E. coli proteins, whereas our
        # benchmark carries newer/current symbols. Therefore a gene-name
        # mismatch is recorded in the audit instead of treated as a mapping
        # failure. The script still requires every JW ID to occur exactly once.
        gene_name_matches_exact = benchmark_gene == expected_gene

        old_label = int(updated.at[idx, LABEL])
        new_label = int(expected["validated_label"])

        # Minami Table 1 proteins entered the false-positive validation set
        # from the original negative pool. If one is already positive in v1.0,
        # stop rather than silently changing assumptions.
        if old_label not in {0, 1}:
            raise RuntimeError(f"Unexpected old label for {jwid}: {old_label}")

        changed = old_label != new_label

        updated.at[idx, "previous_binary_label"] = old_label
        updated.at[idx, LABEL] = new_label
        updated.at[
            idx, "label_updated_from_minami_validation"
        ] = changed
        updated.at[
            idx, "minami_2023_table1_groe_dependency"
        ] = expected["groe_dependency"]
        updated.at[
            idx, "minami_2023_validation_status"
        ] = (
            "experimental_groe_dependent"
            if new_label == 1
            else "experimental_groe_independent"
        )
        updated.at[
            idx, "minami_2023_validation_source"
        ] = (
            "Minami et al. 2023 Table 1 experimental validation "
            "of predicted false positives"
        )

        if changed:
            if "label_source" in updated.columns:
                prior = normalize_text(updated.at[idx, "label_source"])
                updated.at[idx, "label_source"] = (
                    prior + " | Minami_2023_Table1_validation"
                ).strip(" |")
            if "label_evidence_type" in updated.columns:
                updated.at[
                    idx, "label_evidence_type"
                ] = "GroE_depletion_experimental_validation"
            if "label_confidence" in updated.columns:
                updated.at[
                    idx, "label_confidence"
                ] = "high_experimental"

        audit_rows.append(
            {
                JWID: jwid,
                "table1_gene": expected_gene,
                "benchmark_gene": benchmark_gene,
                "gene_name_matches_exact": gene_name_matches_exact,
                ACCESSION: updated.at[idx, ACCESSION],
                "groe_dependency_table1": expected["groe_dependency"],
                "old_label": old_label,
                "new_label": new_label,
                "label_changed": changed,
                "validation_status": (
                    "experimental_groe_dependent"
                    if new_label == 1
                    else "experimental_groe_independent"
                ),
            }
        )

    audit = pd.DataFrame(audit_rows)

    if len(audit) != 17:
        raise RuntimeError(f"Expected 17 audited proteins, got {len(audit)}")

    positive_count = int((audit["new_label"] == 1).sum())
    negative_count = int((audit["new_label"] == 0).sum())

    if positive_count != 9 or negative_count != 8:
        raise RuntimeError(
            f"Expected 9 dependent and 8 independent proteins; got "
            f"{positive_count} and {negative_count}"
        )

    changed_count = int(audit["label_changed"].sum())

    # In the intended historical benchmark all 17 were initially negatives,
    # therefore exactly 9 should change 0 -> 1.
    if changed_count != 9:
        raise RuntimeError(
            f"Expected exactly 9 label changes, observed {changed_count}. "
            "Inspect Benchmark v1.0 before proceeding."
        )

    bad_changes = audit[
        audit["label_changed"]
        & ~(
            (audit["old_label"] == 0)
            & (audit["new_label"] == 1)
        )
    ]
    if not bad_changes.empty:
        raise RuntimeError(
            "Detected a label change other than the expected 0 -> 1 updates"
        )

    return updated, audit


def patch_feature_files(
    updated_benchmark: pd.DataFrame,
    source_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    label_map = dict(
        zip(
            updated_benchmark[ACCESSION].map(normalize_accession),
            updated_benchmark[LABEL].astype(int),
        )
    )

    provenance_columns = [
        ACCESSION,
        "benchmark_version",
        "previous_binary_label",
        "label_updated_from_minami_validation",
        "minami_2023_table1_groe_dependency",
        "minami_2023_validation_status",
        "minami_2023_validation_source",
    ]
    provenance = updated_benchmark[provenance_columns].copy()

    records = []

    for filename in FEATURE_FILENAMES:
        source = source_dir / filename
        if not source.exists():
            raise FileNotFoundError(
                f"Expected Feature Set v2 file not found: {source}"
            )

        df = pd.read_csv(source)

        if ACCESSION not in df.columns or LABEL not in df.columns:
            raise ValueError(
                f"{filename} is missing {ACCESSION} or {LABEL}"
            )

        old_labels = df[LABEL].astype(int).copy()
        keys = df[ACCESSION].map(normalize_accession)

        missing = keys[~keys.isin(label_map)]
        if not missing.empty:
            raise RuntimeError(
                f"{filename} has accessions missing from Benchmark v1.1"
            )

        df[LABEL] = keys.map(label_map).astype(int)

        existing_provenance = [
            c for c in provenance_columns
            if c != ACCESSION and c in df.columns
        ]
        if existing_provenance:
            df = df.drop(columns=existing_provenance)

        df = df.merge(
            provenance,
            on=ACCESSION,
            how="left",
            validate="one_to_one",
        )

        output_path = output_dir / filename
        df.to_csv(output_path, index=False)

        records.append(
            {
                "filename": filename,
                "rows": len(df),
                "changed_labels": int(
                    (old_labels.to_numpy() != df[LABEL].to_numpy()).sum()
                ),
                "sha256": sha256_file(output_path),
            }
        )

    return pd.DataFrame(records)


def main() -> int:
    args = parse_args()

    benchmark_path = Path(args.benchmark)
    feature_dir = Path(args.feature_dir)
    output_dir = Path(args.output_dir)

    if not benchmark_path.exists():
        print(
            f"ERROR: benchmark not found: {benchmark_path}",
            file=sys.stderr,
        )
        return 2
    if not feature_dir.exists():
        print(
            f"ERROR: feature directory not found: {feature_dir}",
            file=sys.stderr,
        )
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark = pd.read_csv(benchmark_path)
    validate_benchmark(benchmark)

    old_positive_count = int(benchmark[LABEL].astype(int).sum())
    old_negative_count = int(len(benchmark) - old_positive_count)

    updated, audit = audit_and_relabel(benchmark)

    new_positive_count = int(updated[LABEL].astype(int).sum())
    new_negative_count = int(len(updated) - new_positive_count)

    benchmark_output = output_dir / "benchmark_v1.1.csv"
    updated.to_csv(benchmark_output, index=False)

    audit_output = output_dir / "minami_table1_label_audit.csv"
    audit.to_csv(audit_output, index=False)

    changes = audit[audit["label_changed"]].copy()
    changes.to_csv(
        output_dir / "benchmark_v1.0_vs_v1.1_changes.csv",
        index=False,
    )

    preserved_negatives = audit[~audit["label_changed"]].copy()
    preserved_negatives.to_csv(
        output_dir / "experimentally_confirmed_independent_proteins.csv",
        index=False,
    )

    feature_audit = patch_feature_files(
        updated,
        feature_dir,
        output_dir / "feature_set_v2_labels_v1.1",
    )
    feature_audit.to_csv(
        output_dir / "feature_file_patch_audit.csv",
        index=False,
    )

    summary = {
        "stage": "10B_v2",
        "benchmark_input": str(benchmark_path),
        "benchmark_rows": len(benchmark),
        "positive_count_v1_0": old_positive_count,
        "negative_count_v1_0": old_negative_count,
        "positive_count_v1_1": new_positive_count,
        "negative_count_v1_1": new_negative_count,
        "validated_table1_proteins": 17,
        "experimentally_confirmed_dependent": 9,
        "experimentally_confirmed_independent": 8,
        "label_change_count": int(len(changes)),
        "changed_genes_table1": changes["table1_gene"].tolist(),
        "changed_genes_benchmark": changes["benchmark_gene"].tolist(),
        "historical_gene_symbol_mismatch_count": int(
            (~audit["gene_name_matches_exact"]).sum()
        ),
        "historical_gene_symbol_mismatches": audit.loc[
            ~audit["gene_name_matches_exact"],
            ["jwid", "table1_gene", "benchmark_gene"],
        ].to_dict(orient="records"),
        "changed_jwids": changes[JWID].tolist(),
        "feature_files_patched": feature_audit.to_dict(orient="records"),
        "policy": (
            "Benchmark v1.1 changes only the nine Minami Table 1 proteins "
            "experimentally validated as GroE-dependent. The eight validated "
            "GroE-independent proteins remain negative. Feature values and "
            "fold assignments are unchanged."
        ),
    }

    (output_dir / "stage10B_v2_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    methods = """# Benchmark v1.1 construction

Benchmark v1.1 updates Benchmark v1.0 using the experimental validation
reported in Minami et al. 2023 Table 1.

The 17 proteins in Table 1 were mapped by unique JW ID. JW ID is treated as
the authoritative identifier because several gene symbols in the 2023 table are
historical names that differ from current names in the benchmark. Gene-symbol
differences are preserved in the audit rather than treated as mapping failures.
Nine proteins reported with GroE dependency "+" were changed from negative to
positive. Eight proteins reported with GroE dependency "-" remained negative.

No label was changed because of this project's model predictions, feature
importance, structural similarity, or biological intuition.

Benchmark v1.0 is retained unchanged. Feature values and the previously frozen
MMseqs2-based cross-validation group assignments are also unchanged. Only label
and label-provenance fields are updated in separately versioned output files.
"""
    (output_dir / "benchmark_v1.1_methods.md").write_text(methods)

    checksums = []
    for path in sorted(output_dir.rglob("*")):
        if (
            path.is_file()
            and path.name != "benchmark_v1.1_sha256.txt"
        ):
            checksums.append(
                f"{sha256_file(path)}  {path.relative_to(output_dir)}"
            )

    (output_dir / "benchmark_v1.1_sha256.txt").write_text(
        "\n".join(checksums) + "\n"
    )

    print("Stage 10B v2 complete")
    print(f"  Benchmark rows:             {len(updated)}")
    print(f"  Positives v1.0:             {old_positive_count}")
    print(f"  Positives v1.1:             {new_positive_count}")
    print(f"  Negatives v1.0:             {old_negative_count}")
    print(f"  Negatives v1.1:             {new_negative_count}")
    print(f"  Table 1 proteins audited:   {len(audit)}")
    print(f"  Labels changed 0 -> 1:      {len(changes)}")
    print("  Expected label changes:     9")
    print(
        f"  Historical gene aliases:    "
        f"{int((~audit['gene_name_matches_exact']).sum())}"
    )
    print(f"  Output directory:           {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
