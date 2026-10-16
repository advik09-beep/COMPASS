#!/usr/bin/env python3
"""
Stage 10B: Audit Minami experimental-validation labels and build Benchmark v1.1.

Purpose
-------
The Minami et al. benchmark initially treated some proteins as negatives, then
experimentally tested predicted false positives in GroE-depleted cells. This
script preserves Benchmark v1.0 and creates a separately versioned Benchmark
v1.1 only from manually verified experimental outcomes.

The script never infers labels from model predictions. A protein is relabeled
from 0 to 1 only when the validation table explicitly records:
    validation_outcome = dependent
    evidence_level = experimental_confirmed

Workflow
--------
1. Generate a manual-review template:
   python3 scripts/10B_audit_minami_validation_labels.py \
       --benchmark data/frozen/benchmark_v1_0/benchmark_v1.0.csv \
       --high-confidence-fps results/stage10A/high_confidence_false_positives.csv \
       --make-template results/stage10B/minami_validation_review_template.csv

2. Fill the template from Minami Table 1 and its accompanying text.

3. Build Benchmark v1.1:
   python3 scripts/10B_audit_minami_validation_labels.py \
       --benchmark data/frozen/benchmark_v1_0/benchmark_v1.0.csv \
       --feature-dir data/frozen/feature_set_v2 \
       --validation-table data/manual/minami_2023_validation_labels.csv \
       --high-confidence-fps results/stage10A/high_confidence_false_positives.csv \
       --output-dir data/frozen/benchmark_v1_1

Important
---------
- Benchmark v1.0 is never overwritten.
- Frozen feature values are never recomputed or altered.
- Existing folds are not changed.
- All label changes are written to an explicit audit file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ACCESSION = "current_uniprot_accession"
GENE = "gene_minami"
JWID = "jwid"
LABEL = "binary_label"

ALLOWED_OUTCOMES = {
    "dependent",
    "independent",
    "inconclusive",
    "not_tested",
}

ALLOWED_EVIDENCE = {
    "experimental_confirmed",
    "experimental_independent",
    "inconclusive",
    "not_tested",
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
        description="Audit Minami 2023 validation labels and build Benchmark v1.1."
    )
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--feature-dir", default=None)
    parser.add_argument("--validation-table", default=None)
    parser.add_argument("--high-confidence-fps", default=None)
    parser.add_argument("--make-template", default=None)
    parser.add_argument(
        "--output-dir",
        default="data/frozen/benchmark_v1_1",
    )
    parser.add_argument(
        "--source-citation",
        default=(
            "Minami et al. 2023, Prediction of chaperonin GroE substrates "
            "using small structural patterns of proteins, Table 1"
        ),
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_gene(value: Any) -> str:
    return normalize_text(value).lower()


def normalize_accession(value: Any) -> str:
    return normalize_text(value).upper()


def normalize_jwid(value: Any) -> str:
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
        duplicates = df.loc[df[ACCESSION].duplicated(), ACCESSION].tolist()
        raise ValueError(f"Duplicate benchmark accessions: {duplicates}")

    labels = set(pd.to_numeric(df[LABEL], errors="raise").astype(int).unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f"Unexpected binary labels: {sorted(labels)}")


def make_review_template(
    benchmark: pd.DataFrame,
    high_confidence_fp_path: Path | None,
    output_path: Path,
    source_citation: str,
) -> None:
    candidates = benchmark.copy()

    candidates["_acc_key"] = candidates[ACCESSION].map(normalize_accession)
    candidates["_gene_key"] = candidates[GENE].map(normalize_gene)
    candidates["_jwid_key"] = candidates[JWID].map(normalize_jwid)

    candidate_keys: set[str] = set()

    if high_confidence_fp_path is not None and high_confidence_fp_path.exists():
        fp = pd.read_csv(high_confidence_fp_path)
        for column in [ACCESSION, GENE, JWID]:
            if column not in fp.columns:
                continue
            normalizer = {
                ACCESSION: normalize_accession,
                GENE: normalize_gene,
                JWID: normalize_jwid,
            }[column]
            candidate_keys.update(
                normalizer(value)
                for value in fp[column]
                if normalize_text(value)
            )

    if candidate_keys:
        mask = (
            candidates["_acc_key"].isin(candidate_keys)
            | candidates["_gene_key"].isin(candidate_keys)
            | candidates["_jwid_key"].isin(candidate_keys)
        )
        candidates = candidates[mask].copy()
    else:
        # If no FP file is supplied, include every original negative for review.
        candidates = candidates[candidates[LABEL].astype(int) == 0].copy()

    candidates = candidates.drop_duplicates(ACCESSION)

    template = pd.DataFrame(
        {
            ACCESSION: candidates[ACCESSION],
            GENE: candidates[GENE],
            JWID: candidates[JWID],
            "benchmark_v1_0_label": candidates[LABEL].astype(int),
            "tested_in_minami_2023": "",
            "validation_outcome": "",
            "evidence_level": "",
            "table_or_figure": "Table 1",
            "source_citation": source_citation,
            "manual_reviewer": "",
            "review_date": "",
            "evidence_notes": "",
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output_path, index=False)

    print(f"Review template written: {output_path}")
    print(f"Candidate proteins: {len(template)}")
    print(
        "Fill tested_in_minami_2023, validation_outcome, evidence_level, "
        "manual_reviewer, review_date, and evidence_notes."
    )


def validate_manual_table(df: pd.DataFrame) -> None:
    required = {
        "tested_in_minami_2023",
        "validation_outcome",
        "evidence_level",
        "source_citation",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Validation table missing required columns: {sorted(missing)}"
        )

    if not any(column in df.columns for column in [ACCESSION, GENE, JWID]):
        raise ValueError(
            "Validation table needs at least one identifier column: "
            f"{ACCESSION}, {GENE}, or {JWID}"
        )

    reviewed = df[
        df["tested_in_minami_2023"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"yes", "true", "1", "y"})
    ].copy()

    bad_outcomes = sorted(
        set(reviewed["validation_outcome"].map(normalize_gene))
        - ALLOWED_OUTCOMES
    )
    if bad_outcomes:
        raise ValueError(
            "Invalid validation_outcome values: "
            + ", ".join(bad_outcomes)
        )

    bad_evidence = sorted(
        set(reviewed["evidence_level"].map(normalize_gene))
        - ALLOWED_EVIDENCE
    )
    if bad_evidence:
        raise ValueError(
            "Invalid evidence_level values: "
            + ", ".join(bad_evidence)
        )

    dependent = reviewed[
        reviewed["validation_outcome"].map(normalize_gene) == "dependent"
    ]
    bad_dependent = dependent[
        dependent["evidence_level"].map(normalize_gene)
        != "experimental_confirmed"
    ]
    if not bad_dependent.empty:
        raise ValueError(
            "Every dependent outcome must use "
            "evidence_level=experimental_confirmed"
        )

    independent = reviewed[
        reviewed["validation_outcome"].map(normalize_gene) == "independent"
    ]
    bad_independent = independent[
        independent["evidence_level"].map(normalize_gene)
        != "experimental_independent"
    ]
    if not bad_independent.empty:
        raise ValueError(
            "Every independent outcome must use "
            "evidence_level=experimental_independent"
        )


def build_identifier_maps(
    benchmark: pd.DataFrame,
) -> tuple[dict[str, int], dict[str, list[int]], dict[str, list[int]]]:
    accession_map = {
        normalize_accession(value): index
        for index, value in benchmark[ACCESSION].items()
        if normalize_accession(value)
    }

    gene_map: dict[str, list[int]] = {}
    for index, value in benchmark[GENE].items():
        key = normalize_gene(value)
        if key:
            gene_map.setdefault(key, []).append(index)

    jwid_map: dict[str, list[int]] = {}
    for index, value in benchmark[JWID].items():
        key = normalize_jwid(value)
        if key:
            jwid_map.setdefault(key, []).append(index)

    return accession_map, gene_map, jwid_map


def resolve_manual_row(
    row: pd.Series,
    accession_map: dict[str, int],
    gene_map: dict[str, list[int]],
    jwid_map: dict[str, list[int]],
) -> tuple[int | None, str, str]:
    matches: list[tuple[str, int]] = []

    accession = normalize_accession(row.get(ACCESSION, ""))
    if accession and accession in accession_map:
        matches.append(("accession", accession_map[accession]))

    gene = normalize_gene(row.get(GENE, ""))
    if gene and gene in gene_map:
        for index in gene_map[gene]:
            matches.append(("gene", index))

    jwid = normalize_jwid(row.get(JWID, ""))
    if jwid and jwid in jwid_map:
        for index in jwid_map[jwid]:
            matches.append(("jwid", index))

    unique_indices = sorted({index for _, index in matches})

    if not unique_indices:
        return None, "not_found", ""
    if len(unique_indices) > 1:
        detail = ";".join(
            f"{kind}:{index}" for kind, index in matches
        )
        return None, "ambiguous_identifier", detail

    matched_by = ";".join(
        sorted({kind for kind, index in matches if index == unique_indices[0]})
    )
    return unique_indices[0], "matched", matched_by


def apply_validation_labels(
    benchmark: pd.DataFrame,
    manual: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    updated = benchmark.copy()

    for column, default in [
        ("benchmark_version", "1.1"),
        ("label_updated_from_validation", False),
        ("validation_outcome", ""),
        ("validation_evidence_level", ""),
        ("validation_source_citation", ""),
        ("validation_table_or_figure", ""),
        ("validation_evidence_notes", ""),
        ("previous_binary_label", np.nan),
    ]:
        if column not in updated.columns:
            updated[column] = default

    accession_map, gene_map, jwid_map = build_identifier_maps(updated)

    audit_rows = []
    unresolved_rows = []
    seen_benchmark_indices: set[int] = set()

    tested_mask = (
        manual["tested_in_minami_2023"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"yes", "true", "1", "y"})
    )

    for manual_index, row in manual[tested_mask].iterrows():
        benchmark_index, status, matched_by = resolve_manual_row(
            row,
            accession_map,
            gene_map,
            jwid_map,
        )

        base_audit = {
            "manual_row_index": int(manual_index),
            ACCESSION: normalize_text(row.get(ACCESSION, "")),
            GENE: normalize_text(row.get(GENE, "")),
            JWID: normalize_text(row.get(JWID, "")),
            "validation_outcome": normalize_gene(
                row.get("validation_outcome", "")
            ),
            "evidence_level": normalize_gene(
                row.get("evidence_level", "")
            ),
            "source_citation": normalize_text(
                row.get("source_citation", "")
            ),
            "table_or_figure": normalize_text(
                row.get("table_or_figure", "")
            ),
            "evidence_notes": normalize_text(
                row.get("evidence_notes", "")
            ),
            "mapping_status": status,
            "matched_by": matched_by,
        }

        if benchmark_index is None:
            unresolved_rows.append(base_audit)
            continue

        if benchmark_index in seen_benchmark_indices:
            base_audit["mapping_status"] = "duplicate_manual_mapping"
            unresolved_rows.append(base_audit)
            continue
        seen_benchmark_indices.add(benchmark_index)

        old_label = int(updated.at[benchmark_index, LABEL])
        outcome = base_audit["validation_outcome"]
        evidence = base_audit["evidence_level"]

        proposed_label = old_label
        change_reason = "no_label_change"

        if outcome == "dependent" and evidence == "experimental_confirmed":
            proposed_label = 1
            if old_label == 0:
                change_reason = "negative_to_experimentally_confirmed_positive"
            else:
                change_reason = "already_positive_confirmed"
        elif outcome == "independent" and evidence == "experimental_independent":
            # Do not flip an established positive to negative automatically.
            if old_label == 1:
                change_reason = "positive_conflict_requires_manual_resolution"
            else:
                proposed_label = 0
                change_reason = "negative_experimentally_supported"
        elif outcome == "inconclusive":
            change_reason = "inconclusive_no_change"
        elif outcome == "not_tested":
            change_reason = "not_tested_no_change"

        if change_reason == "positive_conflict_requires_manual_resolution":
            base_audit.update(
                {
                    "benchmark_index": benchmark_index,
                    "benchmark_accession": updated.at[
                        benchmark_index, ACCESSION
                    ],
                    "old_label": old_label,
                    "proposed_label": proposed_label,
                    "label_changed": False,
                    "change_reason": change_reason,
                }
            )
            unresolved_rows.append(base_audit)
            continue

        updated.at[benchmark_index, "previous_binary_label"] = old_label
        updated.at[benchmark_index, LABEL] = proposed_label
        updated.at[
            benchmark_index, "label_updated_from_validation"
        ] = bool(proposed_label != old_label)
        updated.at[
            benchmark_index, "validation_outcome"
        ] = outcome
        updated.at[
            benchmark_index, "validation_evidence_level"
        ] = evidence
        updated.at[
            benchmark_index, "validation_source_citation"
        ] = base_audit["source_citation"]
        updated.at[
            benchmark_index, "validation_table_or_figure"
        ] = base_audit["table_or_figure"]
        updated.at[
            benchmark_index, "validation_evidence_notes"
        ] = base_audit["evidence_notes"]

        # Preserve original source fields while appending the validation source.
        if proposed_label != old_label:
            if "label_source" in updated.columns:
                previous_source = normalize_text(
                    updated.at[benchmark_index, "label_source"]
                )
                updated.at[benchmark_index, "label_source"] = (
                    previous_source
                    + " | Minami_2023_experimental_validation"
                ).strip(" |")
            if "label_evidence_type" in updated.columns:
                updated.at[
                    benchmark_index, "label_evidence_type"
                ] = "GroE_depletion_experimental_validation"
            if "label_confidence" in updated.columns:
                updated.at[
                    benchmark_index, "label_confidence"
                ] = "high_experimental"

        audit_rows.append(
            {
                **base_audit,
                "benchmark_index": benchmark_index,
                "benchmark_accession": updated.at[
                    benchmark_index, ACCESSION
                ],
                "benchmark_gene": updated.at[benchmark_index, GENE],
                "old_label": old_label,
                "new_label": proposed_label,
                "label_changed": bool(proposed_label != old_label),
                "change_reason": change_reason,
            }
        )

    audit = pd.DataFrame(audit_rows)
    unresolved = pd.DataFrame(unresolved_rows)

    changes = (
        audit[audit["label_changed"]].copy()
        if not audit.empty
        else pd.DataFrame()
    )

    return updated, audit, unresolved


def patch_feature_files(
    updated_benchmark: pd.DataFrame,
    source_feature_dir: Path,
    output_feature_dir: Path,
) -> list[dict[str, Any]]:
    output_feature_dir.mkdir(parents=True, exist_ok=True)

    label_map = dict(
        zip(
            updated_benchmark[ACCESSION].map(normalize_accession),
            updated_benchmark[LABEL].astype(int),
        )
    )

    provenance_columns = [
        ACCESSION,
        LABEL,
        "benchmark_version",
        "label_updated_from_validation",
        "validation_outcome",
        "validation_evidence_level",
        "validation_source_citation",
        "validation_table_or_figure",
        "validation_evidence_notes",
        "previous_binary_label",
    ]
    provenance = updated_benchmark[provenance_columns].copy()

    records = []

    for filename in FEATURE_FILENAMES:
        source = source_feature_dir / filename
        if not source.exists():
            records.append(
                {
                    "filename": filename,
                    "status": "missing_source_file",
                    "rows": 0,
                    "changed_labels": 0,
                }
            )
            continue

        df = pd.read_csv(source)
        if ACCESSION not in df.columns or LABEL not in df.columns:
            raise ValueError(
                f"{filename} missing {ACCESSION} or {LABEL}"
            )

        old_labels = df[LABEL].astype(int).copy()
        accession_keys = df[ACCESSION].map(normalize_accession)
        missing = accession_keys[~accession_keys.isin(label_map)]
        if not missing.empty:
            raise ValueError(
                f"{filename} contains accessions absent from updated benchmark: "
                + ", ".join(sorted(set(missing)))
            )

        df[LABEL] = accession_keys.map(label_map).astype(int)

        extra_provenance = provenance.drop(columns=[LABEL])
        drop_existing = [
            column
            for column in extra_provenance.columns
            if column != ACCESSION and column in df.columns
        ]
        if drop_existing:
            df = df.drop(columns=drop_existing)

        df = df.merge(
            extra_provenance,
            on=ACCESSION,
            how="left",
            validate="one_to_one",
        )

        output = output_feature_dir / filename
        df.to_csv(output, index=False)

        records.append(
            {
                "filename": filename,
                "status": "written",
                "rows": len(df),
                "changed_labels": int(
                    (old_labels.to_numpy() != df[LABEL].to_numpy()).sum()
                ),
                "sha256": sha256_file(output),
            }
        )

    return records


def main() -> int:
    args = parse_args()
    benchmark_path = Path(args.benchmark)

    if not benchmark_path.exists():
        print(f"ERROR: benchmark not found: {benchmark_path}", file=sys.stderr)
        return 2

    benchmark = pd.read_csv(benchmark_path)
    validate_benchmark(benchmark)

    if args.make_template:
        fp_path = (
            Path(args.high_confidence_fps)
            if args.high_confidence_fps
            else None
        )
        make_review_template(
            benchmark,
            fp_path,
            Path(args.make_template),
            args.source_citation,
        )

        if not args.validation_table:
            return 0

    if not args.validation_table:
        print(
            "ERROR: --validation-table is required unless only "
            "--make-template is requested",
            file=sys.stderr,
        )
        return 2

    validation_path = Path(args.validation_table)
    if not validation_path.exists():
        print(
            f"ERROR: validation table not found: {validation_path}",
            file=sys.stderr,
        )
        return 2

    manual = pd.read_csv(validation_path, dtype=str, keep_default_na=False)
    validate_manual_table(manual)

    updated, audit, unresolved = apply_validation_labels(
        benchmark,
        manual,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmark_output = output_dir / "benchmark_v1.1.csv"
    audit_output = output_dir / "label_update_audit.csv"
    changes_output = output_dir / "benchmark_v1.0_vs_v1.1_changes.csv"
    unresolved_output = output_dir / "unresolved_validation_rows.csv"
    validated_output = output_dir / "validated_false_positive_analysis.csv"

    updated.to_csv(benchmark_output, index=False)
    audit.to_csv(audit_output, index=False)

    changes = (
        audit[audit["label_changed"]].copy()
        if not audit.empty
        else pd.DataFrame()
    )
    changes.to_csv(changes_output, index=False)
    unresolved.to_csv(unresolved_output, index=False)

    validated = (
        audit[
            audit["validation_outcome"].isin(
                ["dependent", "independent"]
            )
        ].copy()
        if not audit.empty
        else pd.DataFrame()
    )
    validated.to_csv(validated_output, index=False)

    feature_records: list[dict[str, Any]] = []
    if args.feature_dir:
        source_feature_dir = Path(args.feature_dir)
        if not source_feature_dir.exists():
            raise FileNotFoundError(source_feature_dir)

        feature_records = patch_feature_files(
            updated,
            source_feature_dir,
            output_dir / "feature_set_v2_labels_v1.1",
        )

    old_positive = int(benchmark[LABEL].astype(int).sum())
    new_positive = int(updated[LABEL].astype(int).sum())

    summary = {
        "stage": "10B",
        "benchmark_input": str(benchmark_path),
        "validation_table": str(validation_path),
        "benchmark_v1_0_rows": len(benchmark),
        "benchmark_v1_1_rows": len(updated),
        "positive_count_v1_0": old_positive,
        "positive_count_v1_1": new_positive,
        "negative_count_v1_0": int(len(benchmark) - old_positive),
        "negative_count_v1_1": int(len(updated) - new_positive),
        "tested_validation_rows": int(len(audit) + len(unresolved)),
        "resolved_validation_rows": int(len(audit)),
        "unresolved_validation_rows": int(len(unresolved)),
        "label_change_count": int(len(changes)),
        "changed_accessions": (
            changes["benchmark_accession"].tolist()
            if not changes.empty
            else []
        ),
        "feature_files": feature_records,
        "policy": (
            "Only explicit experimental_confirmed dependent outcomes can "
            "change a benchmark label from 0 to 1. Existing positives are "
            "never automatically changed to 0."
        ),
    }

    (output_dir / "stage10B_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    methods = """# Benchmark v1.1 label-update policy

Benchmark v1.0 is preserved unchanged.

Benchmark v1.1 incorporates only manually verified experimental outcomes from
Minami et al. 2023 Table 1. A label changes from negative to positive only when
the manual validation table records both:

- `validation_outcome = dependent`
- `evidence_level = experimental_confirmed`

Predictions, permutation importance, model confidence, structural similarity,
and biological plausibility are never used as relabeling evidence.

Experimentally independent results strengthen an existing negative label but do
not alter an existing positive label automatically. Inconclusive, untested,
unmapped, ambiguous, and conflicting records are retained in an audit table and
do not modify the benchmark.

Feature values and frozen fold assignments are unchanged. Only label and
provenance columns are patched into separately versioned feature tables.
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

    print("Stage 10B complete")
    print(f"  Benchmark rows:         {len(updated)}")
    print(f"  Positives v1.0:         {old_positive}")
    print(f"  Positives v1.1:         {new_positive}")
    print(f"  Label changes:          {len(changes)}")
    print(f"  Unresolved rows:        {len(unresolved)}")
    print(f"  Output directory:       {output_dir}")

    if len(unresolved):
        print(
            "WARNING: unresolved validation rows remain. Inspect "
            "unresolved_validation_rows.csv before modeling."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
