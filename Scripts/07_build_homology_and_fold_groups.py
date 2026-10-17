#!/usr/bin/env python3
"""
Stage 07: Build MMseqs2 homology clusters, parse SCCS fold annotations,
and generate audited group assignments for downstream cross-validation.

Inputs
------
1. Frozen benchmark v1.0 CSV (contains accession, label, SCCS)
2. Original merged feature CSV (contains FASTA paths)

Outputs
-------
- all_181_sequences.fasta
- MMseqs2 clustering results at 30% and 40% identity
- per-protein homology/fold assignments
- cluster and fold composition audits
- deterministic grouped-CV fold assignments when feasible
- Stage 07 summary JSON and methods log

Primary clustering definition
-----------------------------
MMseqs2 easy-cluster with:
    --min-seq-id 0.30
    -c 0.80
    --cov-mode 0

cov-mode 0 requires the alignment to cover at least 80% of the longer sequence,
which is appropriate for clustering full-length proteins while preserving
multi-domain architecture.

Example
-------
python3 scripts/07_build_homology_and_fold_groups.py \
  --benchmark data/frozen/benchmark_v1_0/benchmark_v1.0.csv \
  --source-features data/modeling/minami_binary_benchmark_features_181.csv \
  --output-dir data/grouping/stage07
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


EXPECTED_ROWS = 181
EXPECTED_POSITIVES = 64
EXPECTED_NEGATIVES = 117


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MMseqs2 homology clusters and fold-aware group assignments."
    )
    parser.add_argument("--benchmark", required=True, help="Frozen benchmark_v1.0.csv")
    parser.add_argument(
        "--source-features",
        required=True,
        help="Merged 181-protein feature CSV containing fasta_path",
    )
    parser.add_argument(
        "--output-dir",
        default="data/grouping/stage07",
        help="Output directory",
    )
    parser.add_argument(
        "--primary-identity",
        type=float,
        default=0.30,
        help="Primary MMseqs2 minimum sequence identity",
    )
    parser.add_argument(
        "--sensitivity-identity",
        type=float,
        default=0.40,
        help="Sensitivity-analysis identity threshold",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.80,
        help="Minimum full-length bidirectional coverage",
    )
    parser.add_argument(
        "--cov-mode",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="MMseqs2 coverage mode; 0 is bidirectional full-length coverage",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="MMseqs2 thread count",
    )
    parser.add_argument(
        "--requested-folds",
        type=int,
        default=5,
        help="Requested number of grouped CV folds",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Seed for deterministic group assignment",
    )
    parser.add_argument(
        "--force-mmseqs",
        action="store_true",
        help="Rerun MMseqs2 even when expected TSV outputs already exist",
    )
    return parser.parse_args()


def read_fasta(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"FASTA missing: {path}")
    sequence = "".join(
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and not line.startswith(">")
    ).upper()
    if not sequence:
        raise ValueError(f"Empty FASTA: {path}")
    invalid = sorted(set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"))
    if invalid:
        raise ValueError(f"Non-standard residues {invalid} in {path}")
    return sequence


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def validate_inputs(benchmark: pd.DataFrame, source: pd.DataFrame) -> None:
    required_benchmark = {
        "current_uniprot_accession",
        "binary_label",
        "cohort_label",
        "sccs",
        "jwid",
    }
    missing = required_benchmark - set(benchmark.columns)
    if missing:
        raise ValueError(f"Benchmark missing columns: {sorted(missing)}")
    if "fasta_path" not in source.columns:
        raise ValueError("Source feature CSV must contain fasta_path")
    if "current_uniprot_accession" not in source.columns:
        raise ValueError("Source feature CSV must contain current_uniprot_accession")

    if len(benchmark) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} benchmark rows, observed {len(benchmark)}")
    positives = int((pd.to_numeric(benchmark["binary_label"]) == 1).sum())
    negatives = int((pd.to_numeric(benchmark["binary_label"]) == 0).sum())
    if positives != EXPECTED_POSITIVES or negatives != EXPECTED_NEGATIVES:
        raise ValueError(
            f"Expected {EXPECTED_POSITIVES}/{EXPECTED_NEGATIVES} positive/negative, "
            f"observed {positives}/{negatives}"
        )


def centralize_sequences(
    benchmark: pd.DataFrame,
    source: pd.DataFrame,
    output_fasta: Path,
    sequence_manifest_path: Path,
) -> pd.DataFrame:
    source_lookup = source.set_index("current_uniprot_accession", drop=False)
    records: list[dict[str, Any]] = []
    fasta_lines: list[str] = []

    for _, row in benchmark.iterrows():
        accession = str(row["current_uniprot_accession"]).strip()
        if accession not in source_lookup.index:
            raise ValueError(f"No source feature row for {accession}")
        source_row = source_lookup.loc[accession]
        if isinstance(source_row, pd.DataFrame):
            raise ValueError(f"Duplicate source feature rows for {accession}")

        fasta_path = Path(str(source_row["fasta_path"]))
        sequence = read_fasta(fasta_path)
        sequence_hash = sha256_text(sequence)
        fasta_lines.extend([f">{accession}", sequence])
        records.append(
            {
                "current_uniprot_accession": accession,
                "jwid": row["jwid"],
                "binary_label": int(row["binary_label"]),
                "fasta_path": str(fasta_path),
                "sequence_length": len(sequence),
                "sequence_sha256": sequence_hash,
            }
        )

    manifest = pd.DataFrame(records)
    if manifest["current_uniprot_accession"].duplicated().any():
        raise ValueError("Duplicate accessions in centralized sequence manifest")
    if manifest["sequence_sha256"].duplicated().any():
        duplicate = manifest[manifest["sequence_sha256"].duplicated(keep=False)]
        raise ValueError(
            "Identical sequences appeared despite frozen benchmark duplicate audit:\n"
            + duplicate.to_string(index=False)
        )

    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    output_fasta.write_text("\n".join(fasta_lines) + "\n")
    manifest.to_csv(sequence_manifest_path, index=False)
    return manifest


def check_mmseqs() -> str:
    executable = shutil.which("mmseqs")
    if not executable:
        raise RuntimeError(
            "MMseqs2 executable 'mmseqs' was not found in PATH. "
            "Verify installation with: mmseqs version"
        )
    result = subprocess.run(
        [executable, "version"], capture_output=True, text=True, check=False
    )
    version_text = (result.stdout or result.stderr).strip()
    if result.returncode != 0:
        raise RuntimeError(f"MMseqs2 version check failed: {version_text}")
    return version_text


def threshold_tag(identity: float) -> str:
    return f"id{int(round(identity * 100)):02d}"


def run_mmseqs_easy_cluster(
    executable: str,
    input_fasta: Path,
    output_dir: Path,
    identity: float,
    coverage: float,
    cov_mode: int,
    threads: int,
    force: bool,
) -> Path:
    tag = threshold_tag(identity)
    run_dir = output_dir / f"mmseqs_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    prefix = run_dir / f"cluster_{tag}"
    tmp_dir = run_dir / "tmp"
    cluster_tsv = Path(str(prefix) + "_cluster.tsv")
    log_path = run_dir / "mmseqs_command.log"

    if cluster_tsv.exists() and not force:
        return cluster_tsv

    if force:
        for path in run_dir.glob("cluster_*"):
            if path.is_file():
                path.unlink()
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    command = [
        executable,
        "easy-cluster",
        str(input_fasta),
        str(prefix),
        str(tmp_dir),
        "--min-seq-id",
        str(identity),
        "-c",
        str(coverage),
        "--cov-mode",
        str(cov_mode),
        "--threads",
        str(threads),
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    log_path.write_text(
        "COMMAND\n"
        + " ".join(command)
        + "\n\nSTDOUT\n"
        + result.stdout
        + "\n\nSTDERR\n"
        + result.stderr
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"MMseqs2 failed at identity {identity}. See {log_path}"
        )
    if not cluster_tsv.exists():
        raise RuntimeError(
            f"MMseqs2 completed but expected output is missing: {cluster_tsv}"
        )
    return cluster_tsv


def parse_cluster_tsv(
    tsv_path: Path,
    all_accessions: set[str],
    cluster_column: str,
) -> pd.DataFrame:
    raw = pd.read_csv(
        tsv_path,
        sep="\t",
        header=None,
        names=["representative", "member"],
        dtype=str,
    )
    if raw.empty:
        raise ValueError(f"Empty MMseqs cluster TSV: {tsv_path}")

    unknown_members = set(raw["member"]) - all_accessions
    unknown_reps = set(raw["representative"]) - all_accessions
    if unknown_members or unknown_reps:
        raise ValueError(
            f"Unknown MMseqs IDs. Members={sorted(unknown_members)}, reps={sorted(unknown_reps)}"
        )

    duplicate_members = raw[raw["member"].duplicated(keep=False)]
    if not duplicate_members.empty:
        raise ValueError(
            "MMseqs member assigned more than once:\n"
            + duplicate_members.to_string(index=False)
        )

    missing = all_accessions - set(raw["member"])
    if missing:
        raise ValueError(f"MMseqs TSV omitted proteins: {sorted(missing)}")

    reps = sorted(raw["representative"].unique())
    rep_to_cluster = {rep: f"{cluster_column}_{i:03d}" for i, rep in enumerate(reps, 1)}
    raw[cluster_column] = raw["representative"].map(rep_to_cluster)
    return raw[["member", "representative", cluster_column]].rename(
        columns={"member": "current_uniprot_accession"}
    )


def parse_sccs(value: Any) -> dict[str, Any]:
    text = "" if pd.isna(value) else str(value).strip()
    tokens = [token.strip() for token in text.split(";") if token.strip() and token.strip() != "nan"]
    classes: set[str] = set()
    folds: set[str] = set()
    superfamilies: set[str] = set()

    for token in tokens:
        parts = token.split(".")
        if parts and re.fullmatch(r"[a-z]", parts[0]):
            classes.add(parts[0])
        if len(parts) >= 2 and re.fullmatch(r"[a-z]", parts[0]):
            folds.add(".".join(parts[:2]))
        if len(parts) >= 3 and re.fullmatch(r"[a-z]", parts[0]):
            superfamilies.add(".".join(parts[:3]))

    is_tim = "c.1" in folds
    return {
        "scop_class_set": ";".join(sorted(classes)),
        "scop_fold_set": ";".join(sorted(folds)),
        "scop_superfamily_set": ";".join(sorted(superfamilies)),
        "scop_fold_count": len(folds),
        "scop_superfamily_count": len(superfamilies),
        "is_multifold": len(folds) > 1,
        "is_tim_barrel": is_tim,
        "primary_scop_class": sorted(classes)[0] if classes else "",
        "primary_scop_fold": sorted(folds)[0] if folds else "",
        "primary_scop_superfamily": sorted(superfamilies)[0] if superfamilies else "",
    }


def cluster_audit(assignments: pd.DataFrame, cluster_column: str) -> pd.DataFrame:
    records = []
    for cluster_id, group in assignments.groupby(cluster_column, sort=True):
        positives = int((group["binary_label"] == 1).sum())
        negatives = int((group["binary_label"] == 0).sum())
        records.append(
            {
                cluster_column: cluster_id,
                "representative_accession": group[f"representative_{cluster_column}"].iloc[0],
                "cluster_size": len(group),
                "positive_count": positives,
                "negative_count": negatives,
                "mixed_label_cluster": positives > 0 and negatives > 0,
                "members": ";".join(sorted(group["current_uniprot_accession"])),
            }
        )
    return pd.DataFrame(records).sort_values(
        ["cluster_size", "positive_count"], ascending=[False, False]
    )


def fold_audit(assignments: pd.DataFrame) -> pd.DataFrame:
    exploded = []
    for _, row in assignments.iterrows():
        folds = [f for f in str(row["scop_fold_set"]).split(";") if f]
        if not folds:
            folds = ["UNASSIGNED"]
        for fold in folds:
            exploded.append(
                {
                    "scop_fold": fold,
                    "current_uniprot_accession": row["current_uniprot_accession"],
                    "binary_label": row["binary_label"],
                    "is_tim_barrel": row["is_tim_barrel"],
                }
            )
    exploded_df = pd.DataFrame(exploded)
    records = []
    for fold, group in exploded_df.groupby("scop_fold", sort=True):
        positives = int((group["binary_label"] == 1).sum())
        negatives = int((group["binary_label"] == 0).sum())
        records.append(
            {
                "scop_fold": fold,
                "protein_count": group["current_uniprot_accession"].nunique(),
                "positive_count": positives,
                "negative_count": negatives,
                "mixed_label_fold": positives > 0 and negatives > 0,
                "tim_barrel_fold": fold == "c.1",
            }
        )
    return pd.DataFrame(records).sort_values(
        ["protein_count", "positive_count"], ascending=[False, False]
    )


def assign_grouped_folds(
    assignments: pd.DataFrame,
    cluster_column: str,
    requested_folds: int,
    seed: int,
) -> tuple[pd.Series, pd.DataFrame, int]:
    """
    Assign complete homology clusters to approximately stratified folds.

    Uses scikit-learn StratifiedGroupKFold so no homology cluster can cross
    train/test partitions. Fold count is reduced automatically if a requested
    split does not give every test fold at least one positive and one negative.
    Labels are used only for stratification; no predictor values are used.
    """
    y = pd.to_numeric(assignments["binary_label"], errors="raise").astype(int)
    groups = assignments[cluster_column].astype(str)

    unique_group_count = groups.nunique()
    maximum_folds = min(int(requested_folds), int(unique_group_count))

    for n_folds in range(maximum_folds, 1, -1):
        splitter = StratifiedGroupKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=seed,
        )

        fold_values = pd.Series(index=assignments.index, dtype="Int64")
        audit_rows = []
        valid = True

        try:
            split_iterator = splitter.split(
                X=np.zeros((len(assignments), 1)),
                y=y.to_numpy(),
                groups=groups.to_numpy(),
            )

            for fold, (_, test_idx) in enumerate(split_iterator):
                test_index = assignments.index[test_idx]
                fold_values.loc[test_index] = fold

                test_labels = y.loc[test_index]
                test_groups = sorted(groups.loc[test_index].unique().tolist())
                positive_count = int((test_labels == 1).sum())
                negative_count = int((test_labels == 0).sum())

                if positive_count == 0 or negative_count == 0:
                    valid = False

                audit_rows.append(
                    {
                        "fold": fold,
                        "protein_count": int(len(test_index)),
                        "positive_count": positive_count,
                        "negative_count": negative_count,
                        "cluster_count": int(len(test_groups)),
                        "clusters": ";".join(test_groups),
                    }
                )
        except ValueError:
            valid = False

        if valid and fold_values.notna().all():
            fold_values = fold_values.astype(int)
            audit = pd.DataFrame(audit_rows).sort_values("fold").reset_index(drop=True)
            return fold_values, audit, n_folds

    # Provide useful diagnostics rather than implying MMseqs2 failed.
    cluster_stats = (
        assignments.groupby(cluster_column)["binary_label"]
        .agg(protein_count="size", positive_count="sum")
        .reset_index()
    )
    cluster_stats["negative_count"] = (
        cluster_stats["protein_count"] - cluster_stats["positive_count"]
    )
    largest = cluster_stats.sort_values("protein_count", ascending=False).head(10)
    diagnostic = largest.to_dict(orient="records")
    raise RuntimeError(
        "Could not create at least two homology-grouped folds with both classes. "
        f"Unique groups={unique_group_count}; largest groups={diagnostic}"
    )

def verify_no_cluster_leakage(
    assignments: pd.DataFrame, cluster_column: str, fold_column: str
) -> None:
    counts = assignments.groupby(cluster_column)[fold_column].nunique()
    leaking = counts[counts > 1]
    if not leaking.empty:
        raise RuntimeError(
            f"Cluster leakage detected for {cluster_column}: {leaking.index.tolist()}"
        )


def make_joint_components(assignments: pd.DataFrame, cluster_column: str) -> pd.Series:
    """
    Build connected components where proteins are linked if they share either
    a homology cluster or any SCOP fold. A component must remain intact to
    satisfy both no-homology-leakage and no-fold-leakage constraints.
    """
    accessions = assignments["current_uniprot_accession"].tolist()
    parent = {a: a for a in accessions}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for _, group in assignments.groupby(cluster_column):
        members = group["current_uniprot_accession"].tolist()
        for member in members[1:]:
            union(members[0], member)

    fold_to_members: dict[str, list[str]] = defaultdict(list)
    for _, row in assignments.iterrows():
        folds = [f for f in str(row["scop_fold_set"]).split(";") if f]
        for fold in folds:
            fold_to_members[fold].append(row["current_uniprot_accession"])
    for members in fold_to_members.values():
        for member in members[1:]:
            union(members[0], member)

    roots = [find(a) for a in accessions]
    unique_roots = sorted(set(roots))
    root_to_id = {root: f"joint_component_{i:03d}" for i, root in enumerate(unique_roots, 1)}
    return pd.Series([root_to_id[root] for root in roots], index=assignments.index)


def main() -> int:
    args = parse_args()
    benchmark_path = Path(args.benchmark)
    source_path = Path(args.source_features)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not benchmark_path.exists() or not source_path.exists():
        print("ERROR: benchmark or source-feature file is missing", file=sys.stderr)
        return 2

    benchmark = pd.read_csv(benchmark_path)
    source = pd.read_csv(source_path)
    validate_inputs(benchmark, source)
    benchmark["binary_label"] = pd.to_numeric(benchmark["binary_label"], errors="raise").astype(int)

    combined_fasta = output_dir / "all_181_sequences.fasta"
    sequence_manifest_path = output_dir / "sequence_manifest.csv"
    centralize_sequences(benchmark, source, combined_fasta, sequence_manifest_path)

    version = check_mmseqs()
    executable = shutil.which("mmseqs")
    assert executable is not None

    threshold_results = {}
    for identity in (args.primary_identity, args.sensitivity_identity):
        tag = threshold_tag(identity)
        tsv = run_mmseqs_easy_cluster(
            executable=executable,
            input_fasta=combined_fasta,
            output_dir=output_dir,
            identity=identity,
            coverage=args.coverage,
            cov_mode=args.cov_mode,
            threads=args.threads,
            force=args.force_mmseqs,
        )
        cluster_column = f"homology_cluster_{int(round(identity * 100))}"
        parsed = parse_cluster_tsv(
            tsv,
            set(benchmark["current_uniprot_accession"]),
            cluster_column,
        )
        parsed = parsed.rename(
            columns={"representative": f"representative_{cluster_column}"}
        )
        threshold_results[cluster_column] = parsed

    assignments = benchmark[
        [
            "current_uniprot_accession",
            "gene_minami",
            "jwid",
            "binary_label",
            "cohort_label",
            "sccs",
        ]
    ].copy()

    for cluster_column, parsed in threshold_results.items():
        assignments = assignments.merge(
            parsed,
            on="current_uniprot_accession",
            how="left",
            validate="one_to_one",
        )

    sccs_features = assignments["sccs"].apply(parse_sccs).apply(pd.Series)
    assignments = pd.concat([assignments, sccs_features], axis=1)

    # Primary homology-grouped fold assignment.
    primary_cluster = f"homology_cluster_{int(round(args.primary_identity * 100))}"
    fold_series, cv_audit, actual_folds = assign_grouped_folds(
        assignments,
        primary_cluster,
        args.requested_folds,
        args.random_seed,
    )
    assignments["homology_cv_fold"] = fold_series
    verify_no_cluster_leakage(assignments, primary_cluster, "homology_cv_fold")

    # Sensitivity-threshold fold assignment.
    sensitivity_cluster = f"homology_cluster_{int(round(args.sensitivity_identity * 100))}"
    sens_fold_series, sens_cv_audit, sens_actual_folds = assign_grouped_folds(
        assignments,
        sensitivity_cluster,
        args.requested_folds,
        args.random_seed,
    )
    assignments["homology_cv_fold_sensitivity"] = sens_fold_series
    verify_no_cluster_leakage(
        assignments, sensitivity_cluster, "homology_cv_fold_sensitivity"
    )

    # Joint components are audited for feasibility; not forced into final CV yet.
    assignments["joint_homology_fold_component"] = make_joint_components(
        assignments, primary_cluster
    )

    # Subset flags for planned experiments.
    assignments["analysis_subset_full"] = True
    assignments["analysis_subset_non_tim"] = ~assignments["is_tim_barrel"]
    assignments["analysis_subset_tim_only"] = assignments["is_tim_barrel"]

    # Audits.
    for cluster_column in threshold_results:
        audit = cluster_audit(assignments, cluster_column)
        audit.to_csv(output_dir / f"{cluster_column}_audit.csv", index=False)

    fold_audit_df = fold_audit(assignments)
    fold_audit_df.to_csv(output_dir / "scop_fold_audit.csv", index=False)
    cv_audit.to_csv(output_dir / "homology_cv_fold_audit.csv", index=False)
    sens_cv_audit.to_csv(
        output_dir / "homology_cv_fold_sensitivity_audit.csv", index=False
    )

    joint_audit = (
        assignments.groupby("joint_homology_fold_component")["binary_label"]
        .agg(protein_count="size", positive_count="sum")
        .reset_index()
    )
    joint_audit["negative_count"] = (
        joint_audit["protein_count"] - joint_audit["positive_count"]
    )
    joint_audit.to_csv(output_dir / "joint_component_audit.csv", index=False)

    assignment_path = output_dir / "fold_assignments_v1.0.csv"
    assignments.to_csv(assignment_path, index=False)

    cluster30_audit = pd.read_csv(output_dir / f"{primary_cluster}_audit.csv")
    cluster40_audit = pd.read_csv(output_dir / f"{sensitivity_cluster}_audit.csv")

    summary = {
        "mmseqs_version": version,
        "protein_count": len(assignments),
        "positive_count": int((assignments["binary_label"] == 1).sum()),
        "negative_count": int((assignments["binary_label"] == 0).sum()),
        "clustering": {
            "primary_identity": args.primary_identity,
            "sensitivity_identity": args.sensitivity_identity,
            "coverage": args.coverage,
            "cov_mode": args.cov_mode,
            "coverage_interpretation": (
                "bidirectional coverage of the longer sequence"
                if args.cov_mode == 0
                else "see MMseqs2 cov-mode definition"
            ),
            "primary_cluster_count": int(cluster30_audit[primary_cluster].nunique()),
            "primary_singleton_count": int((cluster30_audit["cluster_size"] == 1).sum()),
            "primary_mixed_label_cluster_count": int(
                cluster30_audit["mixed_label_cluster"].sum()
            ),
            "sensitivity_cluster_count": int(cluster40_audit[sensitivity_cluster].nunique()),
            "sensitivity_singleton_count": int((cluster40_audit["cluster_size"] == 1).sum()),
            "sensitivity_mixed_label_cluster_count": int(
                cluster40_audit["mixed_label_cluster"].sum()
            ),
        },
        "homology_cv": {
            "requested_folds": args.requested_folds,
            "actual_primary_folds": actual_folds,
            "actual_sensitivity_folds": sens_actual_folds,
            "primary_cluster_leakage": False,
            "sensitivity_cluster_leakage": False,
        },
        "fold_structure": {
            "unique_scop_folds": int(
                len({f for value in assignments["scop_fold_set"] for f in str(value).split(";") if f})
            ),
            "unassigned_scop_count": int((assignments["scop_fold_set"] == "").sum()),
            "tim_barrel_count": int(assignments["is_tim_barrel"].sum()),
            "non_tim_count": int((~assignments["is_tim_barrel"]).sum()),
            "joint_component_count": int(assignments["joint_homology_fold_component"].nunique()),
            "largest_joint_component": int(joint_audit["protein_count"].max()),
        },
        "status": "READY_FOR_BASELINE_MODELING_GROUPED_CV",
    }
    (output_dir / "stage07_summary.json").write_text(json.dumps(summary, indent=2))

    methods = f"""# Stage 07 Grouping Methods v1.0

## Sequence clustering
Canonical sequences for all 181 proteins were centralized into one FASTA file.
MMseqs2 version: `{version}`.

Primary clustering used:

```bash
mmseqs easy-cluster all_181_sequences.fasta cluster_id30 tmp \\
  --min-seq-id {args.primary_identity} \\
  -c {args.coverage} \\
  --cov-mode {args.cov_mode} \\
  --threads {args.threads}
```

A sensitivity clustering used minimum sequence identity
`{args.sensitivity_identity}` with the same coverage definition.

Coverage mode 0 requires the aligned region to cover at least {args.coverage:.0%}
of the longer sequence in both directions. This full-length criterion was chosen
to avoid grouping proteins solely because they share one short domain.

## Fold parsing
SCCS strings were parsed into SCOP class, fold, and superfamily sets. Proteins
with multiple SCCS entries retained all fold assignments. TIM barrels were
defined by presence of SCOP fold `c.1`.

## Cross-validation assignment
Complete primary homology clusters were assigned to {actual_folds} folds using a
deterministic greedy balancing procedure based only on cluster size and class
labels. No structural or sequence predictor values were used. A leakage audit
confirmed that no homology cluster crossed folds.

## Joint constraints
Connected components were created linking proteins that shared either a primary
homology cluster or any SCOP fold. These components are retained for feasibility
audit of a future joint homology-and-fold-held-out analysis; they are not yet
used as the primary cross-validation scheme.
"""
    (output_dir / "grouping_methods_v1.0.md").write_text(methods)

    print("Stage 07 complete")
    print(f"  MMseqs2 version:       {version}")
    print(f"  Proteins:              {len(assignments)}")
    print(f"  30% clusters:          {cluster30_audit[primary_cluster].nunique()}")
    print(f"  40% clusters:          {cluster40_audit[sensitivity_cluster].nunique()}")
    print(f"  Primary CV folds:      {actual_folds}")
    print(f"  TIM / non-TIM:         {assignments['is_tim_barrel'].sum()} / {(~assignments['is_tim_barrel']).sum()}")
    print(f"  Joint components:      {assignments['joint_homology_fold_component'].nunique()}")
    print(f"  Assignments:           {assignment_path}")
    print(f"  Summary:               {output_dir / 'stage07_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
