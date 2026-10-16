#!/usr/bin/env python3
"""
Extract interpretable structural features from AlphaFold PDB models.

Input:
    data/download_manifest.csv

Expected files:
    data/structures/<ACCESSION>.pdb
    data/sequences/<ACCESSION>.fasta

Outputs:
    data/features/class_iv_structural_features.csv
    data/features/class_iv_feature_failures.csv

Features include:
- sequence/structure length agreement
- AlphaFold pLDDT summaries
- radius of gyration and molecular dimensions
- CA-contact density and contact-order metrics
- local/medium/long-range contact fractions
- CA-based secondary-structure fractions using Biotite P-SEA
- a simple sequence-cut contact bottleneck proxy for domain separation

Example:
    python3 03_extract_structural_features.py \
        --manifest data/download_manifest.csv \
        --output-dir data/features
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

import biotite.structure as struc
from biotite.structure.io import load_structure


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract interpretable features from AlphaFold structures."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to data/download_manifest.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/features",
        help="Directory for feature outputs.",
    )
    parser.add_argument(
        "--output-name",
        default="class_iv_structural_features.csv",
        help="Feature CSV filename.",
    )
    parser.add_argument(
        "--failures-name",
        default="class_iv_feature_failures.csv",
        help="Failure CSV filename.",
    )
    parser.add_argument(
        "--contact-cutoff",
        type=float,
        default=8.0,
        help="CA-CA contact cutoff in angstroms (default: 8.0).",
    )
    parser.add_argument(
        "--min-separation",
        type=int,
        default=3,
        help="Minimum sequence separation for a nonlocal contact.",
    )
    parser.add_argument(
        "--long-range-separation",
        type=int,
        default=24,
        help="Minimum sequence separation for a long-range contact.",
    )
    parser.add_argument(
        "--break-distance",
        type=float,
        default=4.5,
        help="Consecutive CA distance marking a chain break.",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def read_fasta_length(path: Path) -> int:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and not line.startswith(">")
    ]
    sequence = "".join(lines).replace(" ", "")
    if not sequence:
        raise ValueError(f"No sequence found in FASTA: {path}")
    return len(sequence)


def safe_fraction(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def finite_or_nan(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def load_single_model(path: Path):
    atoms = load_structure(path, model=1, extra_fields=["b_factor"])
    if hasattr(atoms, "stack_depth"):
        atoms = atoms[0]
    amino = atoms[struc.filter_amino_acids(atoms)]
    if amino.array_length() == 0:
        raise ValueError("No amino-acid atoms found")
    return amino


def unique_residue_ca_atoms(amino):
    ca = amino[amino.atom_name == "CA"]
    if ca.array_length() == 0:
        raise ValueError("No CA atoms found")

    # AlphaFold PDBs should have one CA per residue. Keep the first if a file
    # unexpectedly contains duplicate residue identifiers.
    keys = list(
        zip(
            ca.chain_id.tolist(),
            ca.res_id.tolist(),
            ca.ins_code.tolist(),
        )
    )
    keep = np.ones(len(keys), dtype=bool)
    seen = set()
    for i, key in enumerate(keys):
        if key in seen:
            keep[i] = False
        else:
            seen.add(key)
    return ca[keep]


def principal_axis_features(coords: np.ndarray) -> dict[str, float]:
    centered = coords - coords.mean(axis=0)
    covariance = np.cov(centered, rowvar=False, bias=True)
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalues = np.sort(np.clip(eigenvalues, 0.0, None))[::-1]

    l1, l2, l3 = eigenvalues
    total = eigenvalues.sum()

    return {
        "principal_variance_1": float(l1),
        "principal_variance_2": float(l2),
        "principal_variance_3": float(l3),
        "axis_ratio_1_to_2": safe_fraction(l1, l2),
        "axis_ratio_1_to_3": safe_fraction(l1, l3),
        "asphericity": (
            float(
                ((l1 - l2) ** 2 + (l2 - l3) ** 2 + (l3 - l1) ** 2)
                / (2.0 * total**2)
            )
            if total > 0
            else float("nan")
        ),
    }


def secondary_structure_features(amino, ca_count: int) -> dict[str, float]:
    """
    Biotite annotate_sse() uses the CA-based P-SEA algorithm.
    Codes are 'a' (helix), 'b' (sheet), and 'c' (coil).
    """
    try:
        sse = np.asarray(struc.annotate_sse(amino))
    except Exception as exc:
        return {
            "ss_helix_fraction": float("nan"),
            "ss_sheet_fraction": float("nan"),
            "ss_coil_fraction": float("nan"),
            "ss_assignment_count": 0,
            "ss_note": f"secondary_structure_failed:{exc}",
        }

    if len(sse) == 0:
        return {
            "ss_helix_fraction": float("nan"),
            "ss_sheet_fraction": float("nan"),
            "ss_coil_fraction": float("nan"),
            "ss_assignment_count": 0,
            "ss_note": "secondary_structure_empty",
        }

    # annotate_sse() returns one assignment per amino-acid residue.
    n = len(sse)
    return {
        "ss_helix_fraction": float(np.mean(sse == "a")),
        "ss_sheet_fraction": float(np.mean(sse == "b")),
        "ss_coil_fraction": float(np.mean(sse == "c")),
        "ss_assignment_count": int(n),
        "ss_note": "" if n == ca_count else f"sse_count_{n}_ca_count_{ca_count}",
    }


def contact_features(
    coords: np.ndarray,
    cutoff: float,
    min_separation: int,
    long_range_separation: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    n = len(coords)
    distances = squareform(pdist(coords))
    indices = np.arange(n)
    sequence_separation = np.abs(indices[:, None] - indices[None, :])

    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    eligible = upper & (sequence_separation >= min_separation)
    contacts = eligible & (distances <= cutoff)

    contact_i, contact_j = np.where(contacts)
    separations = sequence_separation[contact_i, contact_j].astype(float)
    contact_count = len(separations)
    eligible_pair_count = int(np.count_nonzero(eligible))

    local_upper = max(min_separation, 12)
    local_count = int(
        np.count_nonzero(
            contacts
            & (sequence_separation >= min_separation)
            & (sequence_separation < local_upper)
        )
    )
    medium_count = int(
        np.count_nonzero(
            contacts
            & (sequence_separation >= local_upper)
            & (sequence_separation < long_range_separation)
        )
    )
    long_count = int(
        np.count_nonzero(
            contacts & (sequence_separation >= long_range_separation)
        )
    )

    degree = np.zeros(n, dtype=int)
    if contact_count:
        np.add.at(degree, contact_i, 1)
        np.add.at(degree, contact_j, 1)

    features = {
        "eligible_ca_pair_count": eligible_pair_count,
        "ca_contact_count": int(contact_count),
        "ca_contact_density_per_residue": safe_fraction(contact_count, n),
        "ca_contact_pair_fraction": safe_fraction(contact_count, eligible_pair_count),
        "mean_contact_degree": float(degree.mean()),
        "median_contact_degree": float(np.median(degree)),
        "max_contact_degree": int(degree.max()) if n else 0,
        "zero_contact_residue_fraction": float(np.mean(degree == 0)),
        "mean_contact_sequence_separation": (
            float(np.mean(separations)) if contact_count else float("nan")
        ),
        "median_contact_sequence_separation": (
            float(np.median(separations)) if contact_count else float("nan")
        ),
        "max_contact_sequence_separation": (
            float(np.max(separations)) if contact_count else float("nan")
        ),
        "relative_contact_order": (
            float(np.sum(separations) / (n * contact_count))
            if n > 0 and contact_count > 0
            else float("nan")
        ),
        "absolute_contact_order": (
            float(np.mean(separations)) if contact_count else float("nan")
        ),
        "local_contact_count": local_count,
        "medium_range_contact_count": medium_count,
        "long_range_contact_count": long_count,
        "local_contact_fraction": safe_fraction(local_count, contact_count),
        "medium_range_contact_fraction": safe_fraction(medium_count, contact_count),
        "long_range_contact_fraction": safe_fraction(long_count, contact_count),
    }
    return features, contacts, distances


def sequence_cut_bottleneck(
    contacts: np.ndarray,
    minimum_side_fraction: float = 0.20,
) -> dict[str, float]:
    """
    Find the sequence cut with the fewest cross-cut contacts after normalizing
    by the size of the smaller side.

    A low value can indicate a weakly connected domain boundary, but it is only
    a proxy and should not be interpreted as a formal domain assignment.
    """
    n = contacts.shape[0]
    if n < 10:
        return {
            "minimum_cut_cross_contacts": float("nan"),
            "minimum_cut_cross_contact_density": float("nan"),
            "best_cut_position": float("nan"),
            "best_cut_fraction": float("nan"),
        }

    start = max(1, int(math.ceil(n * minimum_side_fraction)))
    stop = min(n - 1, int(math.floor(n * (1.0 - minimum_side_fraction))))

    best = None
    for cut in range(start, stop + 1):
        cross_count = int(np.count_nonzero(contacts[:cut, cut:]))
        smaller_side = min(cut, n - cut)
        density = cross_count / smaller_side if smaller_side else float("inf")

        candidate = (density, cross_count, cut)
        if best is None or candidate < best:
            best = candidate

    if best is None:
        return {
            "minimum_cut_cross_contacts": float("nan"),
            "minimum_cut_cross_contact_density": float("nan"),
            "best_cut_position": float("nan"),
            "best_cut_fraction": float("nan"),
        }

    density, cross_count, cut = best
    return {
        "minimum_cut_cross_contacts": int(cross_count),
        "minimum_cut_cross_contact_density": float(density),
        "best_cut_position": int(cut),
        "best_cut_fraction": float(cut / n),
    }


def extract_one(
    row: pd.Series,
    contact_cutoff: float,
    min_separation: int,
    long_range_separation: int,
    break_distance: float,
) -> dict[str, Any]:
    accession = clean_text(row.get("accession") or row.get("uniprot_accession"))
    gene = clean_text(row.get("gene"))
    b_number = clean_text(row.get("b_number"))

    pdb_path = Path(clean_text(row.get("pdb_path")))
    fasta_path = Path(clean_text(row.get("fasta_path")))

    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB not found: {pdb_path}")
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")

    amino = load_single_model(pdb_path)
    ca = unique_residue_ca_atoms(amino)

    coords = np.asarray(ca.coord, dtype=float)
    plddt = np.asarray(ca.b_factor, dtype=float)
    n = len(coords)
    fasta_length = read_fasta_length(fasta_path)

    if n < 3:
        raise ValueError(f"Only {n} CA atoms found")

    center = coords.mean(axis=0)
    radial_squared = np.sum((coords - center) ** 2, axis=1)
    radius_of_gyration = float(np.sqrt(np.mean(radial_squared)))

    pairwise_condensed = pdist(coords)
    maximum_span = float(pairwise_condensed.max()) if len(pairwise_condensed) else 0.0

    consecutive_distances = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    chain_break_count = int(np.count_nonzero(consecutive_distances > break_distance))

    result: dict[str, Any] = {
        "accession": accession,
        "gene": gene,
        "b_number": b_number,
        "pdb_path": str(pdb_path),
        "fasta_path": str(fasta_path),
        "fasta_length": fasta_length,
        "structure_ca_length": n,
        "fasta_structure_length_match": bool(fasta_length == n),
        "chain_count": int(len(np.unique(ca.chain_id))),
        "chain_break_count": chain_break_count,
        "mean_consecutive_ca_distance": float(np.mean(consecutive_distances)),
        "max_consecutive_ca_distance": float(np.max(consecutive_distances)),
        "radius_of_gyration": radius_of_gyration,
        "radius_of_gyration_normalized_n_cuberoot": safe_fraction(
            radius_of_gyration, n ** (1 / 3)
        ),
        "maximum_ca_span": maximum_span,
        "compactness_n_over_rg_cubed": safe_fraction(n, radius_of_gyration**3),
        "mean_plddt": float(np.mean(plddt)),
        "median_plddt": float(np.median(plddt)),
        "minimum_plddt": float(np.min(plddt)),
        "maximum_plddt": float(np.max(plddt)),
        "plddt_std": float(np.std(plddt)),
        "fraction_plddt_below_50": float(np.mean(plddt < 50)),
        "fraction_plddt_below_70": float(np.mean(plddt < 70)),
        "fraction_plddt_at_least_90": float(np.mean(plddt >= 90)),
        "contact_cutoff_angstrom": contact_cutoff,
        "minimum_contact_sequence_separation": min_separation,
        "long_range_sequence_separation": long_range_separation,
    }

    result.update(principal_axis_features(coords))
    result.update(secondary_structure_features(amino, n))

    contact_result, contacts, distances = contact_features(
        coords,
        contact_cutoff,
        min_separation,
        long_range_separation,
    )
    result.update(contact_result)
    result.update(sequence_cut_bottleneck(contacts))

    return result


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_path = output_dir / args.output_name
    failures_path = output_dir / args.failures_name

    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}", file=sys.stderr)
        return 2

    manifest = pd.read_csv(manifest_path)

    if "overall_status" in manifest.columns:
        eligible = manifest[
            manifest["overall_status"].astype(str).str.lower().isin(
                ["complete", "partial"]
            )
        ].copy()
    else:
        eligible = manifest.copy()

    if eligible.empty:
        print("ERROR: No eligible manifest rows.", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    features: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    print(f"Eligible structures: {len(eligible)}")
    for index, (_, row) in enumerate(eligible.iterrows(), start=1):
        accession = clean_text(
            row.get("accession") or row.get("uniprot_accession")
        )
        gene = clean_text(row.get("gene"))
        print(f"[{index:02d}/{len(eligible)}] {accession} {gene}", flush=True)

        try:
            result = extract_one(
                row,
                contact_cutoff=args.contact_cutoff,
                min_separation=args.min_separation,
                long_range_separation=args.long_range_separation,
                break_distance=args.break_distance,
            )
            features.append(result)
            print(
                f"    length={result['structure_ca_length']} "
                f"mean_pLDDT={result['mean_plddt']:.1f} "
                f"RCO={result['relative_contact_order']:.4f}",
                flush=True,
            )
        except Exception as exc:
            failures.append(
                {
                    "accession": accession,
                    "gene": gene,
                    "b_number": clean_text(row.get("b_number")),
                    "pdb_path": clean_text(row.get("pdb_path")),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)

    feature_df = pd.DataFrame(features)
    failure_df = pd.DataFrame(
        failures,
        columns=[
            "accession",
            "gene",
            "b_number",
            "pdb_path",
            "error_type",
            "error",
        ],
    )

    feature_df.to_csv(output_path, index=False)
    failure_df.to_csv(failures_path, index=False)

    print("\nFeature extraction summary:")
    print(f"  successful: {len(feature_df)}")
    print(f"  failed:     {len(failure_df)}")
    print(f"\nFeatures: {output_path}")
    print(f"Failures: {failures_path}")

    if len(feature_df):
        mismatches = int(
            (~feature_df["fasta_structure_length_match"].astype(bool)).sum()
        )
        print(f"  sequence/structure length mismatches: {mismatches}")

    if failures:
        return 1

    print(f"\nComplete: {len(feature_df)}/{len(eligible)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
