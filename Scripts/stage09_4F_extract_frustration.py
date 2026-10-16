#!/usr/bin/env python3

"""
Stage 09.4F — Energetic Frustration Extraction

Dataset:
    Frozen GroEL/GroES benchmark v1.1
    181 E. coli proteins

Method:
    Python Frustratometer 0.3.2
    AWSEM energetic model

Calculations:
    Mutational frustration
    Configurational frustration

Analysis contacts:
    C-beta distance < 8 Angstrom
    Sequence separation >= 4
    Unique residue pairs (i < j)

Outputs:
    results/stage09_4F/raw/*.npz
    results/stage09_4F/frustration_features.csv
    results/stage09_4F/frustration_qc.csv
    results/stage09_4F/extraction_log.csv

Original benchmark and PDB files are never modified.
"""

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
import traceback
import warnings

from pathlib import Path

import numpy as np
import pandas as pd

import frustratometer


ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "results/stage09_4F"
RAW_DIR = RESULTS / "raw"

MANIFEST = RESULTS / "structure_manifest.csv"

FEATURES_FILE = RESULTS / "frustration_features.csv"
QC_FILE = RESULTS / "frustration_qc.csv"
LOG_FILE = RESULTS / "extraction_log.csv"

CONFIG_FILE = RESULTS / "extraction_config.json"

CONTACT_CUTOFF = 8.0
MIN_SEQUENCE_SEPARATION = 4

# Conventional Frustratometer thresholds.
# Interpretation must be validated against the implementation.
HIGH_THRESHOLD = -1.0
MIN_THRESHOLD = 0.78

# Configurational calculation defaults to 4000 decoys.
CONFIGURATIONAL_DECOYS = 4000

SEED = 20261012


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def append_log(record):
    df = pd.DataFrame([record])

    df.to_csv(
        LOG_FILE,
        mode="a",
        header=not LOG_FILE.exists(),
        index=False
    )


def save_table(path, records):
    if not records:
        return

    df = pd.DataFrame(records)
    df.to_csv(path, index=False)


def safe_float(value):
    if np.isfinite(value):
        return float(value)

    return np.nan


def summarize_values(values, prefix):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    output = {
        f"{prefix}_n": int(len(values)),
        f"{prefix}_mean": np.nan,
        f"{prefix}_std": np.nan,
        f"{prefix}_high_fraction": np.nan,
        f"{prefix}_minimal_fraction": np.nan,
    }

    if len(values) == 0:
        return output

    output.update({
        f"{prefix}_mean": safe_float(np.mean(values)),
        f"{prefix}_std": safe_float(np.std(values)),
        f"{prefix}_high_fraction": safe_float(
            np.mean(values < HIGH_THRESHOLD)
        ),
        f"{prefix}_minimal_fraction": safe_float(
            np.mean(values > MIN_THRESHOLD)
        ),
    })

    return output


def residue_high_fraction(n_residues, ii, jj, values):
    """
    Fraction of residues participating in at least one
    highly frustrated analyzed contact.

    Residues without an analyzed contact are included
    in the denominator.
    """

    if n_residues == 0:
        return np.nan

    high = values < HIGH_THRESHOLD

    involved = np.zeros(n_residues, dtype=bool)

    involved[ii[high]] = True
    involved[jj[high]] = True

    return float(involved.mean())


def extract_features(accession, model, mut, conf):
    distances = np.asarray(model.distance_matrix)

    n = model.N

    if distances.shape != (n, n):
        raise ValueError("Unexpected distance matrix dimensions")

    if mut.shape != (n, n):
        raise ValueError("Unexpected mutational dimensions")

    if conf.shape != (n, n):
        raise ValueError("Unexpected configurational dimensions")

    i, j = np.indices((n, n))

    sequence_separation = np.abs(i - j)

    contacts = (
        np.isfinite(distances)
        & (distances < CONTACT_CUTOFF)
        & (sequence_separation >= MIN_SEQUENCE_SEPARATION)
        & (i < j)
    )

    ii, jj = np.where(contacts)

    mut_values = mut[ii, jj]
    conf_values = conf[ii, jj]

    mut_valid = np.isfinite(mut_values)
    conf_valid = np.isfinite(conf_values)

    joint_valid = mut_valid & conf_valid

    if len(ii) == 0:
        raise ValueError("Protein has zero analysis contacts")

    # Require complete numerical coverage for this initial extraction.
    # A protein failing this check is logged, not silently imputed.
    if not np.all(joint_valid):
        raise ValueError(
            f"Incomplete contact coverage: "
            f"{joint_valid.sum()}/{len(joint_valid)}"
        )

    features = {
        "accession": accession,
        "n_residues": n,
        "n_contacts": len(ii),
    }

    for name, values in [
        ("mut", mut_values),
        ("conf", conf_values),
    ]:
        features.update(
            summarize_values(values, name)
        )

        features[f"{name}_high_residue_fraction"] = (
            residue_high_fraction(n, ii, jj, values)
        )

        long_range = (
            (jj - ii) >= 24
        )

        if np.any(long_range):
            features[f"{name}_long_range_high_fraction"] = (
                float(
                    np.mean(
                        values[long_range] < HIGH_THRESHOLD
                    )
                )
            )
        else:
            features[f"{name}_long_range_high_fraction"] = np.nan

    qc = {
        "accession": accession,
        "n_residues": n,
        "n_contacts": len(ii),
        "mut_finite_contacts": int(mut_valid.sum()),
        "conf_finite_contacts": int(conf_valid.sum()),
        "joint_finite_contacts": int(joint_valid.sum()),
        "mut_finite_fraction": float(mut_valid.mean()),
        "conf_finite_fraction": float(conf_valid.mean()),
        "distance_cutoff": CONTACT_CUTOFF,
        "min_sequence_separation": MIN_SEQUENCE_SEPARATION,
    }

    raw = {
        "accession": np.array(accession),
        "sequence": np.array(model.sequence),
        "distance_matrix": distances.astype(np.float32),
        "mutational_matrix": mut.astype(np.float32),
        "configurational_matrix": conf.astype(np.float32),
        "contact_i": ii.astype(np.int32),
        "contact_j": jj.astype(np.int32),
        "contact_distance": distances[ii, jj].astype(np.float32),
        "contact_mutational": mut_values.astype(np.float32),
        "contact_configurational": conf_values.astype(np.float32),
    }

    return features, qc, raw


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N proteins"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Recalculate existing results"
    )

    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)

    manifest = pd.read_csv(MANIFEST)

    if len(manifest) != 181:
        raise ValueError("Manifest must contain 181 proteins")

    if not manifest["accession"].is_unique:
        raise ValueError("Duplicate accessions in manifest")

    if not manifest["status"].eq("found").all():
        raise ValueError("Manifest contains missing structures")

    config = {
        "stage": "09.4F",
        "benchmark": "v1.1",
        "frustratometer_version": "0.3.2",
        "energy_model": "AWSEM",
        "contact_cutoff_angstrom": CONTACT_CUTOFF,
        "minimum_sequence_separation": MIN_SEQUENCE_SEPARATION,
        "high_threshold": HIGH_THRESHOLD,
        "minimal_threshold": MIN_THRESHOLD,
        "configurational_default_decoys": CONFIGURATIONAL_DECOYS,
        "seed": SEED,
        "python_version": sys.version,
        "platform": platform.platform(),
    }

    if not CONFIG_FILE.exists():
        CONFIG_FILE.write_text(
            json.dumps(config, indent=2)
        )

    np.random.seed(SEED)

    print("=" * 72, flush=True)
    print("STAGE 09.4F — ENERGETIC FRUSTRATION EXTRACTION", flush=True)
    print("=" * 72, flush=True)

    print(f"Benchmark proteins: {len(manifest)}", flush=True)
    print(f"Analysis contact cutoff: {CONTACT_CUTOFF} Å", flush=True)
    print(f"Minimum sequence separation: {MIN_SEQUENCE_SEPARATION}", flush=True)
    print(f"Results directory: {RESULTS}", flush=True)

    if args.limit is not None:
        manifest = manifest.head(args.limit)

    feature_records = []
    qc_records = []

    # Preserve existing feature/QC rows on resumed runs.
    if FEATURES_FILE.exists() and not args.force:
        feature_records = pd.read_csv(
            FEATURES_FILE
        ).to_dict("records")

    if QC_FILE.exists() and not args.force:
        qc_records = pd.read_csv(
            QC_FILE
        ).to_dict("records")

    completed = {
        row["accession"]
        for row in feature_records
        if (RAW_DIR / f"{row['accession']}.npz").exists()
    }

    for index, row in enumerate(manifest.itertuples(index=False), 1):
        accession = row.accession

        if accession in completed and not args.force:
            print(
                f"[{index}/{len(manifest)}] {accession}: SKIPPED",
                flush=True
            )
            continue

        start = time.perf_counter()

        print(
            f"\n[{index}/{len(manifest)}] {accession}",
            flush=True
        )

        try:
            # Deterministic, accession-specific random seed.
            # Independent of processing order and interrupted runs.
            accession_hash = hashlib.sha256(
                accession.encode("utf-8")
            ).digest()

            protein_seed = (
                SEED + int.from_bytes(accession_hash[:4], "big")
            ) % (2**32)

            np.random.seed(protein_seed)

            pdb_path = ROOT / row.pdb_path

            if not pdb_path.exists():
                raise FileNotFoundError(pdb_path)

            if sha256_file(pdb_path) != row.pdb_sha256:
                raise ValueError("PDB checksum mismatch")

            structure = frustratometer.Structure.full_pdb(
                str(pdb_path),
                repair_pdb=False
            )

            if len(structure.sequence) != int(row.seq_length):
                raise ValueError(
                    f"Sequence length mismatch: "
                    f"{len(structure.sequence)} vs {row.seq_length}"
                )

            model = frustratometer.AWSEM(structure)

            if model.N != int(row.structure_ca_length):
                raise ValueError(
                    f"Structural length mismatch: "
                    f"{model.N} vs {row.structure_ca_length}"
                )

            t0 = time.perf_counter()

            mut = np.asarray(
                model.frustration(kind="mutational")
            )

            mut_seconds = time.perf_counter() - t0

            t0 = time.perf_counter()

            conf = np.asarray(
                model.frustration(kind="configurational")
            )

            conf_seconds = time.perf_counter() - t0

            features, qc, raw = extract_features(
                accession, model, mut, conf
            )

            qc["random_seed"] = protein_seed
            raw["random_seed"] = np.array(protein_seed, dtype=np.uint64)
            raw["configurational_decoys"] = np.array(
                CONFIGURATIONAL_DECOYS,
                dtype=np.int32
            )
            

            raw_path = RAW_DIR / f"{accession}.npz"
            temp_path = RAW_DIR / f"{accession}.tmp.npz"

            np.savez_compressed(temp_path, **raw)
            os.replace(temp_path, raw_path)

            # Remove an earlier row if --force was used.
            feature_records = [
                r for r in feature_records
                if r["accession"] != accession
            ]

            qc_records = [
                r for r in qc_records
                if r["accession"] != accession
            ]

            feature_records.append(features)
            qc_records.append(qc)

            save_table(FEATURES_FILE, feature_records)
            save_table(QC_FILE, qc_records)

            elapsed = time.perf_counter() - start

            append_log({
                "accession": accession,
                "status": "success",
                "mutational_seconds": mut_seconds,
                "configurational_seconds": conf_seconds,
                "total_seconds": elapsed,
                "error": "",
            })

            print(
                f"SUCCESS | "
                f"N={model.N} | "
                f"contacts={features['n_contacts']} | "
                f"mut={mut_seconds:.2f}s | "
                f"conf={conf_seconds:.2f}s | "
                f"total={elapsed:.2f}s",
                flush=True
            )

        except Exception as exc:
            elapsed = time.perf_counter() - start

            append_log({
                "accession": accession,
                "status": "failed",
                "mutational_seconds": np.nan,
                "configurational_seconds": np.nan,
                "total_seconds": elapsed,
                "error": repr(exc),
            })

            print(
                f"FAILED: {accession}: {exc}",
                flush=True
            )

            traceback.print_exc()

        finally:
            for variable in [
                "structure", "model", "mut", "conf",
                "features", "qc", "raw"
            ]:
                if variable in locals():
                    del locals()[variable]

            gc.collect()

    print("\n" + "=" * 72, flush=True)
    print("EXTRACTION RUN FINISHED", flush=True)
    print("=" * 72, flush=True)

    if FEATURES_FILE.exists():
        final = pd.read_csv(FEATURES_FILE)

        print(
            f"Successful proteins: {final['accession'].nunique()}/181",
            flush=True
        )

    print(f"Features: {FEATURES_FILE}", flush=True)
    print(f"QC: {QC_FILE}", flush=True)
    print(f"Log: {LOG_FILE}", flush=True)


if __name__ == "__main__":
    main()
