#!/usr/bin/env python3
"""
Download canonical UniProt sequences and current AlphaFold DB structures.

Input:
    fujiwara_class_iv_identifiers.csv

Outputs:
    data/sequences/<ACCESSION>.fasta
    data/structures/<ACCESSION>.cif
    data/structures/<ACCESSION>.pdb              (optional, default enabled)
    data/alphafold_metadata/<ACCESSION>.json
    data/download_manifest.csv
    data/download_failures.csv

The script requests AlphaFold's prediction API first and uses the download URLs
returned by the API. This avoids hard-coding a model version such as v4 or v6.

Example:
    python3 02_download_sequences_structures.py \
        --input fujiwara_class_iv_identifiers.csv \
        --output-dir data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import pandas as pd
import requests

UNIPROT_FASTA_URL = "https://rest.uniprot.org/uniprotkb/{accession}.fasta"

# AlphaFold has moved its public site while retaining API compatibility.
# Try the current host first, then the legacy EBI host.
ALPHAFOLD_API_ENDPOINTS = (
    "https://alphafold.com/api/prediction/{accession}",
    "https://alphafold.ebi.ac.uk/api/prediction/{accession}",
)

ACCESSION_RE = re.compile(r"^[A-Z0-9]+(?:-\d+)?$")
FASTA_SEQUENCE_RE = re.compile(r"^[A-Z*]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download UniProt FASTA and AlphaFold DB files."
    )
    parser.add_argument("--input", required=True, help="Identifier CSV.")
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Root output directory (default: data).",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Manifest CSV path; defaults to OUTPUT_DIR/download_manifest.csv.",
    )
    parser.add_argument(
        "--failures",
        default=None,
        help="Failure CSV path; defaults to OUTPUT_DIR/download_failures.csv.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.20,
        help="Delay between proteins in seconds.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=4,
        help="Maximum attempts per HTTP request.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload existing valid files.",
    )
    parser.add_argument(
        "--skip-pdb",
        action="store_true",
        help="Download mmCIF only and skip PDB files.",
    )
    parser.add_argument(
        "--include-unconfirmed",
        action="store_true",
        help="Process rows not marked include=yes and mapping_status=confirmed.",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_fasta_text(text: str, expected_accession: str) -> tuple[bool, int, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2 or not lines[0].startswith(">"):
        return False, 0, "Missing FASTA header or sequence"

    sequence = "".join(lines[1:]).replace(" ", "").upper()
    if not sequence or not FASTA_SEQUENCE_RE.fullmatch(sequence):
        return False, len(sequence), "Invalid FASTA sequence characters"

    # Accession may appear in forms such as sp|P77475|...
    if expected_accession not in lines[0]:
        return True, len(sequence), "valid_sequence_header_accession_not_found"

    return True, len(sequence), ""


def looks_like_cif(data: bytes) -> bool:
    head = data[:5000].decode("utf-8", errors="ignore")
    return (
        head.lstrip().startswith("data_")
        and ("_atom_site." in head or "_entry.id" in head)
    )


def looks_like_pdb(data: bytes) -> bool:
    head = data[:10000].decode("utf-8", errors="ignore")
    return any(
        line.startswith(("ATOM  ", "HETATM", "HEADER", "MODEL "))
        for line in head.splitlines()
    )


def request_with_retries(
    session: requests.Session,
    url: str,
    *,
    timeout: float,
    retries: int,
    accept: str | None = None,
) -> requests.Response:
    headers = {}
    if accept:
        headers["Accept"] = accept

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = session.get(
                url,
                headers=headers,
                timeout=timeout,
                allow_redirects=True,
            )

            if response.status_code == 404:
                return response

            if response.status_code in {429, 500, 502, 503, 504}:
                retry_after = response.headers.get("Retry-After")
                wait = (
                    float(retry_after)
                    if retry_after and retry_after.replace(".", "", 1).isdigit()
                    else min(2 ** (attempt - 1), 12)
                )
                if attempt < retries:
                    time.sleep(wait)
                    continue

            response.raise_for_status()
            return response

        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 12))

    raise RuntimeError(f"Request failed after {retries} attempts: {url}: {last_error}")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(data)
    temporary.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def existing_file_valid(path: Path, kind: str, accession: str = "") -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False

    if kind == "fasta":
        valid, _, _ = validate_fasta_text(
            path.read_text(encoding="utf-8", errors="replace"),
            accession,
        )
        return valid

    data = path.read_bytes()
    if kind == "cif":
        return looks_like_cif(data)
    if kind == "pdb":
        return looks_like_pdb(data)
    if kind == "json":
        try:
            json.loads(data)
            return True
        except json.JSONDecodeError:
            return False
    return False


def flatten_url_candidates(value: Any, key_path: str = "") -> list[tuple[str, str]]:
    """Recursively collect HTTP(S) URLs from arbitrary API JSON."""
    found: list[tuple[str, str]] = []

    if isinstance(value, dict):
        for key, child in value.items():
            next_path = f"{key_path}.{key}" if key_path else str(key)
            found.extend(flatten_url_candidates(child, next_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            next_path = f"{key_path}[{index}]"
            found.extend(flatten_url_candidates(child, next_path))
    elif isinstance(value, str) and value.startswith(("http://", "https://")):
        found.append((key_path, value))

    return found


def score_download_url(key_path: str, url: str, kind: str) -> int:
    key = key_path.lower()
    lower_url = url.lower()
    path = urlparse(url).path.lower()
    score = 0

    extension_map = {
        "cif": (".cif", ".cif.gz"),
        "pdb": (".pdb", ".pdb.gz"),
        "confidence": (".json", ".json.gz"),
    }

    if path.endswith(extension_map[kind]):
        score += 100

    if kind == "cif":
        for token in ("cifurl", "cif_url", "mmcif", "cif"):
            if token in key:
                score += 40
        if "bcif" in key or path.endswith(".bcif"):
            score -= 100
    elif kind == "pdb":
        for token in ("pdburl", "pdb_url", "pdb"):
            if token in key:
                score += 40
    elif kind == "confidence":
        for token in (
            "confidence",
            "plddt",
            "pae",
            "predictionconfidence",
            "confidenceurl",
        ):
            if token in key:
                score += 40
        # Avoid selecting generic API or coordinate JSON URLs.
        if any(token in key for token in ("api", "entry", "metadata")):
            score -= 10

    if "download" in key or "files" in key:
        score += 10

    return score


def select_url(metadata: Any, kind: str) -> str:
    candidates = flatten_url_candidates(metadata)
    scored = [
        (score_download_url(key, url, kind), key, url)
        for key, url in candidates
    ]
    scored = [item for item in scored if item[0] > 0]
    if not scored:
        return ""
    scored.sort(key=lambda item: (item[0], item[2]), reverse=True)
    return scored[0][2]


def first_record(metadata: Any) -> dict[str, Any]:
    if isinstance(metadata, list):
        return metadata[0] if metadata and isinstance(metadata[0], dict) else {}
    if isinstance(metadata, dict):
        # Some APIs wrap the records.
        for key in ("results", "predictions", "entries", "models"):
            value = metadata.get(key)
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return value[0]
        return metadata
    return {}


def pick_value(record: dict[str, Any], possible_keys: Iterable[str]) -> Any:
    normalized = {
        re.sub(r"[^a-z0-9]", "", str(key).lower()): value
        for key, value in record.items()
    }
    for key in possible_keys:
        normalized_key = re.sub(r"[^a-z0-9]", "", key.lower())
        if normalized_key in normalized:
            return normalized[normalized_key]
    return None


def fetch_alphafold_metadata(
    session: requests.Session,
    accession: str,
    timeout: float,
    retries: int,
) -> tuple[Any | None, str, str]:
    errors: list[str] = []

    for endpoint in ALPHAFOLD_API_ENDPOINTS:
        url = endpoint.format(accession=accession)
        try:
            response = request_with_retries(
                session,
                url,
                timeout=timeout,
                retries=retries,
                accept="application/json",
            )
            if response.status_code == 404:
                errors.append(f"{url}:404")
                continue

            metadata = response.json()
            if metadata in (None, [], {}):
                errors.append(f"{url}:empty_response")
                continue

            return metadata, url, ""

        except (RuntimeError, requests.RequestException, json.JSONDecodeError) as exc:
            errors.append(f"{url}:{exc}")

    return None, "", " | ".join(errors)


def download_url_to_path(
    session: requests.Session,
    url: str,
    path: Path,
    *,
    kind: str,
    timeout: float,
    retries: int,
    overwrite: bool,
) -> tuple[str, str]:
    """
    Return (status, note), where status is downloaded, existing, missing, or failed.
    """
    if not overwrite and existing_file_valid(path, kind):
        return "existing", ""

    try:
        response = request_with_retries(
            session,
            url,
            timeout=timeout,
            retries=retries,
        )
        if response.status_code == 404:
            return "missing", "HTTP 404"

        data = response.content
        encoding = response.headers.get("Content-Encoding", "").lower()
        # requests usually decompresses transfer encodings automatically.
        if urlparse(url).path.lower().endswith(".gz") and encoding != "gzip":
            import gzip
            try:
                data = gzip.decompress(data)
            except OSError:
                # The server may already have transparently decompressed it.
                pass

        valid = (
            looks_like_cif(data)
            if kind == "cif"
            else looks_like_pdb(data)
            if kind == "pdb"
            else True
        )
        if not valid:
            content_type = response.headers.get("Content-Type", "")
            return "failed", f"Downloaded content failed {kind} validation ({content_type})"

        atomic_write_bytes(path, data)
        return "downloaded", ""

    except Exception as exc:
        return "failed", str(exc)


def process_protein(
    session: requests.Session,
    row: pd.Series,
    *,
    output_dir: Path,
    timeout: float,
    retries: int,
    overwrite: bool,
    download_pdb: bool,
) -> dict[str, Any]:
    accession = clean_text(row["uniprot_accession"]).upper()
    gene = clean_text(row.get("gene"))
    b_number = clean_text(row.get("b_number"))
    expected_length = row.get("sequence_length", pd.NA)

    sequence_dir = output_dir / "sequences"
    structure_dir = output_dir / "structures"
    metadata_dir = output_dir / "alphafold_metadata"

    fasta_path = sequence_dir / f"{accession}.fasta"
    cif_path = structure_dir / f"{accession}.cif"
    pdb_path = structure_dir / f"{accession}.pdb"
    metadata_path = metadata_dir / f"{accession}.json"

    result: dict[str, Any] = {
        "accession": accession,
        "gene": gene,
        "b_number": b_number,
        "expected_sequence_length": expected_length,
        "fasta_status": "",
        "fasta_sequence_length": pd.NA,
        "sequence_length_matches_identifiers": pd.NA,
        "fasta_path": str(fasta_path),
        "fasta_sha256": "",
        "alphafold_api_status": "",
        "alphafold_api_url": "",
        "alphafold_entry_id": "",
        "alphafold_model_version": "",
        "alphafold_sequence_length": pd.NA,
        "alphafold_sequence_length_matches_uniprot": pd.NA,
        "cif_status": "",
        "cif_url": "",
        "cif_path": str(cif_path),
        "cif_bytes": pd.NA,
        "cif_sha256": "",
        "pdb_status": "skipped" if not download_pdb else "",
        "pdb_url": "",
        "pdb_path": str(pdb_path) if download_pdb else "",
        "pdb_bytes": pd.NA,
        "pdb_sha256": "",
        "confidence_json_status": "",
        "confidence_json_url": "",
        "confidence_json_path": "",
        "metadata_path": str(metadata_path),
        "overall_status": "",
        "notes": "",
    }
    notes: list[str] = []

    if not ACCESSION_RE.fullmatch(accession):
        result["overall_status"] = "failed"
        result["notes"] = "Invalid or missing UniProt accession"
        return result

    # ---------- UniProt sequence ----------
    if not overwrite and existing_file_valid(fasta_path, "fasta", accession):
        fasta_text = fasta_path.read_text(encoding="utf-8", errors="replace")
        valid, observed_length, note = validate_fasta_text(fasta_text, accession)
        result["fasta_status"] = "existing"
        result["fasta_sequence_length"] = observed_length
        if note:
            notes.append(f"fasta:{note}")
    else:
        fasta_url = UNIPROT_FASTA_URL.format(accession=accession)
        try:
            response = request_with_retries(
                session,
                fasta_url,
                timeout=timeout,
                retries=retries,
                accept="text/plain",
            )
            if response.status_code == 404:
                result["fasta_status"] = "missing"
                notes.append("UniProt FASTA returned HTTP 404")
            else:
                valid, observed_length, note = validate_fasta_text(
                    response.text,
                    accession,
                )
                result["fasta_sequence_length"] = observed_length
                if valid:
                    atomic_write_text(fasta_path, response.text)
                    result["fasta_status"] = "downloaded"
                else:
                    result["fasta_status"] = "failed"
                if note:
                    notes.append(f"fasta:{note}")
        except Exception as exc:
            result["fasta_status"] = "failed"
            notes.append(f"UniProt FASTA error:{exc}")

    if fasta_path.exists() and existing_file_valid(fasta_path, "fasta", accession):
        fasta_text = fasta_path.read_text(encoding="utf-8", errors="replace")
        _, observed_length, _ = validate_fasta_text(fasta_text, accession)
        result["fasta_sequence_length"] = observed_length
        result["fasta_sha256"] = sha256_file(fasta_path)
        try:
            result["sequence_length_matches_identifiers"] = (
                int(observed_length) == int(expected_length)
            )
        except (TypeError, ValueError):
            result["sequence_length_matches_identifiers"] = pd.NA

    # ---------- AlphaFold metadata ----------
    metadata, api_url, metadata_error = fetch_alphafold_metadata(
        session,
        accession,
        timeout,
        retries,
    )

    if metadata is None:
        result["alphafold_api_status"] = "missing_or_failed"
        result["cif_status"] = "not_attempted"
        if download_pdb:
            result["pdb_status"] = "not_attempted"
        notes.append(f"AlphaFold API:{metadata_error}")
    else:
        result["alphafold_api_status"] = "found"
        result["alphafold_api_url"] = api_url
        metadata_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            metadata_path,
            json.dumps(metadata, indent=2, sort_keys=True),
        )

        record = first_record(metadata)
        entry_id = pick_value(
            record,
            ("entryId", "modelEntityId", "predictionId", "modelId"),
        )
        version = pick_value(
            record,
            ("latestVersion", "modelVersion", "version"),
        )
        af_length = pick_value(
            record,
            ("sequenceLength", "uniprotStart", "length"),
        )

        result["alphafold_entry_id"] = clean_text(entry_id)
        result["alphafold_model_version"] = clean_text(version)
        if af_length not in (None, ""):
            result["alphafold_sequence_length"] = af_length

        cif_url = select_url(metadata, "cif")
        pdb_url = select_url(metadata, "pdb")
        confidence_url = select_url(metadata, "confidence")

        result["cif_url"] = cif_url
        result["pdb_url"] = pdb_url if download_pdb else ""
        result["confidence_json_url"] = confidence_url

        if cif_url:
            status, note = download_url_to_path(
                session,
                cif_url,
                cif_path,
                kind="cif",
                timeout=timeout,
                retries=retries,
                overwrite=overwrite,
            )
            result["cif_status"] = status
            if note:
                notes.append(f"cif:{note}")
        else:
            result["cif_status"] = "url_not_found"
            notes.append("No mmCIF URL discovered in AlphaFold API response")

        if download_pdb:
            if pdb_url:
                status, note = download_url_to_path(
                    session,
                    pdb_url,
                    pdb_path,
                    kind="pdb",
                    timeout=timeout,
                    retries=retries,
                    overwrite=overwrite,
                )
                result["pdb_status"] = status
                if note:
                    notes.append(f"pdb:{note}")
            else:
                result["pdb_status"] = "url_not_found"
                notes.append("No PDB URL discovered in AlphaFold API response")

        # Preserve confidence metadata separately when a dedicated URL exists.
        if confidence_url:
            confidence_path = metadata_dir / f"{accession}_confidence.json"
            result["confidence_json_path"] = str(confidence_path)
            try:
                response = request_with_retries(
                    session,
                    confidence_url,
                    timeout=timeout,
                    retries=retries,
                    accept="application/json",
                )
                if response.status_code == 404:
                    result["confidence_json_status"] = "missing"
                else:
                    # Validate JSON before saving. It may be a list or dict.
                    confidence_payload = response.json()
                    atomic_write_text(
                        confidence_path,
                        json.dumps(confidence_payload, indent=2, sort_keys=True),
                    )
                    result["confidence_json_status"] = "downloaded"
            except Exception as exc:
                result["confidence_json_status"] = "failed"
                notes.append(f"confidence_json:{exc}")
        else:
            result["confidence_json_status"] = "url_not_found"

        if cif_path.exists() and existing_file_valid(cif_path, "cif"):
            result["cif_bytes"] = cif_path.stat().st_size
            result["cif_sha256"] = sha256_file(cif_path)

        if download_pdb and pdb_path.exists() and existing_file_valid(pdb_path, "pdb"):
            result["pdb_bytes"] = pdb_path.stat().st_size
            result["pdb_sha256"] = sha256_file(pdb_path)

        try:
            af_len_int = int(result["alphafold_sequence_length"])
            fasta_len_int = int(result["fasta_sequence_length"])
            result["alphafold_sequence_length_matches_uniprot"] = (
                af_len_int == fasta_len_int
            )
        except (TypeError, ValueError):
            result["alphafold_sequence_length_matches_uniprot"] = pd.NA

    fasta_ok = result["fasta_status"] in {"downloaded", "existing"}
    cif_ok = result["cif_status"] in {"downloaded", "existing"}
    pdb_ok = (
        not download_pdb
        or result["pdb_status"] in {"downloaded", "existing"}
    )

    if fasta_ok and cif_ok and pdb_ok:
        result["overall_status"] = "complete"
    elif fasta_ok and cif_ok:
        result["overall_status"] = "partial"
    else:
        result["overall_status"] = "failed"

    result["notes"] = " | ".join(notes)
    return result


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else output_dir / "download_manifest.csv"
    )
    failures_path = (
        Path(args.failures)
        if args.failures
        else output_dir / "download_failures.csv"
    )

    if not input_path.exists():
        print(f"ERROR: Input does not exist: {input_path}", file=sys.stderr)
        return 2

    try:
        identifiers = pd.read_csv(input_path)
    except Exception as exc:
        print(f"ERROR: Could not read identifier CSV: {exc}", file=sys.stderr)
        return 2

    if "uniprot_accession" not in identifiers.columns:
        print(
            "ERROR: Input must contain uniprot_accession.",
            file=sys.stderr,
        )
        return 2

    if args.include_unconfirmed:
        proteins = identifiers.copy()
    else:
        include_mask = (
            identifiers.get(
                "include",
                pd.Series("yes", index=identifiers.index),
            )
            .astype(str)
            .str.lower()
            .eq("yes")
        )
        confirmed_mask = (
            identifiers.get(
                "mapping_status",
                pd.Series("confirmed", index=identifiers.index),
            )
            .astype(str)
            .str.lower()
            .eq("confirmed")
        )
        proteins = identifiers[include_mask & confirmed_mask].copy()

    proteins = proteins[
        proteins["uniprot_accession"].notna()
        & proteins["uniprot_accession"].astype(str).str.strip().ne("")
    ].copy()

    if proteins.empty:
        print("ERROR: No eligible proteins found.", file=sys.stderr)
        return 2

    for directory in (
        output_dir / "sequences",
        output_dir / "structures",
        output_dir / "alphafold_metadata",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "GroEL-ClassIV-Downloader/1.0 "
                "(academic research; UniProt and AlphaFold DB)"
            )
        }
    )

    print(f"Eligible proteins: {len(proteins)}")
    print(f"Output directory:  {output_dir.resolve()}")
    print()

    rows: list[dict[str, Any]] = []
    for index, (_, row) in enumerate(proteins.iterrows(), start=1):
        accession = clean_text(row["uniprot_accession"]).upper()
        gene = clean_text(row.get("gene"))
        print(
            f"[{index:02d}/{len(proteins)}] {accession} {gene}",
            flush=True,
        )

        result = process_protein(
            session,
            row,
            output_dir=output_dir,
            timeout=args.timeout,
            retries=max(args.retries, 1),
            overwrite=args.overwrite,
            download_pdb=not args.skip_pdb,
        )
        rows.append(result)

        print(
            "    "
            f"FASTA={result['fasta_status']}  "
            f"CIF={result['cif_status']}  "
            f"PDB={result['pdb_status']}  "
            f"overall={result['overall_status']}",
            flush=True,
        )
        time.sleep(max(args.delay, 0.0))

    manifest = pd.DataFrame(rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    failures = manifest[manifest["overall_status"] != "complete"].copy()
    failures.to_csv(failures_path, index=False)

    print("\nDownload summary:")
    for status, count in manifest["overall_status"].value_counts().items():
        print(f"  {status}: {count}")

    print(f"\nManifest: {manifest_path}")
    print(f"Failures: {failures_path}")

    complete = int((manifest["overall_status"] == "complete").sum())
    if complete != len(manifest):
        print(
            "\nSome entries were incomplete. Re-running the same command is safe: "
            "valid existing files are reused."
        )
        return 1

    print(f"\nComplete: {complete}/{len(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
