#!/usr/bin/env python3
"""
Map Fujiwara Class IV E. coli K-12 proteins to UniProtKB records.

The script:
1. reads a CSV containing Gene and b_number columns,
2. removes non-protein heading rows,
3. queries the official UniProt REST API using each b-number,
4. prefers reviewed E. coli K-12 reference-proteome entries,
5. verifies the reported gene name,
6. writes a clean identifier table and an audit report.

Example:
    python 01_map_uniprot.py \
        --input "fujiwara_class_iv_proteins_only.csv" \
        --output "fujiwara_class_iv_identifiers.csv" \
        --audit "fujiwara_class_iv_mapping_audit.csv"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
ECOLI_K12_TAXON_ID = 83333
ECOLI_K12_PROTEOME = "UP000000625"

B_NUMBER_RE = re.compile(r"^b\d{4}$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map Fujiwara Class IV b-numbers to UniProtKB."
    )
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument(
        "--output",
        default="fujiwara_class_iv_identifiers.csv",
        help="Clean identifier-table output CSV.",
    )
    parser.add_argument(
        "--audit",
        default="fujiwara_class_iv_mapping_audit.csv",
        help="Detailed mapping-audit output CSV.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.15,
        help="Delay between API calls in seconds.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds.",
    )
    return parser.parse_args()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize expected column names without altering source values."""
    aliases = {
        "Gene": "gene",
        "gene": "gene",
        "b_number": "b_number",
        "B_number": "b_number",
        "source": "source",
        "Source": "source",
        "Function": "function",
        "function": "function",
    }

    renamed = {}
    for column in df.columns:
        stripped = str(column).strip()
        renamed[column] = aliases.get(stripped, stripped)

    df = df.rename(columns=renamed)

    required = {"gene", "b_number"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Missing required column(s): " + ", ".join(sorted(missing))
        )

    if "source" not in df.columns:
        df["source"] = "Fujiwara_2010"
    if "function" not in df.columns:
        df["function"] = pd.NA

    return df


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def canonical_b_number(value: Any) -> str:
    text = clean_text(value).lower()
    return text if B_NUMBER_RE.fullmatch(text) else ""


def extract_recommended_name(result: dict[str, Any]) -> str:
    desc = result.get("proteinDescription", {})

    recommended = desc.get("recommendedName", {})
    full_name = recommended.get("fullName", {})
    if isinstance(full_name, dict) and full_name.get("value"):
        return full_name["value"]

    submission = desc.get("submissionNames", [])
    if submission:
        full_name = submission[0].get("fullName", {})
        if isinstance(full_name, dict):
            return full_name.get("value", "")

    alternatives = desc.get("alternativeNames", [])
    if alternatives:
        full_name = alternatives[0].get("fullName", {})
        if isinstance(full_name, dict):
            return full_name.get("value", "")

    return ""


def extract_gene_names(result: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for gene_record in result.get("genes", []) or []:
        primary = gene_record.get("geneName", {}).get("value")
        if primary:
            names.append(primary)

        for synonym in gene_record.get("synonyms", []) or []:
            value = synonym.get("value")
            if value:
                names.append(value)

        for ordered in gene_record.get("orderedLocusNames", []) or []:
            value = ordered.get("value")
            if value:
                names.append(value)

        for orf_name in gene_record.get("orfNames", []) or []:
            value = orf_name.get("value")
            if value:
                names.append(value)

    # Stable de-duplication
    seen = set()
    unique = []
    for name in names:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            unique.append(name)
    return unique


def primary_gene_name(result: dict[str, Any]) -> str:
    genes = result.get("genes", []) or []
    if not genes:
        return ""
    return genes[0].get("geneName", {}).get("value", "")


def is_reviewed(result: dict[str, Any]) -> bool:
    return "reviewed" in str(result.get("entryType", "")).lower()


def has_reference_proteome_keyword(result: dict[str, Any]) -> bool:
    keywords = result.get("keywords", []) or []
    return any(
        str(keyword.get("name", "")).lower() == "reference proteome"
        for keyword in keywords
        if isinstance(keyword, dict)
    )


def query_uniprot(
    session: requests.Session,
    b_number: str,
    timeout: float,
) -> tuple[list[dict[str, Any]], str]:
    """
    Search exact ordered-locus name first, then a broader gene search.
    Returns (results, query_used).
    """
    queries = [
        (
            f"(gene_exact:{b_number}) AND "
            f"(organism_id:{ECOLI_K12_TAXON_ID}) AND "
            f"(proteome:{ECOLI_K12_PROTEOME})"
        ),
        (
            f"(gene:{b_number}) AND "
            f"(organism_id:{ECOLI_K12_TAXON_ID}) AND "
            f"(proteome:{ECOLI_K12_PROTEOME})"
        ),
        (
            f"(gene_exact:{b_number}) AND "
            f"(organism_id:{ECOLI_K12_TAXON_ID})"
        ),
    ]

    for query in queries:
        response = session.get(
            UNIPROT_SEARCH_URL,
            params={
                "query": query,
                "format": "json",
                "size": 25,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", [])
        if results:
            return results, query

    return [], queries[-1]


def score_candidate(
    result: dict[str, Any],
    b_number: str,
    paper_gene: str,
) -> tuple[int, list[str]]:
    names = [name.lower() for name in extract_gene_names(result)]
    score = 0
    reasons: list[str] = []

    if b_number.lower() in names:
        score += 100
        reasons.append("exact_b_number")

    if paper_gene and paper_gene.lower() in names:
        score += 30
        reasons.append("exact_gene")

    # Fujiwara table uses superscript footnote letters that can appear as
    # trailing plain-text letters after CSV transcription (e.g., serCf).
    if paper_gene and len(paper_gene) > 1:
        stripped = paper_gene[:-1].lower()
        if stripped in names and paper_gene.lower() not in names:
            score += 15
            reasons.append("gene_matches_after_possible_footnote_suffix_removal")

    organism_id = result.get("organism", {}).get("taxonId")
    if organism_id == ECOLI_K12_TAXON_ID:
        score += 20
        reasons.append("ecoli_k12_taxon")

    if is_reviewed(result):
        score += 10
        reasons.append("reviewed")

    if has_reference_proteome_keyword(result):
        score += 5
        reasons.append("reference_proteome")

    return score, reasons


def determine_gene_match(
    paper_gene: str,
    all_names: list[str],
) -> tuple[str, str]:
    lower_names = {name.lower() for name in all_names}
    paper_lower = paper_gene.lower()

    if paper_lower in lower_names:
        return "exact", paper_gene

    if len(paper_gene) > 1 and paper_gene[:-1].lower() in lower_names:
        return "possible_fujiwara_footnote_suffix", paper_gene[:-1]

    return "mismatch", ""


def map_row(
    session: requests.Session,
    row: pd.Series,
    timeout: float,
) -> dict[str, Any]:
    paper_gene = clean_text(row["gene"])
    b_number = canonical_b_number(row["b_number"])
    source = clean_text(row.get("source")) or "Fujiwara_2010"
    function = clean_text(row.get("function"))

    base = {
        "gene_raw": paper_gene,
        "gene": "",
        "b_number": b_number,
        "protein_name": "",
        "uniprot_accession": "",
        "uniprot_entry_name": "",
        "sequence_length": pd.NA,
        "organism": "",
        "taxon_id": pd.NA,
        "proteome_id": ECOLI_K12_PROTEOME,
        "reviewed": pd.NA,
        "label": 1,
        "label_definition": "Obligate GroEL/GroES-dependent Class IV substrate",
        "source": source,
        "cohort_role": "historical_training_positive",
        "mapping_status": "",
        "gene_match_status": "",
        "mapping_notes": "",
        "include": "no",
        "exclusion_reason": "",
        "fujiwara_function": function,
        "uniprot_query": "",
        "candidate_count": 0,
        "candidate_accessions": "",
        "candidate_scores": "",
    }

    if not b_number:
        base["mapping_status"] = "invalid_b_number"
        base["exclusion_reason"] = "Missing or invalid b-number"
        return base

    try:
        candidates, query_used = query_uniprot(session, b_number, timeout)
    except requests.RequestException as exc:
        base["mapping_status"] = "api_error"
        base["mapping_notes"] = str(exc)
        base["exclusion_reason"] = "UniProt API request failed"
        return base

    base["uniprot_query"] = query_used
    base["candidate_count"] = len(candidates)

    if not candidates:
        base["mapping_status"] = "not_found"
        base["exclusion_reason"] = "No UniProtKB result found"
        return base

    scored = []
    for candidate in candidates:
        score, reasons = score_candidate(candidate, b_number, paper_gene)
        scored.append((score, reasons, candidate))

    scored.sort(
        key=lambda item: (
            item[0],
            is_reviewed(item[2]),
            item[2].get("primaryAccession", ""),
        ),
        reverse=True,
    )

    base["candidate_accessions"] = ";".join(
        item[2].get("primaryAccession", "") for item in scored
    )
    base["candidate_scores"] = ";".join(str(item[0]) for item in scored)

    best_score, best_reasons, best = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else None

    all_gene_names = extract_gene_names(best)
    gene_match_status, corrected_gene = determine_gene_match(
        paper_gene, all_gene_names
    )

    base.update(
        {
            "gene": primary_gene_name(best) or corrected_gene or paper_gene,
            "protein_name": extract_recommended_name(best),
            "uniprot_accession": best.get("primaryAccession", ""),
            "uniprot_entry_name": best.get("uniProtkbId", ""),
            "sequence_length": best.get("sequence", {}).get("length", pd.NA),
            "organism": best.get("organism", {}).get("scientificName", ""),
            "taxon_id": best.get("organism", {}).get("taxonId", pd.NA),
            "reviewed": is_reviewed(best),
            "gene_match_status": gene_match_status,
        }
    )

    ambiguous_tie = second_score is not None and second_score == best_score
    exact_b_match = "exact_b_number" in best_reasons
    correct_taxon = best.get("organism", {}).get("taxonId") == ECOLI_K12_TAXON_ID

    notes = [
        f"selected_score={best_score}",
        "selection_reasons=" + ",".join(best_reasons),
    ]

    if ambiguous_tie:
        base["mapping_status"] = "ambiguous"
        base["exclusion_reason"] = "Top UniProt candidates have equal scores"
        notes.append("equal_top_candidate_scores")
    elif not exact_b_match:
        base["mapping_status"] = "ambiguous"
        base["exclusion_reason"] = "Selected record does not contain exact b-number"
        notes.append("missing_exact_b_number_in_gene_names")
    elif not correct_taxon:
        base["mapping_status"] = "ambiguous"
        base["exclusion_reason"] = "Selected record is not E. coli K-12 taxon 83333"
        notes.append("wrong_taxon")
    elif gene_match_status == "mismatch":
        # b-number is the principal identity key, so retain the record for review
        # rather than rejecting it automatically.
        base["mapping_status"] = "review_required"
        base["exclusion_reason"] = "Paper gene text does not match UniProt gene aliases"
        notes.append(
            "paper_gene="
            + paper_gene
            + ";uniprot_gene_names="
            + "|".join(all_gene_names)
        )
    else:
        base["mapping_status"] = "confirmed"
        base["include"] = "yes"
        base["exclusion_reason"] = ""
        if gene_match_status == "possible_fujiwara_footnote_suffix":
            notes.append(
                f"paper_gene_probably_contains_footnote_suffix:{paper_gene}->{corrected_gene}"
            )

    base["mapping_notes"] = ";".join(notes)
    return base


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"ERROR: Input file does not exist: {input_path}", file=sys.stderr)
        return 2

    try:
        raw = pd.read_csv(input_path)
        raw = normalize_columns(raw)
    except Exception as exc:
        print(f"ERROR: Could not read input CSV: {exc}", file=sys.stderr)
        return 2

    # Remove category headings such as "Metabolic reactions" and "Unknown".
    proteins = raw[
        raw["b_number"].map(canonical_b_number).astype(bool)
    ].copy()

    print(
        f"Loaded {len(raw)} rows; {len(proteins)} rows contain valid b-numbers."
    )

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "GroEL-ClassIV-Mapping/1.0 "
                "(academic research; UniProt REST API)"
            ),
            "Accept": "application/json",
        }
    )

    mapped_rows = []
    for index, (_, row) in enumerate(proteins.iterrows(), start=1):
        b_number = canonical_b_number(row["b_number"])
        gene = clean_text(row["gene"])
        print(f"[{index:02d}/{len(proteins)}] {b_number} {gene}", flush=True)

        mapped_rows.append(map_row(session, row, args.timeout))
        time.sleep(max(args.delay, 0.0))

    audit = pd.DataFrame(mapped_rows)

    identifier_columns = [
        "gene_raw",
        "gene",
        "b_number",
        "protein_name",
        "uniprot_accession",
        "uniprot_entry_name",
        "sequence_length",
        "organism",
        "taxon_id",
        "proteome_id",
        "reviewed",
        "label",
        "label_definition",
        "source",
        "cohort_role",
        "mapping_status",
        "gene_match_status",
        "mapping_notes",
        "include",
        "exclusion_reason",
    ]

    audit_path = Path(args.audit)
    output_path = Path(args.output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    audit.to_csv(audit_path, index=False)
    audit[identifier_columns].to_csv(output_path, index=False)

    status_counts = audit["mapping_status"].value_counts(dropna=False)
    print("\nMapping summary:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")

    confirmed = int((audit["mapping_status"] == "confirmed").sum())
    print(f"\nConfirmed: {confirmed}/{len(audit)}")
    print(f"Identifier table: {output_path}")
    print(f"Audit report:     {audit_path}")

    if confirmed != len(audit):
        print(
            "\nSome rows require review. Open the audit CSV and inspect "
            "mapping_status, candidate_accessions, and mapping_notes."
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
