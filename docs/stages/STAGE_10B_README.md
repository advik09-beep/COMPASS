# Stage 10B — Minami experimental-label audit

This stage corrects label provenance without overwriting Benchmark v1.0.

## Important

Do not relabel proteins based on model probability, structural similarity, or
biological intuition. Only manually verified experimental outcomes from Minami
Table 1 can change labels.

The included starter CSV contains only three outcomes directly supported by the
paper text already reviewed:

- BglB: dependent
- AscB: independent
- YcdM: independent

The accession and JWID fields are intentionally blank. Verify them against the
benchmark before use. Complete all 17 experimentally tested proteins from Table
1 before creating Benchmark v1.1.

## 1. Create a review template from the current high-confidence false positives

```bash
python3 scripts/10B_audit_minami_validation_labels.py \
  --benchmark data/frozen/benchmark_v1_0/benchmark_v1.0.csv \
  --high-confidence-fps results/stage10A/high_confidence_false_positives.csv \
  --make-template results/stage10B/minami_validation_review_template.csv
```

This is only a candidate-review sheet. It is not sufficient by itself because
Minami tested 17 proteins, including proteins that may not appear among our
high-confidence errors.

## 2. Build the complete manual validation table

Use the included starter file as a schema:

```text
minami_2023_validation_labels_starter.csv
```

Allowed values:

```text
tested_in_minami_2023:
  yes
  no

validation_outcome:
  dependent
  independent
  inconclusive
  not_tested

evidence_level:
  experimental_confirmed
  experimental_independent
  inconclusive
  not_tested
```

For every tested protein, record:

- accession, gene, or JW ID;
- experimental outcome;
- Table 1 or figure reference;
- citation;
- reviewer;
- review date;
- concise evidence note.

## 3. Audit and create Benchmark v1.1

```bash
python3 scripts/10B_audit_minami_validation_labels.py \
  --benchmark data/frozen/benchmark_v1_0/benchmark_v1.0.csv \
  --feature-dir data/frozen/feature_set_v2 \
  --validation-table data/manual/minami_2023_validation_labels.csv \
  --high-confidence-fps results/stage10A/high_confidence_false_positives.csv \
  --output-dir data/frozen/benchmark_v1_1
```

## Outputs

```text
data/frozen/benchmark_v1_1/
├── benchmark_v1.1.csv
├── label_update_audit.csv
├── benchmark_v1.0_vs_v1.1_changes.csv
├── unresolved_validation_rows.csv
├── validated_false_positive_analysis.csv
├── stage10B_summary.json
├── benchmark_v1.1_methods.md
├── benchmark_v1.1_sha256.txt
└── feature_set_v2_labels_v1.1/
    ├── feature_set_v2_sequence_final.csv
    ├── feature_set_v2_structure_minimal_final.csv
    ├── feature_set_v2_structure_core_final.csv
    ├── feature_set_v2_sequence_plus_structure_minimal.csv
    └── feature_set_v2_sequence_plus_structure_core.csv
```

## Stop condition

Do not rerun models until:

- all 17 tested proteins are represented;
- `unresolved_validation_rows.csv` is empty;
- every dependent label has `experimental_confirmed`;
- every independent label has `experimental_independent`;
- the count of experimentally confirmed dependent proteins matches the paper.
