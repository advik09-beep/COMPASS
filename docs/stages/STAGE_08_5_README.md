# Stage 08.5 — Feature audit and Feature Set v2 freeze

## Install

```bash
python3 -m pip install -r requirements_08_5.txt
```

## Run

```bash
python3 scripts/08_5_audit_and_freeze_features.py \
  --benchmark data/frozen/benchmark_v1_0/benchmark_v1.0.csv \
  --output-dir data/frozen/feature_set_v2
```

## Main outputs

```text
data/frozen/feature_set_v2/
├── feature_set_v2_sequence_final.csv
├── feature_set_v2_structure_minimal_final.csv
├── feature_set_v2_structure_core_final.csv
├── feature_set_v2_sequence_plus_structure_minimal.csv
├── feature_set_v2_sequence_plus_structure_core.csv
├── feature_audit_v2.csv
├── feature_set_v2_manifest.csv
├── exact_duplicate_feature_pairs.csv
├── near_perfect_correlation_pairs.csv
├── feature_set_v2_summary.json
├── feature_set_v2_methods.md
└── feature_set_v2_sha256.txt
```

No model is trained. Feature decisions are frozen before Random Forest
development.
