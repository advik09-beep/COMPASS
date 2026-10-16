# Stage 08.6 — Logistic rerun on Feature Set v2

This stage repeats the grouped logistic-regression comparison after feature
deduplication and Feature Set v2 freezing.

## Install

```bash
python3 -m pip install -r requirements_08_6.txt
```

## Run

```bash
python3 scripts/08_6_rerun_logistic_feature_set_v2.py \
  --feature-dir data/frozen/feature_set_v2 \
  --fold-assignments data/grouping/stage07/fold_assignments_v1.0.csv \
  --output-dir results/stage08_6 \
  --include-subset-analyses
```

The default is 5,000 bootstrap iterations. For a faster diagnostic run:

```bash
--bootstrap-iterations 500
```

## Compared feature sets

- sequence_final
- structure_minimal_final
- structure_core_final
- sequence_plus_structure_minimal
- sequence_plus_structure_core

## Outputs

```text
results/stage08_6/
├── stage08_6_oof_predictions.csv
├── stage08_6_fold_metrics.csv
├── stage08_6_summary_metrics.csv
├── stage08_6_fold_coefficients.csv
├── stage08_6_coefficient_summary.csv
├── stage08_6_feature_manifest.csv
└── stage08_6_methods.json
```
