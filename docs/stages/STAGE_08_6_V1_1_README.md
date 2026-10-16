# Stage 08.6 v1.1 — Logistic rerun after label correction

This is the first model rerun using Benchmark v1.1.

Nothing except the labels has changed:
- same 181 proteins;
- same Feature Set v2 values;
- same frozen 30% and 40% grouped folds;
- same logistic-regression settings.

## Install

```bash
python3 -m pip install -r requirements_08_6_v1_1.txt
```

## Run

```bash
python3 scripts/08_6_v1_1_rerun_logistic.py \
  --feature-dir data/frozen/benchmark_v1_1/feature_set_v2_labels_v1.1 \
  --fold-assignments data/grouping/stage07/fold_assignments_v1.0.csv \
  --output-dir results/stage08_6_v1_1 \
  --include-subset-analyses
```

The script requires:
- 181 proteins in every feature file;
- 73 positives in Benchmark v1.1;
- complete frozen fold assignments.

## Outputs

```text
results/stage08_6_v1_1/
├── stage08_6_v1_1_oof_predictions.csv
├── stage08_6_v1_1_fold_metrics.csv
├── stage08_6_v1_1_summary_metrics.csv
├── stage08_6_v1_1_fold_coefficients.csv
├── stage08_6_v1_1_coefficient_summary.csv
├── stage08_6_v1_1_feature_manifest.csv
└── stage08_6_v1_1_methods.json
```

## Upload afterward

Please upload:

```text
stage08_6_v1_1_summary_metrics.csv
stage08_6_v1_1_fold_metrics.csv
stage08_6_v1_1_oof_predictions.csv
```

The key comparison is against the historical v1.0 Stage 08.6 results.
