# Stage 09 v2 — Final Random Forest pipeline

This is the frozen Benchmark v1.1 / Feature Set v3 Random Forest analysis.

## What it does

- validates Benchmark v1.1 and Feature Set v3;
- uses frozen 30% and 40% homology-grouped outer folds;
- performs grouped nested hyperparameter tuning;
- saves the best model for every outer fold;
- records the top 5 inner-CV configurations for every outer fold;
- generates strictly OOF probabilities;
- reports fixed-0.5 and inner-selected thresholds;
- calculates held-out permutation importance;
- calculates calibration diagnostics and Brier scores;
- generates publication-ready figures;
- writes leakage, OOF, model, feature, and checksum audits.

## Install

```bash
python3 -m pip install -r requirements_09_v2.txt
```

## Recommended diagnostic run

Start with the three primary models and fewer candidates:

```bash
python3 scripts/09_v2_final_random_forest.py \
  --feature-dir data/frozen/feature_set_v3 \
  --fold-assignments data/grouping/stage07/fold_assignments_v1.0.csv \
  --output-dir results/stage09_v2_test \
  --include-non-tim \
  --primary-only \
  --n-iter 10 \
  --permutation-repeats 20 \
  --bootstrap-iterations 500
```

If this completes cleanly, run the final analysis.

## Final run

```bash
python3 scripts/09_v2_final_random_forest.py \
  --feature-dir data/frozen/feature_set_v3 \
  --fold-assignments data/grouping/stage07/fold_assignments_v1.0.csv \
  --output-dir results/stage09_v2 \
  --include-non-tim
```

Defaults:

```text
Inner folds:                 3
Randomized candidates:       80 per outer fold
Permutation repeats:         100 per feature per outer fold
Bootstrap iterations:        5000
Primary grouping:            30% sequence identity
Sensitivity grouping:        40% sequence identity
```

The full run is substantially larger than the old Stage 09 analysis and may
take several hours.

## Main outputs

```text
results/stage09_v2/
├── rf_summary_metrics.csv
├── rf_fold_metrics.csv
├── rf_oof_predictions.csv
├── rf_best_hyperparameters.csv
├── rf_top5_hyperparameters_by_outer_fold.csv
├── rf_inner_search_results.csv
├── rf_thresholds.csv
├── rf_leakage_audit.csv
├── permutation_importance.csv
├── permutation_importance_by_fold.csv
├── hyperparameter_stability.csv
├── calibration_by_outer_fold.csv
├── calibration_pooled_oof.csv
├── rf_feature_manifest.csv
├── saved_model_manifest.csv
├── oof_prediction_audit.csv
├── methods.json
├── reproducibility_report.json
├── sha256_manifest.txt
├── saved_models/
└── figures/
```

## Required success conditions

Before interpreting results:

```text
leakage_failures = 0
oof_prediction_audit_failures = 0
protein_count = 181
positive_count = 73
negative_count = 108
```

## Upload after the final run

Start with:

```text
reproducibility_report.json
rf_summary_metrics.csv
rf_best_hyperparameters.csv
permutation_importance.csv
```

Then, if those pass, upload:

```text
rf_fold_metrics.csv
rf_oof_predictions.csv
rf_top5_hyperparameters_by_outer_fold.csv
hyperparameter_stability.csv
```

Do not modify Benchmark v1.1 or Feature Set v3 after this run.
