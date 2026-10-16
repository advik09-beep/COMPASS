# Stage 08A — Baseline models

This stage evaluates three simple baselines using the frozen grouped folds:

1. Majority-class baseline
2. Length-only logistic regression
3. Relative-contact-order-only logistic regression

No hyperparameter search occurs. All predictions are out-of-fold.

## Install

```bash
python3 -m pip install -r requirements_08A.txt
```

## Run

```bash
python3 scripts/08A_baseline_models.py \
  --benchmark data/frozen/benchmark_v1_0/benchmark_v1.0.csv \
  --fold-assignments data/grouping/stage07/fold_assignments_v1.0.csv \
  --output-dir results/stage08A
```

To additionally run exploratory non-TIM and TIM-only analyses:

```bash
python3 scripts/08A_baseline_models.py \
  --benchmark data/frozen/benchmark_v1_0/benchmark_v1.0.csv \
  --fold-assignments data/grouping/stage07/fold_assignments_v1.0.csv \
  --output-dir results/stage08A \
  --include-subset-analyses
```

## Outputs

```text
results/stage08A/
├── stage08A_oof_predictions.csv
├── stage08A_fold_metrics.csv
├── stage08A_summary_metrics.csv
├── stage08A_logistic_coefficients.csv
└── stage08A_methods.json
```

The primary comparison is based on the 30%-identity grouped folds. The
40%-identity folds are a sensitivity analysis.
