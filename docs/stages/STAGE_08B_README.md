# Stage 08B — Multivariable logistic baselines

This stage compares four interpretable feature groups:

1. Sequence only
2. Structural minimal
3. Structural core
4. Sequence plus structural core

All models use fixed frozen grouped folds. No hyperparameter tuning occurs.

## Install

```bash
python3 -m pip install -r requirements_08B.txt
```

## Run

```bash
python3 scripts/08B_multivariable_logistic_baselines.py \
  --sequence-dataset data/frozen/benchmark_v1_0/benchmark_v1.0_sequence_only.csv \
  --structural-minimal data/frozen/benchmark_v1_0/benchmark_v1.0_structural_minimal.csv \
  --structural-core data/frozen/benchmark_v1_0/benchmark_v1.0_structural_core.csv \
  --combined-core data/frozen/benchmark_v1_0/benchmark_v1.0_sequence_plus_structure_core.csv \
  --fold-assignments data/grouping/stage07/fold_assignments_v1.0.csv \
  --output-dir results/stage08B \
  --include-subset-analyses
```

The script uses 5,000 bootstrap iterations by default. To perform a quicker
diagnostic run, add:

```bash
--bootstrap-iterations 500
```

## Outputs

```text
results/stage08B/
├── stage08B_oof_predictions.csv
├── stage08B_fold_metrics.csv
├── stage08B_summary_metrics.csv
├── stage08B_fold_coefficients.csv
├── stage08B_coefficient_summary.csv
├── stage08B_feature_manifest.csv
└── stage08B_methods.json
```
