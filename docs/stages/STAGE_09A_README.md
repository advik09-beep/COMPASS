# Stage 09A — Nested grouped Random Forest

This stage runs the primary nonlinear feature-group ablation.

## Models

- sequence final
- structural minimal final
- sequence plus structural minimal

## Evaluation

- 30% MMseqs2 homology-grouped outer folds: primary
- 40% MMseqs2 homology-grouped outer folds: sensitivity
- 3-fold grouped inner CV
- 50 randomized hyperparameter candidates per outer fold
- MCC-based hyperparameter selection
- fixed 0.5 and training-selected thresholds
- full and optional non-TIM subsets

## Install

```bash
python3 -m pip install -r requirements_09A.txt
```

## Full recommended run

```bash
python3 scripts/09A_nested_random_forest.py \
  --feature-dir data/frozen/feature_set_v2 \
  --fold-assignments data/grouping/stage07/fold_assignments_v1.0.csv \
  --output-dir results/stage09 \
  --include-non-tim
```

This run may take several hours on a laptop because it performs nested grouped
cross-validation with randomized hyperparameter search.

## Faster diagnostic run

```bash
python3 scripts/09A_nested_random_forest.py \
  --feature-dir data/frozen/feature_set_v2 \
  --fold-assignments data/grouping/stage07/fold_assignments_v1.0.csv \
  --output-dir results/stage09_test \
  --include-non-tim \
  --n-iter 10 \
  --bootstrap-iterations 500
```

## Outputs

```text
results/stage09/
├── stage09_oof_predictions.csv
├── stage09_outer_fold_metrics.csv
├── stage09_summary_metrics.csv
├── stage09_selected_hyperparameters.csv
├── stage09_inner_search_results.csv
├── stage09_thresholds.csv
├── stage09_feature_manifest.csv
├── stage09_leakage_audit.csv
├── stage09_methods.json
└── stage09_run_summary.json
```

Do not edit Feature Set v2 during or after this analysis.
