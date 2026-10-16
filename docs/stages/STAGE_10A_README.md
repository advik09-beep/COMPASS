# Stage 10A — Held-out permutation importance and error analysis

## Install

```bash
python3 -m pip install -r requirements_10A.txt
```

## Run

```bash
python3 scripts/10A_permutation_importance_and_error_analysis.py \
  --feature-dir data/frozen/feature_set_v2 \
  --fold-assignments data/grouping/stage07/fold_assignments_v1.0.csv \
  --oof-predictions results/stage09/stage09_oof_predictions.csv \
  --selected-hyperparameters results/stage09/stage09_selected_hyperparameters.csv \
  --output-dir results/stage10A
```

The default uses 100 permutation repeats per feature per outer test fold and
ROC-AUC as the importance score.

For a faster diagnostic run:

```bash
python3 scripts/10A_permutation_importance_and_error_analysis.py \
  --feature-dir data/frozen/feature_set_v2 \
  --fold-assignments data/grouping/stage07/fold_assignments_v1.0.csv \
  --oof-predictions results/stage09/stage09_oof_predictions.csv \
  --selected-hyperparameters results/stage09/stage09_selected_hyperparameters.csv \
  --output-dir results/stage10A_test \
  --permutation-repeats 20
```

## Outputs

```text
results/stage10A/
├── permutation_importance_by_fold.csv
├── permutation_importance_summary.csv
├── permutation_importance_fold_audit.csv
├── error_analysis_all_proteins.csv
├── high_confidence_false_positives.csv
├── high_confidence_false_negatives.csv
├── model_disagreement_cases.csv
├── sufS_csdA_case.csv
├── stage10A_summary.json
├── stage10A_methods.json
└── figures/
```

Permutation importance is calculated only on untouched outer test folds.
