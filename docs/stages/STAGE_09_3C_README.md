# Stage 09.3C — Evaluation Reconciliation + Fold 3 Audit

This is an **audit**, not another optimization experiment.

It addresses two unresolved issues:

1. Why did Structural Core move from approximately **MCC 0.537 in Stage 09 v2**
   to **0.559 in Stage 09.3B**?
2. Why does **fold 3** appear to be the weakest fold across essentially every
   Stage 09.3B model?

## What the script does

### A. Exact v2 vs 09.3B reconciliation

It reconstructs the Stage 09 v2 Structural Core out-of-fold predictions
directly from the saved fold models and compares them protein-by-protein with
the Stage 09.3B Structural Core predictions.

It reports:

- changed probabilities;
- changed 0.5-threshold classifications;
- v2-wrong → 09.3B-correct proteins;
- v2-correct → 09.3B-wrong proteins;
- fold-specific metric changes;
- selected hyperparameters in each run.

This should tell us whether the 0.537 → 0.559 movement is simply due to
different random hyperparameter samples / fitted RF seeds or whether there is a
deeper pipeline mismatch.

### B. Fold 3 difficulty

Across **every Stage 09.3B feature-set trial**, it calculates fold-level:

- MCC;
- ROC-AUC;
- PR-AUC;
- balanced accuracy;
- recall;
- specificity.

Then it asks whether fold 3 is unusual in:

- class balance;
- TIM-barrel frequency;
- homology-cluster composition;
- structural-feature distributions;
- repeatedly misclassified proteins.

### Why this matters

A single consistently weak fold can occur because that fold contains a harder
biological subpopulation, not because the model is unstable.

MCC is also relatively volatile within small folds: a few additional false
positives or false negatives can move fold-specific MCC substantially. The
audit therefore looks at ROC-AUC, PR-AUC, recall, specificity, feature shift,
and repeated protein-level errors rather than treating fold MCC alone as the
explanation.

## Run

```bash
python3 scripts/09_3C_evaluation_and_fold_audit.py \
  --feature-dir data/frozen/feature_set_v3 \
  --fold-assignments data/grouping/stage07/fold_assignments_v1.0.csv \
  --stage09-v2-dir results/stage09_v2 \
  --stage09-3b-dir results/stage09_3B \
  --output-dir results/stage09_3C \
  --focus-fold 3
```

## Most important outputs

Upload:

```text
stage09_3C_summary.json
fold_difficulty_summary.csv
fold_composition_summary.csv
v2_vs_3B_fold_metrics.csv
v2_vs_3B_hyperparameters.csv
fold3_feature_shift_vs_others.csv
fold3_repeated_errors.csv
```

Optional if we need protein-level reconciliation:

```text
v2_vs_3B_structural_core_prediction_comparison.csv
```

## Interpretation rule

Do **not** change fold assignments because fold 3 is difficult.

The folds were frozen before these results were observed. If fold 3 represents
a genuinely harder region of protein space, that is useful information about
model generalization and should be reported rather than engineered away.
