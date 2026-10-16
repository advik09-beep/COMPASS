# Stage 08 Figures

This script converts the baseline and frozen Feature Set v2 results into
publication/poster-ready figures.

## Install

```bash
python3 -m pip install -r requirements_stage08_figures.txt
```

## Run

```bash
python3 scripts/08_figures.py \
  --stage08a-summary results/stage08A/stage08A_summary_metrics.csv \
  --stage086-summary results/stage08_6/stage08_6_summary_metrics.csv \
  --stage086-oof results/stage08_6/stage08_6_oof_predictions.csv \
  --stage086-fold-metrics results/stage08_6/stage08_6_fold_metrics.csv \
  --stage086-coefficients results/stage08_6/stage08_6_coefficient_summary.csv \
  --output-dir figures/stage08
```

## Outputs

Each figure is saved as both 300-dpi PNG and vector PDF.

1. `figure_01_baseline_performance_ladder`
   - Best current poster figure.
   - Shows how prediction improves from majority and length baselines through
     contact order, sequence, structure, and integrated features.

2. `figure_02_feature_set_performance`
   - MCC, ROC-AUC, and PR-AUC with bootstrap confidence intervals.
   - Best concise summary for a paper Results section.

3. `figure_03_oof_roc_curves`
   - Uses pooled out-of-fold probabilities.
   - Shows threshold-independent discrimination.

4. `figure_04_oof_precision_recall_curves`
   - Particularly useful because positives are the minority class.

5. `figure_05_non_tim_robustness`
   - Compares full and non-TIM datasets.
   - Important evidence that the model is not only detecting TIM barrels.

6. `figure_06_fold_stability`
   - Shows fold-to-fold MCC variation.
   - Useful as a supplementary robustness figure.

7. `figure_07_coefficient_stability`
   - Top standardized logistic coefficients with variability across folds.
   - Use cautiously because correlated predictors can destabilize coefficients.
