# Stage 09.3A — SHAP + Leave-One-Feature-Out Ablation

This is the first diagnostic stage after the frozen Stage 09 v2 Random Forest.

It answers:

1. Which Structural Core features matter most and least?
2. Which Combined Core features matter most and least?
3. Which features actually hurt held-out performance?
4. Are the sequence variables helping or hurting Combined Core?
5. Do SHAP and retraining-based LOFO ablation agree?

## Install

```bash
python3 -m pip install -r requirements_09_3A.txt
```

## Primary run

```bash
python3 scripts/09_3A_shap_lofo_analysis.py \
  --feature-dir data/frozen/feature_set_v3 \
  --fold-assignments data/grouping/stage07/fold_assignments_v1.0.csv \
  --stage09-v2-dir results/stage09_v2 \
  --output-dir results/stage09_3A
```

This analyzes:

- Structural Core
- Sequence + Structural Core
- Full benchmark
- Primary 30% homology grouping

## Optional sensitivity run

Only after reviewing the primary result:

```bash
python3 scripts/09_3A_shap_lofo_analysis.py \
  --feature-dir data/frozen/feature_set_v3 \
  --fold-assignments data/grouping/stage07/fold_assignments_v1.0.csv \
  --stage09-v2-dir results/stage09_v2 \
  --output-dir results/stage09_3A_sensitivity \
  --include-sensitivity
```

## LOFO interpretation

```text
delta = performance(feature removed) - performance(full model)

negative delta = feature was useful
near zero      = little detectable effect
positive delta = removing the feature improved performance
```

## Outputs

```text
results/stage09_3A/
├── shap_values_oof_long.csv
├── shap_feature_summary.csv
├── lofo_oof_predictions.csv
├── lofo_feature_summary.csv
├── shap_lofo_consensus.csv
├── baseline_stage09_v2_predictions_reconstructed.csv
├── model_audit.csv
├── stage09_3A_summary.json
├── stage09_3A_methods.md
└── figures/
```

## Upload afterward

Start with:

```text
stage09_3A_summary.json
shap_feature_summary.csv
lofo_feature_summary.csv
shap_lofo_consensus.csv
```

Do not remove features yet.

The next stage will test why Combined Core loses to Structural Core by adding
sequence feature families back to Structural Core systematically. Only after
09.3A + that family-level experiment should Stage 09 v3 be frozen.
