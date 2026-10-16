# Stage 09.3A — Patch for `feature_origin` merge error

The error:

```text
KeyError: "['feature_origin'] not in index"
```

came from the final SHAP/LOFO merge. Both `shap_summary` and `lofo_summary`
contained a `feature_origin` column, so pandas renamed them internally to:

```text
feature_origin_x
feature_origin_y
```

The later code still asked for a column literally named `feature_origin`.

The patched script removes the duplicate LOFO copy before merging and keeps the
SHAP-side `feature_origin` as the canonical column.

## Run

```bash
python3 scripts/09_3A_shap_lofo_analysis_FIXED.py \
  --feature-dir data/frozen/feature_set_v3 \
  --fold-assignments data/grouping/stage07/fold_assignments_v1.0.csv \
  --stage09-v2-dir results/stage09_v2 \
  --output-dir results/stage09_3A
```

No methods, labels, folds, models, or scientific logic changed. This is only a
pandas column-collision fix.
