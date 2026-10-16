# Stage 08.9 — Feature redundancy audit / Feature Set v3

This stage performs the last predictor-space cleanup before the final Random
Forest.

The pruning is deliberately conservative and **does not use the target label or
model performance**.

## Install

```bash
python3 -m pip install -r requirements_08_9.txt
```

## Run

```bash
python3 scripts/08_9_feature_redundancy_audit.py \
  --feature-dir data/frozen/benchmark_v1_1/feature_set_v2_labels_v1.1 \
  --output-dir data/frozen/feature_set_v3
```

Defaults:

```text
Pearson flag threshold:       |r| >= 0.95
Spearman flag threshold:      |rho| >= 0.95
Same-family auto-prune:       |rho| >= 0.97
Effectively exact threshold:  >= 0.999999
```

## Important design rule

A high correlation alone does not automatically remove a predictor.

The script only auto-prunes:
- exact/near-exact redundancies; or
- extremely correlated variables within the same biological/measurement family.

Cross-family correlations are kept and flagged because two biologically
different variables can correlate without being interchangeable.

## Outputs

```text
data/frozen/feature_set_v3/
├── feature_set_v3_sequence_final.csv
├── feature_set_v3_structure_minimal_final.csv
├── feature_set_v3_structure_core_final.csv
├── feature_set_v3_sequence_plus_structure_minimal.csv
├── feature_set_v3_sequence_plus_structure_core.csv
├── pearson_correlation_matrix.csv
├── spearman_correlation_matrix.csv
├── high_correlation_pairs.csv
├── feature_redundancy_groups.csv
├── feature_pruning_decisions.csv
├── feature_set_v3_audit.csv
├── feature_set_v3_manifest.csv
├── feature_set_v3_export_summary.csv
├── stage08_9_summary.json
├── feature_set_v3_methods.md
├── feature_set_v3_sha256.txt
└── figures/
    ├── pearson_correlation_heatmap.png
    ├── pearson_correlation_heatmap.pdf
    ├── spearman_correlation_heatmap.png
    └── spearman_correlation_heatmap.pdf
```

## Upload afterward

Please upload:

```text
stage08_9_summary.json
high_correlation_pairs.csv
feature_pruning_decisions.csv
feature_set_v3_export_summary.csv
```

We should inspect those four before freezing v3 and starting Stage 09 v2.
