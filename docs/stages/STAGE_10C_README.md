# Stage 10C — Benchmark v1.1 impact analysis

This stage quantifies what changed between Benchmark v1.0 and v1.1 **before**
any models are rerun.

## Install

```bash
python3 -m pip install -r requirements_10C.txt
```

## Run

```bash
python3 scripts/10C_benchmark_v1_1_impact_analysis.py \
  --benchmark-v1 data/frozen/benchmark_v1_0/benchmark_v1.0.csv \
  --benchmark-v11 data/frozen/benchmark_v1_1/benchmark_v1.1.csv \
  --fold-assignments data/grouping/stage07/fold_assignments_v1.0.csv \
  --feature-dir-v11 data/frozen/benchmark_v1_1/feature_set_v2_labels_v1.1 \
  --output-dir results/stage10C
```

## Outputs

```text
results/stage10C/
├── benchmark_change_table_all_proteins.csv
├── nine_corrected_proteins.csv
├── fold_composition_v1_0_vs_v1_1.csv
├── corrected_cluster_impact.csv
├── corrected_protein_feature_effects.csv
├── class_weight_changes.csv
├── stage10C_summary.json
├── stage10C_methods.md
└── figures/
    ├── fold_positive_composition.png
    ├── fold_positive_composition.pdf
    ├── corrected_protein_feature_profile.png
    └── corrected_protein_feature_profile.pdf
```

## What this answers

- Are the 9 corrections concentrated in one outer fold?
- Are they concentrated in TIM barrels?
- Do they cluster into a small number of MMseqs2 groups?
- How much does class balance change?
- How do class weights change?
- Which frozen features are unusual among the nine corrected proteins?

No model fitting occurs in Stage 10C.

## Upload afterward

Please upload:

```text
stage10C_summary.json
fold_composition_v1_0_vs_v1_1.csv
corrected_cluster_impact.csv
corrected_protein_feature_effects.csv
```
