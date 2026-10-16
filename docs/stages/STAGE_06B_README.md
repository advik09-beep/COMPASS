# Stage 06B — Prepare and Freeze Benchmark v1.0

## Install

```bash
python3 -m pip install -r requirements_06B.txt
```

## Run

```bash
python3 scripts/06B_prepare_and_freeze_benchmark.py \
  --input data/modeling/minami_binary_benchmark_features_181.csv \
  --merge-audit data/modeling/minami_binary_benchmark_merge_audit.csv \
  --output-dir data/frozen/benchmark_v1_0
```

This stage adds label evidence/confidence, audits exact duplicate sequences,
calculates sequence features, audits AlphaFold-confidence confounding, removes
predetermined redundant structural variables, computes exploratory univariate
AUC statistics, and freezes benchmark version 1.0 with checksums.
