# Stage 10B v2.2 — Pandas dtype fix

The previous run failed because `previous_binary_label` was initialized as an
empty-string column. Newer pandas versions infer this as a strict string dtype,
so assigning integer labels such as `0` raises a `TypeError`.

This revision initializes provenance columns with explicit nullable dtypes:

- `previous_binary_label`: `Int64`
- `label_updated_from_minami_validation`: `boolean`
- text provenance fields: `string`

No scientific logic or labels changed.

## Run

```bash
python3 scripts/10B_build_benchmark_v1_1_from_minami_table1_v3.py \
  --benchmark data/frozen/benchmark_v1_0/benchmark_v1.0.csv \
  --feature-dir data/frozen/feature_set_v2 \
  --output-dir data/frozen/benchmark_v1_1
```

If `data/frozen/benchmark_v1_1` contains partial files from the failed run,
that is okay: this run rewrites the expected outputs.

Expected success checks:

```text
Table 1 proteins audited:   17
Labels changed 0 -> 1:      9
Expected label changes:     9
```
