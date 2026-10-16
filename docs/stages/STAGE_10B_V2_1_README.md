# Stage 10B v2.1 — Historical gene-name aware correction

The previous script stopped on `JW0843` because Minami Table 1 calls the gene
`rumB`, whereas the current benchmark calls the same JW locus `rlmC`.

That is not a wrong mapping. `JW0843` is the same locus; `rumB` is an older gene
symbol and `rlmC` is the current symbol. The revised script therefore uses the
unique JW ID as the authoritative mapping key and records gene-name differences
in the audit instead of failing.

This also protects against other historical/current symbol changes in Table 1.

## Run

Replace the old Stage 10B script with this revised file, or run it directly:

```bash
python3 scripts/10B_build_benchmark_v1_1_from_minami_table1_v2.py \
  --benchmark data/frozen/benchmark_v1_0/benchmark_v1.0.csv \
  --feature-dir data/frozen/feature_set_v2 \
  --output-dir data/frozen/benchmark_v1_1
```

## Required success checks

The run must still report:

```text
Table 1 proteins audited:   17
Labels changed 0 -> 1:      9
Expected label changes:     9
```

It may also report one or more historical gene aliases. That is expected.

After completion, upload:

```text
data/frozen/benchmark_v1_1/stage10B_v2_summary.json
data/frozen/benchmark_v1_1/minami_table1_label_audit.csv
data/frozen/benchmark_v1_1/benchmark_v1.0_vs_v1.1_changes.csv
data/frozen/benchmark_v1_1/feature_file_patch_audit.csv
```
