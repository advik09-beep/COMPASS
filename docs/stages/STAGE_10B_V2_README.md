# Stage 10B v2 — Deterministic Benchmark v1.1 correction

You do **not** need to edit `minami_2023_validation_labels.csv`.

The 17 experimental outcomes from Minami et al. Table 1 are encoded directly
in the script by unique JW ID.

## Table 1 outcomes encoded

### GroE-dependent (+)

```text
JW0843  rumB
JW3699  bglB
JW0760  bioC
JW2308  ubiX
JW2650  nrdE
JW2781  csdA
JW3488  yhjB
JW4221  idnR
JW5454  yqeH
```

### GroE-independent (-)

```text
JW0495  gcl
JW0762  uvrB
JW1439  ydcW
JW2686  ascB
JW0997  ycdM
JW1201  hemA
JW1612  malI
JW3958  thiC
```

## Install

```bash
python3 -m pip install -r requirements_10B_v2.txt
```

## Run

```bash
python3 scripts/10B_build_benchmark_v1_1_from_minami_table1.py \
  --benchmark data/frozen/benchmark_v1_0/benchmark_v1.0.csv \
  --feature-dir data/frozen/feature_set_v2 \
  --output-dir data/frozen/benchmark_v1_1
```

## The script will stop if

- any of the 17 JW IDs is missing;
- any JW ID appears more than once;
- a JW ID maps to the wrong gene;
- the benchmark does not contain exactly nine 0 -> 1 label corrections;
- an unexpected label change is required;
- a frozen Feature Set v2 file is missing.

## Outputs

```text
data/frozen/benchmark_v1_1/
├── benchmark_v1.1.csv
├── minami_table1_label_audit.csv
├── benchmark_v1.0_vs_v1.1_changes.csv
├── experimentally_confirmed_independent_proteins.csv
├── feature_file_patch_audit.csv
├── stage10B_v2_summary.json
├── benchmark_v1.1_methods.md
├── benchmark_v1.1_sha256.txt
└── feature_set_v2_labels_v1.1/
    ├── feature_set_v2_sequence_final.csv
    ├── feature_set_v2_structure_minimal_final.csv
    ├── feature_set_v2_structure_core_final.csv
    ├── feature_set_v2_sequence_plus_structure_minimal.csv
    └── feature_set_v2_sequence_plus_structure_core.csv
```

## After the run

Before retraining anything, inspect:

```text
stage10B_v2_summary.json
benchmark_v1.0_vs_v1.1_changes.csv
minami_table1_label_audit.csv
feature_file_patch_audit.csv
```

Expected result:

```text
Table 1 proteins audited: 17
Labels changed 0 -> 1: 9
```

Do not delete Benchmark v1.0 or the old Stage 08/09/10A results. They remain the
historical pre-correction analysis.
