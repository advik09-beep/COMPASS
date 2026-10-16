# Stage 07 — MMseqs2 Homology and Fold Grouping

## Confirm MMseqs2

```bash
mmseqs version
```

## Install Python dependencies

```bash
python3 -m pip install -r requirements_07.txt
```

## Run from the project root

```bash
python3 scripts/07_build_homology_and_fold_groups.py \
  --benchmark data/frozen/benchmark_v1_0/benchmark_v1.0.csv \
  --source-features data/modeling/minami_binary_benchmark_features_181.csv \
  --output-dir data/grouping/stage07
```

The script automatically:

- centralizes all 181 canonical FASTA sequences;
- runs MMseqs2 at 30% and 40% identity;
- requires 80% bidirectional full-length coverage (`--cov-mode 0 -c 0.80`);
- parses MMseqs2 cluster TSV files;
- parses SCCS into SCOP class/fold/superfamily sets;
- marks TIM barrels (`c.1`);
- generates balanced homology-grouped CV folds;
- verifies zero homology-cluster leakage;
- audits stricter joint homology-and-fold components.

## Main outputs

```text
data/grouping/stage07/
├── all_181_sequences.fasta
├── sequence_manifest.csv
├── fold_assignments_v1.0.csv
├── homology_cluster_30_audit.csv
├── homology_cluster_40_audit.csv
├── homology_cv_fold_audit.csv
├── homology_cv_fold_sensitivity_audit.csv
├── scop_fold_audit.csv
├── joint_component_audit.csv
├── stage07_summary.json
├── grouping_methods_v1.0.md
├── mmseqs_id30/
└── mmseqs_id40/
```

If an MMseqs2 run is interrupted, rerun the same command. Existing cluster TSV
files are reused. Add `--force-mmseqs` only when intentionally regenerating them.
