# Stage 06 - Dataset QC and exploratory analysis

This stage audits the 181-protein Minami benchmark before any machine-learning
model is trained.

## Install dependencies

```bash
python3 -m pip install -r requirements_qc.txt
```

## Run from the project root

```bash
python3 scripts/06_dataset_qc_and_eda.py \
  --input data/modeling/minami_binary_benchmark_features_181.csv \
  --output-dir results/qc
```

## Main outputs

```text
results/qc/
├── QC_Report.pdf
├── qc_summary.json
├── ml_dataset_structure_broad.csv
├── ml_dataset_structure_core.csv
├── figures/
│   └── feature_distributions/
└── tables/
    ├── integrity_audit.csv
    ├── feature_role_manifest.csv
    ├── missingness_report.csv
    ├── constant_features.csv
    ├── high_missing_features.csv
    ├── univariate_feature_statistics.csv
    ├── spearman_correlation_matrix.csv
    ├── high_correlation_pairs.csv
    ├── scop_class_composition.csv
    ├── primary_sccs_composition.csv
    ├── tim_barrel_enrichment.json
    └── core_feature_list.csv
```

## Scientific rules encoded in the script

The following are not supplied to a classifier:

- labels;
- identifiers and file paths;
- Minami's published relative contact order;
- Minami's RW and RW+RR predictions;
- Minami's original structure source;
- SCCS fold labels.

SCCS is retained only for fold-bias analysis and future fold-aware validation.

AlphaFold confidence and pipeline-QC variables are excluded from the primary
biological feature set. They can be used later in sensitivity analyses.

The script exports a broad feature set and a smaller, restrained core set. It
does not train a model. A successful audit means the next stage is homology and
fold grouping, not an ordinary random train/test split.
