# Stage 09.3B — Sequence-Family Add-Back

Stage 09.3A gave us two important observations:

1. Structural Core remains the strongest current RF representation.
2. Some sequence variables are useful individually, even though adding all
   sequence variables reduces overall performance.

09.3B tests **why**.

## Principle

We do NOT remove arbitrary individual sequence variables based on the previous
test-set results.

Instead, sequence variables are assigned to predefined biological families,
then each family is added to the same Structural Core baseline.

The experiment also tests the specific structural concern identified by 09.3A:

- Structural Core − asphericity
- Structural Core − global shape family

The shape-family test is intentionally narrow:
`asphericity` + `axis_ratio_1_to_2`.

Normalized radius of gyration stays in the model because 09.3A showed strong
positive contribution.

## Candidate models

```text
Structural Core
Structural Core − asphericity
Structural Core − shape family

Structural Core + amino-acid composition
Structural Core + charge/electrostatics
Structural Core + hydrophobicity
Structural Core + sequence complexity
Structural Core + other sequence physicochemical descriptors

Combined Core control
```

No arbitrary combinations of whichever families happen to score highest are
generated in this stage.

## Run

```bash
python3 scripts/09_3B_sequence_family_addback.py \
  --feature-dir data/frozen/feature_set_v3 \
  --fold-assignments data/grouping/stage07/fold_assignments_v1.0.csv \
  --output-dir results/stage09_3B
```

This may take several hours because every candidate representation receives a
fresh nested grouped-CV Random Forest evaluation.

## Outputs to upload

Start with:

```text
stage09_3B_summary.json
stage09_3B_summary_metrics.csv
stage09_3B_paired_bootstrap_vs_structural_core.csv
feature_family_manifest.csv
```

If we need fold-level diagnosis afterward, also upload:

```text
stage09_3B_fold_metrics.csv
```

## Decision after 09.3B

We will not simply select the highest MCC.

We will examine:

- MCC
- ROC-AUC
- PR-AUC
- paired bootstrap difference from Structural Core
- fold consistency
- biological coherence
- whether the result agrees with 09.3A SHAP/LOFO

Only then will we define a candidate Stage 09 v3 feature representation.

After Stage 09 v3 is frozen, the next development cycle moves from global
features to **local structural and local sequence features**.
