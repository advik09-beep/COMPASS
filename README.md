# COMPASS

**A CHAPERONIN OBLIGATE-DEPENDENCE MODEL USING PROTEIN ARCHITECTURE, SEQUENCE AND STRUCTURE**

COMPASS is a research pipeline studying whether global protein structural features and energetic frustration help predict *obligate dependence* on the bacterial GroEL/GroES chaperonin system. The project includes benchmark construction, leakage-controlled modeling, feature ablations, and model interpretation.

> **Research status (2026):** Curated source-code release. The repository does **not** contain all frozen input tables or full analysis outputs. Reported scores are from completed local experiments and **have not been independently reproduced from this GitHub checkout**.

## Study at a glance

| Design choice | Specification |
| --- | --- |
| Organism | *Escherichia coli* |
| Frozen benchmark | 181 proteins: 73 obligate and 108 non-obligate |
| Structural descriptors | 15 predictors describing global fold topology and secondary structure |
| Energetic-frustration descriptors | 6 mutational and 6 configurational predictors |
| Outer validation | 5 fixed, 30%-identity homology-grouped folds |
| Inner model selection | 3-fold stratified, homology-group-aware CV |
| Estimator | Random forest; 80 randomized hyperparameter candidates per outer fold |
| Classification threshold | Fixed at 0.5 (inner-selected thresholds are not applied) |
| Primary descriptive metric | Pooled out-of-fold Matthews correlation coefficient (MCC) |
| Main reported environment | Python 3.10; scikit-learn 1.7.2; seed 42 |

## Observed exploratory results

All configurations use the same 181-protein benchmark and frozen outer-fold assignments. **These are descriptive, exploratory comparisons:** feature sets were examined iteratively using the same outer folds; therefore, the best observed value is not an unbiased estimate of a model chosen after these comparisons.

| Feature configuration | Predictors | Pooled MCC | Balanced accuracy | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Structural + all mutational frustration | 21 | **0.5482** | **0.7706** | 0.8330 | 0.7384 |
| Structural + configurational frustration | 21 | 0.5159 | 0.7567 | 0.8189 | 0.7602 |
| Structural + mutational distribution (`mean`, `std`) | 17 | 0.5159 | 0.7567 | **0.8380** | 0.7476 |
| Structural + mutational spatial descriptors | 17 | 0.5126 | 0.7523 | 0.8229 | 0.7466 |
| Structural + four reduced mutational descriptors | 19 | 0.5126 | 0.7523 | 0.8318 | **0.7635** |
| Structural core | 15 | 0.5003 | 0.7454 | 0.8258 | 0.7523 |
| Structural + all 12 frustration descriptors | 27 | 0.4535 | 0.7225 | 0.8214 | 0.7368 |
| Frustration only | 12 | 0.3130 | 0.6595 | 0.6894 | 0.5987 |

The highest **observed** MCC comes from the 15 structural features plus six mutational frustration features (0.5482 vs. 0.5003 for structural only). A paired classification comparison on the same held-out proteins showed six corrected and two newly incorrect classifications. Performance gains are metric-dependent: the full mutational model has *lower PR-AUC* than structural only (0.7384 vs. 0.7523), and the reduced feature set has higher PR-AUC than the full mutational model. These results do not establish causality or performance on an independent external cohort.

A separate five-fold held-out permutation-importance analysis ranked **minimum-cut cross-contact density**, **Cα contact-pair fraction**, and **sequence length** among the highest contributors for the structural + mutational model. Correlation and permutation analyses are descriptive and sensitive to correlated predictors.

## Repository guide

- [`Scripts/`](Scripts/) — **19 selected research scripts**: input construction and label audits, the current modeling implementation, frustration extraction, feature ablations and evaluation diagnostics. Poster graphics, preliminary models and superseded script versions are excluded. `09_v2_final_random_forest_FIXED.py` supplies shared model-building functions.
- [`docs/methods.md`](docs/methods.md) — benchmark, features, CV, and scientific interpretation.
- [`docs/reproducibility.md`](docs/reproducibility.md) — required local inputs, environment, and exact commands.
- [`docs/data_provenance.md`](docs/data_provenance.md) — source-data and redistribution considerations.
- [`docs/stages/`](docs/stages/) — eight selected contemporaneous methodology and provenance notes, retained as historical records; some refer to earlier script names or benchmark versions.
- [`requirements-analysis.txt`](requirements-analysis.txt) — principal analysis dependencies (not a tested, fully pinned environment); preprocessing and frustration extraction have additional tool-specific dependencies described in the scripts and reproducibility guide.
- [`results/stage09_4F/all_feature_configurations_summary.csv`](results/stage09_4F/all_feature_configurations_summary.csv) — user-reported consolidated experimental metrics (rounded as originally displayed).

## Quick start

From the repository root:

```bash
conda create -n compass python=3.10 -y
conda activate compass
python -m pip install -r requirements-analysis.txt
python -m compileall -q Scripts
```

**Full modeling requires the unpublished input tables** listed in [reproducibility instructions](docs/reproducibility.md); running the pipeline without them will raise `FileNotFoundError` as intended. When the tables are available:

```bash
python Scripts/stage09_4F_evaluate_frustration.py --audit-only
python Scripts/stage09_4F_evaluate_frustration.py --n-iter 80 --n-jobs 4 --seed 42
python Scripts/stage09_4F_group_ablation.py --n-iter 80 --n-jobs 4 --seed 42
python Scripts/stage09_4F_targeted_ablation.py --n-iter 80 --n-jobs 4 --seed 42
```

**Important label warning:** `data/grouping/stage07/fold_assignments_v1.0.csv` contains a stale `binary_label` field corresponding to an earlier benchmark (64 positives). The authoritative corrected v1.1 labels (73 positives) are in `data/frozen/feature_set_v3/feature_set_v3_structure_core_final.csv`. The current modeling scripts merge **only accession, fold, and homology cluster** from the old fold file; do not merge its labels. This is checked in the modeling input audit.

## Limitations and research ethics

The benchmark is small, class labels depend on literature-derived assignments, and homology grouping does not remove every possible dataset bias. Multiple feature configurations were evaluated on shared folds; independent homology-disjoint confirmation is required after selecting a model. The observed improvement is not evidence that frustration *causes* GroEL/GroES dependence. Energetic features depend on Frustratometer/AWSEM settings and structure availability. Upstream source tables and downloaded structure files are not redistributed until their licenses and provenance have been reviewed.

## Licensing and citation

A project-specific software license and complete source-data citations are **pending review**. No open-source license has been granted by this initial repository upload. Do not infer third-party data redistribution rights from this repository's existence. See [data provenance](docs/data_provenance.md).
