# Methods: benchmark, structural topology, energetic frustration, and evaluation

## Research question

Does energetic frustration convey predictive information about *E. coli* obligate GroEL/GroES dependence beyond global structural topology?

## Dataset and labels

The frozen corrected benchmark v1.1 contains **181 proteins** with a binary label: **73 obligate** and **108 non-obligate**. The updated labels reside in the frozen feature-set v3 file, not in the older fold-assignment CSV. The latter's label column predates the v1.1 correction and reflects only 64 positives. Scripts deliberately merge three fields from it: `current_uniprot_accession`, `homology_cv_fold`, and `homology_cluster_30`.

Original cohort construction and subsequent label-correction work are preserved in `Scripts/04_*`, `Scripts/06*`, `Scripts/07_*`, and `Scripts/10B_*` and their stage notes. Review these for exact external source extraction and labeling rules.

## Structural core: 15 predictors

`structure_ca_length`, `relative_contact_order`, `median_contact_sequence_separation`, `long_range_contact_fraction`, `medium_range_contact_fraction`, `ca_contact_density_per_residue`, `ca_contact_pair_fraction`, `zero_contact_residue_fraction`, `radius_of_gyration_normalized_n_cuberoot`, `axis_ratio_1_to_2`, `asphericity`, `ss_helix_fraction`, `ss_sheet_fraction`, `minimum_cut_cross_contact_density`, `best_cut_fraction`.

The core includes protein size, long-range contact organization, contact-network connectivity, shape/compactness, and secondary-structure composition.

## Energetic frustration: 12 predictors

The extractor (`Scripts/stage09_4F_extract_frustration.py`) uses Python Frustratometer 0.3.2 with an AWSEM-based model. Analysis contacts are unique residue pairs separated by at least four positions along the sequence, with Cβ distances below 8 Å. The script defines high frustration below an index of -1.0 and minimal frustration above 0.78. A long-range descriptor applies a sequence-separation cutoff of 24. The recorded configurational-decoy default is 4,000; see code and generated configuration for the exact run conditions.

For each of **mutational** and **configurational** frustration, the model uses six descriptors: mean, standard deviation, fraction of highly frustrated contacts, fraction of minimally frustrated contacts, fraction of residues participating in a highly frustrated contact, and fraction of highly frustrated long-range contacts. Counts and molecule size from the extraction table are **not** included among the 12 frustration model predictors.

## Model configurations

- Structural core (15).
- Frustration only (12).
- Structural + all frustration (27).
- Structural + mutational (21).
- Structural + configurational (21).
- Structural + distribution (`mut_mean`, `mut_std`; 17).
- Structural + spatial (`mut_high_residue_fraction`, `mut_long_range_high_fraction`; 17).
- Structural + reduced mutational (the four preceding mutational fields; 19).

The reduced configuration excludes `mut_minimal_fraction` and `mut_high_fraction` following correlation and permutation checks. This choice was made *after* examining performance on the shared benchmark, so it is exploratory rather than pre-registered.

## Cross-validation and modeling

Five preassigned outer folds group proteins by 30%-sequence-identity homology clusters. For each outer fold, model hyperparameters are selected using 3-fold `StratifiedGroupKFold` on the outer training partition only, searching 80 randomized random-forest configurations. Test-fold probabilities come from the refitted outer-training model. Predictions are produced with a **fixed 0.5 decision threshold**, not a threshold optimized on the held-out data. The random seed is 42, with fold-specific deterministic seeds.

The main evaluation uses `Scripts/09_v2_final_random_forest_FIXED.py` as the shared implementation. As of these experiments, the relevant scikit-learn version is 1.7.2; older stage experiments used other versions and should not be equated with the current within-study baseline.

**Pooled out-of-fold MCC** is computed across all 181 unique held-out predictions per model. ROC-AUC and PR-AUC are computed from pooled probabilities, not averages of fold AUCs. Balanced accuracy and fold-level metrics are also reported.

## Diagnostics and exploratory interpretation

- Leakage audits confirm no 30%-identity homology-cluster overlap between outer train and test partitions.
- A reconstruction audit refit saved fold-specific random forests and matched original out-of-fold classifications on all 181 proteins, with probability differences at floating-point precision.
- Held-out permutation importance uses 30 repeats per predictor for the structural + mutational configuration; correlations between inputs can obscure individual importance.
- Spearman correlations were examined between structural and mutational descriptors and within the mutational group. Several mutational descriptors are highly correlated.
- The full mutational add-back improves pooled MCC relative to structural core but slightly reduces PR-AUC; reduced and alternative subsets differ by metric.

## Inference caveat

Repeatedly inspecting eight configurations on the same five outer folds creates model-selection optimism. The reported best MCC is the **best exploratory observation**, not an externally validated deployment estimate. Testing a prespecified configuration on an independent, homology-disjoint benchmark is the appropriate next stage. No causal claim is implied.
