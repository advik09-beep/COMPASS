# Data provenance and redistribution policy

## Dataset construction

The project harmonizes published GroEL/GroES-dependence classifications (including sources described in the original Stage 02–10 documentation) with UniProt accessions and structural descriptors extracted from protein structures. The codebase retains its original scripts for mapping identifiers, downloading sequences and structures, filtering labels, constructing homology groups, and auditing revised benchmark v1.1 labels.

Original source-table files, structure downloads, and generated benchmark tables have **not** been provided in this GitHub checkout. See the historical stage READMEs for the project's internal derivation chronology. Specific paper citations, versions, and source license verification should be completed against the original tables rather than inferred from filenames.

## Frozen benchmark v1.1

The latest frozen table has 181 records: 73 positive and 108 negative. The original homology-fold CSV predates the label correction (64 positive) and must be joined without its label column. Sequence-identity grouping uses the frozen 30% homology clusters; no fold definitions were changed during the frustration add-back study.

## Energetic feature derivation

The 12 frustration predictors derive from per-protein mutational/configurational frustration calculations using Python Frustratometer 0.3.2 and AWSEM. Source structure paths, exact files, configs and contacts are recorded by the extraction pipeline where the local generated metadata is available. They have not been included in the source release.

## Redistribution

The presence of identifier-mapping or download code does not itself authorize redistribution of every upstream dataset or structure. Until the original scholarly supplementary material, UniProt/AlphaFold downloads and any other underlying sources are checked for their respective reuse terms, do not publish the raw source tables, derived label tables, or structure files from the workstation by default. Instead, document lawful retrieval/reconstruction steps or share only verified redistributable artifacts.

### Pending provenance checklist

- Record full citations and DOI/URL for Fujiwara, Minami and other referenced primary sources, confirming precisely which tables/labels were used.
- Record each original table's licensing and reuse permissions.
- Document the exact download/accession mapping versions, and allowed derivative releases.
- Verify whether the proposed 181-row structural/label table, fold CSV, and derived frustration table can be redistributed.
- Publish SHA-256 hashes of release artifacts and validate against modeling audit outputs.

No claims of public benchmark availability are made by the initial GitHub source upload.
