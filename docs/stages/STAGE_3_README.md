# Stage 3: Structural feature extraction

## Install dependencies

```bash
python3 -m pip install -r requirements_features.txt
```

## Run

From the `GroElLabels` project directory:

```bash
python3 03_extract_structural_features.py \
  --manifest data/download_manifest.csv \
  --output-dir data/features
```

## Expected outputs

```text
data/features/
├── class_iv_structural_features.csv
└── class_iv_feature_failures.csv
```

A clean run should end with:

```text
Complete: 57/57
```

## Notes

- pLDDT is read from the AlphaFold PDB B-factor column.
- Secondary structure is assigned from CA geometry with Biotite's P-SEA
  implementation, so no external DSSP executable is needed.
- Contacts use CA distance <= 8 Å by default and exclude residue pairs with
  sequence separation below 3.
- Long-range contacts use sequence separation >= 24 by default.
- The sequence-cut bottleneck is a domain-separation proxy, not a formal domain
  prediction.
