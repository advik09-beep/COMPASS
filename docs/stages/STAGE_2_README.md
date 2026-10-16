# Stage 2: Download sequences and AlphaFold structures

This stage uses the validated `fujiwara_class_iv_identifiers.csv`.

## Install

```bash
python3 -m pip install -r requirements_downloads.txt
```

## Run

```bash
python3 02_download_sequences_structures.py \
  --input fujiwara_class_iv_identifiers.csv \
  --output-dir data
```

The script downloads both mmCIF and PDB coordinate files by default. To keep
only mmCIF:

```bash
python3 02_download_sequences_structures.py \
  --input fujiwara_class_iv_identifiers.csv \
  --output-dir data \
  --skip-pdb
```

## Outputs

```text
data/
├── sequences/
├── structures/
├── alphafold_metadata/
├── download_manifest.csv
└── download_failures.csv
```

A successful run should end with:

```text
Complete: 57/57
```

The command can be re-run safely. Existing valid files are reused unless
`--overwrite` is supplied.
