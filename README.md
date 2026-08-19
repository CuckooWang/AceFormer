# AceFormer

AceFormer is a deep-learning predictor for lysine acetylation. It provides two prediction modes:

- `general`: predicts whether a lysine site is likely to be acetylated.
- `enzyme_specific`: predicts whether a lysine site is likely to be regulated by a selected acetyltransferase or deacetylase.
- `enzyme_all`: applies all packaged enzyme-specific models to the same input sites.

The model uses a two-branch neural network. One branch encodes ESM2 protein-context embeddings and the other branch encodes the local 31-mer peptide sequence; the two branches are fused by a gating module before binary classification.

## Installation

```bash
conda create -n aceformer python=3.9
conda activate aceformer
pip install -r requirements.txt
```

The first prediction run downloads the ESM2 model `facebook/esm2_t12_35M_UR50D` through Hugging Face Transformers unless it is already cached.

## Quick Start

List available enzyme-specific models:

```bash
python aceformer_predict.py --list_enzymes --model_dir models
```

Predict all lysine sites in a FASTA file with the general model:

```bash
python aceformer_predict.py \
  --mode general \
  --input examples/example_proteins.fasta \
  --input_type fasta \
  --predict all_k \
  --model_dir models \
  --output outputs/example_general_predictions.tsv
```

Predict one selected site from FASTA:

```bash
python aceformer_predict.py \
  --mode general \
  --input examples/example_proteins.fasta \
  --input_type fasta \
  --predict selected_sites \
  --protein_id TP53 \
  --position 120 \
  --model_dir models \
  --output outputs/example_TP53_K120_general.tsv
```

Predict selected sites with one enzyme-specific model:

```bash
python aceformer_predict.py \
  --mode enzyme_specific \
  --enzyme SIRT3 \
  --input examples/example_sites.tsv \
  --input_type site_table \
  --predict selected_sites \
  --model_dir models \
  --output outputs/example_SIRT3_predictions.tsv
```

Apply all enzyme-specific models to selected sites:

```bash
python aceformer_predict.py \
  --mode enzyme_all \
  --input examples/example_sites.tsv \
  --input_type site_table \
  --predict selected_sites \
  --model_dir models \
  --top_n 3 \
  --output outputs/example_all_enzyme_top3.tsv
```

## Input Files

FASTA input can contain one or more protein sequences. AceFormer scans all lysines when `--predict all_k` is used.

Selected-site tables must be tab-delimited and contain:

- `protein_id`: protein accession or identifier.
- `sequence`: full protein sequence.
- `position`: 1-based site position.

Optional columns such as `gene`, `residue`, and `pep_31mer` are retained in the output.

## Output

The prediction table includes input site annotations plus:

- `model_mode`
- `enzyme`
- `probability`
- `prediction_label`

By default, `prediction_label` uses a probability threshold of 0.5. Use `--threshold` to change this.

## Useful Parameters

Run:

```bash
python aceformer_predict.py --help
```

Common options:

- `--mode`: `general`, `enzyme_specific`, or `enzyme_all`.
- `--input_type`: `fasta` or `site_table`.
- `--predict`: `all_k` or `selected_sites`.
- `--enzyme`: enzyme name for `enzyme_specific` mode.
- `--device`: `auto`, `cuda`, or `cpu`.
- `--batch_size`: inference batch size.
- `--limit_sites`: only predict the first N sites for testing.
- `--min_prob`: only keep predictions above a probability cutoff.
- `--save_features`: save generated ESM2 and peptide features.

## Model Files

Packaged model checkpoints are stored in:

```text
models/
  general/
  enzyme_specific/
```

The general model is a 5-fold ensemble. Each enzyme-specific model is also predicted as an ensemble of five fine-tuned checkpoints.

## Test

After installation, run a small CPU smoke test:

```bash
bash tests/run_smoke_test.sh
```

For a very fast structure-only test that does not load ESM2 or model checkpoints:

```bash
python tests/test_aceformer_predict_helpers.py
```
