#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs

python aceformer_predict.py \
  --mode general \
  --input examples/example_proteins.fasta \
  --input_type fasta \
  --predict all_k \
  --model_dir models \
  --limit_sites 3 \
  --device cpu \
  --output outputs/smoke_general.tsv

python aceformer_predict.py \
  --mode enzyme_specific \
  --enzyme SIRT3 \
  --input examples/example_sites.tsv \
  --input_type site_table \
  --predict selected_sites \
  --model_dir models \
  --device cpu \
  --output outputs/smoke_SIRT3.tsv

echo "Smoke test completed."
