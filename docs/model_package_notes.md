# AceFormer Model Package Notes

This folder contains inference-ready AceFormer assets for GitHub release and downstream proteome-scale prediction.

## Included Assets

- `aceformer_predict.py`: main command-line prediction script.
- `models/general`: final general acetylation ensemble checkpoints and summary files.
- `models/enzyme_specific`: final enzyme-specific fine-tuned checkpoints grouped by enzyme.
- `examples`: small FASTA and selected-site inputs for demonstration.
- `tests`: smoke-test scripts.

## Prediction Strategy

AceFormer generates a 31-mer peptide centered on the queried lysine and combines one-hot peptide encoding with ESM2 context embeddings. For very long proteins, ESM2 input is cropped around the queried site while keeping the local residue centered in the embedding context.

## Reproducibility

The packaged model corresponds to the final retained AceFormer model archive in `/Users/chenwei/Desktop/acetylation/3.model_training`. Training data and manuscript source tables are maintained separately in the manuscript directory.
