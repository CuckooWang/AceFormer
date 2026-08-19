#!/usr/bin/env python3
"""Command-line prediction utility for AceFormer."""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, EsmModel


AA_LIST = list("ACDEFGHIKLMNPQRSTVWYX")
AA_DICT = {aa: i for i, aa in enumerate(AA_LIST)}


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class DeepFusionNet(nn.Module):
    def __init__(self, esm_dim, pep_dim, dropout=0.3):
        super().__init__()
        self.esm_branch = nn.Sequential(
            nn.Linear(esm_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            ResidualBlock(512, dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pep_branch = nn.Sequential(
            nn.Linear(pep_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            ResidualBlock(512, dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            ResidualBlock(256, dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x_esm, x_pep):
        h_esm = self.esm_branch(x_esm)
        h_pep = self.pep_branch(x_pep)
        h = torch.cat([h_esm, h_pep], dim=1)
        return self.classifier(h * self.gate(h)).squeeze(1)


class PredictionDataset(Dataset):
    def __init__(self, feature_df):
        self.esm_cols = [c for c in feature_df.columns if c.startswith("esm_")]
        self.pep_cols = [c for c in feature_df.columns if c.startswith("pep_") and c != "pep_31mer"]
        self.x_esm = feature_df[self.esm_cols].to_numpy(dtype=np.float32)
        self.x_pep = feature_df[self.pep_cols].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.x_esm)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x_esm[idx]), torch.from_numpy(self.x_pep[idx])


def parse_fasta(path):
    records = []
    header = None
    seq_parts = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append(record_from_header(header, "".join(seq_parts)))
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
    if header is not None:
        records.append(record_from_header(header, "".join(seq_parts)))
    return records


def record_from_header(header, sequence):
    protein_id = header.split()[0]
    gene_match = re.search(r"\bGN=([^\s]+)", header)
    gene = gene_match.group(1) if gene_match else ""
    if "|" in protein_id:
        fields = protein_id.split("|")
        if len(fields) >= 2:
            protein_id = fields[1]
    return {
        "protein_id": protein_id,
        "gene": gene,
        "description": header,
        "sequence": sequence.upper(),
    }


def make_31mer(sequence, position):
    idx = int(position) - 1
    return "".join(sequence[i] if 0 <= i < len(sequence) else "X" for i in range(idx - 15, idx + 16))


def one_hot(peptide):
    arr = np.zeros((len(peptide), len(AA_LIST)), dtype=np.float32)
    for i, aa in enumerate(peptide):
        arr[i, AA_DICT.get(aa, AA_DICT["X"])] = 1.0
    return arr.reshape(-1)


def sites_from_fasta(fasta_path, predict, protein_id=None, position=None):
    records = parse_fasta(fasta_path)
    rows = []
    for rec in records:
        if protein_id and rec["protein_id"] != protein_id and rec["gene"] != protein_id:
            continue
        sequence = rec["sequence"]
        if predict == "selected_sites":
            if position is None:
                raise ValueError("--position is required when --input_type fasta --predict selected_sites")
            positions = [int(position)]
        else:
            positions = [i for i, aa in enumerate(sequence, start=1) if aa == "K"]
        for pos in positions:
            if pos < 1 or pos > len(sequence):
                raise ValueError(f"Position {pos} is outside sequence length for {rec['protein_id']}")
            rows.append(
                {
                    "protein_id": rec["protein_id"],
                    "gene": rec["gene"],
                    "position": pos,
                    "residue": sequence[pos - 1],
                    "sequence": sequence,
                    "pep_31mer": make_31mer(sequence, pos),
                }
            )
    return pd.DataFrame(rows)


def sites_from_table(path):
    df = pd.read_csv(path, sep="\t")
    if "protein_id" not in df.columns and "upid" in df.columns:
        df = df.rename(columns={"upid": "protein_id"})
    required = {"protein_id", "sequence", "position"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Site table is missing required columns: {', '.join(missing)}")
    df["sequence"] = df["sequence"].astype(str).str.upper()
    df["position"] = df["position"].astype(int)
    if "gene" not in df.columns:
        df["gene"] = ""
    if "residue" not in df.columns:
        df["residue"] = [seq[pos - 1] if 1 <= pos <= len(seq) else "" for seq, pos in zip(df["sequence"], df["position"])]
    if "pep_31mer" not in df.columns:
        df["pep_31mer"] = [make_31mer(seq, pos) for seq, pos in zip(df["sequence"], df["position"])]
    return df


def crop_sequence_for_esm(sequence, position, max_context):
    if len(sequence) <= max_context:
        return sequence, position
    half = max_context // 2
    start = max(0, min(position - half - 1, len(sequence) - max_context))
    end = start + max_context
    return sequence[start:end], position - start


def generate_features(site_df, esm_model_name, window, max_context, batch_message_every, device):
    tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
    esm_model = EsmModel.from_pretrained(esm_model_name).to(device)
    esm_model.eval()
    cache = {}
    half = window // 2
    esm_k, esm_mean, esm_max, esm_std, pep = [], [], [], [], []

    with torch.no_grad():
        for i, row in site_df.iterrows():
            sequence = str(row["sequence"]).upper()
            position = int(row["position"])
            esm_sequence, esm_position = crop_sequence_for_esm(sequence, position, max_context)
            cache_key = esm_sequence
            if cache_key not in cache:
                inputs = tokenizer(esm_sequence, return_tensors="pt", truncation=True, max_length=max_context + 2)
                inputs = {key: value.to(device) for key, value in inputs.items()}
                cache[cache_key] = esm_model(**inputs).last_hidden_state[0].cpu()
            hidden = cache[cache_key]
            start = max(1, esm_position - half)
            end = min(len(esm_sequence), esm_position + half)
            local = hidden[start : end + 1]
            esm_k.append(hidden[esm_position].numpy())
            esm_mean.append(local.mean(dim=0).numpy())
            esm_max.append(local.max(dim=0).values.numpy())
            esm_std.append(local.std(dim=0, unbiased=False).numpy())
            pep.append(one_hot(str(row["pep_31mer"])))
            if batch_message_every and (i + 1) % batch_message_every == 0:
                print(f"Generated features for {i + 1} sites | cached ESM sequences: {len(cache)}")

    blocks = []
    for prefix, values in [
        ("esm_k", esm_k),
        ("esm_window_mean", esm_mean),
        ("esm_window_max", esm_max),
        ("esm_window_std", esm_std),
        ("pep", pep),
    ]:
        block = pd.DataFrame(np.asarray(values, dtype=np.float32))
        block.columns = [f"{prefix}_{j}" for j in range(block.shape[1])]
        blocks.append(block)
    return pd.concat([site_df.reset_index(drop=True)] + blocks, axis=1)


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def find_checkpoints(model_dir, mode, enzyme=None):
    base = Path(model_dir)
    if mode == "general":
        paths = sorted((base / "general").glob("fold_*_model_*.pt"))
    else:
        paths = sorted((base / "enzyme_specific" / enzyme).glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No model checkpoints found for {mode}" + (f" / {enzyme}" if enzyme else ""))
    return paths


def available_enzymes(model_dir):
    enzyme_root = Path(model_dir) / "enzyme_specific"
    if not enzyme_root.exists():
        return []
    return sorted([p.name for p in enzyme_root.iterdir() if p.is_dir() and list(p.glob("*.pt"))])


def predict_with_checkpoints(feature_df, checkpoints, batch_size, dropout, device):
    dataset = PredictionDataset(feature_df)
    esm_dim = len(dataset.esm_cols)
    pep_dim = len(dataset.pep_cols)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_probs = []
    for checkpoint in checkpoints:
        model = DeepFusionNet(esm_dim=esm_dim, pep_dim=pep_dim, dropout=dropout).to(device)
        model.load_state_dict(safe_torch_load(checkpoint, device))
        model.eval()
        probs = []
        with torch.no_grad():
            for x_esm, x_pep in loader:
                logits = model(x_esm.to(device), x_pep.to(device))
                probs.extend(torch.sigmoid(logits).cpu().numpy())
        all_probs.append(np.asarray(probs, dtype=np.float32))
    return np.mean(np.vstack(all_probs), axis=0)


def add_predictions(feature_df, model_dir, mode, enzymes, batch_size, threshold, dropout, device, include_sequence_output):
    base_cols = [c for c in feature_df.columns if not c.startswith("esm_") and not (c.startswith("pep_") and c != "pep_31mer")]
    if not include_sequence_output:
        base_cols = [c for c in base_cols if c != "sequence"]
    output_base = feature_df[base_cols].copy()
    rows = []
    if mode == "general":
        probs = predict_with_checkpoints(feature_df, find_checkpoints(model_dir, "general"), batch_size, dropout, device)
        out = output_base.copy()
        out["model_mode"] = "general"
        out["enzyme"] = "general"
        out["probability"] = probs
        out["prediction_label"] = (probs >= threshold).astype(int)
        return out
    for enzyme in enzymes:
        probs = predict_with_checkpoints(feature_df, find_checkpoints(model_dir, "enzyme_specific", enzyme), batch_size, dropout, device)
        out = output_base.copy()
        out["model_mode"] = "enzyme_specific"
        out["enzyme"] = enzyme
        out["probability"] = probs
        out["prediction_label"] = (probs >= threshold).astype(int)
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def write_default_config(path):
    config = {
        "model": "AceFormer",
        "architecture": "Two-branch ESM2 and peptide one-hot fusion neural network",
        "esm_model": "facebook/esm2_t12_35M_UR50D",
        "peptide_window": 31,
        "threshold": 0.5,
        "general_model": "5-fold ensemble",
        "enzyme_specific_model": "5-checkpoint ensemble per enzyme",
    }
    with open(path, "w") as handle:
        json.dump(config, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="AceFormer prediction for general acetylation and enzyme-specific acetylation/deacetylation models.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python aceformer_predict.py --mode general --input proteins.fasta --input_type fasta --predict all_k --output general.tsv\n"
            "  python aceformer_predict.py --mode enzyme_specific --enzyme SIRT3 --input sites.tsv --input_type site_table --output SIRT3.tsv\n"
            "  python aceformer_predict.py --mode enzyme_all --input sites.tsv --input_type site_table --top_n 3 --output enzyme_top3.tsv"
        ),
    )
    parser.add_argument("--mode", choices=["general", "enzyme_specific", "enzyme_all"], default="general", help="Prediction mode.")
    parser.add_argument("--input", metavar="FILE", help="Input FASTA or tab-delimited selected-site table.")
    parser.add_argument("--input_type", choices=["fasta", "site_table"], default="fasta", help="Input file format.")
    parser.add_argument("--predict", choices=["all_k", "selected_sites"], default="all_k", help="Predict all lysines or selected sites.")
    parser.add_argument("--protein_id", metavar="ID_OR_GENE", help="Protein accession or gene name for single-site FASTA prediction.")
    parser.add_argument("--position", metavar="POS", type=int, help="1-based lysine position for single-site FASTA prediction.")
    parser.add_argument("--enzyme", metavar="ENZYME_NAME", help="Enzyme name for enzyme_specific mode, for example SIRT3.")
    parser.add_argument("--model_dir", metavar="DIR", default="models", help="Directory containing AceFormer model checkpoints. Default: models")
    parser.add_argument("--output", metavar="TSV", default="aceformer_predictions.tsv", help="Output prediction table. Default: aceformer_predictions.tsv")
    parser.add_argument("--threshold", metavar="FLOAT", type=float, default=0.5, help="Probability threshold for binary prediction. Default: 0.5")
    parser.add_argument("--batch_size", metavar="N", type=int, default=64, help="Batch size for neural-network inference. Default: 64")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Computation device.")
    parser.add_argument("--esm_model", metavar="MODEL_NAME", default="facebook/esm2_t12_35M_UR50D", help="ESM2 model used for feature generation. Default: facebook/esm2_t12_35M_UR50D")
    parser.add_argument("--window", metavar="N", type=int, default=31, help="Peptide and local ESM window size. Default: 31")
    parser.add_argument("--esm_context", metavar="N", type=int, default=1022, help="Maximum amino-acid context passed to ESM2 for each site. Default: 1022")
    parser.add_argument("--dropout", metavar="FLOAT", type=float, default=0.3, help="Dropout value used to instantiate the trained model architecture. Default: 0.3")
    parser.add_argument("--top_n", metavar="N", type=int, help="For enzyme_all mode, keep the top N enzymes for each site.")
    parser.add_argument("--min_prob", metavar="FLOAT", type=float, help="Only keep predictions with probability at or above this value.")
    parser.add_argument("--limit_sites", metavar="N", type=int, help="Predict only the first N sites; useful for quick testing.")
    parser.add_argument("--chunk_size", metavar="N", type=int, default=50000, help="Number of sites processed per output chunk. Default: 50000")
    parser.add_argument("--include_sequence_output", action="store_true", help="Keep full protein sequences in the output table.")
    parser.add_argument("--save_features", metavar="TSV", help="Optional path to save generated feature table.")
    parser.add_argument("--list_enzymes", action="store_true", help="List available enzyme-specific models and exit.")
    parser.add_argument("--write_config", metavar="JSON", help="Write a small JSON model configuration file and exit.")
    args = parser.parse_args()

    if args.write_config:
        write_default_config(args.write_config)
        return
    if args.list_enzymes:
        print("\n".join(available_enzymes(args.model_dir)))
        return
    if not args.input:
        parser.error("--input is required unless --list_enzymes or --write_config is used")
    if args.mode == "enzyme_specific" and not args.enzyme:
        parser.error("--enzyme is required for --mode enzyme_specific")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if args.input_type == "fasta":
        site_df = sites_from_fasta(args.input, args.predict, args.protein_id, args.position)
    else:
        site_df = sites_from_table(args.input)
    if args.limit_sites:
        site_df = site_df.head(args.limit_sites).copy()
    if site_df.empty:
        raise ValueError("No prediction sites were found.")
    if any(site_df["residue"].astype(str).str.upper() != "K"):
        print("Warning: at least one selected site is not lysine (K). Predictions will still be produced.")

    print(f"Prediction sites: {len(site_df)}")
    print(f"Device: {device}")

    if args.mode == "general":
        enzymes = []
    elif args.mode == "enzyme_specific":
        enzymes = [args.enzyme]
    else:
        enzymes = available_enzymes(args.model_dir)
        if not enzymes:
            raise FileNotFoundError("No enzyme-specific model folders were found.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_path = Path(args.save_features) if args.save_features else None
    if feature_path:
        feature_path.parent.mkdir(parents=True, exist_ok=True)

    first_chunk = True
    chunk_size = args.chunk_size or len(site_df)
    for start in range(0, len(site_df), chunk_size):
        end = min(start + chunk_size, len(site_df))
        print(f"Processing sites {start + 1}-{end} of {len(site_df)}")
        chunk_df = site_df.iloc[start:end].copy()
        feature_df = generate_features(chunk_df, args.esm_model, args.window, args.esm_context, 100, device)
        if feature_path:
            feature_df.to_csv(feature_path, sep="\t", index=False, mode="w" if first_chunk else "a", header=first_chunk)
        pred_df = add_predictions(
            feature_df,
            args.model_dir,
            args.mode,
            enzymes,
            args.batch_size,
            args.threshold,
            args.dropout,
            device,
            args.include_sequence_output,
        )
        if args.min_prob is not None:
            pred_df = pred_df[pred_df["probability"] >= args.min_prob].copy()
        if args.top_n and args.mode == "enzyme_all":
            pred_df = (
                pred_df.sort_values(["protein_id", "position", "probability"], ascending=[True, True, False])
                .groupby(["protein_id", "position"], as_index=False)
                .head(args.top_n)
            )
        pred_df.to_csv(output_path, sep="\t", index=False, mode="w" if first_chunk else "a", header=first_chunk)
        first_chunk = False
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
