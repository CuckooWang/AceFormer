import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("aceformer_predict", ROOT / "aceformer_predict.py")
aceformer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aceformer)


def test_make_31mer_length_and_center():
    sequence = "M" + "A" * 14 + "K" + "G" * 20
    peptide = aceformer.make_31mer(sequence, 16)
    assert len(peptide) == 31
    assert peptide[15] == "K"


def test_fasta_parser_examples():
    records = aceformer.parse_fasta(ROOT / "examples" / "example_proteins.fasta")
    genes = {record["gene"] for record in records}
    assert {"TP53", "MYC", "SIRT3"}.issubset(genes)


if __name__ == "__main__":
    test_make_31mer_length_and_center()
    test_fasta_parser_examples()
    print("Helper tests passed.")
