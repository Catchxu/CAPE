import anndata as ad
import numpy as np

from src.models.scbert.utils import align_adata_to_gene_order, matrix_to_scbert_sequences
from src.models.scgpt.tokenizer import GeneVocab, tokenize_and_pad_batch
from src.models.scgpt.utils import filter_adata_by_vocab


def test_scbert_alignment_and_sequence_conversion():
    adata = ad.AnnData(np.array([[1.2, 0.0, 3.9], [0.0, 2.1, 1.4]], dtype=np.float32))
    adata.var_names = ["g2", "g1", "g3"]
    aligned = align_adata_to_gene_order(adata, ["g1", "g2", "g4"])
    sequences = matrix_to_scbert_sequences(aligned, max_bin=5)
    assert sequences.shape == (2, 4)
    assert sequences[0, -1] == 0


def test_scgpt_vocab_filter_and_tokenization():
    adata = ad.AnnData(np.array([[0, 2, 1], [5, 0, 3]], dtype=np.float32))
    adata.var_names = ["g1", "g2", "g3"]
    vocab = GeneVocab.from_dict({"<pad>": 0, "<cls>": 1, "g1": 2, "g3": 3})
    filtered, _, gene_ids = filter_adata_by_vocab(adata, vocab)
    tokenized = tokenize_and_pad_batch(
        data=filtered.X,
        gene_ids=gene_ids,
        max_len=4,
        vocab=vocab,
        pad_token="<pad>",
        pad_value=0,
        append_cls=True,
        include_zero_gene=False,
    )
    assert tokenized["genes"].shape == (2, 3)
    assert tokenized["values"].shape == (2, 3)
