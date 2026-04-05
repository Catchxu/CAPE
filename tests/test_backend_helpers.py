import anndata as ad
import numpy as np

from src.models.scbert.processing_scbert import ScBertProcessor
from src.models.scgpt.processing_scgpt import ScGptProcessor


def test_scbert_processor_alignment_and_sequence_conversion():
    adata = ad.AnnData(np.array([[1.2, 0.0, 3.9], [0.0, 2.1, 1.4]], dtype=np.float32))
    adata.var_names = ["g2", "g1", "g3"]
    processor = ScBertProcessor(
        vocab={"g1": 0, "g2": 1, "g4": 2},
        bin_num=5,
        append_eos_token=True,
        eos_token_id=0,
        pad_token_id=0,
    )
    encoded = processor.encode_adata(adata, return_tensors="np")
    assert encoded["input_ids"].shape == (2, 4)
    assert encoded["input_ids"][0, -1] == 0


def test_scgpt_processor_vocab_filter_and_tokenization():
    adata = ad.AnnData(np.array([[0, 2, 1], [5, 0, 3]], dtype=np.float32))
    adata.var_names = ["g1", "g2", "g3"]
    processor = ScGptProcessor(
        vocab={"<pad>": 0, "<cls>": 1, "<eoc>": 2, "g1": 3, "g3": 4},
        max_seq_len=4,
        pad_token="<pad>",
        cls_token="<cls>",
        pad_value=0,
        mask_value=-1,
        append_cls=True,
        include_zero_gene=False,
        n_input_bins=5,
    )
    encoded = processor.encode_adata(adata, return_tensors="pt")
    assert encoded["gene_ids"].shape == (2, 3)
    assert encoded["values"].shape == (2, 3)
