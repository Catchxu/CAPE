import json
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

from .pretrained import resolve_pretrained_dir


class ScBertProcessor:
    model_input_names = ["input_ids"]

    def __init__(
        self,
        vocab,
        bin_num=5,
        append_eos_token=True,
        eos_token_id=0,
        pad_token_id=0,
        vocab_file="vocab.json",
    ):
        self.vocab = {str(token): int(idx) for token, idx in vocab.items()}
        self.bin_num = int(bin_num)
        self.append_eos_token = bool(append_eos_token)
        self.eos_token_id = int(eos_token_id)
        self.pad_token_id = int(pad_token_id)
        self.vocab_file = vocab_file
        self.ordered_genes = [gene for gene, _ in sorted(self.vocab.items(), key=lambda item: item[1])]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        cache_dir = kwargs.pop("cache_dir", None)
        revision = kwargs.pop("revision", None)
        local_files_only = kwargs.pop("local_files_only", False)
        base_path = resolve_pretrained_dir(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=local_files_only,
        )
        preprocessor_path = base_path / "preprocessor_config.json"
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Missing scBERT preprocessor config: {preprocessor_path}")
        preprocessor_config = json.loads(preprocessor_path.read_text(encoding="utf-8"))
        vocab_path = base_path / preprocessor_config.get("vocab_file", "vocab.json")
        if not vocab_path.exists():
            raise FileNotFoundError(f"Missing scBERT vocab file: {vocab_path}")
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        preprocessor_config.pop("processor_class", None)
        preprocessor_config.update(kwargs)
        return cls(vocab=vocab, **preprocessor_config)

    def save_pretrained(self, save_directory):
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        vocab_path = save_path / self.vocab_file
        vocab_path.write_text(json.dumps(self.vocab, indent=2, sort_keys=True), encoding="utf-8")
        preprocessor_payload = {
            "processor_class": self.__class__.__name__,
            "vocab_file": self.vocab_file,
            "bin_num": self.bin_num,
            "append_eos_token": self.append_eos_token,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
        }
        (save_path / "preprocessor_config.json").write_text(
            json.dumps(preprocessor_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _align_matrix(self, matrix, gene_names):
        gene_to_idx = {str(gene): idx for idx, gene in enumerate(gene_names)}
        n_cells = matrix.shape[0]
        n_genes = len(self.ordered_genes)
        if sparse.issparse(matrix):
            matrix = matrix.tocsr()
            aligned = sparse.lil_matrix((n_cells, n_genes), dtype=np.float32)
            for ref_idx, gene in enumerate(self.ordered_genes):
                src_idx = gene_to_idx.get(gene)
                if src_idx is not None:
                    aligned[:, ref_idx] = matrix[:, src_idx]
            return aligned.tocsr()

        dense = np.asarray(matrix, dtype=np.float32)
        aligned = np.zeros((n_cells, n_genes), dtype=np.float32)
        for ref_idx, gene in enumerate(self.ordered_genes):
            src_idx = gene_to_idx.get(gene)
            if src_idx is not None:
                aligned[:, ref_idx] = dense[:, src_idx]
        return aligned

    def _matrix_to_sequences(self, aligned_matrix):
        dense = aligned_matrix.toarray() if sparse.issparse(aligned_matrix) else np.asarray(aligned_matrix)
        dense = np.nan_to_num(dense, copy=False)
        dense[dense < 0] = 0
        dense[dense > self.bin_num] = self.bin_num
        tokens = dense.astype(np.int64)
        if self.append_eos_token:
            eos = np.full((tokens.shape[0], 1), self.eos_token_id, dtype=np.int64)
            tokens = np.concatenate([tokens, eos], axis=1)
        return tokens

    def encode_matrix(self, matrix, gene_names, return_tensors="np"):
        aligned = self._align_matrix(matrix, gene_names)
        sequences = self._matrix_to_sequences(aligned)
        if return_tensors == "pt":
            return {"input_ids": torch.from_numpy(sequences).long()}
        return {"input_ids": sequences}

    def encode_adata(self, adata, gene_column=None, return_tensors="np"):
        if gene_column:
            if gene_column not in adata.var:
                raise ValueError(f"Gene column '{gene_column}' not found in AnnData.var")
            gene_names = adata.var[gene_column].astype(str).tolist()
        else:
            gene_names = adata.var_names.astype(str).tolist()
        return self.encode_matrix(adata.X, gene_names, return_tensors=return_tensors)
