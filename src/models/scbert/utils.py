import json
from pathlib import Path

import numpy as np
from scipy import sparse


def load_gene_list(path: str):
    path_obj = Path(path)
    if path_obj.suffix == ".json":
        return json.loads(path_obj.read_text(encoding="utf-8"))
    return [line.strip() for line in path_obj.read_text(encoding="utf-8").splitlines() if line.strip()]


def get_gene_names(adata, gene_column=None):
    if gene_column:
        if gene_column not in adata.var:
            raise ValueError(f"Gene column '{gene_column}' not found in AnnData.var")
        return adata.var[gene_column].astype(str).tolist()
    return adata.var_names.astype(str).tolist()


def align_adata_to_gene_order(adata, reference_genes, gene_column=None):
    source_genes = get_gene_names(adata, gene_column)
    gene_to_idx = {gene: idx for idx, gene in enumerate(source_genes)}
    matrix = adata.X.tocsr() if sparse.issparse(adata.X) else np.asarray(adata.X)
    n_cells = adata.n_obs
    n_genes = len(reference_genes)

    if sparse.issparse(matrix):
        aligned = sparse.lil_matrix((n_cells, n_genes), dtype=np.float32)
        for ref_idx, gene in enumerate(reference_genes):
            src_idx = gene_to_idx.get(gene)
            if src_idx is not None:
                aligned[:, ref_idx] = matrix[:, src_idx]
        return aligned.tocsr()

    aligned = np.zeros((n_cells, n_genes), dtype=np.float32)
    for ref_idx, gene in enumerate(reference_genes):
        src_idx = gene_to_idx.get(gene)
        if src_idx is not None:
            aligned[:, ref_idx] = matrix[:, src_idx]
    return aligned


def matrix_to_scbert_sequences(aligned_matrix, max_bin: int):
    dense = aligned_matrix.toarray() if sparse.issparse(aligned_matrix) else np.asarray(aligned_matrix)
    dense = np.nan_to_num(dense, copy=False)
    dense[dense < 0] = 0
    dense[dense > max_bin] = max_bin
    tokens = dense.astype(np.int64)
    eos = np.zeros((tokens.shape[0], 1), dtype=np.int64)
    return np.concatenate([tokens, eos], axis=1)
