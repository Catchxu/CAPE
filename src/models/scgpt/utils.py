import numpy as np
from scipy import sparse


def get_gene_names(adata, gene_column=None):
    if gene_column:
        if gene_column not in adata.var:
            raise ValueError(f"Gene column '{gene_column}' not found in AnnData.var")
        return adata.var[gene_column].astype(str).tolist()
    return adata.var_names.astype(str).tolist()


def filter_adata_by_vocab(adata, vocab, gene_column=None):
    gene_names = get_gene_names(adata, gene_column)
    keep_mask = np.array([gene in vocab for gene in gene_names])
    if keep_mask.sum() == 0:
        raise ValueError("No genes from the input AnnData matched the scGPT vocabulary")
    filtered = adata[:, keep_mask].copy()
    filtered_gene_names = [gene for gene, keep in zip(gene_names, keep_mask) if keep]
    gene_ids = np.array([vocab[gene] for gene in filtered_gene_names], dtype=np.int64)
    return filtered, filtered_gene_names, gene_ids


def get_batch_ids(adata, batch_column=None):
    if batch_column and batch_column in adata.obs:
        categories = adata.obs[batch_column].astype("category")
        return categories.cat.codes.to_numpy(), list(categories.cat.categories)
    return np.zeros(adata.n_obs, dtype=np.int64), ["default"]


def get_input_matrix(adata):
    return adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)


def digitize_row(row: np.ndarray, bins: np.ndarray):
    left_digits = np.digitize(row, bins)
    right_digits = np.digitize(row, bins, right=True)
    rands = np.random.rand(len(row))
    digits = rands * (right_digits - left_digits) + left_digits
    return np.ceil(digits).astype(np.int64)


def bin_matrix(matrix: np.ndarray, n_bins: int):
    binned_rows = []
    for row in matrix:
        if row.max() == 0:
            binned_rows.append(np.zeros_like(row, dtype=np.int64))
            continue
        if row.min() <= 0:
            non_zero_ids = row.nonzero()[0]
            non_zero_row = row[non_zero_ids]
            bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
            digits = digitize_row(non_zero_row, bins)
            binned = np.zeros_like(row, dtype=np.int64)
            binned[non_zero_ids] = digits
        else:
            bins = np.quantile(row, np.linspace(0, 1, n_bins - 1))
            binned = digitize_row(row, bins)
        binned_rows.append(binned)
    return np.stack(binned_rows)
