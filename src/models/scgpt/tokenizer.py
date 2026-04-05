import json
from pathlib import Path

import numpy as np
import torch


class GeneVocab:
    def __init__(self, token_to_idx):
        self.token_to_idx = {token: int(idx) for token, idx in token_to_idx.items()}

    @classmethod
    def from_file(cls, file_path):
        path = Path(file_path)
        with path.open("r", encoding="utf-8") as handle:
            token_to_idx = json.load(handle)
        return cls.from_dict(token_to_idx)

    @classmethod
    def from_dict(cls, token_to_idx):
        ordered = {token: int(idx) for token, idx in sorted(token_to_idx.items(), key=lambda item: item[1])}
        return cls(ordered)

    def __contains__(self, token):
        return token in self.token_to_idx

    def __getitem__(self, token):
        return self.token_to_idx[token]

    def __len__(self):
        return len(self.token_to_idx)

    def get_stoi(self):
        return dict(self.token_to_idx)


def tokenize_batch(
    data: np.ndarray,
    gene_ids: np.ndarray,
    append_cls: bool = True,
    include_zero_gene: bool = False,
    cls_id: int = 0,
):
    tokenized = []
    for row in data:
        if include_zero_gene:
            values = row
            genes = gene_ids
        else:
            idx = np.nonzero(row)[0]
            values = row[idx]
            genes = gene_ids[idx]
        if append_cls:
            genes = np.insert(genes, 0, cls_id)
            values = np.insert(values, 0, 0)
        tokenized.append((torch.from_numpy(genes).long(), torch.from_numpy(values).float()))
    return tokenized


def pad_batch(batch, max_len: int, pad_token_id: int, pad_value: int, cls_appended: bool = True):
    max_len = min(max(len(item[0]) for item in batch), max_len)
    padded_genes = []
    padded_values = []
    for genes, values in batch:
        if len(genes) > max_len:
            if cls_appended:
                keep = np.random.choice(len(genes) - 1, max_len - 1, replace=False) + 1
                keep = np.insert(keep, 0, 0)
            else:
                keep = np.random.choice(len(genes), max_len, replace=False)
            genes = genes[keep]
            values = values[keep]
        elif len(genes) < max_len:
            genes = torch.cat(
                [
                    genes,
                    torch.full((max_len - len(genes),), pad_token_id, dtype=genes.dtype),
                ]
            )
            values = torch.cat(
                [
                    values,
                    torch.full((max_len - len(values),), pad_value, dtype=values.dtype),
                ]
            )
        padded_genes.append(genes)
        padded_values.append(values)
    return {
        "genes": torch.stack(padded_genes, dim=0),
        "values": torch.stack(padded_values, dim=0),
    }


def tokenize_and_pad_batch(
    data: np.ndarray,
    gene_ids: np.ndarray,
    max_len: int,
    vocab: GeneVocab,
    pad_token: str,
    pad_value: int,
    append_cls: bool = True,
    include_zero_gene: bool = False,
    cls_token: str = "<cls>",
):
    batch = tokenize_batch(
        data=data,
        gene_ids=gene_ids,
        append_cls=append_cls,
        include_zero_gene=include_zero_gene,
        cls_id=vocab[cls_token],
    )
    return pad_batch(
        batch=batch,
        max_len=max_len,
        pad_token_id=vocab[pad_token],
        pad_value=pad_value,
        cls_appended=append_cls,
    )


def random_mask_value(values, mask_ratio: float = 0.15, mask_value: int = -1, pad_value: int = 0):
    if isinstance(values, torch.Tensor):
        values = values.clone().detach().numpy()
    else:
        values = values.copy()
    for idx in range(len(values)):
        row = values[idx]
        non_padding_idx = np.nonzero(row - pad_value)[0]
        n_mask = int(len(non_padding_idx) * mask_ratio)
        if n_mask == 0:
            continue
        mask_idx = np.random.choice(non_padding_idx, n_mask, replace=False)
        row[mask_idx] = mask_value
    return torch.from_numpy(values).float()
