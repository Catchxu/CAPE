import json
from pathlib import Path
from typing import Literal, Sequence

import torch
from safetensors.torch import load_file
from torch import Tensor, nn

from ..pretrained import resolve_pretrained_dir
from ..scbert.configuration_scbert import ScBertConfig
from ..scgpt.configuration_scgpt import ScGptConfig
from .modules import RMSNorm


EncoderBackend = Literal["scbert", "scgpt"]


def _load_asset_bundle(pretrained_model_name_or_path: str) -> tuple[Path, dict[str, Tensor]]:
    base_path = resolve_pretrained_dir(pretrained_model_name_or_path)
    state_dict = load_file(str(base_path / "model.safetensors"))
    return base_path, state_dict


def _ordered_scgpt_gene_ids(base_path: Path, vocab_file: str, num_genes: int) -> list[int]:
    vocab = json.loads((base_path / vocab_file).read_text(encoding="utf-8"))
    ordered_tokens = sorted(vocab.items(), key=lambda item: item[1])
    gene_ids = [idx for token, idx in ordered_tokens if not token.startswith("<")]
    if len(gene_ids) < num_genes:
        raise ValueError(
            f"scGPT vocab only exposes {len(gene_ids)} non-special tokens, "
            f"but num_genes={num_genes}"
        )
    return [int(idx) for idx in gene_ids[:num_genes]]


def _copy_linear(weight: Tensor, bias: Tensor | None = None) -> nn.Linear:
    layer = nn.Linear(weight.size(1), weight.size(0), bias=bias is not None)
    with torch.no_grad():
        layer.weight.copy_(weight)
        if bias is not None:
            layer.bias.copy_(bias)
    return layer


class ScBertInputEncoder(nn.Module):
    """scBERT-style discrete tokenization plus gene2vec positional embedding."""

    def __init__(
        self,
        num_genes: int,
        embed_dim: int,
        pretrained_model_name_or_path: str,
        dropout: float = 0.0,
        gene_token_ids: Sequence[int] | None = None,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        _, state_dict = _load_asset_bundle(pretrained_model_name_or_path)
        config = ScBertConfig.from_pretrained(pretrained_model_name_or_path)

        token_weight = state_dict["scbert.token_emb.weight"].detach().clone()
        full_gene2vec_weight = state_dict[config.gene_embedding_key].detach().clone()
        if gene_token_ids is None:
            gene_indices = torch.arange(num_genes, dtype=torch.long)
        else:
            gene_indices = torch.tensor(list(gene_token_ids), dtype=torch.long)
            if gene_indices.numel() != num_genes:
                raise ValueError(f"Expected {num_genes} scBERT gene ids, got {gene_indices.numel()}")
        if gene_indices.max().item() >= full_gene2vec_weight.size(0):
            raise ValueError("scBERT gene ids exceed the available gene2vec rows")
        gene2vec_weight = full_gene2vec_weight.index_select(0, gene_indices)
        source_dim = int(config.hidden_size)

        self.num_genes = num_genes
        self.bin_num = int(config.bin_num)
        self.token_embedding = nn.Embedding.from_pretrained(token_weight, freeze=freeze_backbone)
        self.gene2vec_embedding = nn.Embedding.from_pretrained(gene2vec_weight, freeze=freeze_backbone)
        self.output_projection = (
            nn.Identity() if source_dim == embed_dim else nn.Linear(source_dim, embed_dim)
        )
        self.norm = RMSNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _discretize(self, x: Tensor) -> Tensor:
        tokens = torch.nan_to_num(x.float(), nan=0.0)
        tokens = torch.clamp(tokens, min=0.0, max=float(self.bin_num))
        return tokens.round().long()

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected input shape (B, L), got {tuple(x.shape)}")
        if x.size(1) != self.num_genes:
            raise ValueError(
                f"Expected {self.num_genes} genes, received {x.size(1)}"
            )

        token_ids = self._discretize(x)
        token_embed = self.token_embedding(token_ids)
        positions = torch.arange(self.num_genes, device=x.device)
        gene_embed = self.gene2vec_embedding(positions).unsqueeze(0).expand(x.size(0), -1, -1)
        h = token_embed + gene_embed
        h = self.output_projection(h)
        h = self.norm(h)
        return self.dropout(h)


class ScGptInputEncoder(nn.Module):
    """scGPT-style initial embedding using gene embedding and value encoder only."""

    def __init__(
        self,
        num_genes: int,
        embed_dim: int,
        pretrained_model_name_or_path: str,
        dropout: float = 0.0,
        gene_token_ids: Sequence[int] | None = None,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        base_path, state_dict = _load_asset_bundle(pretrained_model_name_or_path)
        config = ScGptConfig.from_pretrained(pretrained_model_name_or_path)

        token_ids = list(gene_token_ids) if gene_token_ids is not None else _ordered_scgpt_gene_ids(
            base_path,
            config.vocab_file,
            num_genes,
        )
        if len(token_ids) != num_genes:
            raise ValueError(f"Expected {num_genes} gene token ids, got {len(token_ids)}")

        source_dim = int(config.hidden_size)
        self.num_genes = num_genes
        self.input_emb_style = str(config.input_emb_style)
        self.n_input_bins = int(config.n_input_bins)
        self.gene_embedding = nn.Embedding(
            config.vocab_size,
            source_dim,
            padding_idx=config.pad_token_id,
        )
        with torch.no_grad():
            self.gene_embedding.weight.copy_(state_dict["scgpt.encoder.embedding.weight"])
        self.gene_norm = RMSNorm(source_dim)
        with torch.no_grad():
            self.gene_norm.weight.copy_(state_dict["scgpt.encoder.enc_norm.weight"])

        if self.input_emb_style == "continuous":
            self.value_linear1 = _copy_linear(
                state_dict["scgpt.value_encoder.linear1.weight"],
                state_dict["scgpt.value_encoder.linear1.bias"],
            )
            self.value_linear2 = _copy_linear(
                state_dict["scgpt.value_encoder.linear2.weight"],
                state_dict["scgpt.value_encoder.linear2.bias"],
            )
            self.value_activation = nn.ReLU()
            self.value_norm = RMSNorm(source_dim)
            with torch.no_grad():
                self.value_norm.weight.copy_(state_dict["scgpt.value_encoder.norm.weight"])
            self.value_embedding = None
        elif self.input_emb_style == "category":
            value_weight = state_dict["scgpt.value_encoder.embedding.weight"]
            self.value_embedding = nn.Embedding(value_weight.size(0), value_weight.size(1))
            with torch.no_grad():
                self.value_embedding.weight.copy_(value_weight)
            self.value_norm = RMSNorm(source_dim)
            with torch.no_grad():
                self.value_norm.weight.copy_(state_dict["scgpt.value_encoder.enc_norm.weight"])
            self.value_linear1 = None
            self.value_linear2 = None
            self.value_activation = None
        else:
            self.value_embedding = None
            self.value_linear1 = None
            self.value_linear2 = None
            self.value_activation = None
            self.value_norm = None

        if freeze_backbone:
            for parameter in self.gene_embedding.parameters():
                parameter.requires_grad = False
            for parameter in self.gene_norm.parameters():
                parameter.requires_grad = False
            if self.value_embedding is not None:
                for parameter in self.value_embedding.parameters():
                    parameter.requires_grad = False
            if self.value_linear1 is not None:
                for parameter in self.value_linear1.parameters():
                    parameter.requires_grad = False
            if self.value_linear2 is not None:
                for parameter in self.value_linear2.parameters():
                    parameter.requires_grad = False
            if self.value_norm is not None:
                for parameter in self.value_norm.parameters():
                    parameter.requires_grad = False

        self.output_projection = (
            nn.Identity() if source_dim == embed_dim else nn.Linear(source_dim, embed_dim)
        )
        self.norm = RMSNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "gene_token_ids",
            torch.tensor(token_ids, dtype=torch.long),
            persistent=False,
        )

    def _encode_values(self, values: Tensor) -> Tensor:
        if self.input_emb_style == "scaling":
            return values.unsqueeze(-1)
        if self.input_emb_style == "continuous":
            value_embed = values.unsqueeze(-1)
            value_embed = self.value_activation(self.value_linear1(value_embed))
            value_embed = self.value_linear2(value_embed)
            return self.value_norm(value_embed)
        if self.input_emb_style == "category":
            discrete = values.round().long().clamp(min=0, max=self.n_input_bins - 1)
            return self.value_norm(self.value_embedding(discrete))
        raise ValueError(f"Unsupported scGPT input_emb_style: {self.input_emb_style}")

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected input shape (B, L), got {tuple(x.shape)}")
        if x.size(1) != self.num_genes:
            raise ValueError(
                f"Expected {self.num_genes} genes, received {x.size(1)}"
            )

        batch_gene_ids = self.gene_token_ids.unsqueeze(0).expand(x.size(0), -1)
        gene_embed = self.gene_norm(self.gene_embedding(batch_gene_ids))
        values = x.float()

        if self.input_emb_style == "scaling":
            h = gene_embed * self._encode_values(values)
        else:
            h = gene_embed + self._encode_values(values)

        h = self.output_projection(h)
        h = self.norm(h)
        return self.dropout(h)


class FoundationEncoder(nn.Module):
    """Select scBERT or scGPT front-end embeddings without running the full backbone."""

    def __init__(
        self,
        num_genes: int,
        embed_dim: int,
        dropout: float = 0.0,
        backend: EncoderBackend = "scbert",
        pretrained_model_name_or_path: str | None = None,
        freeze_backbone: bool = False,
        gene_token_ids: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if pretrained_model_name_or_path is None:
            raise ValueError(f"{backend} encoder requires pretrained_model_name_or_path")

        if backend == "scbert":
            self.impl = ScBertInputEncoder(
                num_genes=num_genes,
                embed_dim=embed_dim,
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                dropout=dropout,
                gene_token_ids=gene_token_ids,
                freeze_backbone=freeze_backbone,
            )
        elif backend == "scgpt":
            self.impl = ScGptInputEncoder(
                num_genes=num_genes,
                embed_dim=embed_dim,
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                dropout=dropout,
                gene_token_ids=gene_token_ids,
                freeze_backbone=freeze_backbone,
            )
        else:
            raise ValueError(f"Unsupported encoder backend: {backend}")

    def forward(self, x: Tensor) -> Tensor:
        return self.impl(x)
