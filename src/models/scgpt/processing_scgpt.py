import json
from pathlib import Path

import torch

from .pretrained import resolve_pretrained_dir
from .tokenizer import GeneVocab, random_mask_value, tokenize_and_pad_batch
from .utils import bin_matrix, filter_adata_by_vocab, get_input_matrix


class ScGptProcessor:
    model_input_names = ["gene_ids", "values"]

    def __init__(
        self,
        vocab,
        max_seq_len=1200,
        pad_token="<pad>",
        cls_token="<cls>",
        pad_value=-2,
        mask_value=-1,
        append_cls=True,
        include_zero_gene=False,
        n_input_bins=51,
        vocab_file="vocab.json",
    ):
        self.vocab = vocab if isinstance(vocab, GeneVocab) else GeneVocab.from_dict(vocab)
        self.max_seq_len = int(max_seq_len)
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.pad_value = int(pad_value)
        self.mask_value = int(mask_value)
        self.append_cls = bool(append_cls)
        self.include_zero_gene = bool(include_zero_gene)
        self.n_input_bins = int(n_input_bins)
        self.vocab_file = vocab_file

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
            raise FileNotFoundError(f"Missing scGPT preprocessor config: {preprocessor_path}")
        preprocessor_config = json.loads(preprocessor_path.read_text(encoding="utf-8"))
        vocab_path = base_path / preprocessor_config.get("vocab_file", "vocab.json")
        if not vocab_path.exists():
            raise FileNotFoundError(f"Missing scGPT vocab file: {vocab_path}")
        vocab = GeneVocab.from_file(vocab_path)
        preprocessor_config.pop("processor_class", None)
        preprocessor_config.update(kwargs)
        return cls(vocab=vocab, **preprocessor_config)

    def save_pretrained(self, save_directory):
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        vocab_path = save_path / self.vocab_file
        vocab_path.write_text(
            json.dumps(self.vocab.get_stoi(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        preprocessor_payload = {
            "processor_class": self.__class__.__name__,
            "vocab_file": self.vocab_file,
            "max_seq_len": self.max_seq_len,
            "pad_token": self.pad_token,
            "cls_token": self.cls_token,
            "pad_value": self.pad_value,
            "mask_value": self.mask_value,
            "append_cls": self.append_cls,
            "include_zero_gene": self.include_zero_gene,
            "n_input_bins": self.n_input_bins,
        }
        (save_path / "preprocessor_config.json").write_text(
            json.dumps(preprocessor_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def encode_adata(self, adata, gene_column=None, return_tensors="np", mask_ratio=0.0):
        filtered, _, gene_ids = filter_adata_by_vocab(
            adata,
            self.vocab,
            gene_column=gene_column,
        )
        matrix = get_input_matrix(filtered)
        binned = bin_matrix(matrix, self.n_input_bins)
        tokenized = tokenize_and_pad_batch(
            data=binned,
            gene_ids=gene_ids,
            max_len=self.max_seq_len,
            vocab=self.vocab,
            pad_token=self.pad_token,
            pad_value=self.pad_value,
            append_cls=self.append_cls,
            include_zero_gene=self.include_zero_gene,
            cls_token=self.cls_token,
        )
        values = random_mask_value(
            tokenized["values"],
            mask_ratio=mask_ratio,
            mask_value=self.mask_value,
            pad_value=self.pad_value,
        )
        if return_tensors == "pt":
            return {
                "gene_ids": tokenized["genes"].long(),
                "values": values.float(),
                "adata": filtered,
            }
        return {
            "gene_ids": tokenized["genes"].numpy(),
            "values": values.numpy(),
            "adata": filtered,
        }

    def build_padding_mask(self, gene_ids: torch.Tensor) -> torch.Tensor:
        return gene_ids.eq(self.vocab[self.pad_token])
