import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from .configuration_scgpt import ScGptConfig
from .model import TransformerModel
from ..pretrained import resolve_pretrained_from_kwargs


class ScGptPreTrainedModel(PreTrainedModel):
    config_class = ScGptConfig
    base_model_prefix = "scgpt"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class ScGptModel(ScGptPreTrainedModel):
    def __init__(self, config: ScGptConfig):
        super().__init__(config)
        vocab = {
            config.pad_token: config.pad_token_id,
            config.cls_token: config.cls_token_id,
            config.eoc_token: config.eos_token_id,
        }
        self.scgpt = TransformerModel(
            ntoken=config.vocab_size,
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            d_hid=config.intermediate_size,
            nlayers=config.num_hidden_layers,
            nlayers_cls=config.nlayers_cls,
            n_cls=1,
            vocab=vocab,
            dropout=config.dropout,
            pad_token=config.pad_token,
            pad_value=config.pad_value,
            do_mvc=False,
            do_dab=False,
            use_batch_labels=bool(config.use_batch_labels),
            num_batch_labels=None,
            domain_spec_batchnorm=config.domain_spec_batchnorm,
            input_emb_style=config.input_emb_style,
            n_input_bins=config.n_input_bins,
            cell_emb_style=config.cell_emb_style,
            use_fast_transformer=bool(config.use_fast_transformer),
            fast_transformer_backend=config.fast_transformer_backend,
            pre_norm=bool(config.pre_norm),
        )
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        resolved = resolve_pretrained_from_kwargs(pretrained_model_name_or_path, kwargs)
        return super().from_pretrained(str(resolved), *model_args, **kwargs)

    def forward(
        self,
        input_gene_ids: torch.LongTensor,
        values: torch.FloatTensor,
        src_key_padding_mask: torch.BoolTensor,
        batch_labels: torch.LongTensor | None = None,
        external_positions: torch.LongTensor | None = None,
        return_dict: bool = True,
        **kwargs,
    ):
        hidden_states = self.scgpt._encode(
            input_gene_ids,
            values,
            src_key_padding_mask,
            batch_labels=batch_labels,
            external_positions=external_positions,
        )
        pooled = self.scgpt._get_cell_emb_from_layer(hidden_states, values)

        if not return_dict:
            return hidden_states, pooled

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=None,
            attentions=None,
        )
