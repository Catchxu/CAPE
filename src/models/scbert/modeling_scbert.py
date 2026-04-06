import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from .configuration_scbert import ScBertConfig
from .performer_pytorch import PerformerLM
from ..pretrained import resolve_pretrained_from_kwargs
from .reversible import ReversibleSequence, SequentialSequence  # noqa: F401


class ScBertPreTrainedModel(PreTrainedModel):
    config_class = ScBertConfig
    base_model_prefix = "scbert"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class ScBertModel(ScBertPreTrainedModel):
    def __init__(self, config: ScBertConfig):
        super().__init__(config)
        self.scbert = PerformerLM(
            num_tokens=config.num_tokens,
            dim=config.hidden_size,
            depth=config.num_hidden_layers,
            max_seq_len=config.max_position_embeddings,
            heads=config.num_attention_heads,
            local_attn_heads=config.local_attn_heads,
            g2v_position_emb=True,
            gene2vec_weight=None,
            emb_dropout=config.hidden_dropout_prob,
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.scbert.token_emb

    def set_input_embeddings(self, value):
        self.scbert.token_emb = value

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        resolved = resolve_pretrained_from_kwargs(pretrained_model_name_or_path, kwargs)
        return super().from_pretrained(str(resolved), *model_args, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor,
        external_positions: torch.LongTensor | None = None,
        output_attentions: bool = False,
        return_dict: bool = True,
        **kwargs,
    ):
        if output_attentions:
            hidden_states, attentions = self.scbert(
                input_ids,
                external_positions=external_positions,
                return_encodings=True,
                output_attentions=True,
                **kwargs,
            )
        else:
            hidden_states = self.scbert(
                input_ids,
                external_positions=external_positions,
                return_encodings=True,
                output_attentions=False,
                **kwargs,
            )
            attentions = None

        if not return_dict:
            output = (hidden_states,)
            if attentions is not None:
                output += (attentions,)
            return output

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=None,
            attentions=(attentions,) if attentions is not None else None,
        )
