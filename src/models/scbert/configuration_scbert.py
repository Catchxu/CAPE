from transformers import PretrainedConfig

from ..pretrained import resolve_pretrained_from_kwargs


class ScBertConfig(PretrainedConfig):
    model_type = "scbert"

    def __init__(
        self,
        vocab_size=16906,
        hidden_size=200,
        num_hidden_layers=6,
        num_attention_heads=10,
        max_position_embeddings=16907,
        num_tokens=7,
        bin_num=5,
        local_attn_heads=0,
        classifier_hidden_dim=128,
        hidden_dropout_prob=0.0,
        classifier_dropout=0.0,
        gene_embedding_included=True,
        gene_embedding_key="scbert.pos_emb.emb.weight",
        vocab_file="vocab.json",
        weight_file="model.safetensors",
        preprocessor_file="preprocessor_config.json",
        classifier_type="cell_type_annotation",
        pad_token_id=0,
        eos_token_id=0,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.num_tokens = num_tokens
        self.bin_num = bin_num
        self.local_attn_heads = local_attn_heads
        self.classifier_hidden_dim = classifier_hidden_dim
        self.hidden_dropout_prob = hidden_dropout_prob
        self.classifier_dropout = classifier_dropout
        self.gene_embedding_included = gene_embedding_included
        self.gene_embedding_key = gene_embedding_key
        self.vocab_file = vocab_file
        self.weight_file = weight_file
        self.preprocessor_file = preprocessor_file
        self.classifier_type = classifier_type

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        resolved = resolve_pretrained_from_kwargs(pretrained_model_name_or_path, kwargs)
        return super().from_pretrained(str(resolved), **kwargs)
