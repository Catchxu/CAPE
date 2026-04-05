from transformers import PretrainedConfig

from .pretrained import resolve_pretrained_dir


class ScGptConfig(PretrainedConfig):
    model_type = "scgpt"

    def __init__(
        self,
        vocab_size=60697,
        hidden_size=512,
        num_hidden_layers=12,
        num_attention_heads=8,
        intermediate_size=512,
        max_position_embeddings=1200,
        nlayers_cls=3,
        dropout=0.2,
        pad_token="<pad>",
        cls_token="<cls>",
        eoc_token="<eoc>",
        pad_token_id=0,
        cls_token_id=1,
        eos_token_id=2,
        pad_value=-2,
        mask_value=-1,
        input_emb_style="continuous",
        n_input_bins=51,
        cell_emb_style="cls",
        use_fast_transformer=True,
        fast_transformer_backend="flash",
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        pre_norm=False,
        vocab_file="vocab.json",
        weight_file="model.safetensors",
        preprocessor_file="preprocessor_config.json",
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.nlayers_cls = nlayers_cls
        self.dropout = dropout
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.eoc_token = eoc_token
        self.cls_token_id = cls_token_id
        self.pad_value = pad_value
        self.mask_value = mask_value
        self.input_emb_style = input_emb_style
        self.n_input_bins = n_input_bins
        self.cell_emb_style = cell_emb_style
        self.use_fast_transformer = use_fast_transformer
        self.fast_transformer_backend = fast_transformer_backend
        self.use_batch_labels = use_batch_labels
        self.domain_spec_batchnorm = domain_spec_batchnorm
        self.pre_norm = pre_norm
        self.vocab_file = vocab_file
        self.weight_file = weight_file
        self.preprocessor_file = preprocessor_file

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        cache_dir = kwargs.get("cache_dir")
        revision = kwargs.get("revision")
        local_files_only = kwargs.get("local_files_only", False)
        resolved = resolve_pretrained_dir(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=local_files_only,
        )
        return super().from_pretrained(str(resolved), **kwargs)
