import torch
import transformers
import math
from typing import Any

from .multi_modal_model import MultiModalModel, SpecialTokenIds
from .image_encoders import ClipImageEncoder, ClipImageEncoderConfig, ImageEncoderBase
from ..harness import ModuleConfig
from .decoder import DecoderLayers, DecoderLayersConfig, DecoderBlockConfig

class BertMultiModalModelConfig(ModuleConfig):
    num_layers: int
    heads_per_layer: int
    attention_kq_dimension: int
    attention_v_dimension: int
    rope_enabled: bool

    mlp_hidden_dimension: int
    mlp_dropout: float

    freeze_image_weights: bool
    freeze_bert_weights: bool

class BertMultiModalModel(MultiModalModel):
    def __init__(self, config: BertMultiModalModelConfig):
        super().__init__()

        self.config = config

        # NOTE - We'll use [SEP] to mark the start and end of the caption, as per BERT's
        # pretrained convention.
        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
        prev_token_count = len(tokenizer)
        tokenizer.add_tokens(["[IMAGE]"], special_tokens=True)
        self._image_start_token_id = prev_token_count
        self.tokenizer = tokenizer

        self.bert_model = transformers.BertForMaskedLM.from_pretrained("bert-base-uncased")
        self.bert_model.resize_token_embeddings(len(tokenizer))  # Resize to accommodate new tokens

        self._special_token_ids = SpecialTokenIds(
            section_start = self.tokenizer.convert_tokens_to_ids("[SEP]"),
            section_end = self.tokenizer.convert_tokens_to_ids("[SEP]"),
            image = self.tokenizer.convert_tokens_to_ids("[IMAGE]"),
            caption = self.tokenizer.convert_tokens_to_ids("[CLS]"),
            padding=self.tokenizer.pad_token_id,
        )
        print(f"Special token ids: {self._special_token_ids}")

        if self.config.freeze_bert_weights:
            # TODO: Could make the new token ids trainable, like with the Qwen model
            self.bert_model.requires_grad_(False)

        self.embedding_dimension: int = self.bert_model.config.hidden_size

        self._image_encoder = ClipImageEncoder(ClipImageEncoderConfig(
            # CLIP encodes to a 512-dimensional vector, so make sure we have enough space here
            tokens_per_image=math.ceil(512 / self.embedding_dimension),
            model_embedding_dimension=self.embedding_dimension,
            freeze_visual_model=config.freeze_image_weights,
        ))
        self.decoder = DecoderLayers(DecoderLayersConfig(
            decoder_layers = self.config.num_layers,
            decoder_block_config = DecoderBlockConfig(
                embedding_dimension = self.embedding_dimension,
                num_heads = self.config.heads_per_layer,
                kq_dimension = self.config.attention_kq_dimension,
                v_dimension = self.config.attention_v_dimension,
                rope_enabled = self.config.rope_enabled,

                mlp_hidden_dimension = self.config.mlp_hidden_dimension,
                mlp_dropout = self.config.mlp_dropout,
            ),
        ))

    ## OVERRIDES

    @property
    def special_token_ids(self) -> SpecialTokenIds:
        return self._special_token_ids

    @property
    def image_encoder(self) -> ImageEncoderBase:
        return self._image_encoder

    def tokenize_no_padding_without_special_chars(self, texts: list[str]) -> list[list[int]]:
        return self.tokenizer(
            texts,
            return_tensors=None,
            add_special_tokens=False,
            padding=transformers.utils.PaddingStrategy.DO_NOT_PAD,
        )["input_ids"]

    def token_ids_to_text(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def embed_token_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.to(self.bert_model.device)
        return self.bert_model.bert.embeddings.word_embeddings(token_ids)

    def unembed_to_token_id_logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.bert_model.cls(hidden_state)

    def run_model(self, input_embeds: torch.Tensor, cache: Any = None) -> tuple[torch.Tensor, Any]:
        return self.decoder(input_embeds), None
