from enum import StrEnum

import torch
import os
import transformers
from typing import Optional, Any
from peft import LoraConfig, TaskType, get_peft_model

from .multi_modal_model import MultiModalModel, SpecialTokenIds
from .image_encoders import ClipImageEncoder, ClipImageEncoderConfig, ImageEncoderBase
from ..harness import ModuleConfig
from .custom_lora import CustomLora, CustomLoraConfig

class EmbeddingLearningStrategy(StrEnum):
    LEARN_ALL = "learn_all"
    FREEZE_ALL = "freeze"
    LEARN_NEW_WITH_GRAD_FILTERING = "learn_new_with_grad_filtering"
    LORA = "lora"

class SpecialTokensStrategy(StrEnum):
    CREATE_NEW = "new_tokens"
    REUSE_EXISTING_UNTRAINED = "reuse_existing_untrained"

class QwenMultiModalModelConfig(ModuleConfig):
    freeze_visual_model: bool
    special_tokens_strategy: SpecialTokensStrategy = SpecialTokensStrategy.CREATE_NEW
    embedding_learning_strategy: Optional[EmbeddingLearningStrategy] = None
    freeze_new_special_token_embeddings: Optional[bool] = False
    """Included for back-compat only. Use embedding_learning_strategy instead"""
    apply_lora_to_mlp_layers: bool = False
    lora_alpha: int = 16
    lora_rank: int = 16
    lora_dropout: float = 0.05

class QwenMultiModalModel(MultiModalModel):
    def __init__(self, config: QwenMultiModalModelConfig):
        super().__init__()
        model_name = "Qwen/Qwen3-0.6B-Base"

        self.config = config

        if config.embedding_learning_strategy is None:
            if config.freeze_new_special_token_embeddings is None:
                raise ValueError("Please set new_token_embedding_learning_strategy")
            # Backwards compatibility
            if config.freeze_new_special_token_embeddings:
                config.embedding_learning_strategy = EmbeddingLearningStrategy.FREEZE_ALL
            else:
                config.embedding_learning_strategy = EmbeddingLearningStrategy.LEARN_NEW_WITH_GRAD_FILTERING

        # Avoids an annoying error message, and the rust tokenizer is so fast that it's
        # fine not to parallelize. See https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.tokenizer: transformers.Qwen2TokenizerFast = transformers.AutoTokenizer.from_pretrained(model_name)
        self.auto_model: transformers.Qwen2ForCausalLM = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        self.qwen_model: transformers.Qwen2Model = self.auto_model.model

        # SPECIAL TOKENS
        match config.special_tokens_strategy:
            case SpecialTokensStrategy.CREATE_NEW:
                # <|im_start|>, <|im_end|>, <|user|>, <|assistant|>:
                # > Chat/Instruct models: Used and understood.
                # > Base model: Present in vocab, but not used for chat structure by default.
                # > <|endoftext|>: Used by both.

                # Let's use the following structure
                # <|im_start|><|image|>   ... <|im_end|>
                # <|im_start|><|caption|> ... <|im_end|>

                # Add custom tokens
                self.tokenizer.add_tokens(["<|image|>", "<|caption|>"], special_tokens=True)
                # Note - There is some special weight-tying in the auto-model/qwen-model, meaning
                # the lm_head decoder has its weights tied encoder's embeddings, and this resizes both.
                self.qwen_model.resize_token_embeddings(len(self.tokenizer))
                assert self.qwen_model.embed_tokens.weight.shape[0] == len(self.tokenizer)
                assert self.auto_model.lm_head.weight.shape[0] == len(self.tokenizer)

                self._special_token_ids = SpecialTokenIds(
                    section_start = self.tokenizer.convert_tokens_to_ids("<|im_start|>"),
                    section_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                    image = self.tokenizer.convert_tokens_to_ids("<|image|>"),
                    caption = self.tokenizer.convert_tokens_to_ids("<|caption|>"),
                    padding = self.tokenizer.pad_token_id,
                )
                print(f"Special token ids: {self._special_token_ids}")
            case SpecialTokensStrategy.REUSE_EXISTING_UNTRAINED:
                # Re-use existing special tokens which are untrained in the base model.
                # This means we don't need to expand our vocabulary, and the default "save only learnable parameters"
                # works okay
                self._special_token_ids = SpecialTokenIds(
                    section_start = self.tokenizer.convert_tokens_to_ids("<|im_start|>"),
                    section_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                    image = self.tokenizer.convert_tokens_to_ids("<|vision_start|>"),
                    caption = self.tokenizer.convert_tokens_to_ids("<|box_start|>"),
                    padding = self.tokenizer.pad_token_id,
                )
                print(f"Special token ids: {self._special_token_ids}")

        # Enable LORA on the Qwen model. get_peft_model actually changes it in-place
        # We have to change it after the lm_head is resized by adding new tokens, else it won't work.
        lora_target_modules = ["q_proj", "v_proj"]
        if config.apply_lora_to_mlp_layers:
            lora_target_modules.append("mlp.up_proj")
            lora_target_modules.append("mlp.down_proj")

        # NOTE: We cannot apply LoRA to the lm_head layer using PEFT, because it is tied to the embed_tokens layer.
        # If we need to do that, we'll need to do it ourselves via adding our own learnable weights to the lm_head layer.
        # (see EmbeddingLearningStrategy.LORA below)

        self.peft_model = get_peft_model(
            self.auto_model,    
            LoraConfig(
                r=config.lora_rank,
                target_modules=lora_target_modules,
                task_type=TaskType.CAUSAL_LM,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
            ),
        )

        self.lm_head_lora: Optional[CustomLora] = None
        match config.embedding_learning_strategy:
            case EmbeddingLearningStrategy.LEARN_ALL:
                self.qwen_model.embed_tokens.requires_grad_(True)
                self.auto_model.lm_head.requires_grad_(True)
            case EmbeddingLearningStrategy.FREEZE_ALL:
                pass # Already frozen via the LoRA
            case EmbeddingLearningStrategy.LEARN_NEW_WITH_GRAD_FILTERING:
                self.qwen_model.embed_tokens.requires_grad_(True)
                self.auto_model.lm_head.requires_grad_(True)
                self.only_accept_gradients_for_these_token_ids([
                    self._special_token_ids.image,
                    self._special_token_ids.caption,
                    self._special_token_ids.section_start,
                    self._special_token_ids.section_end,
                ])
            case EmbeddingLearningStrategy.LORA:
                # For LoRA strategy, we freeze the base embeddings and lm_head
                # and create a custom LoRA layer for the lm_head
                self.qwen_model.embed_tokens.requires_grad_(False)
                self.auto_model.lm_head.requires_grad_(False)
                
                # Create custom LoRA for lm_head since PEFT can't handle tied weights
                self.lm_head_lora = CustomLora(CustomLoraConfig(
                    input_dim=self.qwen_model.config.hidden_size,
                    output_dim=self.auto_model.lm_head.weight.shape[0],  # vocab size
                    rank=config.lora_rank,  # configurable rank
                    dropout=config.lora_dropout,
                    alpha=config.lora_alpha,
                ))
                

        self.embedding_dimension: int = self.qwen_model.config.hidden_size

        self._image_encoder = ClipImageEncoder(ClipImageEncoderConfig(
            tokens_per_image=1,  # CLIP encodes to a 512-dimensional vector already
            model_embedding_dimension=self.embedding_dimension,
            freeze_visual_model=self.config.freeze_visual_model,
        ))

    def only_accept_gradients_for_these_token_ids(self, trainable_token_ids) -> None:
        # Allow the model to learn (just) these embeddings
        # See this stack overflow post: https://stackoverflow.com/a/79621033
        # This should hopefully allow for predicting the end of section tokens at the end of captions

        # I don't quite know how the auto-tying logic works, so I'll just apply masking to both
        @torch.utils.hooks.unserializable_hook
        def mask_embedding_gradients(grad):
            mask = torch.zeros_like(grad)
            mask[trainable_token_ids] = 1.0
            return grad * mask

        self.qwen_model.embed_tokens.weight.register_hook(mask_embedding_gradients)
        self.auto_model.lm_head.weight.register_hook(mask_embedding_gradients)

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
            padding=transformers.utils.PaddingStrategy.DO_NOT_PAD,
        )["input_ids"]

    def token_ids_to_text(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def embed_token_ids(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.to(self.qwen_model.device)
        return self.qwen_model.embed_tokens(token_ids)

    def unembed_to_token_id_logits(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if self.lm_head_lora is not None:
            return self.auto_model.lm_head(hidden_state) + self.lm_head_lora(hidden_state)
        else:
            return self.auto_model.lm_head(hidden_state)

    def run_model(self, input_embeds: torch.Tensor, cache: Any = None) -> tuple[torch.Tensor, Any]:
        result = self.qwen_model(
            inputs_embeds=input_embeds,
            use_cache=True,
            past_key_values=cache,
        )
        return result.last_hidden_state, result.past_key_values
