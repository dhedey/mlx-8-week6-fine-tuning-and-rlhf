import torch.nn as nn
import torch
from .transformer import MaskedSelfAttentionConfig, MaskedSelfAttention, MultiLayerPerceptron
from ..harness import ModuleConfig


class DecoderBlockConfig(ModuleConfig):
    embedding_dimension: int
    num_heads: int
    kq_dimension: int
    v_dimension: int
    rope_enabled: bool

    mlp_hidden_dimension: int
    mlp_dropout: float

class DecoderBlock(nn.Module):
    def __init__(self, config: DecoderBlockConfig):
        super().__init__()
        embedding_dimension = config.embedding_dimension
        self.masked_self_attention = MaskedSelfAttention(MaskedSelfAttentionConfig(
            kq_dimension=config.kq_dimension,
            v_dimension=config.v_dimension,
            embedding_dimension=embedding_dimension,
            num_heads=config.num_heads,
            rope_enabled=config.rope_enabled,
        ))
        self.ln_1 = nn.LayerNorm(embedding_dimension)
        self.mlp = MultiLayerPerceptron(
            embedding_dimension=embedding_dimension,
            hidden_dimension=config.mlp_hidden_dimension,
            dropout=config.mlp_dropout,
        )
        self.ln_2 = nn.LayerNorm(embedding_dimension)

    def forward(self, residual_stream: torch.Tensor) -> torch.Tensor:
        residual_stream = residual_stream + self.masked_self_attention(residual_stream)
        residual_stream = self.ln_1(residual_stream)
        residual_stream = residual_stream + self.mlp(residual_stream)
        residual_stream = self.ln_2(residual_stream)
        return residual_stream

class DecoderLayersConfig(ModuleConfig):
    decoder_layers: int
    decoder_block_config: DecoderBlockConfig

class DecoderLayers(nn.Module):
    def __init__(self, config: DecoderLayersConfig):
        super().__init__()
        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(config.decoder_block_config) for _ in range(config.decoder_layers)]
        )

    def forward(self, residual_stream) -> torch.Tensor:
        for decoder in self.decoder_blocks:
            residual_stream = decoder(residual_stream)
        return residual_stream