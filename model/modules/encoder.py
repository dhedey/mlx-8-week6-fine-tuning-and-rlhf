import torch.nn as nn
from typing import Optional
from .transformer import UnmaskedAttentionConfig, UnmaskedAttention, MultiLayerPerceptron
from model.harness import ModuleConfig

class SelfEncoderBlockConfig(ModuleConfig):
    kq_dimension: int
    v_dimension: int
    embedding_dimension: int
    num_heads: int
    mlp_hidden_dimension: Optional[int]
    mlp_dropout: float

class SelfEncoderBlock(nn.Module):
    def __init__(self, config: SelfEncoderBlockConfig):
        super().__init__()

        attention_config = UnmaskedAttentionConfig(
            kq_dimension=config.kq_dimension,
            v_dimension=config.v_dimension,
            encoder_embedding_dimension=config.embedding_dimension,
            decoder_embedding_dimension=config.embedding_dimension,
            num_heads=config.num_heads,
        )
        self.attention = UnmaskedAttention(attention_config)

        self.layer_norm_1 = nn.LayerNorm(config.embedding_dimension)

        mlp_hidden_dimension = config.mlp_hidden_dimension

        if mlp_hidden_dimension is None:
            mlp_hidden_dimension = config.embedding_dimension * 4 # Commonly used in transformers

        self.mlp = MultiLayerPerceptron(
            embedding_dimension=config.embedding_dimension,
            hidden_dimension=mlp_hidden_dimension,
            dropout=config.mlp_dropout,
        )
        self.layer_norm_2 = nn.LayerNorm(config.embedding_dimension)

    def forward(self, residual_stream):
        # Shape: (Batch, Sequence Length, Embedding Dimension)
        residual_stream = residual_stream + self.attention(residual_stream, residual_stream)
        residual_stream = self.layer_norm_1(residual_stream)
        residual_stream = residual_stream + self.mlp(residual_stream)         
        residual_stream = self.layer_norm_2(residual_stream)

        return residual_stream
