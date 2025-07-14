import torch.nn as nn
import math
from model.harness import ModuleConfig


class CustomLoraConfig(ModuleConfig):
    input_dim: int
    output_dim: int
    rank: int
    alpha: float = 32.0  # scaling factor
    dropout: float = 0.1

class CustomLora(nn.Module):
    """
    Standard LoRA implementation: lora_B(lora_A(dropout(x))) * alpha
    """
    def __init__(self, config: CustomLoraConfig):
        super().__init__()
        self.config = config
        self.rank = config.rank
        self.alpha = config.alpha
        self.dropout = nn.Dropout(config.dropout)
        
        # Low-rank decomposition: W = A @ B
        # A: (input_dim, rank), B: (rank, output_dim)
        self.lora_A = nn.Linear(config.input_dim, config.rank, bias=False)
        self.lora_B = nn.Linear(config.rank, config.output_dim, bias=False)  # No bias in B
        
        # Initialize weights properly for LoRA
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)  # Initialize B to zero
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        x = self.dropout(x)
        x = self.lora_A(x)  # (batch_size, seq_len, rank)
        x = self.lora_B(x)  # (batch_size, seq_len, output_dim)
        return x * self.alpha


