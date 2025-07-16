import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import load_dataset
import copy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Reward Model
class RewardModel(nn.Module):
    def __init__(self, base_model, freeze_ratio=0.7):
        super().__init__()
        self.transformer = base_model.model
        self.config = base_model.config
        self.v_head = nn.Linear(self.config.hidden_size, 1, bias=False)
        layers = self.transformer.model.layers
        num_layers = len(layers)
        num_frozen = int(freeze_ratio * num_layers)
        for layer in layers[:num_frozen]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states[-1]  # (batch, seq, hidden)
        rewards = self.v_head(hidden_states).squeeze(-1)  # (batch, seq)
        final_rewards = rewards[:, -1]  # Only the final token
        return final_rewards

# Load SFT Model and Tokenizer
sft_path = "trained/tldr_fine_tuned"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")

# Load base model with value head, then apply LoRA adapter
base_policy = AutoModelForCausalLMWithValueHead.from_pretrained("Qwen/Qwen3-0.6B-Base").to(DEVICE)
policy_model = PeftModel.from_pretrained(base_policy, sft_path).to(DEVICE)

# Reference and value models (no LoRA, just base with value head)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained("Qwen/Qwen3-0.6B-Base").to(DEVICE)
value_model = AutoModelForCausalLMWithValueHead.from_pretrained("Qwen/Qwen3-0.6B-Base").to(DEVICE)

# Build Reward Model
reward_model = RewardModel(policy_model.base_model).to(DEVICE)
reward_model.eval()

# Load dataset
comparison_dataset = load_dataset("CarperAI/openai_summarize_comparisons", split="train")

# PPO dataset: returns dict with 'input_ids' and 'attention_mask'
class TokenizedPromptDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        prompt = self.dataset[idx]["prompt"]
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}

ppo_dataset = TokenizedPromptDataset(comparison_dataset, tokenizer)

ppo_config = PPOConfig(
    learning_rate=1e-5,
    batch_size=32,
    mini_batch_size=1,
    gradient_accumulation_steps=1,
)

# Set up PPOTrainer (no custom DataLoader or collate_fn needed)
ppo_trainer = PPOTrainer(
    ppo_config,         # args
    tokenizer,          # processing_class
    policy_model,       # model (policy, with value head and LoRA)
    ref_model,          # ref_model (with value head)
    reward_model,       # reward_model
    ppo_dataset,        # train_dataset
    value_model         # value_model (with value head)
)

# Run PPO training
ppo_trainer.train()

# Save the PPO-updated model
policy_model.save_pretrained("trained/tldr_fine_tuned/ppo_rlhf_tuned")

