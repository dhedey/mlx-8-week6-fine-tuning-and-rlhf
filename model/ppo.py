
import json
import argparse
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import Dataset
import os
from .harness import print_detailed_parameter_counts

import random
from peft import LoraConfig, TaskType, get_peft_model

import evaluate
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    TextIteratorStreamer,
    GenerationConfig,
    TrainerCallback, EvalPrediction,
)
from threading import Thread
from peft import PeftModel
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

### NOTE:
### This comes in large part from this blog post: https://medium.com/@Uvwxyz/rlhf-on-a-budget-gpt-2-for-summarization-39f9d016202b
### And in particular this code: https://github.com/KookyGhost/RLHF-Summarize-GPT2-Small/blob/master/SFT-gpt2-sum.ipynb

from .model_common import TLDRDataset, set_seed, stream_generate_summary, test_streaming_inference

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument(
    '--resume',
    action='store_true',
)
args = parser.parse_args()
if args.resume:
    print("Resuming...")
    resume = True
else:
    print("No --resume flag was supplied, so starting from scratch...")
    resume = False

# Config
sft_path = os.path.join(os.path.dirname(__file__), "trained/tldr_fine_tuned")
output_dir = os.path.join(os.path.dirname(__file__), "snapshots", "colour_ppo")
trained_output_dir = os.path.join(os.path.dirname(__file__), "trained", "colour_ppo")
train_batch_size = 8
gradient_accumulation_steps = 1
learning_rate = 1e-5
eval_batch_size = 1
eval_steps = 500
eval_dataset_size = 400
max_input_token_length = 700
save_steps = 500
num_train_epochs = 1
train_dataset_size = 16000
random.seed(42)

# Reward Model
class TextBasedRewardModel(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask=None):
        texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        final_rewards = []
        target_words = {"red", "green", "yellow", "blue", "orange", "purple", "violet", "mauve", "color", "colorful",
                        "color", "light", "rainbow", "cyan", "magenta"}
        for text in texts:
            words = text.split()
            score = 0
            for word in words:
                if word.lower() in target_words:
                    score += 1
            word_count = len(words)
            final_rewards.append(score / word_count if word_count > 0 else 0)

        return final_rewards

## See https://whimsical.com/week-6-ppo-16R4j1A2455itf65D4HWkH for details

print("Loading Qwen Base Model...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base", use_cache=False)
tokenizer.pad_token = tokenizer.eos_token
base_model.resize_token_embeddings(len(tokenizer))
tokenizer.pad_token_id = tokenizer.eos_token_id
base_model.config.end_token_id = tokenizer.eos_token_id
base_model.config.pad_token_id = base_model.config.eos_token_id

# Load SFT Model and Tokenizer
# Load base model with value head, then apply LoRA adapter
print("Loading the base policy, and adding a value head to act as the value model")
base_policy = AutoModelForCausalLMWithValueHead(PeftModel.from_pretrained(base_model, sft_path).merge_and_unload()).to(DEVICE)
policy_model = get_peft_model(
    base_policy,
    LoraConfig(
        r=32,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
        lora_alpha=16,
        lora_dropout=0.05,
    ),
)

print("Loading the reference policy...")
reference_policy = PeftModel.from_pretrained(base_model, sft_path).merge_and_unload().to(DEVICE)

print("\n Reference policy parameters:")
print_detailed_parameter_counts(reference_policy, "reference_policy")

print("\n Training policy parameters:")
policy_model.print_trainable_parameters()
print_detailed_parameter_counts(reference_policy, "policy")

print()

# Build Reward Model
reward_model = TextBasedRewardModel(policy_model.base_model).to(DEVICE)
reward_model.eval()

data_path = "CarperAI/openai_summarize_tldr"
train_dataset = TLDRDataset(
    data_path,
    tokenizer,
    "train",
    size_cap=train_dataset_size,
    max_token_length=max_input_token_length,
)
print(f"Train dataset size: {len(train_dataset)}. Expected batches: {(len(train_dataset) // train_batch_size)}")
eval_dataset = TLDRDataset(
    data_path,
    tokenizer,
    "valid",
    size_cap=eval_dataset_size,
    max_token_length=max_input_token_length,
)
print(f"Eval dataset size: {len(eval_dataset)}. Expected batches: {(len(eval_dataset) // eval_batch_size)}")

# Set up the metric
rouge = evaluate.load("rouge")

ppo_config = PPOConfig(
    output_dir=output_dir,
    eval_strategy="steps",
    eval_accumulation_steps=1,
    learning_rate=learning_rate,
    batch_size=train_batch_size,
    mini_batch_size=1,
    gradient_accumulation_steps=gradient_accumulation_steps,
    adam_beta1=0.9,
    adam_beta2=0.95,
    bf16=True,
    num_train_epochs=num_train_epochs,
    warmup_steps=100,
    eval_steps=eval_steps,
    save_steps=save_steps,
    load_best_model_at_end=True,
    logging_steps=50,
)

class PrintEvalCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        test_streaming_inference(model, tokenizer)
        print("\nEvaluation metrics:", metrics)

print("\n🚀 Before PPO starts... Let's test streaming inference...")
test_streaming_inference(policy_model, tokenizer)

# Set up PPOTrainer (no custom DataLoader or collate_fn needed)
ppo_trainer = PPOTrainer(
    ppo_config,         # args
    tokenizer,          # processing_class
    model=policy_model,              # policy model (policy, with value head and LoRA)
    ref_model=reference_policy,      # reference model
    reward_model=reward_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    value_model=policy_model,        # value_model (with value head)
    callbacks=[PrintEvalCallback()],
)

# Run PPO training
ppo_trainer.train(
    resume_from_checkpoint=resume,
)

# Save the PPO-updated model
policy_model.save_pretrained(trained_output_dir)

