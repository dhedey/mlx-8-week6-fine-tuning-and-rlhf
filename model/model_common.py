import json
import argparse
import pandas as pd
import torch
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

### NOTE:
### This comes in large part from this blog post: https://medium.com/@Uvwxyz/rlhf-on-a-budget-gpt-2-for-summarization-39f9d016202b
### And in particular this code: https://github.com/KookyGhost/RLHF-Summarize-GPT2-Small/blob/master/SFT-gpt2-sum.ipynb

class TLDRDataset(Dataset):
    def __init__(self, train_path, tokenizer, split, max_token_length, size_cap=None):
        self.prompts = []
        self.labels = []
        dataset = load_dataset(train_path, split=split)
        for sample in dataset:
            self.prompts.append(sample["prompt"])
            self.labels.append(sample["label"])
        if size_cap is not None:
            self.prompts = self.prompts[:size_cap]
            self.labels = self.labels[:size_cap]
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        label = self.labels[idx]
        # Tokenize prompt and label separately
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        label_ids = self.tokenizer(label, add_special_tokens=False)["input_ids"]
        # Concatenate and truncate
        input_ids = (prompt_ids + label_ids)[:self.max_token_length]
        attn_mask = [1] * len(input_ids)
        # Pad if necessary
        pad_len = self.max_token_length - len(input_ids)
        if pad_len > 0:
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            attn_mask += [0] * pad_len
        # Mask prompt tokens in labels
        labels = input_ids.copy()
        labels[:min(len(prompt_ids), self.max_token_length)] = [-100] * min(len(prompt_ids), self.max_token_length)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attn_mask),
            "labels": torch.tensor(labels),
        }
    
def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)

def stream_generate_summary(model, tokenizer, prompt, max_new_tokens=50):
    """
    Stream generate a summary using the fine-tuned model.
    
    Args:
        model: Fine-tuned causal LM model
        tokenizer: Tokenizer
        prompt: Input prompt for summarization
        max_new_tokens: Maximum tokens to generate
        
    Yields:
        Generated text tokens as they are produced
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Create streaming iterator
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=60.0
    )
    
    # Configuration optimized for summarization
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Start generation in background thread
    generation_kwargs = {
        **inputs,
        "generation_config": generation_config,
        "streamer": streamer,
    }

    with torch.no_grad():
        model.gradient_checkpointing_disable()

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
    
        try:
            for token in streamer:
                yield token
        finally:
            thread.join()
            model.gradient_checkpointing_enable()

def test_streaming_inference(model, tokenizer):
    """Test the streaming capabilities of the trained model"""
    print("\n" + "="*60)
    print("TESTING STREAMING INFERENCE")
    print("="*60)
    
    # Test prompts based on your TLDR training data format
    test_prompts = [
        "SUBREDDIT: r/MachineLearning\n\nTITLE: New breakthrough in transformer efficiency\n\nPOST: Researchers have developed a novel attention mechanism that reduces computational complexity by 40% while maintaining performance across multiple NLP benchmarks. The method uses sparse attention patterns and dynamic head selection to achieve these gains.\n\nTL;DR:",
        
        "SUBREDDIT: r/technology\n\nTITLE: Revolutionary quantum computing advance\n\nPOST: Scientists at a major university have demonstrated a quantum computer that can maintain coherence for 100 times longer than previous systems. This breakthrough could accelerate practical quantum computing applications in cryptography and optimization.\n\nTL;DR:",
        
        "SUBREDDIT: r/science\n\nTITLE: Climate study reveals urgent findings\n\nPOST: A comprehensive analysis of global temperature data shows that warming is accelerating faster than previous models predicted. The study examined satellite data from the past 30 years and found concerning trends in polar ice loss.\n\nTL;DR:"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}:")
        print(f"Input: {prompt[:100]}...")
        print("Streaming Summary: ", end="", flush=True)
        
        # Stream the generated summary
        for token in stream_generate_summary(model, tokenizer, prompt):
            print(token, end="", flush=True)
        
        print("\n" + "-"*50)
