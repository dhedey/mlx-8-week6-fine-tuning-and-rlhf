import json

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
)

### NOTE:
### This comes in large part from this blog post: https://medium.com/@Uvwxyz/rlhf-on-a-budget-gpt-2-for-summarization-39f9d016202b
### And in particular this code: https://github.com/KookyGhost/RLHF-Summarize-GPT2-Small/blob/master/SFT-gpt2-sum.ipynb

class TLDRDataset(Dataset):
    def __init__(self, train_path, tokenizer, split, max_token_length):
        self.post_list = []
        dataset = load_dataset(train_path, split=split)
        for sample in dataset:
            self.post_list.append(sample["prompt"] + sample["label"])
        if "valid" in split:
            self.post_list = self.post_list[0:2000]
        self.tokenizer = tokenizer
        self.max_token_length = max_token_length
        self.input_ids = []
        self.attn_masks = []

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        txt = self.post_list[idx]
        encodings_dict = self.tokenizer(txt, truncation=True, max_length=self.max_token_length, padding="max_length")
        input_ids = torch.tensor(encodings_dict["input_ids"])
        attn_masks = torch.tensor(encodings_dict["attention_mask"])

        return {
            "input_ids": input_ids,
            "attention_mask": attn_masks,
            "labels": input_ids,
        }
    
def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)

def main():
    output_dir = os.path.join(os.path.dirname(__file__), "snapshots", "tldr_fine_tuned")
    train_batch_size = 4
    gradient_accumulation_steps = 1
    learning_rate = 1e-5
    eval_batch_size = 1
    eval_steps = 500
    max_input_token_length = 1028
    save_steps = 1000
    num_train_epochs = 1
    random.seed(42)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base", use_cache=False)
    tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    # Enable LoRa
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=32,
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
            lora_alpha=16,
            lora_dropout=0.05,
        ),
    )

    model.print_trainable_parameters()

    print_detailed_parameter_counts(model, "model")

    data_path = "CarperAI/openai_summarize_tldr"
    train_dataset = TLDRDataset(
        data_path,
        tokenizer,
        "train",
        max_token_length=max_input_token_length,
    )
    print(f"Train dataset size: {len(train_dataset)}. Expected batches: {(len(train_dataset) // train_batch_size)}")
    dev_dataset = TLDRDataset(
        data_path,
        tokenizer,
        "valid",
        max_token_length=max_input_token_length,
    )

    # Set up the metric
    rouge = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        labels_ids = eval_preds.label_ids
        pred_ids = eval_preds.predictions
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
        result = rouge.compute(predictions=pred_str, references=label_str)
        return result

    # Create a preprocessing function to extract out the proper logits from the model output
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    # Prepare the trainer and start training
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_accumulation_steps=1,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_checkpointing=True,
        bf16=True, # Requires GPU
        # use_mps_device=False,
        adam_beta1=0.9,
        adam_beta2=0.95,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        warmup_steps=100,
        eval_steps=eval_steps,
        save_steps=save_steps,
        load_best_model_at_end=True,
        logging_steps=50,
        max_steps=100, # Comment out after testing
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    
if __name__ == "__main__":
    main()