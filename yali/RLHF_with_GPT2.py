"""
https://github.com/KookyGhost/RLHF-Summarize-GPT2-Small/blob/master/SFT-gpt2-sum.ipynb
************* step 1: fine-tuning gpt-2 for summarization  ***************
"""


"""
dataset
"""
import json

import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

class TLDRDataset(Dataset):
    def __init__(self, train_path, tokenizer, split, max_length=550):
        self.post_list = []
        dataset = load_dataset(train_path, split=split)
        for sample in dataset:
            self.post_list.append(sample["prompt"] + sample["label"])
        if "valid" in split:
            self.post_list = self.post_list[0:2000]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_ids = []
        self.attn_masks = []

    def __len__(self):
        return len(self.post_list)

    def __getitem__(self, idx):
        txt = self.post_list[idx]
        encodings_dict = self.tokenizer(txt, truncation=True, max_length=self.max_length, padding="max_length")
        input_ids = torch.tensor(encodings_dict["input_ids"])
        attn_masks = torch.tensor(encodings_dict["attention_mask"])

        return {
            "input_ids": input_ids,
            "attention_mask": attn_masks,
            "labels": input_ids,
        }

    
import random
import evaluate
import numpy as np
import torch
# from summarize_dataset import TLDRDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)


def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)




output_dir = "./gpt2-supervised-summarize-checkpoint"
train_batch_size = 24
gradient_accumulation_steps = 2
learning_rate = 1e-5
eval_batch_size = 1
eval_steps = 2000
max_input_length = 550
save_steps = 4000
num_train_epochs = 1
random.seed(42)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", use_cache=False)
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))
tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.end_token_id = tokenizer.eos_token_id
model.config.pad_token_id = model.config.eos_token_id

if torch.cuda.is_available():
    model = model.to("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("CUDA not available, using CPU.")


data_path = "CarperAI/openai_summarize_tldr"
train_dataset = TLDRDataset(
    data_path,
    tokenizer,
    "train",
    max_length=max_input_length,
)
dev_dataset = TLDRDataset(
    data_path,
    tokenizer,
    "valid",
    max_length=max_input_length,
)

dataset = load_dataset(data_path, split='train')




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
    half_precision_backend=True,
    fp16=True,
    adam_beta1=0.9,
    adam_beta2=0.95,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=num_train_epochs,
    warmup_steps=100,
    eval_steps=eval_steps,
    save_steps=save_steps,
    max_steps=29000,
    load_best_model_at_end=True,
    logging_steps=200,
    # deepspeed="/notebooks/ds_config_gptj.json",
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



