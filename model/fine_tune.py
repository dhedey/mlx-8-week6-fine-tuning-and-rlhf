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

from .model_common import TLDRDataset, set_seed, stream_generate_summary, test_streaming_inference

def main():

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

    output_dir = os.path.join(os.path.dirname(__file__), "snapshots", "tldr_fine_tuned")
    trained_output_dir = os.path.join(os.path.dirname(__file__), "trained", "tldr_fine_tuned")
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

    def compute_metrics(eval_preds: EvalPrediction):
        labels_to_use = eval_preds.label_ids
        preds_to_use = eval_preds.predictions

        # Filter out the original texts
        preds_to_use[labels_to_use == -100] = tokenizer.pad_token_id
        # Prevent the tokenizer throwing an exception with -100...
        labels_to_use[labels_to_use == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(preds_to_use, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(labels_to_use, skip_special_tokens=True)
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
        bf16=True,
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
        # max_steps=100, # Comment out after testing
    )

    class PrintEvalCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
            test_streaming_inference(model, tokenizer)
            print("\nEvaluation metrics:", metrics)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[PrintEvalCallback()],
    )

    print("\n🚀 Before training starts... Let's test streaming inference...")
    test_streaming_inference(model, tokenizer)

    trainer.train(
        resume_from_checkpoint=resume,
    )
    
    # Test streaming inference with the trained model
    print("\n🚀 Training completed! Testing streaming inference...")
    test_streaming_inference(model, tokenizer)

    print("\nSaving model...")
    trainer.save_model(trained_output_dir)
    
if __name__ == "__main__":
    main()