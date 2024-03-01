import os
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments,BitsAndBytesConfig
from datasets import load_dataset
from trl import SFTTrainer
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from utils import find_all_linear_names, print_trainable_parameters
from dataclasses import field
import random
import pdb
from datasets import Dataset

output_dir="./Results/"
model_name ="NousResearch/Llama-2-7b-hf"

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--language", type=str, default="English")
parser.add_argument("--task", type=str, default="Wiki")

args = parser.parse_args()
print(args)

dataset = load_dataset("json", data_files="German_Wiki.json",split="train")


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(base_model)


tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

# Change the LORA hyperparameters accordingly to fit your use case
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules = [
        "q_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "k_proj",
        "down_proj",
        "up_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)



base_model = get_peft_model(base_model, peft_config)

print_trainable_parameters(base_model)


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        text = f"```{example['prompt'][i]}```{example['completion'][i]}"
        # text = f"```{example['prompt'][i]}{example['completion'][i]}```"
        output_texts.append(text)
    return output_texts

# Parameters for training arguments details => https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py#L158
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    gradient_checkpointing =True,
    max_grad_norm= 0.3,
    num_train_epochs=3, 
    learning_rate=2e-4,
    bf16=True,
    save_total_limit=3,
    logging_steps=10,
    output_dir=output_dir,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
)

trainer = SFTTrainer(
    base_model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_seq_length=1024,
    formatting_func=formatting_prompts_func,
    args=training_args
)

trainer.train() 
trainer.save_model(output_dir)

output_dir = os.path.join(output_dir, f"{args.language}_{args.task}_lora")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)