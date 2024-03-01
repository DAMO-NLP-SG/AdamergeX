import os
from dataclasses import field, dataclass
from typing import Optional, Any

import torch
import transformers
from transformers import Trainer
import evaluate

from dataset import Seq2SeqDataset, Seq2SeqCollator
from datasets import load_dataset
import numpy as np
import random
from itertools import groupby
from peft import PeftModel
import pdb

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    MT5ForConditionalGeneration,
    ByT5Tokenizer,
    PreTrainedTokenizer,
    T5TokenizerFast as T5Tokenizer,
    MT5TokenizerFast as MT5Tokenizer,
)

random.seed(112)

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from typing import List
import logging
logging.basicConfig(level=logging.INFO)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: Optional[str] = field(default="t5-base")
    data_paths: List[str] = field(default_factory=lambda: ["./alpaca_data.json"], metadata={"help": "Path to the training data."})
    instruction_length: int = 160
    output_length: int = 40
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    load_in_8bit: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    # lora arguments
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q", "v",])

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def train():
    parser = transformers.HfArgumentParser(TrainingArguments)
    args = parser.parse_args_into_dataclasses()[0]

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False
    )

    device_map = "auto"

    if args.model_name_or_path == "google/flan-t5-xxl" and args.load_in_8bit == False:
        logging.info("You are training flan-t5-xxl with float32 data type. "
                     "To save the memory, you may set load_in_8bit to True.")


    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        load_in_8bit=False,
        use_cache=False,
        torch_dtype=torch.float16,
        cache_dir=args.cache_dir,
        device_map=device_map,
    )

    print(model)



    # model = PeftModel.from_pretrained(model, '/home/bizon/Desktop/multilingual-vicuna/ckpts/English_Wiki_T5/', adapter_name="English_Wiki_T5")

    # if args.load_in_8bit:
    #     model = prepare_model_for_int8_training(model)


    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="English_Wiki_mT5_LORA",
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
     

    dataset = load_dataset("blo05/cleaned_wiki_en_80-100")["train"]


    print(dataset[0])

    dataset = Seq2SeqDataset(dataset)
    collator = Seq2SeqCollator(tokenizer, args.instruction_length, args.output_length)

    trainer = Trainer(
        model,
        args=args,
        data_collator=collator,
        train_dataset=dataset,
    )

    trainer.train()

    model.save_pretrained(args.output_dir+"/English_Wiki_mT5_LORA")


if __name__ == "__main__":
    train()
