import os
import pdb
import torch
from datasets import load_dataset
from peft import (LoraConfig, PeftType, PrefixTuningConfig,
                  PromptEncoderConfig, get_peft_config, get_peft_model,
                  get_peft_model_state_dict, set_peft_model_state_dict)
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm
from transformers import (AutoModelForSequenceClassification, AutoConfig, AutoTokenizer,
                          Trainer, TrainingArguments,
                          get_linear_schedule_with_warmup, set_seed)
     
MODEL_NAME_OR_PATH = "xlm-roberta-base"
MAX_LENGTH=128
DEVICE = "cuda"
NUM_EPOCHS = 5
PADDING_SIDE= "right"
EPOCHS = 3
LR = 2e-5
TRAIN_BS = 128
EVAL_BS = TRAIN_BS * 2
     

datasets = load_dataset("xnli", 'en')

for split in datasets:
    dataset = datasets[split]
    if 'label' in dataset.features:
        dataset = dataset.rename_column('label', 'labels')
        datasets[split] = dataset


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, padding_side=PADDING_SIDE)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

def tokenize_function(examples):
    new_prompt = []
    for i in range(len(examples['premise'])):
        prompt = 'You should judge whether the hypothesis is true (entailment), false (contradiction), or undetermined (neutral) given the premise. The relationship can be chosen from entailment, contradiction, and neutral.\nPremise: ' + examples['premise'][i] + '\nHypothesis: ' + examples['hypothesis'][i] + '\nRelationship: '
        new_prompt.append(prompt)
    examples['premise'] = new_prompt
    outputs = tokenizer(examples['premise'], truncation=True, max_length=MAX_LENGTH)
    return outputs

tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=['premise', 'hypothesis'],
)


all_langs = sorted(list(set([0, 1, 2])))

id2label = {idx: all_langs[idx] for idx in range(len(all_langs))}
label2id = {v: k for k, v in id2label.items()}

tokenized_datasets = tokenized_datasets.map(lambda example: {"labels": example["labels"]})
tok_train = tokenized_datasets['train']
tok_valid = tokenized_datasets['validation']
tok_test = tokenized_datasets['test']

print(f"Train / valid / test samples: {len(tok_train)} / {len(tok_valid)} / {len(tok_test)}")


def collate_fn(examples):
    return tokenizer.pad(examples, padding=True, return_tensors="pt")
    
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {
        "accuracy": acc,
        "f1": f1
        }


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME_OR_PATH,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True, 
    return_dict=True
)
peft_config = LoraConfig(
    task_type="SEQ_CLS",
    inference_mode=False,
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)
     
lora_model = get_peft_model(model, peft_config)
lora_model.print_trainable_parameters()

lora_model.to(DEVICE)

logging_steps = len(tokenized_datasets["train"]) // TRAIN_BS
output_dir = "./ckpts/English_XNLI_XLMR"

args = TrainingArguments(
    optim='adamw_torch',
    output_dir=output_dir,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    per_device_train_batch_size=TRAIN_BS,
    per_device_eval_batch_size=EVAL_BS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=logging_steps,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    fp16=True,  # Remove if GPU doesn't support it
)
  
trainer = Trainer(
    lora_model,
    args,
    compute_metrics=compute_metrics,
    train_dataset=tok_train,
    eval_dataset=tok_valid,
    data_collator=collate_fn,
    tokenizer=tokenizer,
)

trainer.train()

lora_model.save_pretrained(output_dir)

trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

