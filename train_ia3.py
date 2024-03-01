import torch
import transformers
from datasets import load_dataset
from transformers import Trainer
from dataset import Seq2SeqDataset, Seq2SeqCollator
from transformers import TrainingArguments


from peft import (
    LoraConfig,
    IA3Config,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)


args = TrainingArguments("working_dir")

args = args.set_testing(num_train_epochs=3,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    learning_rate = 5e-4,
    weight_decay = 0.,   
    warmup_ratio = 0.03,
    lr_scheduler_type = "cosine",
    logging_steps = 50,
    tf32 = True,
    )

device_map = "auto"


tokenizer = transformers.AutoTokenizer.from_pretrained(
        "t5-base",
        model_max_length=512,
        padding_side="right",
        use_fast=False
    )


model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
    "t5-base",
    load_in_8bit=False,
    use_cache=False,
    torch_dtype=torch.float16,
    device_map=device_map,
)


config = IA3Config()

model = get_peft_model(model, config)
model.print_trainable_parameters()

dataset = load_dataset("blo05/cleaned_wiki_en_80-100")["train"]

print(dataset[0])

dataset = Seq2SeqDataset(dataset)
collator = Seq2SeqCollator(tokenizer, 40, 160)


trainer = Trainer(
    model,
    data_collator=collator,
    evaluation_strategy = "no",
    save_strategy = "no",
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained("./ckpts/"+"/English_Wiki_T5_IA3")
