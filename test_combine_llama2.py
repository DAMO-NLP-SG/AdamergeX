import os
from dataclasses import field, dataclass
from typing import Optional, Any

import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer

from rouge_score import rouge_scorer
from transformers import AutoTokenizer
import random
from itertools import groupby

import re

from tqdm import tqdm

class GPTTokenizer:
    gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2", max_length=1e5)

    def tokenize(self, s):
        tokens = self.gpt_tokenizer.tokenize(s)
        # GPT2 uses Byte-level BPE, which will include space as part of the word. 
        # But for the first word of a sentence, there is no space before it. 
        # So, we remove all the added spaces ("Ġ"). 
        tokens = [t.lstrip("Ġ") for t in tokens]
        return tokens

default_rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
xlingual_tokenizer = GPTTokenizer()
xlingual_rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], tokenizer=xlingual_tokenizer) 


from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

def rouge(prediction, ground_truth, xlingual=False):
    if xlingual:
        scorer = xlingual_rouge_scorer
    else:
        scorer = default_rouge_scorer
    scores = scorer.score(prediction=str(prediction), target=str(ground_truth))
    rougeL = 100.0 * scores["rougeL"].fmeasure / len(ground_truth)

    return rougeL

from dataset import Seq2SeqDataset, Seq2SeqCollator
from datasets import load_dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

from peft import PeftModel
from typing import List
import logging
logging.basicConfig(level=logging.INFO)

random.seed(112)

import torch


lambda_hyper = 0

tokenizer = LlamaTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")

model1 = LlamaForCausalLM.from_pretrained("NousResearch/Llama-2-7b-hf", device_map="auto")
model2 = LlamaForCausalLM.from_pretrained("NousResearch/Llama-2-7b-hf", device_map="auto")
model3 = LlamaForCausalLM.from_pretrained("NousResearch/Llama-2-7b-hf", device_map="auto")

model1 = PeftModel.from_pretrained(model1, './results/english_gsm_llama2_lora/', adapter_name="English_GSM")
model2 = PeftModel.from_pretrained(model2, './results/swahili_wiki_llama2_lora/', adapter_name="Swahili_Wiki")
model3 = PeftModel.from_pretrained(model3, './results/final_checkpoint/', adapter_name="English_Wiki")


params_model1 = dict(model1.named_parameters())
params_model2 = dict(model2.named_parameters())
params_model3 = dict(model3.named_parameters())


for name, param in tqdm(params_model2.items()):
    # Get corresponding parameter in model2
    if "Swalihi_Wiki" in name:
        name = name.replace("Swalihi_Wiki", "English_Wiki")
        param_model3 = params_model3.get(name, None)
        
        # LoRA
        param.data =  (param.data - param_model3.data)
        
        # IA3
        # param.data =  (param.data / param_model3.data)


for name, param in tqdm(params_model1.items()):
    # Get corresponding parameter in model2
    if "English_GSM" in name:
        name = name.replace("English_GSM", "Swahili_Wiki")
        param_model2 = params_model2.get(name, None)

        # LoRA
        param.data = param.data  + lambda_hyper * param_model2.data

        # IA3
        # param.data = param.data * (lambda_hyper * (param_model2.data - 1) + 1)



def Prompting(model, prompt):

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**{'input_ids':inputs.input_ids, 'max_new_tokens':512})
    answer = tokenizer.decode(outputs[0]).replace('<pad> ', '')
    answer = answer.replace('</s>', '')
    
    return answer


def test():
    dataset = list(load_dataset("juletxara/mgsm", 'sw')["test"])
    dataset = random.sample(dataset, 100)

    correct = 0
    all_index = 0

    for data in tqdm(dataset):
        all_index += 1


        task_instruction = """Let\'s think step by step."""

        prompt = task_instruction + "\n\nQuestion: " + data['question'] + '\nAnswer: '

        answer = Prompting(model1, prompt)
        print(answer)

        try:
            answer = int(re.findall(r'\d+', answer)[-1])
            print(answer)
        except:
            print(answer)
            answer = 0


        if data['answer_number'] == int(answer):
            correct += 1
        
    
    print(correct/all_index)


        



if __name__ == "__main__":
    test()
