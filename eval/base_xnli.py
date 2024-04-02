import os
from dataclasses import field, dataclass
from typing import Optional, Any
import sys

import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer

from rouge_score import rouge_scorer
from transformers import AutoTokenizer
import random
from itertools import groupby
import pdb
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


tokenizer = LlamaTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")

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
import json

def Prompting(model, prompt):

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**{'input_ids':inputs.input_ids, 'max_new_tokens':16})
    answer = tokenizer.decode(outputs[0]).replace('<pad> ', '')
    answer = answer.replace('</s>', '')
    
    return answer


def main(argv):

    language = {'fr': 'French_50K_LM_511_1', 'de': 'German_50K_LM_511_1', 'tr': 'Turkish_50K_LM_511_1', 'hi': 'Hindi_50K_LM_511_1', 'el': 'Greek_50K_LM_511_1', 'ar': 'Arabic_50K_LM_511_1', 'vi': 'Vietnamese_50K_LM_511_1', 'zh' : 'Chinese_50K_LM_511_1', 'es': 'Spanish_50K_LM_511_1', 'ru': 'Russian_50K_LM_511_1', 'sw': 'Swahili_50K_LM_511_1', 'th': 'Thai_50K_LM_511_1'}


    language_name = language[argv[0]]

    model1 = LlamaForCausalLM.from_pretrained("NousResearch/Llama-2-7b-hf", device_map="auto")


    dataset = list(load_dataset("xnli", argv[0])["test"])
    try:
        dataset = random.sample(dataset, 1000)
    except:
        dataset = dataset


    json_str = json.dumps(dataset)

    with open("testset_XNLI_" +language_name.replace('_50K_LM_511_1', '') + ".json", "w") as json_file:
        json_file.write(json_str)

    correct = 0
    all_index = 0

    for data in tqdm(dataset):
        all_index += 1

        source = "I want you to act as a natural language inference expert for" + language_name.replace('_50K_LM_511_1', '') + "." + "\nPremise: " + data['premise'] + "\nHypothesis: " + data['hypothesis'] + "\nYou should retell the premise and hypothesis in English.\nYou should judge whether the hypothesis is true (entailment), false(contradiction), or undetermined (neutral) given the premise. The relationship can be chosen from entailment, contradiction, and neutral.\nYou should step-by-step answer the request. Answer by entailment, contradiction or neutral." + "\nRelationship: "

        print(source)
        if data['label']  == 0:
            target = 'entailment'
        elif data['label']  == 1:
            target = 'neutral'
        elif data['label']  == 2:
            target = 'contradiction'

        answer = Prompting(model1, source)
        print(answer)


        try:
            answer = re.findall(r'Relationship:\s(.+)', answer)[0]
            answer = answer.lower()

        except:
            answer = answer
        
        print(answer)

        
        if target in answer:
            correct += 1
        
    
    print(correct/all_index)

    with open('./Results_Llama2_XLT.txt', 'a', encoding='utf-8') as file:
        file.write('\n')
        file.write('XNLI_' + language_name + '_' + str(argv[1]).replace('_Wiki', '') + '\n' + str(correct/all_index))
        file.write('\n')


        



if __name__ == "__main__":
    main(sys.argv[1:])
