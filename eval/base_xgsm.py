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
import json

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


def Prompting(model, prompt):

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**{'input_ids':inputs.input_ids, 'max_new_tokens':512})
    answer = tokenizer.decode(outputs[0]).replace('<pad> ', '')
    answer = answer.replace('</s>', '')
    
    return answer


def main(argv):

    language = {'fr': 'French_50K_LM_511_1', 'de': 'German_50K_LM_511_1', 'tr': 'Turkish_50K_LM_511_1', 'hi': 'Hindi_50K_LM_511_1', 'el': 'Greek_50K_LM_511_1', 'ar': 'Arabic_50K_LM_511_1', 'vi': 'Vietnamese_50K_LM_511_1', 'zh' : 'Chinese_50K_LM_511_1', 'es': 'Spanish_50K_LM_511_1', 'ru': 'Russian_50K_LM_511_1', 'sw': 'Swahili_50K_LM_511_1', 'th': 'Thai_50K_LM_511_1'}

    language_name = language[argv[0]]

    model1 = LlamaForCausalLM.from_pretrained("NousResearch/Llama-2-7b-hf", device_map="auto")

    dataset = list(load_dataset("juletxara/mgsm", argv[0])["test"])


    correct = 0
    all_index = 0

    for data in tqdm(dataset):
        all_index += 1

        task_instruction =  'I want you to act as act as an arithmetic reasoning expert for'+ language_name.replace('_50K_LM_511_1', '') + '\nYou should retell the question in English.\nYou should answer the question in English in one or a few words.\nYou should step-by-step answer the request.'


        prompt = task_instruction + "\n\nQuestion: " + data['question'] + '\nAnswer: '

        answer = Prompting(model1, prompt)

        try:
            answer = re.findall(r'####\s(.+)', answer)[0]
            prd = re.findall(r"\d+\,?\.?\d*",answer)[-1]
            prd = float(prd.replace(',', '').rstrip('.')) if prd else prd
            prd = int(prd)

        except:
            try:
                prd = re.findall(r"\d+\,?\.?\d*",answer)[-1]
                prd = float(prd.replace(',', '').rstrip('.')) if prd else prd
            except:
                prd = -1
        
        answer = prd

        if data['answer_number'] == answer:
            correct += 1
        
    
    print(correct/all_index)

    with open('./Results_Llama2_XLT.txt', 'a', encoding='utf-8') as file:
        file.write('\n')
        file.write('MGSM_' + language_name + '_' + str(argv[1]).replace('_Wiki', '') + '\n' + str(correct/all_index))
        file.write('\n')


        



if __name__ == "__main__":
    main(sys.argv[1:])
