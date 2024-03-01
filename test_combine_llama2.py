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

model1 = PeftModel.from_pretrained(model1, '/home/masum/junteng/frank/multilingual-vicuna/results/english_gsm_llama2_lora/', adapter_name="English_GSM")
model2 = PeftModel.from_pretrained(model2, '/home/masum/junteng/frank/multilingual-vicuna/results/swahili_wiki_llama2_lora/', adapter_name="Swahili_Wiki")
model3 = PeftModel.from_pretrained(model3, '/home/masum/junteng/frank/multilingual-vicuna/results/final_checkpoint/', adapter_name="English_Wiki")


params_model1 = dict(model1.named_parameters())
params_model2 = dict(model2.named_parameters())
params_model3 = dict(model3.named_parameters())


for name, param in tqdm(params_model2.items()):
    # Get corresponding parameter in model2
    # if "SelfAttention.k.ia3_l.French_Wiki_T5_IA3" in name or 'SelfAttention.v.ia3_l.French_Wiki_T5_IA3' in name:
    if "Swalihi_Wiki" in name:
        name = name.replace("Swalihi_Wiki", "English_Wiki")
        param_model3 = params_model3.get(name, None)
        
        # Average the coefficients
        param.data =  (param.data - param_model3.data)


for name, param in tqdm(params_model1.items()):
    # Get corresponding parameter in model2
    # if "SelfAttention.k.ia3_l.English_Amazon_T5_IA3" in name or "SelfAttention.v.ia3_l.English_Amazon_T5_IA3" in name:
    if "English_GSM" in name:
        name = name.replace("English_GSM", "Swahili_Wiki")
        param_model2 = params_model2.get(name, None)
    
        # Average the coefficients
        # param.data = param.data * (lambda_hyper * (param_model2.data - 1) + 1)
        param.data = param.data  + lambda_hyper * param_model2.data



def Prompting(model, prompt):

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**{'input_ids':inputs.input_ids, 'max_new_tokens':512})
    answer = tokenizer.decode(outputs[0]).replace('<pad> ', '')
    answer = answer.replace('</s>', '')
    
    return answer


def test():
    dataset = list(load_dataset("juletxara/mgsm", 'sw')["test"])
    dataset = random.sample(dataset, 100)

    # sorted_data = sorted(dataset, key=lambda x: x['stars'])
    # grouped_data = {key: list(group) for key, group in groupby(sorted_data,key=lambda x: x['stars'] )}

    # print(len((grouped_data[1])))
    # print(len((grouped_data[2])))
    # print(len((grouped_data[3])))
    # print(len((grouped_data[4])))
    # print(len((grouped_data[5])))

    correct = 0
    all_index = 0

    for data in tqdm(dataset):
        all_index += 1
        # task_instruction = """In this task, you are given a premise sentence, two possible options and a question word. If the question was cause you should select the option that is a possible cause of the premise sentence, and if the question word was effect you should find the option which is a possible effect of the premise sentence. Answer with \"A\" or \"B\"."""
        # task_instruction = """In this task, you are given Yelp reviews. The task is to classify a review as \"1\" if the overall sentiment of the review is positive or as \"0\" if the overall sentiment of the review is negative. Answer by \'0\' or \'1\'"""
        # task_instruction = """You are given a review about a place. You need to provide a rating from \"1\" to \"5\" for this place."""
        # task_instruction = """In this task, you are given Yelp reviews. The task is to classify a review as \"1\" if the overall sentiment of the review is positive or as \"0\" if the overall sentiment of the review is negative. Answer by \'0\' or \'1\'"""
        # task_instruction = """Vous recevez un avis sur un produit. Vous devez fournir une note de « 1 » à « 5 » pour ce produit."""

        # prompt = task_instruction + "\n\nSaisir: " + data['review_body'] + '\nSortir: '

        # task_instruction = """Se le da una reseña sobre un producto. Debe proporcionar una calificación de \"1\" a \"5\" para este producto."""

        # prompt = task_instruction + "\n\nAporte: " + data['review_body'] + '\nProducción: '
    
        # task_instruction = """You are given a review about a product. You need to provide a rating from \"1\" to \"5\" for this product."""
        
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

        # print(answer)

        
        # if data['sentiment'] == int(answer):
        if data['answer_number'] == int(answer):
            correct += 1
        
    
    print(correct/all_index)


        



if __name__ == "__main__":
    test()
