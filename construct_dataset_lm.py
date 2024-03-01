from datasets import load_dataset
import numpy as np
import random
import json
from tqdm import tqdm
from itertools import groupby
import pdb
import json
import os

random.seed(112)

import re

def split_string(string):
    # words = string.split()
    words = string.replace('\n','')
    fragments = []
    curr_fragment = []
    left_words = ''
    for word in words:
        curr_fragment.append(word)
        if len(curr_fragment) >= 512:
            fragments.append(("".join(curr_fragment[:-1]), "".join(curr_fragment[-1:])))
            curr_fragment = []
    if curr_fragment:
        left_words = "".join(curr_fragment)

    
    return fragments, left_words


file = os.path.join("./Chinese_Wiki.json")
with open(file) as fin:
    train_data = json.load(fin)

train_instances = []

left_words = ''

for instance in tqdm(train_data[:10000]):
    all_context =left_words + instance['prompt'] + instance['completion']
    all_context_chunk, left_words = split_string(all_context)
    for pair in all_context_chunk:
        train_instance = {"prompt":pair[0], "completion":pair[1]}
        train_instances.append(train_instance)


file = os.path.join("./Chinese_Wiki_test.json")
with open(file) as fin:
    test_data = json.load(fin)

test_instances  = []


left_words = ''

for instance in tqdm(test_data):
    all_context =left_words + instance['prompt'] + instance['completion']
    all_context_chunk, left_words = split_string(all_context)
    for pair in all_context_chunk:
        test_instance = {"prompt":pair[0], "completion":pair[1]}
        test_instances.append(test_instance)



path_file = "./Chinese_Wiki_10k_LM_511_1.json"

with open(path_file, "w") as fout:
    json.dump(train_instances,fout)


path_file = "./Chinese_Wiki_10k_LM_511_1_test.json"

with open(path_file, "w") as fout:
    json.dump(test_instances,fout)