import transformers
from torch.utils.data import Dataset
import json
import logging
import torch
from tqdm import tqdm
import pdb

# PROMPT_DICT = {
#     "prompt_input": (
#         "Below is an instruction that describes a task, paired with an input that provides further context. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
#     ),
#     "prompt_no_input": (
#         "Below is an instruction that describes a task. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n{instruction}\n\n### Response:"
#     ),
# }

PROMPT_DICT = {
    "prompt_input": (
        "{instruction}\n\n {input}\n\n"
    ),
    "prompt_no_input": (
        "{instruction}\n\n"
    ),
}


class Seq2SeqDataset(Dataset):
    def __init__(self, dataset):
        super(Seq2SeqDataset, self).__init__()

        # list_data_dict = []
        # for data_path in data_paths:
        #     logging.warning(f"Loading data from {data_path}...")
        #     with open(data_path, "r") as f:
        #         list_data_dict.extend(json.load(f))

        # prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

        # logging.warning("Formatting data...")
        # sources = [
        #     prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
        #     for example in list_data_dict
        # ]
        # targets = [f"{example['output']}" if 'output' in example else f"{example['response']}" for example in list_data_dict]

        sources = []
        targets = []
        for data in tqdm(dataset):
            words = data['text'].split()

            sources.append(" ".join(words[:-20]))
            targets.append(" ".join(words[-20:]))


        print(sources[0])
        print(targets[0])

        # self.sources = sources
        # self.targets = targets
        self.sources = sources[:10000]
        self.targets = targets[:10000]

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, item):
        return self.sources[item], self.targets[item]


class Seq2SeqCollator(object):
    def __init__(self, tokenizer, intruction_length=160, output_length=40):
        self.tokenizer = tokenizer
        self.tokenizer.pad_token_id = 0
        self.intruction_length = intruction_length
        self.output_length = output_length

    def __call__(self, batch):
        sources = [ex[0] for ex in batch]
        targets = [ex[1] for ex in batch]

        inputs = self.tokenizer(
            sources,
            max_length=self.intruction_length,
            return_tensors='pt',
            padding=True,
            truncation=True
        )

        labels = self.tokenizer(
            targets,
            max_length=self.output_length,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).input_ids

        inputs['labels'] = labels

        # pdb.set_trace()

        return inputs




