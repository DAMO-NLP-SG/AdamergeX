import transformers
from torch.utils.data import Dataset
import json
import logging
import torch
from tqdm import tqdm
import pdb


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

        sources = []
        targets = []
        for data in tqdm(dataset):
            words = data['text'].split()

            sources.append(" ".join(words[:-20]))
            targets.append(" ".join(words[-20:]))


        print(sources[0])
        print(targets[0])

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


        return inputs




