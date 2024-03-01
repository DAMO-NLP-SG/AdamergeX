# AdaMergeX

This repository contains code for the paper "[AdaMergeX: Cross-Lingual Transfer with Large Language Models via Adaptive Adapter Merging](https://arxiv.org/pdf/2402.18913.pdf) ". Below is the workflow of AdaMergeX.

![./](./adamergex.png)



### Abstract

As an effective alternative to the direct finetuning on target tasks in specific languages, cross-lingual transfer addresses the challenges of limited training data by decoupling "task ability" and "language ability" by fine-tuning on the target task in the source language and another selected task in the target language, respectively. However, they fail to fully separate the task ability from the source language or the language ability from the chosen task. In this paper, we acknowledge the mutual reliance between task ability and language ability and direct our attention toward the gap between the target language and the source language on tasks. As the gap removes the impact of tasks, we assume that it remains consistent across tasks. Based on this assumption, we propose a new cross-lingual transfer method called AdaMergeX that utilizes adaptive adapter merging. By introducing a reference task, we can determine that the divergence of adapters fine-tuned on the reference task in both languages follows the same distribution as the divergence of adapters fine-tuned on the target task in both languages. Hence, we can obtain target adapters by combining the other three adapters. Furthermore, we propose a structureadaptive adapter merging method. Our empirical results demonstrate that our approach yields new and effective cross-lingual transfer, outperforming existing methods across all settings

## Data

To construct the training data for the reference task, i.e., casual language modeling, you can run `construct_dataset_lm.py` 

## Installation

The environment can be installed by running the following command at the root of this repository:

```
conda env create -f environment.yml
```

## Data

