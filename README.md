# SLM-from-Scratch
Let us build a Small Language Model (SLM) from scratch. We will try to keep the parameter size to 50-60 million.  Our goal is to generate creative and coherent text based on the input data.

## How to use?

This repository contains files related to generate, train and use SLM based on the "TinyStories" dataset. The complete process of Model creation is divided in following steps.

### Step 1: Get the dataset
* Get the ___'TinyStories'___ dataset from the `datasets` module. 
> To perform this action, run the `load_dataset.py`.

### Step 2: Tokenization
* We define the BPE tokenization encoding for model gpt2 from the `tiktoken` module. This is used to tokenize dataset into tokenIDs.
* We are generating the tokenized train and validation datasets and storing them on disk (___train.bin___ , ___validation.bin___). To use the optimal way of writing data, *sharding* is used.

* Get the ___'TinyStories'___ dataset from the `datasets` module. To perform this action, run the `load_dataset.py`.
