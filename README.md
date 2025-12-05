# SLM-from-Scratch
Let us build a Small Language Model (SLM) from scratch. We will try to keep the parameter size to 50-60 million.  Our goal is to generate creative and coherent text based on the input data.

## How to Create Model?

This repository contains files related to generate, train and use SLM based on the "TinyStories" dataset. The complete process of Model creation is divided in following steps.

### Step 1: Get the dataset
* Get the ___'TinyStories'___ dataset from the `datasets` module. 
    >
    > To perform this action, run `load_dataset.py`.
    >

### Step 2: Tokenization
* We define __gpt2__ sub-word tokenizer with BPE(Byte Pair Encoding) based encoding from the `tiktoken` module. This is used to tokenize dataset into tokenIDs.
* We are generating the tokenized train and validation datasets and storing them on disk (___train.bin___ , ___validation.bin___). To use the optimal way of writing data, *sharding* is used.
    >
    > To perform this action, run the `tokenization.py`.
    >
* After successful tokenization, ___train.bin___ , ___validation.bin___ are generated. 

### Step 3: Batch Retrieval
* We define the function to obtain the batch of data from the dataset. This function loads the _train/validation_ dataset in memory as per argument 'split'.

### Step 4: GPT Model Architecture
* In this step, we are defining the architecture of the SLM Model.
* Transformer architecture starts after Token Embedding and then Position Embedding retrieval.
* Each Transformer `Block` is having Layer Normalization (`LayerNorm`) at entry point. So inputs (embeddings) are normalized 
* After Layer Normalization, Self Attention (`CausalSelfAttention`) layer is added.
* Result are then Normalized again using Layer Normalization (`LayerNorm`)
* Output of the Normalized results are feed to the 2 layered MLP with dropout. Activation function `GELU()` is used.
* Finally, It calculates the logits and then computes the ___cross_entropy___ loss between the model's predictions and the actual next tokens (targets). This loss value is what's used to update the model's weights during training. 
* To execute model with user query and context, `generate()` method is provided.
    >
    > To create the model, run `model.py`.
    >

### Step 5: Model Training
* In this step, we are creating the Model Training. We are defining training parameters, the Hyperparameters Tuning schedules and loss measurements. Here `LinearLR` schedular is used in warmup epochs and in later stage `CosineAnnealingLR` scheduler is used to update model parameters. For weight update, `AdamW` optimizer is used to make the training more effective by allowing weight decay and learning rate adjustments independently.
* After successful completion of Training, we are able to see the training and validation losses in graphically. Also the model 
* We are saving the best model's parameters in file ___'best_model_params.pt'___. This can be used to test the model with user inputs.
    >
    > To perform model training, run the `training.py`.
    >

## How to use the Model?
* To run the test the created model, 
  1. Open `model_usage.py`
  2. Update *_sentence_* parameter with your topic of interest. 
  3. Save and run `model_usage.py` 
