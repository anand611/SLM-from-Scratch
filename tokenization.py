"""## Step 2: Tokenize the Dataset

In this step, we will do the following:

(1) Tokenize the dataset into tokenIDs.

(2) Create a file called "train.bin" and "validtion.bin" where we will store the tokenIDs from the entire dataset.

(3) We make sure the tokenIDs are stored on a disk, rather than on the RAM for efficient computations.

"""

import os
import numpy as np
from tqdm.auto import tqdm
import multiprocessing as mp
import tiktoken

# IMPORTANT: Don't import heavy globals at module import time on Windows.
# If your dataset object `ds` is created in another module, import it inside main().
# from load_dataset import ds  # ← move inside main()

# Top-level (picklable) tokenizer object is OK on Windows, because child processes re-import the module.
ENC = tiktoken.get_encoding("gpt2")  # open-source GPT-2 BPE (uint16 range OK)

def process(example):
    """Map function: turns a row with 'text' into token ids + length."""
    ids = ENC.encode_ordinary(example["text"])
    return {"ids": ids, "len": len(ids)}

def write_split_memmap(split_name, dset, dtype=np.uint16, total_batches=1024):
    """
    Concatenate variable-length 'ids' from the dataset into one memmap file.
    Handles small datasets and empty shards gracefully.
    """
    # Compute total target length (sum of token lengths)
    # dset['len'] returns a list; use uint64 accumulator to avoid overflow
    arr_len = np.sum(dset["len"], dtype=np.uint64)
    filename = f"{split_name}.bin"
    print(f"Creating {filename} with total length {arr_len}")

    if arr_len == 0:
        # No data to write; create an empty file
        open(filename, "wb").close()
        print(f"{split_name} has zero tokens. Wrote empty {filename}.")
        return

    # Allocate memmap
    arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
    # Choose number of shards: no more than dataset size to avoid empty shards
    num_rows = len(dset)
    num_shards = min(total_batches, max(1, num_rows))

    idx = 0
    for batch_idx in tqdm(range(num_shards), desc=f"Writing {filename}"):
        # contiguous=True keeps order; shard returns a Dataset view
        batch = dset.shard(num_shards=num_shards, index=batch_idx, contiguous=True).with_format("numpy")

        # Extract column; may be empty if tiny dataset
        ids_lists = batch["ids"] if len(batch) > 0 else []
        if len(ids_lists) == 0:
            continue

        # Concatenate variable-length arrays/lists into one 1-D array
        arr_batch = np.concatenate(ids_lists)
        # Safety: ensure dtype fits the token id range (GPT-2 BPE < 65536)
        if arr_batch.dtype != dtype:
            arr_batch = arr_batch.astype(dtype, copy=False)

        # Write slice
        end = idx + len(arr_batch)
        arr[idx:end] = arr_batch
        idx = end

    # Safety: ensure we filled exactly arr_len positions
    if idx != arr_len:
        print(f"Warning: expected to write {arr_len} tokens, actually wrote {idx}. "
              f"Check shard parameters or dataset integrity.")

    arr.flush()
    print(f"Finished writing {filename}")

def main():
    # Import/create dataset inside main (Windows spawn-safe)
    from load_dataset import ds  # ds should be a DatasetDict with splits like 'train', 'validation', etc.

    # Use correct remove_columns (strings, not datasets)
    remove_cols = ds["train"].column_names  # e.g., ['text']

    # Tokenize with multiprocessing
    # If you still hit Windows pickling issues, set num_proc=1.
    NUM_PROC = min(8, os.cpu_count() or 1)

    tokenized = ds.map(
        process,
        remove_columns=remove_cols,
        batched=False,                 # you can set True for speed; then adapt process to handle lists
        desc="Tokenizing splits",
        num_proc=NUM_PROC,
    )

    # Write each split to *.bin file (only if train.bin doesn't exist)
    if not os.path.exists("train.bin"):
        for split_name, dset in tokenized.items():
            write_split_memmap(split_name, dset, dtype=np.uint16, total_batches=1024)
    else:
        print("train.bin already exists; skipping write.")

if __name__ == "__main__":
    # REQUIRED on Windows: use spawn, and call code under main guard
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Start method may already be set; ignore
        pass

    # Optional for PyInstaller/exe packaging
    mp.freeze_support()
    main()
