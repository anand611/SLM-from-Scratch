"""## Step 3: Batch Generation

In this step, we will do the following:

(1) Create Input-Output batches for 'train/validation' dataset 

"""

import numpy as np
import torch


def get_batch(split,block_size,batch_size,device,device_type):
    """
    Return a random batch of input/target sequence pairs sampled from a memory-mapped binary dataset.
    Parameters
    ----------
    split : str
        Dataset split to sample from. Expected values: "train" or "validation".
        This selects the file "train.bin" or "validation.bin" respectively.
    block_size : int
        Length of each sequence (number of tokens) in the returned x and y tensors.
    batch_size : int
        Number of sequences to include in the batch.
    device : torch.device or str
        Destination device for the returned tensors (e.g., torch.device('cuda:0') or 'cpu').
    device_type : str
        Device type string used to decide transfer strategy. If equal to 'cuda', tensors are
        pinned in CPU memory and transferred with non_blocking=True to enable asynchronous copies.
        Otherwise tensors are transferred synchronously via .to(device).
    
    Returns
    -------
    tuple(torch.LongTensor, torch.LongTensor)
        A tuple (x, y) where:
        - x has shape (batch_size, block_size) and contains the input token ids.
        - y has shape (batch_size, block_size) and contains the target token ids (shifted by +1).
        Both tensors have dtype torch.int64.
    
    Behavior / Implementation details
    -------------------------------
    - The function opens the chosen binary file with numpy.memmap(dtype=np.uint16, mode='r')
      on every call to avoid memmap-related memory leaks.
    - It samples `batch_size` random start indices uniformly from the valid range:
      [0, len(data) - block_size).
    - For each start index i:
      - x row = data[i : i + block_size]
      - y row = data[i+1 : i+1 + block_size]
      These slices are cast to int64 numpy arrays and converted to torch tensors, then stacked.
    - If device_type == 'cuda', the CPU tensors are pinned and transferred to `device` with
      non_blocking=True. Otherwise, tensors are moved to `device` synchronously.
    
    Notes and potential errors
    --------------------------
    - Requires the files "train.bin" and "validation.bin" to exist and contain token ids stored
      as uint16 values.
    - If len(data) <= block_size, sampling will fail (invalid range) and an exception will be raised.
    - Missing or corrupted binary files will raise an IOError/ValueError when attempting to memmap.
    - Returned tensors are appropriate for GPU training when device_type == 'cuda' due to pinned
      memory and non-blocking transfers.
    """
    # We recreate np.memmap every time to avoid a memory leak
    if split=="train":
        data = np.memmap('train.bin',dtype=np.uint16,mode='r')
    else:
        data = np.memmap('validation.bin',dtype=np.uint16,mode='r')
    
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64))for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64))for i in ix])
    if device_type == 'cuda':
        x = x.pin_memory().to(device,non_blocking=True)
        y = y.pin_memory().to(device,non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x,y