from model import GPT, GPTConfig
import torch
from tokenization import ENC
import gpt_configs

def load_model(config):
    """
    Load and return a GPT model instantiated with the provided configuration and populated
    with parameters from the on-disk checkpoint 'best_model_params.pt'.
    
    Parameters
    ----------
    config : object
        Configuration object or mapping used to construct the GPT model (passed to GPT(config)).
        The exact expected structure depends on the GPT implementation.
    
    Returns
    -------
    GPT
        An instance of GPT constructed with `config` and whose parameters have been
        loaded from 'best_model_params.pt'. The checkpoint tensors are mapped to the
        device selected at runtime ('cuda' if available, otherwise 'cpu').
    
    Raises
    ------
    FileNotFoundError
        If the checkpoint file 'best_model_params.pt' cannot be found.
    RuntimeError
        If the checkpoint is incompatible with the model architecture or if loading fails.
    ImportError
        If required libraries (e.g. torch) or the GPT class are not available.
    
    Notes
    -----
    - The function chooses the device via torch.cuda.is_available() and passes the
      corresponding map_location to torch.load. It does not modify or return the device
      explicitly.
    - Ensure the working directory or provided path contains 'best_model_params.pt',
      or modify the function to accept a checkpoint path if needed.
    """
    # Recreate model with same config
    best_model = GPT(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_model_params_path = 'best_model_params.pt'

    # Load best model configs
    best_model.load_state_dict(torch.load(best_model_params_path,map_location=torch.device(device=device)))
    return best_model


# Defining model configuration
config = GPTConfig(
    vocab_size=gpt_configs.VOCAB_SIZE,
    block_size=gpt_configs.BLOCK_SIZE,
    n_layer = gpt_configs.N_LAYER,
    n_head = gpt_configs.N_HEAD,
    n_embd = gpt_configs.N_EMBD,
    dropout = gpt_configs.DROPOUT,
    bias=gpt_configs.BIAS
)
# Load the model with configurations
model = load_model(config)
# Input string by user
sentence = "Once upon a time there was a pumpkin"
# Generating the context from the user input text
context = (torch.tensor(ENC.encode_ordinary(sentence)).unsqueeze(dim=0))
# Executing model and generating the context friendly results.
y = model.generate(context, 200)
# print the output
print(ENC.decode(y.squeeze().tolist()))