
"""## Step 4: Define the SLM Model Architecture

In this step, we will do the following:

(1) Define Config class => 'GPTConfig'

(2) Define Layer Normalization class => 'LayerNorm'

(3) Define Self Attention class => 'CausalSelfAttention'

(4) Define Feed forward network class => 'MLP'

(5) Define Multi head attention block => 'Block'

(6) Define SLM model composed of above all classes => 'GPT'

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
import numpy as np
from contextlib import nullcontext
import os

@dataclass
class GPTConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = True

class LayerNorm(nn.Module):
    def __init__(self,ndim,bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    # define the forward pass
    def forward(self,x):
        return F.layer_norm(x,self.weight.shape,self.weight,self.bias,1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                       .view(1, 1, config.block_size, config.block_size))
    
    # define the forward pass
    def forward(self, x):
        B,T,C = x.size()
        q,k,v =self.c_attn(x).split(self.n_embd,dim=2)
        k = k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        q = q.view(B,T,self.n_head, C//self.n_head).transpose(1,2)
        v = v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)
        
        if self.flash:
            y = F.scaled_dot_product_attention(q,k,v,attn_mask=None,dropout_p=self.attn_dropout.p if self.training else 0.0,is_causal=True)
        else:
            att = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(k.size(-1))) # @: matrix multiplication
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0,float('-inf'))
            att = F.softmax(att,dim=-1)
            att = self.att_dropout(att)
            y = att @ v
        
        y = y.transpose(1,2).contiguous().view(B,T,C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd,bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd,config.n_embd,bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
    # define the forward pass
    def forward(self,x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))
    
class Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd,config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2 = LayerNorm(config.n_embd,config.bias)
        self.mlp = MLP(config)
        
    def forward(self,x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    

class GPT(nn.Module):
    def __init__(self,config: GPTConfig):
        """
        Initialize the GPT model components.
        Parameters
        ----------
        config : GPTConfig
            Configuration object containing the model hyperparameters required to
            construct the network. Expected attributes:
            - vocab_size (int): size of the token vocabulary.
            - n_embd (int): embedding dimensionality for tokens and positions.
            - block_size (int): maximum sequence length (used for position embeddings).
            - dropout (float): dropout probability applied to embeddings.
            - n_layer (int): number of transformer blocks to instantiate.
            - bias (bool): whether the final LayerNorm should include a bias term.
        What this initializer does
        --------------------------
        - Builds a ModuleDict named `transformer` with the following entries:
          - wte: token embedding layer (nn.Embedding).
          - wpe: position embedding layer (nn.Embedding).
          - drop: dropout layer (nn.Dropout).
          - h: a ModuleList of `n_layer` transformer Block instances.
          - ln_f: final LayerNorm layer.
        - Creates `lm_head`, a linear layer that maps hidden states to vocabulary
          logits (nn.Linear).
        Weight tying and initialization
        -------------------------------
        - Implements weight tying between input embeddings and output logits by making
          `transformer.wte.weight` reference `lm_head.weight`.
        - Calls `self.apply(self._init_weights)` to run a custom initialization routine
          over all submodules.
        - Applies a specialized initialization for projection weights whose parameter
          names end with 'c_proj.weight': they are initialized from a normal
          distribution with mean 0 and standard deviation 0.02 / sqrt(2 * n_layer).
        Notes
        -----
        - The initializer assumes the provided `config` object exposes the listed
          attributes; it does not perform explicit validation of `config` fields.
        - Weight tying reduces the total number of parameters and enforces consistency
          between input embeddings and output token logits.
        """
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size,config.n_embd),
                wpe =nn.Embedding(config.block_size,config.n_embd),
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = LayerNorm(config.n_embd,config.bias)
                )
            )
        self.lm_head = nn.Linear(config.n_embd,config.vocab_size,bias=False)
        self.transformer.wte.weight = self.lm_head.weight # weight tying
        
        self.apply(self._init_weights)
        
        for pn,p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p,mean=0.0,std=0.02/math.sqrt(2*config.n_layer))
    
    def _init_weights(self,module):
        """
        Initialize weights for a given submodule in-place.

        This function is intended to be passed to torch.nn.Module.apply to initialize parameters
        across a model.

        Behavior:
        - If module is an instance of torch.nn.Linear:
            - Initialize module.weight with a normal distribution (mean=0.0, std=0.02).
            - If module.bias is not None, initialize module.bias to zeros.
        - There is an additional branch intended to initialize torch.nn.Embedding weights
          with a normal distribution (mean=0.0, std=0.02). Note: in the provided implementation
          that branch is placed under the Linear->bias conditional and is therefore unreachable;
          it will not run as written.

        Parameters
        ----------
        module : torch.nn.Module
            The module whose parameters should be initialized.

        Returns
        -------
        None
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            elif isinstance(module,nn.Embedding):
                nn.init.normal_(module.weight,mean=0.0,std=0.02)
    
    # define the forward pass
    def forward(self,idx,targets=None):
        """
        Forward pass of the language model.
        Args:
            idx (torch.LongTensor): Input token indices of shape (B, T). Values are token ids and computation is performed on idx.device.
            targets (torch.LongTensor or None): Optional target token indices of shape (B, T). Positions with the value -1 are ignored when computing loss. If None, no loss is computed and only next-token logits are returned.
        Returns:
            tuple:
                logits (torch.FloatTensor): If targets is provided, logits for all input positions with shape (B, T, V). If targets is None, logits for the last time step with shape (B, 1, V).
                loss (torch.FloatTensor or None): If targets is provided, a scalar cross-entropy loss (computed with torch.nn.functional.cross_entropy and ignore_index=-1). Otherwise None.
        Raises:
            AssertionError: If the input sequence length T exceeds self.config.block_size.
        Behavior:
            - Embeds tokens and positions, sums them, applies dropout, passes through transformer blocks and final layer norm, then projects hidden states to vocabulary logits with self.lm_head.
            - Uses the device of idx for all internal tensors (e.g., positional indices).
        """
        device = idx.device
        b,t = idx.size()
        assert t <= self.config.block_size, "Cannot forward, model block size is exhausted"
        pos = torch.arange(0,t,dtype=torch.long,device=device)
        
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)),targets.view(-1),ignore_index=-1)
            return logits, loss
        else:
            logits = self.lm_head(x[:,[-1],:])
            return logits, None
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0,top_k=None):
        """
        Generate new tokens autoregressively from the model given an initial context.

        Parameters
        ----------
        idx : torch.LongTensor
            Input token indices tensor of shape (B, T) representing the batch of context sequences.
        max_new_tokens : int
            The number of new tokens to generate and append to each sequence in the batch.
        temperature : float, optional
            Sampling temperature used to scale logits before softmax. Values > 1.0 increase randomness,
            values < 1.0 make sampling more greedy. Default: 1.0.
        top_k : int or None, optional
            If provided, restrict sampling at each step to the top_k highest-probability tokens.
            If None, no top-k filtering is applied. Default: None.

        Returns
        -------
        torch.LongTensor
            A tensor of token indices with the generated tokens appended, shape (B, T + max_new_tokens).

        Notes
        -----
        - If the current context length exceeds the model's configured block_size, the context is
          cropped on the left so the final block fed to the model has length <= block_size.
        - At each generation step the model's logits for the next token are temperature-scaled,
          optionally filtered by top-k (tokens below the top-k threshold set to -inf), converted
          to probabilities via softmax, and sampled with torch.multinomial.
        - Sampling is nondeterministic unless a random seed is set externally (e.g., torch.manual_seed).
        """
        """Generate new tokens from the model given a context idx: (B,T)"""
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:,-self.config.block_size:]
            logits,_ = self(idx_cond)
            logits = logits[:,-1,:] / temperature
            if top_k is not None:
                v,_ = torch.topk(logits,min(top_k,logits.size(-1)))
                logits[logits < v[:,[-1]]] = -float('Inf')
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs,num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
        return idx
            
            
        