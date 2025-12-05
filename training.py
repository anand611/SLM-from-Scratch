
"""## Step 5: Model Training and Validation

In this step, we will do the following:

(1) Define Model training parameters and then perform model training.

(2) Showing the model training and validation loss on graph

(3) The best model's parameters are saved in current directory with name 'best_model_params.pt'

"""

import torch
from contextlib import nullcontext
from torch.optim.lr_scheduler import LinearLR,SequentialLR,CosineAnnealingLR
from model import GPT, GPTConfig
from batch import get_batch
from tqdm.auto import tqdm
import utils
import gpt_configs
import training_configs

# function to estimate loss on train and validation sets
def estimate_loss(model):
    out = {}
    model.eval()
    with torch.inference_mode():
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split,block_size=block_size,batch_size=batch_size,device=device,device_type=device_type)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out


# model configuration
config = GPTConfig(
    vocab_size=gpt_configs.VOCAB_SIZE,
    block_size=gpt_configs.BLOCK_SIZE,
    n_layer = gpt_configs.N_LAYER,
    n_head = gpt_configs.N_HEAD,
    n_embd = gpt_configs.N_EMBD,
    dropout = gpt_configs.DROPOUT,
    bias=gpt_configs.BIAS
)

learning_rate = training_configs.LEARNING_RATE # more stable training, earlier 1e-4
max_iters = training_configs.MAX_ITERS # increased from 10k to 20k
warmup_steps = training_configs.WARMUP_STEPS # smoother initial train
min_lr = training_configs.MIN_LR
eval_iters = training_configs.EVAL_ITERS
batch_size = training_configs.BATCH_SIZE
block_size = training_configs.BLOCK_SIZE
gradient_accumulation_steps = training_configs.GRADIENT_ACCUMULATION_STEPS # to simulate larger batch size

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast

dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32,'bfloat16': torch.bfloat16,'float16': torch.float16}[dtype]

ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

torch.set_default_device(device)
torch.manual_seed(42)

# create the model
model = GPT(config)

### PUT in weight decay, changed BETA2 to 0.95
optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate,betas=(0.9,0.95),weight_decay=0.1,eps = 1e-9) # weight decay added for regularization
# Implement linear warmup
scheduler_warmup = LinearLR(optimizer=optimizer,total_iters=warmup_steps)
# Implement cosine decay
scheduler_decay = CosineAnnealingLR(optimizer=optimizer,T_max = max_iters - warmup_steps, eta_min = min_lr)
# Switch from warmup to decay
scheduler = SequentialLR(optimizer=optimizer,schedulers=[scheduler_warmup,scheduler_decay],milestones=[warmup_steps])
# Scaler for mixed precision training
scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))

# Pretrain the SLM
best_val_loss = float('inf')
best_model_params_path = 'best_model_params.pt'
train_loss_list,validation_loss_list = [],[]

# Ensure model is on the right device
model = model.to(device)

# Training loop
for epoch in tqdm(range(max_iters)):
    if epoch % eval_iters == 0 and epoch != 0:
        # Ensure estimate_loss uses the correct device
        losses = estimate_loss(model)
        print(f"\nEpoch {epoch}: train loss {losses['train']:.4f}, val_loss {losses['val']:.4f}")
        print(f"The current learning rate: {optimizer.param_groups[0]['lr']:.5f}")
        train_loss_list += [losses['train']]
        validation_loss_list += [losses['val']]
       
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save(model.state_dict(),best_model_params_path)
            
    # Ensure X and y are on the correct device
    X, y = get_batch('train',block_size=block_size,batch_size=batch_size,device=device,device_type=device_type)
    X, y = X.to(device),y.to(device)
    
    with ctx:
        logits, loss = model(X,y)
        loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()
        
    if ((epoch + 1) % gradient_accumulation_steps == 0) or (epoch + 1 == max_iters):
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    scheduler.step()

utils.plot_loss(train_loss_list,validation_loss_list)  

