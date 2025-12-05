## Model Training configurations

LEARNING_RATE = 1e-4 # more stable training, earlier 1e-4
MAX_ITERS = 20000 # increased from 10k to 20k
WARMUP_STEPS = 1000 # smoother initial train
MIN_LR = 5e-4
EVAL_ITERS = 500
BATCH_SIZE = 32
BLOCK_SIZE = 128
GRADIENT_ACCUMULATION_STEPS = 2