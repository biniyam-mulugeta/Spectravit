import random
import os
import numpy as np
import torch

def seed_everything(seed: int = 42):
    """
    Sets the seed for generating random numbers to allow for reproducible runs.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"[Info] Seeding complete. Seed: {seed} (Deterministic algorithms enabled).")
