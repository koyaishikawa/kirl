import pickle

import torch
import numpy as np


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def gpu_allocate(gpu:int):
    if gpu < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{gpu}')    
    return device
 