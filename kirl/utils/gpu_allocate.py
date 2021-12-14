import torch


def gpu_allocate(gpu:int):
    if gpu < 0:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{gpu}')    
    return device
 