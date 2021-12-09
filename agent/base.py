from copy import Error
import torch
from torch import nn
import numpy as np


class Base:   
    def __init__(self, env, eval_env, network, phi):
        self.network = network
        self.env = env
        self.eval_env = eval_env
        self.phi = phi

    def observe(self):
        raise NotImplementedError()
        
    def update(self):
        raise NotImplementedError()

    def act(self):
        raise NotImplementedError()

    def evaluation(self):
        raise NotImplementedError()

    def _batch_phi(self, state):
        assert len(state.shape) > 1, 'state do not have batch dim.'
        if isinstance(state, torch.Tensor):
            return torch.stack([self.phi(s) for s in state])
        elif isinstance(state, np.ndarray):
            return np.array([self.phi(s) for s in state])
        else:
            raise Error()
