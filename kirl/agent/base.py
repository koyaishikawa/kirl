from copy import Error

import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np


class Base:   
    def __init__(self, env, eval_env, network, phi, gpu):
        self.network = network
        self.env = env
        self.eval_env = eval_env
        self.phi = phi
        if gpu > 0:
            self.device = torch.device(f'cuda:{gpu}')
        else:
            self.device = torch.device('cpu')


    def observe(self):
        raise NotImplementedError()
        
    def update(self):
        raise NotImplementedError()

    def act(self):
        raise NotImplementedError()

    def evaluation(self):  
        self.network.eval()
        R = self._do_one_episode(eval=True)        
        self.network.train()
        return R

    def load_network(self, file_name):
        self.network.load_state_dict(torch.load(file_name))

    def save_network(self, file_name):
        torch.save(self.network.to('cpu').state_dict(), file_name)

    def _do_one_episode(self, eval=False):
        R = 0
        if eval:
            env = self.eval_env
        else:
            env = self.env
        obs = env.reset()
        while True:
            action = self.act(obs, eval)
            obs, reward, done, _ = env.step(action)
            R += reward
            if done:
                break
        return R

    def _batch_phi(self, state):
        assert len(state.shape) > 1, 'state do not have batch dim.'
        if isinstance(state, torch.Tensor):
            return torch.stack([self.phi(s) for s in state])
        elif isinstance(state, np.ndarray):
            return np.array([self.phi(s) for s in state])
        else:
            raise Error()
