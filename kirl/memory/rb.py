import torch 
import numpy as np

class ReplayBuffer:

    def __init__(self, buffer_size, batch_size, state_shape, device:int):
        if device > 0:
            self.device = torch.device(f'cuda:{device}')
        else:
            self.device = torch.device('cpu')
        # GPU上に保存するデータ．
        self.states = torch.empty((buffer_size, *state_shape), dtype=torch.float, device=self.device)
        self.actions = torch.empty((buffer_size, 1), dtype=torch.int64, device=self.device)
        self.rewards = torch.empty((buffer_size, 1), dtype=torch.float, device=self.device)
        self.dones = torch.empty((buffer_size, 1), dtype=torch.bool, device=self.device)
        self.next_states = torch.empty((buffer_size, *state_shape), dtype=torch.float, device=self.device)

        # 次にデータを挿入するインデックス．
        self._p = 0
        # バッファのサイズ．
        self.buffer_size = buffer_size
        self.is_full = False
        self.batch_size = batch_size

    def append(self, state, action, reward, done, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p] = int(action)
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))
        self._p = (self._p + 1) % self.buffer_size
        if self._p == self.buffer_size - 1:
            self.is_full = True
    
    def get(self):
        if self.is_full:
            indices = np.arange(self.batch_size)
        else:
            indices = np.arange(self._p)

        np.random.shuffle(indices)
        batch_index = indices[:self.batch_size]
        return self.states[batch_index], self.actions[batch_index], self.rewards[batch_index], self.dones[batch_index], self.next_states[batch_index]