import copy

import torch
import numpy as np

from kirl.agent.base import Base


class DDQN(Base):
    def __init__(self, env, eval_env, network, buffer, start_update, target_update, eps, gpu=-1, gamma=0.99, lr=0.001, phi=lambda x:x):
        super().__init__(env, eval_env, network, phi, gpu)
        self.buffer = buffer
        self.target_network = copy.deepcopy(network)
        self.start_update = start_update
        self.action_dim = env.action_space.shape
        self._t = 0
        self.gamma = gamma
        self.eps = eps
        self.target_update = target_update

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        
    
    def observe(self, state, action, reward, done, next_state):   
        self._t += 1     
        self.add_memory(state, action, reward, done, next_state)
        if self.do_update_traning():
            self.update()
        if self.do_update_target_network():
            self.update_target_network()

    def act(self, state, eval=False):
        if eval:
            with torch.no_grad():
                torch_state = torch.from_numpy(self.phi(state)).float().unsqueeze(0)
                action_dist = self.network(torch_state)
                action = action_dist.max(1)[1].item()
        else:
            if self.random_action():
                action = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    torch_state = torch.from_numpy(self.phi(state)).float().unsqueeze(0)
                    action_dist = self.network(torch_state)
                    action = action_dist.max(1)[1].item()
        return action
            
    def random_action(self):
        return self.eps < np.random.random()
    
    def add_memory(self, state, action, reward, done, next_state):
        self.buffer.append(state, action, reward, done, next_state)

    def do_update_traning(self):
        return (self._t > self.start_update)

    def update(self):
        # buffer.get() -> Tensor
        state, action, reward, done, n_state = self.buffer.get()
        max_value = self.network(self._batch_phi(state)).gather(1, action)

        with torch.no_grad():
            next_action_indices = self.network(self._batch_phi(n_state)).max(1)[1]
            next_max_value = self.target_network(self._batch_phi(n_state)).gather(1, next_action_indices.unsqueeze(1))
            target = next_max_value * self.gamma * (1 - done.long()) + reward
        
        loss = ((target - max_value) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def do_update_target_network(self):
        return (self._t % self.target_update == 0)

    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())