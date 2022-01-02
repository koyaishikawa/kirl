import copy

import torch
import numpy as np

from kirl.agent.base import Base
from kirl.utils.calculate_advantage import calculate_advantage


class RolloutBuffer:

    def __init__(self, buffer_size, parallel_env_num, state_shape, device=torch.device('cuda')):

        if parallel_env_num == None:
            self.states = torch.empty((buffer_size, *state_shape), dtype=torch.float, device=device)
            self.actions = torch.empty((buffer_size, 1), dtype=torch.int64, device=device)
            self.rewards = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
            self.dones = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
            self.next_states = torch.empty((buffer_size, *state_shape), dtype=torch.float, device=device)
   
        else:
            self.states = torch.empty((buffer_size, parallel_env_num, *state_shape), dtype=torch.float, device=device)
            self.actions = torch.empty((buffer_size, parallel_env_num, 1), dtype=torch.int64, device=device)
            self.rewards = torch.empty((buffer_size, parallel_env_num, 1), dtype=torch.float, device=device)
            self.dones = torch.empty((buffer_size, parallel_env_num, 1), dtype=torch.float, device=device)
            self.next_states = torch.empty((buffer_size, parallel_env_num, *state_shape), dtype=torch.float, device=device)
     
        # 次にデータを挿入するインデックス．
        self._p = 0
        # バッファのサイズ．
        self.buffer_size = buffer_size

    def append(self, state, action, reward, done, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p]= int(action)
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))
        self._p = (self._p + 1) % self.buffer_size
    
    def get(self):
        assert self._p == 0, 'Buffer needs to be full before training.'
        return self.states, self.actions, self.rewards, self.dones, self.next_states

    def check_buffer(self):
        return self._p == 0



class PPO(Base):
    def __init__(self,
                 env,
                 eval_env,
                 network,
                 buffer_size,
                 gpu,
                 num_updates,
                 batch_size,
                 coef_cri=1.0,
                 coef_ent=0.01,
                 clip_eps=0.1,
                 gamma=0.99,
                 lambd=0.95,
                 lr=0.001,
                 max_grad_norm=0.5,
                 parallel_env_num=1,
                 phi=lambda x:x
                 ):

        super().__init__(env, eval_env, network, phi, gpu)
        obs = env.reset()
        self.buffer = RolloutBuffer(buffer_size, parallel_env_num, phi(obs).shape, device=self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.gamma = gamma
        self.lambd = lambd
        self.num_updates = num_updates
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.clip_eps = clip_eps
        self.coef_cri = coef_cri
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm
        self.parallel_env_num = parallel_env_num
        
    def observe(self, state, action, reward, done, next_state):
        self.add_memory(state, action, reward, done, next_state)
        if self.do_update_traning():
            self.update()

    def act(self, state, eval=False):
        # TODO batch_act()の実装 並行化は想定していない
        torch_state = self._as_tensor(state)
        if self.parallel_env_num == 1:
            torch_state = torch_state.unsqueeze(0)
        with torch.no_grad():
            action_dist, _ = self.network(torch_state)

        if eval:
            action = action_dist.max(1)[1].cpu().numpy()
        else:
            dist = torch.distributions.Categorical(action_dist)
            action = dist.sample().cpu().numpy()

        return action

    def add_memory(self, state, action, reward, done, next_state):
        self.buffer.append(state, action, reward, done, next_state)

    def do_update_traning(self):
        return self.buffer.check_buffer()

    def update(self):
        # buffer.get() -> Tensor
        state, action, reward, done, n_state = self.buffer.get()

        with torch.no_grad():
            action_dist, values = self.network(state)
            _, next_values = self.network(n_state)

        old_pis, target, advantages = torch.empty_like(reward), torch.empty_like(reward), torch.empty_like(reward)
        for i in range(self.parallel_env_num):
            old_pis[:, i] = action_dist[:, i].gather(1, action[:, i])
            target[:, i], advantages[:, i] = calculate_advantage(values[:, i], reward[:, i], done[:, i], next_values[:, i], self.gamma, self.lambd)     
        target = target.view(self.buffer_size* self.parallel_env_num, -1)
        advantages = advantages.view(self.buffer_size* self.parallel_env_num, -1)
        old_pis = old_pis.view(self.buffer_size* self.parallel_env_num, -1)
        state = state.view(self.buffer_size* self.parallel_env_num, *state.shape[2:])
        action = action.view(self.buffer_size* self.parallel_env_num, *action.shape[2:])

        # TODO バッファに格納した全てのデータを一回は使うべき?
        index_num = self.buffer_size * self.parallel_env_num
        for _ in range(self.num_updates):
            indices = np.arange(index_num)
            np.random.shuffle(indices)
            for start in range(0, index_num, self.batch_size):
                idxes = indices[start:start+self.batch_size]
                self.update_network(state[idxes], action[idxes], old_pis[idxes], target[idxes], advantages[idxes])

    def update_network(self, state, action, old_pis, target, advantages):
        action_dist, value = self.network(state)
        loss_critic = 0.5 * ((value - target) ** 2).mean()
        pis = action_dist.gather(1, action)
        mean_entropy = -pis.mean()

        ratios = (pis - old_pis).exp_()
        loss_actor1 = -ratios * advantages
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * advantages
        loss = torch.max(loss_actor1, loss_actor2).mean() + self.coef_cri * loss_critic - self.coef_ent * mean_entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
