from enum import IntEnum

import gym
from gym.spaces import Discrete, Box
import numpy as np


class Action(IntEnum):
    SHORT = 0
    NOOP = 1
    LONG = 2


class FinanceEnv(gym.Env):
    def __init__(self, input_data, output_data, cost=0.0):
        self.input_data = input_data
        self.output_data = output_data
        self.data_length = input_data.shape[0]
        self.cost = cost

        _obs = self._reset()
        self.action_space = Discrete(3)
        self.observation_space = Box(input_data.min(), input_data.max(), _obs.shape)

    def __len__(self):
        return self.input_data.shape[0]
        
    def step(self, action):
        self.reward = self.get_reward(action)
        self.done = self.get_done()

        self._t += 1
        self.obs = self._append_action()
        self.info = ""
        self.prev_action = action
        return self.obs, self.reward, self.done, self.info

    def reset(self):
        self._t = 0
        self.prev_action = Action.NOOP
        self.obs = self._append_action()
        self.total_diff = 0
        return self.obs

    def get_reward(self, action):
        if self.prev_action - action == 0:
            # noop
            if action == Action.NOOP:
                reward = 0.0
            
            # keep trading
            else:
                self.total_diff += self.output_data[self._t].item()
                reward = 0.0
                
        else: 
            # finish trading
            if action == Action.NOOP:
                reward = self.total_diff * (self.prev_action - 1)
                self.total_diff = 0
            
            # start trading
            else:
                reward = abs(action - self.prev_action) * self.cost * (-1)

                if self.total_diff == 0:
                    self.total_diff += self.output_data[self._t].item()
                else:
                    reward += self.total_diff * (self.prev_action - 1)
                    self.total_diff = self.output_data[self._t].item()
        
        return reward   

    def get_done(self):
        return not (self.data_length - self._t > 2)

    def _append_action(self):
        return np.append(self.input_data[self._t].flatten(), self.prev_action / 2)
    
    def _reset(self):
        self._t = 0
        self.prev_action = Action.NOOP
        self.obs = self._append_action()
        return self.obs