import gym
import numpy as np


class SingleEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def step(self, action):
        action = action[0]
        obs, reward, done, info = self.env.step(action)

        return np.array(obs), np.array(reward), np.array(done), info
