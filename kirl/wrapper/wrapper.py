import gym


class SingleEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def step(self, action):
        action = action[0]
        return self.env.step(action)
