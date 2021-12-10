import argparse
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
import gym

import kirl
from kirl.agent import PPO


def main(args):
    if args.gpu > 0:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    env = gym.make('CartPole-v0')
    eval_env = gym.make('CartPole-v0')

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Sequential(
                nn.Linear(env.observation_space.shape[0], 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU()
            )
            self.action = nn.Sequential(
                nn.Linear(64, env.action_space.n),
                nn.Sigmoid()
            )
            self.value = nn.Sequential(
                nn.Linear(64, 1)
            )

        def forward(self, x):
            output = self.base(x)
            action = self.action(output)
            value = self.value(output)
            return action, value

    network = Network().to(device)
    phi = lambda x: x
    agent = PPO(env,
                eval_env,
                network,
                buffer_size=args.buffer_size,
                num_updates=args.num_updates,
                batch_size=args.batch_size,
                gpu=args.gpu,
                phi=phi
                )

    for i in tqdm(range(args.epoch)):
        obs = env.reset()
        for _ in range(args.buffer_size):
            action = agent.act(obs)
            n_obs, reward, done, _ = env.step(action)
            agent.observe(obs, action, reward, done, n_obs)
            if done:
                obs = env.reset()
            obs = n_obs
        if i % args.eval_steps == 0:
            R = agent.evaluation()
            tqdm.write(f"episode {i} : R {R}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch-size", default=32)
    parser.add_argument("-gpu", default=0)

    parser.add_argument("-buffer-size", default=1024)
    parser.add_argument("-num-updates", default=32)

    parser.add_argument("-epoch", default=400)
    parser.add_argument("-eval-steps", default=10)


    args = parser.parse_args()
    main(args)
