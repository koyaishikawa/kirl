import argparse
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
import gym

import kirl
from kirl.memory import ReplayBuffer
from kirl.agent import DDQN


def main(args):
    if args.gpu > 0:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')

    env = gym.make('CartPole-v0')
    eval_env = gym.make('CartPole-v0')
    network = nn.Sequential(
        nn.Linear(env.observation_space.shape[0], 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, env.action_space.n)
    ).to(device)
    buffer = ReplayBuffer(args.buffer_size, args.batch_size, env.observation_space.shape, args.gpu)
    phi = lambda x: x
    agent = DDQN(env, eval_env, network, buffer, start_update=args.start_update, target_update = args.target_update, eps=args.eps, phi=phi)

    for i in tqdm(range(args.epoch)):
        obs = env.reset()
        for _ in range(200):
            action = agent.act(obs)
            n_obs, reward, done, _ = env.step(action)
            agent.observe(obs, action, reward, done, n_obs)
            if done:
                break
            obs = n_obs
        if i % args.eval_steps == 0:
            R = agent.evaluation()
            tqdm.write(f"episode {i} : R {R}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-buffer-size", default=10 ** 4)
    parser.add_argument("-batch-size", default=32)
    parser.add_argument("-gpu", default=0)

    parser.add_argument("-start-update", default=500)
    parser.add_argument("-target-update", default=100)
    parser.add_argument("-eps", default=0.5)

    parser.add_argument("-epoch", default=400)
    parser.add_argument("-eval-steps", default=10)


    args = parser.parse_args()
    main(args)
