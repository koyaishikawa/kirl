import argparse
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import gym

import kirl
from kirl.memory import ReplayBuffer
from kirl.agent import DDQN
from kirl.utils import gpu_allocate


def main(args): 
    device = gpu_allocate(args.gpu)
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
    agent = DDQN(env, eval_env, network, buffer, start_update=args.start_update, target_update = args.target_update, eps=args.eps, phi=phi, gpu=args.gpu,)

    save_data_path = 'data_path/ddqn/'

    R_list = []
    obs = env.reset()
    for i in tqdm(range(args.num_steps)):
        action = agent.act(obs)
        n_obs, reward, done, _ = env.step(action)
        agent.observe(obs, action, reward, done, n_obs)
        if done:
            obs = env.reset()
        else:
            obs = n_obs

        if i % args.eval_steps == 0:
            R = agent.evaluation()
            R_list.append(R)
            tqdm.write(f"steps {i} : R {R}")

        if i % args.save_steps == 0:
            agent.save_network(save_data_path + f'ddqn_{i}.pth')
    
    mean_R_list = np.convolve(R_list, np.ones(args.window)/args.window, 'valid')
    plt.plot(R_list, label='raw', alpha=0.6)
    plt.plot(mean_R_list, label='mean')
    plt.legend()
    plt.savefig('performance_with_ddqn.png')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # base
    parser.add_argument("-buffer-size", default=10 ** 4)
    parser.add_argument("-batch-size", default=32)
    parser.add_argument("-gpu", default=0)

    # model
    parser.add_argument("-start-update", default=500)
    parser.add_argument("-target-update", default=100)
    parser.add_argument("-eps", default=0.5)

    # train
    parser.add_argument("-num-steps", default=10000)
    parser.add_argument("-eval-steps", default=100)
    parser.add_argument("-save-steps", default=1000)

    # for performance
    parser.add_argument("-window", default=5)

    args = parser.parse_args()
    main(args)
