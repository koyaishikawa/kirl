import argparse
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import gym

import kirl
from kirl.agent import PPO
from kirl.utils import train_agent_with_evaluation
from kirl.utils import gpu_allocate



def main(args):
    device = gpu_allocate(args.gpu)
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
                gamma=args.gamma,
                lambd=args.lambd,
                lr=args.lr,
                phi=phi
                )

    save_model_dir = 'data_path/ppo/'

    train_agent_with_evaluation(
        env,
        agent,
        args.num_steps, 
        args.eval_steps,
        args.save_steps, 
        save_model_dir,
        'ppo',
        args.window
    )

    """
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
            tqdm.write(f"episode {i} : R {R}")

        if i % args.save_steps == 0:
            agent.save_network(save_data_path + f'ppo_{i}.pth')
    
    mean_R_list = np.convolve(R_list, np.ones(args.window)/args.window, 'valid')
    plt.plot(R_list, label='raw', alpha=0.6)
    plt.plot(mean_R_list, label='mean')
    plt.legend()
    plt.savefig('ppo.png')
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # base
    parser.add_argument("-batch-size", default=32)
    parser.add_argument("-gpu", default=0)

    # model
    parser.add_argument("-buffer-size", default=1024)
    parser.add_argument("-num-updates", default=8)
    parser.add_argument("-gamma", default=0.99)
    parser.add_argument("-lambd", default=0.95)
    parser.add_argument("-lr", default=0.001)


    # train
    parser.add_argument("-num-steps", default=100000)
    parser.add_argument("-eval-steps", default=100)
    parser.add_argument("-save-steps", default=1000)

    # for performance
    parser.add_argument("-window", default=5)

    args = parser.parse_args()
    main(args)
