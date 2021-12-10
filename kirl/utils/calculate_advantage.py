import torch


def calculate_advantage(values, rewards, dones, next_values, gamma=0.995, lambd=0.997):

    deltas = rewards + gamma * next_values * (1 - dones) - values
    advantages = torch.empty_like(rewards)
    advantages[-1] = deltas[-1]

    for t in reversed(range(rewards.size(0) - 1)):
        advantages[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * advantages[t + 1]

    targets = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return targets, advantages