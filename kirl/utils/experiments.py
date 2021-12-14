from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

def train_agent_with_evaluation(env,
                                agent,
                                num_steps,
                                eval_steps,
                                save_steps,
                                save_model_dir, 
                                agent_name,
                                window=5
                                ):
    R_list = []
    obs = env.reset()
    for i in tqdm(range(num_steps)):
        action = agent.act(obs)
        n_obs, reward, done, _ = env.step(action)
        agent.observe(obs, action, reward, done, n_obs)
        if done:
            obs = env.reset()
        else:
            obs = n_obs

        if i % eval_steps == 0:
            R = agent.evaluation()
            R_list.append(R)
            tqdm.write(f"steps {i:8d} : R {R}")

        if i % save_steps == 0:
            agent.save_network(save_model_dir + f'{agent_name}_{i}.pth')
    
    mean_R_list = np.convolve(R_list, np.ones(window)/window, 'valid')
    plt.plot(R_list, label='raw', alpha=0.6)
    plt.plot(mean_R_list, label='mean')
    plt.legend()
    plt.savefig(f'performance_with_{agent_name}.png')