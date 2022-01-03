import matplotlib.pyplot as plt
import csv

import pandas as pd
import numpy as np


def trade_evaluation(env, agent, save_data_dir):
    obs = env.reset()

    trade_action = []
    trade_history = []
    prev_action = 1
    trade_length = 0
    total_profit = 0

    while True:
        action = agent.act(obs, eval=True)
        obs, reward, done, _ = env.step(action)
        total_profit += reward
        trade_history.append(total_profit.item())
        if abs(prev_action - action) == 2:
            trade_action.append([prev_action, trade_length, reward + env.cost])
            trade_length = 1
        elif abs(prev_action - action) == 1:
            if action == 1:
                trade_action.append([prev_action, trade_length, reward - env.cost])
                trade_length = 0
            else:
                trade_length = 1

        else:
            if action == 1:
                pass
            else:
                trade_length += 1

        prev_action = action
        if done:
            break
    
    print(trade_history)
    plt.plot(trade_history)
    plt.savefig(f'{save_data_dir}/trade_history.png')

    df_data = np.array(trade_action)
    df_action = pd.DataFrame(df_data, columns=['prev_action', 'trade_length', 'reward'])

    # buy sell 数
    # 平均トレード長
    # 平均収益　標準偏差
    buy_num = (df_action['prev_action'] == 2).sum()
    sell_num = (df_action['prev_action'] == 0).sum()
    mean_trade_length = df_action['trade_length'].mean()
    mean_profit = df_action['reward'].mean()
    std_profit = df_action['reward'].std()
    trade_density = mean_trade_length * (buy_num + sell_num) / len(env)

    with open(f'{save_data_dir}/trade_history.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['buy num', 'sell num', 'trade length', 'mean profit', 'std profit', 'shape_ratio', 'trade_density'])
        writer.writerow([buy_num, sell_num, mean_trade_length, mean_profit, std_profit, mean_profit/std_profit, trade_density])
