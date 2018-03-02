# -*- encoding:UTF-8 -*-
import numpy as np
import pandas as pd
import time

# np.random.seed(2)

# 一维世界的长度
N_STATES = 8

# 可执行的动作
ACTIONS = ['left', 'right']

# 贪心策略执行的概率
EPSILON = 0.9

# 学习率
ALPHA = 0.1

# 折扣因子
LAMBDA = 0.9

# 最大回合数(最大运行次数)
MAX_EPISODES = N_STATES * 2 + 1

# 走一步需要多少秒(多少秒刷新一次)
FRESH_TIME = 0.3


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states+1, len(actions))),  # n_states行len(actions)列
        columns=actions,
    )
    return table


def choose_action(state, q_table):
    # 根据Q表选择一个操作
    state_actions = q_table.iloc[state, :]
    if np.random.uniform() > EPSILON or state_actions.all() == 0:
        # 探索操作
        action_name = np.random.choice(ACTIONS)
    else:
        # 利用操作
        action_name = state_actions.argmax()
    return action_name


def get_env_feedback(S, A):
    # 获取环境反馈，下一个状态和获得的奖赏
    if A == 'right':
        if S == N_STATES - 2:  # 进行向右操作以后就会到达终点
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:  # 撞到左边的墙
            S_ = S
        else:
            S_ = S - 1
    return S_, R


def update_env(S, episode, step_counter):
    env_list = ['-'] * (N_STATES - 1) + ['T']
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction))
        time.sleep(2)
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction))
        time.sleep(FRESH_TIME)


def rl():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = np.random.randint(0, N_STATES-1)
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:
            A = choose_action(S, q_table)  # 根据当前的状态和Q表选择一个执行动作
            S_, R = get_env_feedback(S, A)  # 在当前状态上执行动作，获取下一个状态和执行动作得到的奖赏
            q_predict = q_table.ix[S, A]  # 根据以往经验，获取在状态S上执行动作A，对往后累积奖励的期望

            # 更新经验值(Q表)
            if S_ != 'terminal':
                q_target = R + LAMBDA * q_table.iloc[S_, :].max()
            else:
                q_target = R
                is_terminated = True
            q_table.ix[S, A] += ALPHA * (q_target - q_predict)
            S = S_  # 更新状态

            update_env(S, episode, step_counter + 1)
            step_counter += 1
    return q_table


if __name__ == '__main__':
    rl()
