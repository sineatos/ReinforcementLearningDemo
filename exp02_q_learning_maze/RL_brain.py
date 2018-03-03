import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        """
        :param actions: 可以执行的动作
        :param learning_rate: 学习率
        :param reward_decay: γ
        :param e_greedy: epsilon
        """
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions)  # 空的DataFrame，用来保存动作和状态之间的关系。

    def choose_action(self, observation):
        """
        根据当前状态选择一个执行的动作
        :param observation: 当前状态
        :return: 执行的动作
        """
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:  # 根据
            state_action = self.q_table.ix[observation, :]
            # 对各个动作的累积奖赏评期望估值进行乱序然后再选出具有最大评估期望的动作，这样做的目的是为了使得具有相同评估期望的动作有机会被选择
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.argmax()
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, state, action, reward, state_):
        self.check_state_exist(state_)  # 检查新状态是否在Q表中
        q_predict = self.q_table.ix[state, action]
        if state_ != 'terminal':
            # 转移到新状态实际的奖励:本次执行动作得到的奖励加上在新状态上有可能得到的最大奖励估计期望
            q_target = reward + self.gamma * self.q_table.loc[state_, :].max()
        else:
            # 新状态就是终止状态，所以实际的奖励期望只有执行动作得到的奖励
            q_target = reward
        # 更新旧状态执行动作action的奖励值，为实际奖励期望与预期奖励期望的差距的百分比
        self.q_table.ix[state,action] += self.lr * (q_target-q_predict)

    def check_state_exist(self, state):
        """
        检查当前状态是否存在于Q表中，如果不存在就往表中添加这个状态
        :param state: 当前状态
        """
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state)
            )
