import numpy as np
import pandas as pd

"""
Q-Learning,Saras,Saras-Lambda的模型：
存在一个环境，这个环境包括多个状态，状态之间通过执行动作进行转移，在执行动作进行转移以后会得到奖励。可以把这个过程看成是客观存在的，即不受agent的影响
agent能做的就是通过在这个环境中活动得到状态之间的关系以及状态与动作之间的关系，从而学习到一个表或函数，这个表可以指导agent得到最大的奖励期望。

区别
Q-Learning:
较激进，不怕犯错掉进强制结束回合的陷阱内。

Saras:
较保守，在尝试过掉进陷阱内以后对陷阱比较畏惧。

Saras-Lambda:
对Saras的优化。
Saras相当于是Saras(0)，单步更新，即只考虑对上一步状态给出到达更大奖励期望的状态的指引。
而Lambda=1代表回合更新，即将整条到达最大奖励期望状态的路径上做标记，使agent知道这条路径只要按照指示走，哪怕多转两个圈都能到达最终的目标状态。
当Lambda在0~1之间代表在到达最大奖励期望状态的路径上存在指数衰减，路径上越靠近最终目标状态的标记的指导效果越强，越远离最终目标的标记知道效果越弱。这样的设置可以避免agent在新回合中在某几个状态之间不停地转圈。
"""


class RL:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        """
        :param actions: 可以执行的动作
        :param learning_rate: 学习率
        :param reward_decay: 奖励衰减值(对将来的状态带来的奖励期望的估计的衰减，使得对越远的未来的估计衰减程度越大)
        :param e_greedy: 使用确定动作还是随机动作的概率
        """
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions,dtype=np.float64)  # 空的DataFrame，用来保存动作和状态之间的关系。

    def check_state_exist(self, state):
        """
        检查当前状态是否存在于Q表中，如果不存在就往表中添加这个状态
        :param state: 当前状态
        """
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state)
            )

    def choose_action(self, observation):
        """
        根据当前状态选择一个执行的动作
        :param observation: 当前状态
        :return: 执行的动作
        """
        self.check_state_exist(observation)
        if np.random.uniform() < self.epsilon:  # 根据
            state_action = self.q_table.loc[observation, :]
            # 对各个动作的累积奖赏评期望估值进行乱序然后再选出具有最大评估期望的动作，这样做的目的是为了使得具有相同评估期望的动作有机会被选择
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args):
        """学习过程"""
        pass


# off-policy
class QLearningTable(RL):
    """
    Q-Learning
    """

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        """
        :param actions: 可以执行的动作
        :param learning_rate: 学习率
        :param reward_decay: 奖励衰减值(对将来的状态带来的奖励期望的估计的衰减，使得对越远的未来的估计衰减程度越大)
        :param e_greedy: 使用确定动作还是随机动作的概率
        """
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

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
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)


class SarasTable(RL):
    """
    Saras Table
    """

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        """
        :param actions: 可以执行的动作
        :param learning_rate: 学习率
        :param reward_decay: 奖励衰减值(对将来的状态带来的奖励期望的估计的衰减，使得对越远的未来的估计衰减程度越大)
        :param e_greedy: 使用确定动作还是随机动作的概率
        """
        super(SarasTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, state, action, reward, state_, action_):
        self.check_state_exist(state_)
        q_predict = self.q_table.ix[state, action]
        if state_ != 'terminal':
            # 实际情况中，agent从旧状态转移到新状态得到的奖励：本次执行动作得到的奖励加上在新状态上将要执行的动作对应的奖励估计期望
            q_target = reward + self.gamma * self.q_table.ix[state_, action_]
        else:
            q_target = reward
        self.q_table.ix[state, action] += self.lr * (q_target - q_predict)


class SarasLambdaTable(RL):
    """
    Saras Lambda
    """

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.9):
        """
        :param actions: 可以执行的动作
        :param learning_rate: 学习率
        :param reward_decay: 奖励衰减值(对将来的状态带来的奖励期望的估计的衰减，使得对越远的未来的估计衰减程度越大)
        :param e_greedy: 使用确定动作还是随机动作的概率
        :param trace_decay: 轨迹延迟值(对过去的状态能导致agent得到最大奖励期望的估计的衰减，使得距离最大奖励期望的状态越远的状态对估计衰减程度越大)
        """
        super(SarasLambdaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()  # 用来记录当前回合的访问轨迹

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            to_be_append = pd.Series(
                [0] * len(self.actions),
                index=self.q_table.columns,
                name=state,
            )
            # 在添加新状态的时候需要更新Q表和轨迹表
            self.q_table = self.q_table.append(to_be_append)
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)

    def learn(self, state, action, reward, state_, action_):
        self.check_state_exist(state_)
        q_predict = self.q_table.loc[state, action]  # 在旧状态估计执行动作以后能得到的奖励期望
        if state_ != 'terminal':
            # 实际情况中，agent从旧状态转移到新状态得到的奖励：本次执行动作得到的奖励加上在新状态上将要执行的动作对应的奖励估计期望
            q_target = reward + self.gamma * self.q_table.loc[state_, action_]
        else:
            q_target = reward
        error = q_target - q_predict

        # 策略 1: 某一个状态-动作对每到达并执行过一次，就加1
        # self.eligibility_trace.loc[state, action] += 1

        # 策略 2: 执行了某一个状态-动作对，就将这个状态的其他动作都标记为没有执行过，而本次执行了的动作则标记为1
        # 这种策略使得当agent在新回合中到达相同的状态，就只会执行这个动作，因为它认为执行了这个动作就一定能取得大奖赏
        # 相当于增强自身并抑制其他信号的表达。
        self.eligibility_trace.loc[state, :] *= 0
        self.eligibility_trace.loc[state, action] = 1

        # 更新Q表，学习率乘以实际奖赏估计与想象中奖赏估计的差再乘以轨迹值
        self.q_table += self.lr * error * self.eligibility_trace

        # 更新轨迹值(这个更多的是对策略1使用)。
        # 轨迹值乘上奖励衰减值和轨迹衰减值
        self.eligibility_trace *= self.gamma * self.lambda_
