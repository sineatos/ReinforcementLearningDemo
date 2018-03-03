"""
@file: test.py
@author: Sineatos
@time: 2018/3/3 10:17
@contact: sineatos@gmail.com
"""

from exp02_q_learning_maze.maze_env import Maze
from exp02_q_learning_maze.RL_brain import QLearningTable

"""
Q-Learning,Saras的模型：
存在一个环境，这个环境包括多个状态，状态之间通过执行动作进行转移，在执行动作进行转移以后会得到奖励。可以把这个过程看成是客观存在的，即不受agent的影响
agent能做的就是通过在这个环境中活动得到状态之间的关系以及状态与动作之间的关系，从而学习到一个表或函数，这个表可以指导agent得到最大的奖励期望。
"""


def update():
    for episode in range(100):
        observation = env.reset()
        while True:
            # 刷新环境
            env.render()

            # 根据当前所在的状态选择一个动作
            action = RL.choose_action(str(observation))

            # 转移到新状态上
            observation_, reward, done = env.step(action)

            # 根据旧状态，执行的动作，得到的奖励，新状态进行学习
            RL.learn(str(observation), action, reward, str(observation_))

            # 更新状态
            observation = observation_

            if done:
                break

    print('game over')
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
