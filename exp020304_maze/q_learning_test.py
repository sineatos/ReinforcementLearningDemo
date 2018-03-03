"""
@file: test.py
@author: Sineatos
@time: 2018/3/3 10:17
@contact: sineatos@gmail.com
"""

from exp020304_maze.maze_env import Maze
from exp020304_maze.RL_brain import QLearningTable


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
    env = Maze('q-learning')
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
