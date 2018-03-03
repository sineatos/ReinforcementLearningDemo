"""
@file: test.py
@author: Sineatos
@time: 2018/3/3 10:17
@contact: sineatos@gmail.com
"""

from exp020304_maze.maze_env import Maze
from exp020304_maze.RL_brain import SarasTable


def update():
    for episode in range(100):
        observation = env.reset()

        # 首先在初始状态选取能够得到积累奖励最高期望的动作
        action = RL.choose_action(str(observation))

        while True:
            # 刷新环境
            env.render()

            # 转移到新状态上
            observation_, reward, done = env.step(action)

            # 选取在新状态上能够得到积累奖励最高期望的动作
            action_ = RL.choose_action(str(observation_))

            # 根据旧状态，执行的动作，得到的奖励，新状态以及新状态中预估能够得到最高奖励期望的动作进行学习
            RL.learn(str(observation), action, reward, str(observation_), action_)

            # 更新状态以及在新状态中准备执行的动作
            observation = observation_
            action = action_

            if done:
                break

    print('game over')
    env.destroy()


if __name__ == "__main__":
    env = Maze('saras')
    RL = SarasTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
