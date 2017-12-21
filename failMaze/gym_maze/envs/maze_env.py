import gym
import os
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import random
from gym_maze.envs.maze import MazeGame


class MazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    id = "maze-v0"

    def __init__(self, width, height, state_type, seed=None, full_deterministic=False, brute=False,
                 reinforcement=False):
        self.env = MazeGame(
            width, height, 1024, 768, state_type, 80, 80, seed=seed, seed_both=full_deterministic, brute=brute,
            reinforcement=reinforcement
        )
        self.observation_space = self.env.get_state().shape
        # print(self.observation_space)
        self.action_space = 4

    def _step(self, action):
        return self.env.step(action)

    def _reset(self):
        return self.env.reset()

    def _render(self, mode='human', close=False):
        if close:
            self.env.quit()
            return None

        return self.env.render()


'''

Full determinstic with the picture pixels as input for the neural network.

'''


class MazeNormalFullDet4x4Env(MazeEnv):
    id = "maze-normal-4x4-full-deterministic-v0"

    def __init__(self):
        super(MazeNormalFullDet4x4Env, self).__init__(4, 4, "normal", 1337, full_deterministic=True)


class MazeNormalFullDet6x6Env(MazeEnv):
    id = "maze-normal-6x6-full-deterministic-v0"

    def __init__(self):
        super(MazeNormalFullDet6x6Env, self).__init__(6, 6, "normal", 1337, full_deterministic=True)


class MazeNormalFullDet7x7Env(MazeEnv):
    id = "maze-normal-7x7-full-deterministic-v0"

    def __init__(self):
        super(MazeNormalFullDet7x7Env, self).__init__(7, 7, "normal", 1337, full_deterministic=True)


class MazeNormalFullDet9x9Env(MazeEnv):
    id = "maze-normal-9x9-full-deterministic-v0"

    def __init__(self):
        super(MazeNormalFullDet9x9Env, self).__init__(9, 9, "normal", 1337, full_deterministic=True)


class MazeNormalFullDet11x11Env(MazeEnv):
    id = "maze-normal-11x11-full-deterministic-v0"

    def __init__(self):
        super(MazeNormalFullDet11x11Env, self).__init__(11, 11, "normal", 1337, full_deterministic=True)


class MazeNormalFullDet19x19Env(MazeEnv):
    id = "maze-normal-19x19-full-deterministic-v0"

    def __init__(self):
        super(MazeNormalFullDet19x19Env, self).__init__(19, 19, "normal", 1337, full_deterministic=True)


class MazeNormalFullDet25x25Env(MazeEnv):
    id = "maze-normal-25x25-full-deterministic-v0"

    def __init__(self):
        super(MazeNormalFullDet25x25Env, self).__init__(25, 25, "normal", 1337, full_deterministic=True)


class MazeNormalFullDet35x35Env(MazeEnv):
    id = "maze-normal-35x35-full-deterministic-v0"

    def __init__(self):
        super(MazeNormalFullDet35x35Env, self).__init__(35, 35, "normal", 1337, full_deterministic=True)


class MazeNormalFullDet55x55Env(MazeEnv):
    id = "maze-normal-55x55-full-deterministic-v0"

    def __init__(self):
        super(MazeNormalFullDet55x55Env, self).__init__(55, 55, "normal", 1337, full_deterministic=True)


'''

Full deterministic, using the puzzle board as input for the neural network. 

'''


class MazeArrFullDet4x4Env(MazeEnv):
    id = "maze-arr-4x4-full-deterministic-v0"

    def __init__(self):
        super(MazeArrFullDet4x4Env, self).__init__(4, 4, "array", 1337, full_deterministic=True)


class MazeArrFullDet6x6Env(MazeEnv):
    id = "maze-arr-6x6-full-deterministic-v0"

    def __init__(self):
        super(MazeArrFullDet6x6Env, self).__init__(6, 6, "array", 1337, full_deterministic=True)


class MazeArrFullDet7x7Env(MazeEnv):
    id = "maze-arr-7x7-full-deterministic-v0"

    def __init__(self):
        super(MazeArrFullDet7x7Env, self).__init__(7, 7, "array", 1337, full_deterministic=True)


class MazeArrFullDet9x9Env(MazeEnv):
    id = "maze-arr-9x9-full-deterministic-v0"

    def __init__(self):
        super(MazeArrFullDet9x9Env, self).__init__(9, 9, "array", 1337, full_deterministic=True)


class MazeArrFullDet11x11Env(MazeEnv):
    id = "maze-arr-11x11-full-deterministic-v0"

    def __init__(self):
        super(MazeArrFullDet11x11Env, self).__init__(11, 11, "array", 1337, full_deterministic=True)


class MazeArrFullDet13x13Env(MazeEnv):
    id = "maze-arr-13x13-full-deterministic-v0"

    def __init__(self):
        super(MazeArrFullDet13x13Env, self).__init__(13, 13, "array", 1337, full_deterministic=True)


class MazeArrFullDet15x15Env(MazeEnv):
    id = "maze-arr-15x15-full-deterministic-v0"

    def __init__(self):
        super(MazeArrFullDet15x15Env, self).__init__(15, 15, "array", 1337, full_deterministic=True)


class MazeArrFullDet17x17Env(MazeEnv):
    id = "maze-arr-17x17-full-deterministic-v0"

    def __init__(self):
        super(MazeArrFullDet17x17Env, self).__init__(17, 17, "array", 1337, full_deterministic=True)


class MazeArrFullDet19x19Env(MazeEnv):
    id = "maze-arr-19x19-full-deterministic-v0"

    def __init__(self):
        super(MazeArrFullDet19x19Env, self).__init__(19, 19, "array", 1337, full_deterministic=True)


class MazeArrFullDet25x25Env(MazeEnv):
    id = "maze-arr-25x25-full-deterministic-v0"

    def __init__(self):
        super(MazeArrFullDet25x25Env, self).__init__(25, 25, "array", 1337, full_deterministic=True)


class MazeArrFullDet35x35Env(MazeEnv):
    id = "maze-arr-35x35-full-deterministic-v0"

    def __init__(self):
        super(MazeArrFullDet35x35Env, self).__init__(35, 35, "array", 1337, full_deterministic=True)


class MazeArrFullDet55x55Env(MazeEnv):
    id = "maze-arr-55x55-full-deterministic-v0"

    def __init__(self):
        super(MazeArrFullDet55x55Env, self).__init__(55, 55, "array", 1337, full_deterministic=True)


'''

Stochastic

'''


class MazeArrRnd9x9Env(MazeEnv):
    id = "maze-arr-9x9-stochastic-v0"

    def __init__(self):
        super(MazeArrRnd9x9Env, self).__init__(9, 9, "array", None)


'''

Brute mazes.

'''


class MazeArrFullDet4x4EnvBrute(MazeEnv):
    id = "maze-arr-4x4-full-deterministic-b0-v0"

    def __init__(self):
        super(MazeArrFullDet4x4EnvBrute, self).__init__(4, 4, "array", 1337, full_deterministic=True, brute=True)


class MazeArrFullDet6x6EnvBrute(MazeEnv):
    id = "maze-arr-6x6-full-deterministic-b0-v0"

    def __init__(self):
        super(MazeArrFullDet6x6EnvBrute, self).__init__(6, 6, "array", 1337, full_deterministic=True, brute=True)


class MazeArrFullDet9x9EnvBrute(MazeEnv):
    id = "maze-arr-9x9-full-deterministic-b0-v0"

    def __init__(self):
        super(MazeArrFullDet9x9EnvBrute, self).__init__(9, 9, "array", 1337, full_deterministic=True, brute=True)


class MazeArrFullDet19x19EnvBrute(MazeEnv):
    id = "maze-arr-19x19-full-deterministic-b0-v0"

    def __init__(self):
        super(MazeArrFullDet19x19EnvBrute, self).__init__(19, 19, "array", 1337, full_deterministic=True, brute=True)


class MazeArrFullDet25x25EnvBrute(MazeEnv):
    id = "maze-arr-25x25-full-deterministic-b0-v0"

    def __init__(self):
        super(MazeArrFullDet25x25EnvBrute, self).__init__(25, 25, "array", 1337, full_deterministic=True, brute=True)


class MazeArrFullDet35x35EnvBrute(MazeEnv):
    id = "maze-arr-35x35-full-deterministic-b0-v0"

    def __init__(self):
        super(MazeArrFullDet35x35EnvBrute, self).__init__(35, 35, "array", 1337, full_deterministic=True, brute=True)


class MazeArrFullDet55x55EnvBrute(MazeEnv):
    id = "maze-arr-55x55-full-deterministic-b0-v0"

    def __init__(self):
        super(MazeArrFullDet55x55EnvBrute, self).__init__(55, 55, "array", 1337, full_deterministic=True, brute=True)


class MazeArrFullDet90x90EnvBrute(MazeEnv):
    id = "maze-arr-90x90-full-deterministic-b0-v0"

    def __init__(self):
        super(MazeArrFullDet90x90EnvBrute, self).__init__(90, 90, "array", 1337, full_deterministic=True, brute=True)


'''

Reinforcement mazes.

'''


class MazeArrFullDet4x4EnvRF(MazeEnv):
    id = "maze-arr-4x4-full-deterministic-rf-v0"

    def __init__(self):
        super(MazeArrFullDet4x4EnvRF, self).__init__(4, 4, "array", 1337, full_deterministic=True, brute=False,
                                                     reinforcement=True)


class MazeArrFullDet6x6EnvRF(MazeEnv):
    id = "maze-arr-6x6-full-deterministic-rf-v0"

    def __init__(self):
        super(MazeArrFullDet6x6EnvRF, self).__init__(6, 6, "array", 1337, full_deterministic=True, brute=False,
                                                     reinforcement=True)


class MazeArrFullDet9x9EnvRF(MazeEnv):
    id = "maze-arr-9x9-full-deterministic-rf-v0"

    def __init__(self):
        super(MazeArrFullDet9x9EnvRF, self).__init__(9, 9, "array", 1337, full_deterministic=True, brute=False,
                                                     reinforcement=True)


class MazeArrFullDet19x19EnvRF(MazeEnv):
    id = "maze-arr-19x19-full-deterministic-rf-v0"

    def __init__(self):
        super(MazeArrFullDet19x19EnvRF, self).__init__(19, 19, "array", 1337, full_deterministic=True, brute=False,
                                                       reinforcement=True)


class MazeArrFullDet25x25EnvRF(MazeEnv):
    id = "maze-arr-25x25-full-deterministic-rf-v0"

    def __init__(self):
        super(MazeArrFullDet25x25EnvRF, self).__init__(25, 25, "array", 1337, full_deterministic=True,
                                                       brute=False,
                                                       reinforcement=True)


class MazeArrFullDet35x35EnvRF(MazeEnv):
    id = "maze-arr-35x35-full-deterministic-rf-v0"

    def __init__(self):
        super(MazeArrFullDet35x35EnvRF, self).__init__(35, 35, "array", 1337, full_deterministic=True,
                                                       brute=False,
                                                       reinforcement=True)


class MazeArrFullDet55x55EnvRF(MazeEnv):
    id = "maze-arr-55x55-full-deterministic-rf-v0"

    def __init__(self):
        super(MazeArrFullDet55x55EnvRF, self).__init__(55, 55, "array", 1337, full_deterministic=True, brute=False,
                                                       reinforcement=True)


class MazeArrFullDet90x90EnvRF(MazeEnv):
    id = "maze-arr-90x90-full-deterministic-rf-v0"

    def __init__(self):
        super(MazeArrFullDet90x90EnvRF, self).__init__(90, 90, "array", 1337, full_deterministic=True, brute=False,
                                                       reinforcement=True)
