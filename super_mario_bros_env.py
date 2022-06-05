from Hyperparameters import *
import gym
import retro
import numpy as np
import cv2
import os


CWD = os.getcwd()
GAME_DIR = CWD + r"\game_data"
STATE_DIR = GAME_DIR + r"\Level_1-"


NO_ACTION = np.array([False, False, False, False, False, False, False, False, False])
LEFT = np.array([False, False, False, False, False, False, True, False, False])
RIGHT = np.array([False, False, False, False, False, False, False, True, False])
A = np.array([False, False, False, False, False, False, False, False, True])
LEFT_A = np.array([False, False, False, False, False, False, True, False, True])
RIGHT_A = np.array([False, False, False, False, False, False, False, True, True])
LEFT_B = np.array([True, False, False, False, False, False, True, False, False])
RIGHT_B = np.array([True, False, False, False, False, False, False, True, False])
LEFT_AB = np.array([True, False, False, False, False, False, True, False, True])
RIGHT_AB = np.array([True, False, False, False, False, False, False, True, True])


class Action_Wrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(Action_Wrapper, self).__init__(env)
        self._actions = [NO_ACTION, LEFT, RIGHT, A, LEFT_A, RIGHT_A, LEFT_B, RIGHT_B, LEFT_AB, RIGHT_AB]
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        return self._actions[a].copy()


class State_Wrapper(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.width = FRAME_WIDTH
        self.height = FRAME_HEIGHT
        self.observation_space = gym.spaces.Box(low = 0,
                                                high = 255,
                                                shape = (self.height, self.width),
                                                dtype = np.uint8)

    def observation(self, frame):
        frame = cv2.resize(frame, (self.width, self.height), interpolation = cv2.INTER_CUBIC)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame


class Method_Wrapper(gym.Wrapper):
    def __init__(self, env, level):
        gym.Wrapper.__init__(self, env)
        self.frame_skip = FRAME_SKIP
        self.level = level
        shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low = 0,
                                                high = 255,
                                                shape = shape,
                                                dtype = np.uint8)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        for i in range(self.frame_skip):
            state, _, _, info = self.env.step(action)
        done = (info["lives"] < 2) or (info["level"] > self.level)
        if (info["lives"] < 2):
            x = -1.
        elif (info["level"] > self.level):
            x = -2.
        else:
            x = 256. * float(info["x1"]) + float(info["x2"])
        return state, x, done, info


def make(level):
    state = STATE_DIR + str(level + 1) + ".state"
    env = retro.make(game = GAME_DIR,
                     state = state)
    env = Action_Wrapper(env)
    env = State_Wrapper(env)
    env = Method_Wrapper(env, level)
    return env