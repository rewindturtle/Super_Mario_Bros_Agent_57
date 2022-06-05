from Hyperparameters import *
import super_mario_bros_env
import numpy as np
import os


def player_process(child_con, epsilon, level=0):
    from Neural_Networks import create_player_predictor
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

    def choose_action(state, past_action):
        s = np.expand_dims(state.copy().astype(np.float32) / 255., axis=0)
        pa = np.zeros((1, NUM_ACTIONS), dtype=np.float32)
        pa[0][past_action] = 1.
        d = np.array([[discount]])
        b = np.array([[beta]])
        a = predictor([s, pa, d, b]).numpy()[0]
        if (np.random.random() < epsilon):
            a = np.random.randint(NUM_ACTIONS)
        return a

    ''' Initialize Player '''
    env = super_mario_bros_env.make(level)
    predictor = create_player_predictor()
    discount = 0.9
    beta = 0.3

    ''' Play Game '''
    while True:
        predictor.reset_states()
        state = env.reset()
        a = 0
        if RENDER:
            env.render()
        while True:
            a = choose_action(state, a)
            state, x, done, info = env.step(a)
            if RENDER:
                env.render()
            if done:
                break