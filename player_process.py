from Hyperparameters import *
import super_mario_bros_env
import numpy as np
import os


def get_reward(x, past_x):
    if x == -1:
        return -1.
    elif x == -2:
        return 1.
    else:
        return x - past_x


def get_d_and_b(dm, ds, bm, bs):
    if np.random.random() < ARM_EPSILON:
        dx = (1. - MIN_DISCOUNT) * np.random.random() + MIN_DISCOUNT
        bx = (MAX_BETA - MIN_BETA) * np.random.random() + MIN_BETA
    else:
        dx = np.random.normal(dm, ds)
        bx = np.random.normal(bm, bs)
        dx = np.clip(dx, MIN_DISCOUNT, 1.)
        bx = np.clip(bx, MIN_BETA, MAX_BETA)
    return dx, bx


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

    ''' Initialize Predictor and Environment '''
    env = super_mario_bros_env.make(level)
    predictor = create_player_predictor()

    discount_mean = (1. - MIN_DISCOUNT) / 2.
    discount_std = (1. - MIN_DISCOUNT) / 2.

    beta_mean = (MAX_BETA - MIN_BETA) / 2.
    beta_std = (MAX_BETA - MIN_BETA) / 2.

    discounts = []
    betas = []
    rewards = []

    ''' Play Game '''
    while True:
        predictor.reset_states()
        state = env.reset()

        discount, beta = get_d_and_b(discount_mean, discount_std, beta_mean, beta_std)
        discounts.append(discount)
        betas.append(beta)

        past_x = 40. / 18.
        past_a = 0

        if RENDER:
            env.render()
        while True:
            a = choose_action(state, past_a)
            state, x, done, info = env.step(a)
            reward = get_reward(x, past_x)
            rewards.append(reward)
            if RENDER:
                env.render()
            if done:
                break
            past_x = x
            past_a = a

        if len(rewards) > ARM_EPISODE_LEN:
            best_idx = np.argsort(rewards)[-BEST_ARM:]
            best_discounts = np.array(discounts)[best_idx]
            best_betas = np.array(betas)[best_idx]

            discount_mean = np.mean(best_discounts)
            discount_std = max(np.std(best_discounts), MIN_DISCOUNT_STD)

            beta_mean = np.mean(best_betas)
            beta_std = max(np.std(best_betas), MIN_BETA_STD)

            discounts = []
            betas = []
            rewards = []