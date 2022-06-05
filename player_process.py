from Hyperparameters import *
import super_mario_bros_env
import numpy as np
import os


def h(x):
    return np.sign(x) * (np.sqrt(abs(x) + 1.) - 1.) + SQUISH * x


def h_inv(x):
    arg = 4 * SQUISH * (abs(x) + SQUISH + 1.) + 1.
    f1 = (1. - np.sqrt(arg)) / (2. * (SQUISH ** 2))
    f2 = (abs(x) + 1) / SQUISH
    return np.sign(x) * (f1 + f2)


def get_reward(x, past_x):
    if x == -1:
        return -2.
    elif x == -2:
        return 2.
    else:
        reward = (x - past_x) / 18.
        if abs(reward) > 1.:
            return 0.
        else:
            return reward


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

    ''' Initialize Predictor and Environment '''
    env = super_mario_bros_env.make(level)
    predictor = create_player_predictor()
    child_con.send(('weights', None))
    weights = child_con.recv()
    predictor.set_weights(weights)

    discount_mean = (1. - MIN_DISCOUNT) / 2.
    discount_std = (1. - MIN_DISCOUNT) / 2.

    beta_mean = (MAX_BETA - MIN_BETA) / 2.
    beta_std = (MAX_BETA - MIN_BETA) / 2.

    discounts = []
    betas = []
    total_rewards = []

    num_games = 0

    ''' Play Game '''
    while True:
        predictor.reset_states()

        discount, beta = get_d_and_b(discount_mean, discount_std, beta_mean, beta_std)
        discounts.append(discount)
        betas.append(beta)

        states = []
        actions = []
        rewards = []
        dones = []
        q_values = []

        steps = 0
        total_reward = 0.

        past_x = 40.
        past_a = 0

        current_state = env.reset()
        if RENDER:
            env.render()

        while True:

            s = np.expand_dims(current_state.copy().astype(np.float32) / 255., axis=0)
            pa = np.zeros((1, NUM_ACTIONS), dtype=np.float32)
            pa[0][past_a] = 1.
            q_value = predictor([s, pa, np.array([[discount]]), np.array([[beta]])]).numpy().squeeze()
            if (np.random.random() < epsilon) or (num_games < PLAYER_WARM_UP):
                action = np.random.randint(NUM_ACTIONS)
            else:
                action = np.argmax(q_value)

            next_state, x, done, info = env.step(action)
            if RENDER:
                env.render()
            reward = get_reward(x, past_x)
            steps += 1
            total_reward += reward

            states.append(current_state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            q_values.append(q_value)

            if done:
                q_values.append(np.zeros(NUM_ACTIONS))
                total_rewards.append(total_reward)
                num_games += 1
                break

            current_state = next_state.copy()
            past_x = x
            past_a = action

        dis = DISCOUNT ** discount
        tds = []
        for i in range(len(rewards)):
            td = abs(h(rewards[i] + dis * h_inv(np.max(q_values[i + 1]))) - q_values[i][actions[i]])
            td = (td ** PER_ALPHA) + PER_EPSILON
            tds.append(td)

        batch = [states, actions, rewards, dones, tds, discount, beta]
        child_con.send(('batch', batch))

        if len(total_rewards) == ARM_EPISODE_LEN:
            best_idx = np.argsort(total_rewards)[-BEST_ARM:]
            best_discounts = np.array(discounts)[best_idx]
            best_betas = np.array(betas)[best_idx]

            discount_mean = np.mean(best_discounts)
            discount_std = max(np.std(best_discounts), MIN_DISCOUNT_STD)

            beta_mean = np.mean(best_betas)
            beta_std = max(np.std(best_betas), MIN_BETA_STD)

            discounts = []
            betas = []
            total_rewards = []

        if (num_games % UPDATE_PLAYER_PERIOD) == 0:
            child_con.send(('weights', None))
            weights = child_con.recv()
            predictor.set_weights(weights)