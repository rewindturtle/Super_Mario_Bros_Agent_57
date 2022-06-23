from Hyperparameters import *
import Neural_Networks as nn
import super_mario_bros_env
import numpy as np
import os


def h(x):
    return np.sign(x) * (np.sqrt(abs(x) + 1.) - 1.) + SQUISH * x


def h_inv(x):
    arg = 4 * SQUISH * (abs(x) + SQUISH + 1.) + 1.
    f1 = (1. - np.sqrt(arg)) / (2. * SQUISH)
    f2 = abs(x) + 1.
    return np.sign(x) * (f1 + f2) / SQUISH


def get_external_reward(x, past_x):
    if x == -1:
        return -2.
    elif x == -2:
        return 2.
    else:
        reward = (x - past_x) / EXTRINSIC_REWARD_NORM
        if abs(reward) > 1.:
            return 0.
        else:
            return reward


def player_process(child_con, player_num, epsilon, level=0):
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))

    ''' Initialize Predictor and Environment '''
    env = super_mario_bros_env.make(level)
    predictor = nn.create_player_predictor()
    intrinsic_calc = nn.create_intrinsic_reward_calc()

    child_con.send(('weights', None))
    weights = child_con.recv()
    predictor.set_weights(weights[0])
    intrinsic_calc.set_weights(weights[1])

    total_reward_window = []

    num_games = 0
    arm_window = []

    ''' Play Game '''
    while True:
        predictor.reset_states()

        if num_games < NUM_ARMS:
            arm = num_games
        elif np.random.random() < ARM_EPSILON:
            arm = np.random.randint(NUM_ARMS)
        else:
            num_window = len(arm_window)
            arm_n = np.zeros((NUM_ARMS, num_window))
            arm_n[arm_window, range(num_window)] = 1.
            arm_n = 1. / (np.sum(arm_n, axis = 1) + 1.)
            arm_r = np.zeros((NUM_ARMS, num_window))
            arm_r[arm_window, range(num_window)] = total_reward_window
            arm_r = np.sum(arm_r, axis = 1)
            arm_mu = arm_n * arm_r + ARM_BETA * np.sqrt(arm_n)
            arm = np.argmax(arm_mu)

        beta = BETAS[arm]
        gamma = GAMMAS[arm]

        states = []
        actions = []
        extrinsic_rewards = []
        dones = []
        q_values = []

        steps = 0
        total_reward = 0.

        past_x = 40.
        past_a = 0
        max_x = 40.

        current_state = env.reset()
        if RENDER:
            env.render()

        while True:
            s = current_state.copy()[None, ...].astype(np.float32) / 255.
            qe, qi = predictor([s, np.array([[past_a]], dtype = np.int32), np.array([[arm]], dtype = np.int32)])
            qe = h_inv(qe.numpy().squeeze())
            qi = h_inv(qi.numpy().squeeze())
            q = h(qe + beta * qi)
            if (np.random.random() < epsilon) or (num_games < PLAYER_WARM_UP):
                action = np.random.randint(NUM_ACTIONS)
            else:
                action = np.argmax(q)
            q_values.append(q)

            next_state, x, done, info = env.step(action)
            if RENDER:
                env.render()
            extrinsic_reward = get_external_reward(x, past_x)
            max_x = max(x, max_x)
            steps += 1
            total_reward += extrinsic_reward

            states.append(current_state)
            actions.append(action)
            extrinsic_rewards.append(extrinsic_reward)
            dones.append(not done)

            if done:
                num_games += 1
                break

            current_state = next_state.copy()
            past_x = x
            past_a = action

        num_samples = np.ceil(len(actions) / REPLAY_PERIOD).astype(int)
        state_array = np.array(states, dtype = np.uint8)
        action_array = np.array(actions, dtype = np.int32)
        e_reward_array = np.array(extrinsic_rewards, dtype = np.float32)
        done_array = np.array(dones, dtype = np.float32)

        q_array = np.array(q_values, dtype = np.float32)
        next_q_array = np.roll(q_array.copy(), -1, axis = 0)
        next_q_array[-1, ...] = np.zeros(NUM_ACTIONS, dtype = np.float32)

        s_array = state_array.copy().astype(np.float32) / 255.
        ns_array = np.roll(s_array.copy(), -1, axis = 0)
        ns_array[-1, ...] = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype = np.float32)
        i_reward_array = intrinsic_calc([s_array, ns_array, action_array]).numpy().squeeze()

        td1 = h(e_reward_array + beta * i_reward_array + gamma * done_array * h_inv(np.max(next_q_array, axis = 1)))
        td2 = h(q_array[range(action_array.shape[0]), action_array])
        td_array = np.absolute(td1 - td2)

        batch_states = []
        batch_actions = []
        batch_e_rewards = []
        batch_dones = []
        batch_tds = []
        batch_arms = []
        for i in range(num_samples):
            trace_states = np.zeros((TRACE_LENGTH + 1, FRAME_HEIGHT, FRAME_WIDTH), dtype = np.uint8)
            trace_actions = np.zeros(TRACE_LENGTH + 1, dtype = np.int32)
            trace_e_rewards = np.zeros(TRACE_LENGTH, dtype = np.float32)
            trace_dones = np.zeros(TRACE_LENGTH, dtype = np.float32)

            end = min(REPLAY_PERIOD * (i + 2), action_array.shape[0])
            s_end = min(REPLAY_PERIOD * (i + 2) + 1, action_array.shape[0])

            trace_end = end - REPLAY_PERIOD * i
            state_end = s_end - REPLAY_PERIOD * i
            action_end = end - REPLAY_PERIOD * i + 1

            trace_states[:state_end, ...] =  state_array[REPLAY_PERIOD * i:s_end].copy()
            if i == 0:
                trace_actions[1:action_end] = action_array[REPLAY_PERIOD * i:end].copy()
            else:
                trace_actions[:action_end] = action_array[(REPLAY_PERIOD * i - 1):end].copy()

            trace_e_rewards[:trace_end] = e_reward_array[REPLAY_PERIOD * i:end].copy()
            trace_dones[:trace_end] = done_array[REPLAY_PERIOD * i:end].copy()
            tds = td_array[REPLAY_PERIOD * i:end].copy()
            trace_td = PER_ETA * np.max(tds) + (1. - PER_ETA) * np.mean(tds) + PER_EPSILON

            batch_states.append(trace_states)
            batch_actions.append(trace_actions)
            batch_e_rewards.append(trace_e_rewards)
            batch_dones.append(trace_dones)
            batch_tds.append(trace_td)
            batch_arms.append(arm)

        total_i_reward = np.sum(i_reward_array)
        total_reward_window.append(total_reward + beta * total_i_reward)
        arm_window.append(arm)
        if len(total_reward_window) > ARM_WINDOW:
            total_reward_window = total_reward_window[1:]
            arm_window = arm_window[1:]

        data = [player_num, num_games, total_reward, total_i_reward, gamma, beta, epsilon, steps, max_x]
        batch = [batch_states, batch_actions, batch_e_rewards, batch_dones, batch_tds, batch_arms, data]
        child_con.send(('batch', batch))
        child_con.send(('weights', None))
        weights = child_con.recv()
        predictor.set_weights(weights[0])