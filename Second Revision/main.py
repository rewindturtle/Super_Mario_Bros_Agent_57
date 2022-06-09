from Hyperparameters import *
import Neural_Networks as nn
from player_process import player_process
import multiprocessing as mp
import threading
import numpy as np
import time
from bisect import bisect_right
from timeit import default_timer as timer


def get_epsilon(num):
    e1 = np.log(MAX_EPSILON)
    e2 = np.log(MIN_EPSILON)
    e3 = (e1 - e2) * num / (NUM_PLAYERS - 1) + e2
    return np.exp(e3)


def h(x):
    return np.sign(x) * (np.sqrt(abs(x) + 1.) - 1.) + SQUISH * x


def h_inv(x):
    x = np.clip(x, -Q_CLIP, Q_CLIP)
    arg = 4 * SQUISH * (abs(x) + SQUISH + 1.) + 1.
    f1 = (1. - np.sqrt(arg)) / (2. * (SQUISH ** 2))
    f2 = (abs(x) + 1) / SQUISH
    return np.sign(x) * (f1 + f2)


class Trainer:
    def __init__(self, connections):
        self.connections = connections
        self.memory_lock = threading.Lock()
        self.network_lock = threading.Lock()
        self.display_lock = threading.Lock()

        self.predictor = nn.create_trainer_predictor()
        self.target = nn.create_target_predictor()
        self.hasher = nn.create_trainer_hasher()
        self.player_hasher = nn.create_player_hasher()
        self.player_hasher.set_weights(self.hasher.get_weights()[:8])
        self.rnd_net = nn.create_trainer_rnd()
        self.rnd_target = nn.create_target_rnd()

        self.states = []
        self.actions = []
        self.rewards = []
        self.tds = []
        self.discounts = []
        self.betas = []
        self.cum_steps = [0]

        self.num_training_episodes = 0

    def update_memory(self, batch):
        states, actions, rewards, tds, discount, beta, data = batch
        self.memory_lock.acquire()
        sz = len(actions)
        self.states = self.states + states
        self.actions = self.actions + actions
        self.rewards = self.rewards + rewards
        self.tds = self.tds + tds
        self.discounts.append(discount)
        self.betas.append(beta)
        self.cum_steps.append(self.cum_steps[-1] + sz)
        self.memory_lock.release()
        self.update_training_data(data)

    def print(self, lines):
        self.display_lock.acquire()
        print('--------------------------')
        for line in lines:
            print(line)
        print(' ')
        self.display_lock.release()

    def update_training_data(self, data):
        player_num, num_games, total_reward, total_i_reward, dis, beta, epsilon, steps, max_x = data
        if PRINT_GAME_DATA:
            lines = []
            lines.append('Player ' + str(player_num))
            lines.append('Games Played: ' + str(num_games))
            lines.append('Total Extrinsic Reward: ' + str(total_reward))
            lines.append('Total Intrinsic Reward: ' + str(total_i_reward))
            lines.append('Total Reward: ' + str(total_reward + beta * total_i_reward))
            lines.append('Discount: ' + str(dis))
            lines.append('Beta: ' + str(beta))
            lines.append('Epsilon: ' + str(epsilon))
            lines.append('Total Steps: ' + str(steps))
            lines.append('Max X: ' + str(max_x))
            lines.append('Memory Size: ' + str(len(self.actions)))
            lines.append('Training Episodes: ' + str(self.num_training_episodes))
            self.print(lines)

    def update_target(self):
        self.network_lock.acquire()
        self.target.set_weights(self.predictor.get_weights())
        self.network_lock.release()

    def get_weights(self):
        self.network_lock.acquire()
        w0 = self.predictor.get_weights()
        w1 = self.hasher.get_weights()[:8]
        w2 = self.rnd_net.get_weights()
        self.network_lock.release()
        weights = [w0, w1, w2]
        return weights

    def get_initial_weights(self):
        self.network_lock.acquire()
        w0 = self.predictor.get_weights()
        w1 = self.player_hasher.get_weights()
        w2 = self.rnd_net.get_weights()
        w3 = self.rnd_target.get_weights()
        self.network_lock.release()
        weights = [w0, w1, w2, w3]
        return weights

    def listen(self):
        i = 0
        while True:
            if self.connections[i].poll():
                cmd, data = self.connections[i].recv()
                if cmd == 'batch':
                    self.update_memory(data)
                elif cmd == 'weights':
                    weights = self.get_weights()
                    self.connections[i].send(weights)
                elif cmd == 'initialize':
                    weights = self.get_initial_weights()
                    self.connections[i].send(weights)
                time.sleep(0.1)
            i = (i + 1) % NUM_PLAYERS

    def get_batch(self, indices):
        actions = np.array([self.actions[i] for i in indices], dtype = int)
        external_rewards = np.array([self.rewards[i] for i in indices], dtype = np.float32)
        hash_states = np.array([self.states[i].copy() for i in indices], dtype = np.float32) / 255.

        current_states = np.zeros((BATCH_SIZE, LSTM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH), dtype = np.float32)
        next_states = np.zeros((BATCH_SIZE, LSTM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH), dtype = np.float32)
        n_states = np.zeros((BATCH_SIZE, LSTM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH), dtype = np.float32)
        past_actions = np.zeros((BATCH_SIZE, LSTM_FRAMES, NUM_ACTIONS), dtype = np.float32)
        next_past_actions = np.zeros((BATCH_SIZE, LSTM_FRAMES, NUM_ACTIONS), dtype = np.float32)
        n_past_actions = np.zeros((BATCH_SIZE, LSTM_FRAMES, NUM_ACTIONS), dtype = np.float32)
        n_external_rewards = np.zeros(BATCH_SIZE, dtype = np.float32)
        n_internal_rewards = np.zeros(BATCH_SIZE, dtype = np.float32)
        internal_rewards = np.zeros(BATCH_SIZE, dtype = np.float32)
        next_hash_states = np.zeros((BATCH_SIZE, FRAME_HEIGHT, FRAME_WIDTH), dtype = np.float32)

        i = 0
        ends = np.zeros(BATCH_SIZE, dtype = int)
        eps = BATCH_SIZE * [0]

        self.network_lock.acquire()
        for idx in indices:
            ep = bisect_right(self.cum_steps, idx)
            start = self.cum_steps[ep - 1]
            end = self.cum_steps[ep]
            ends[i] = end
            eps[i] = ep - 1

            episode_states = np.array(self.states[start:end], dtype = np.float32).copy() / 255.
            episode_action_idx = np.array(self.actions[start:end - 1], dtype = int)
            episode_past_actions = np.zeros((episode_states.shape[0], NUM_ACTIONS), dtype = np.float32)
            episode_past_actions[range(1, episode_past_actions.shape[0]), episode_action_idx] = 1.

            diff = idx - start + 1
            d_low = max(diff - LSTM_FRAMES, 0)
            state_slice = episode_states[d_low:diff, :, :].copy()
            action_slice = episode_past_actions[d_low:diff, :].copy()
            current_states[i, -state_slice.shape[0]:, :, :] = state_slice
            past_actions[i, -action_slice.shape[0]:, :] = action_slice

            d_next = min(diff + 1, end - start)
            d_low = max(d_next - LSTM_FRAMES, 0)
            state_slice = episode_states[d_low:d_next, :, :].copy()
            action_slice = episode_past_actions[d_low:d_next, :].copy()
            next_states[i, -state_slice.shape[0]:, :, :] = state_slice
            next_past_actions[i, -action_slice.shape[0]:, :] = action_slice

            d_n = min(diff + N_STEP, end - start)
            d_low = max(d_n - LSTM_FRAMES, 0)
            state_slice = episode_states[d_low:d_n, :, :].copy()
            action_slice = episode_past_actions[d_low:d_n, :].copy()
            n_states[i, -state_slice.shape[0]:, :, :] = state_slice
            n_past_actions[i, -action_slice.shape[0]:, :] = action_slice

            hash = self.player_hasher(episode_states).numpy()
            rnd = self.rnd_net(episode_states[1:, :, :]).numpy()
            rnd_t = self.rnd_target(episode_states[1:, :, :]).numpy()
            h1 = np.repeat(hash[1:, :].copy(), hash.shape[0], axis = 0)
            h2 = np.tile(hash.copy(), (hash.shape[0] - 1, 1))
            h3 = np.linalg.norm(h1 - h2, axis = 1) ** 2
            h3 = np.reshape(h3[np.newaxis, :], (hash.shape[0] - 1, hash.shape[0]))
            dists = np.sort(h3, axis = 1)[:, 1:(NEAREST_NEIGHBOURS + 1)]
            dists = dists / max(np.mean(dists), 1e-9)
            dists = np.clip(dists - KERNEL_CLUSTER, 0., np.inf)
            kernel = KERNEL_EPSILON / (dists + KERNEL_EPSILON)
            score = np.sqrt(np.sum(kernel, axis = 1)) + KERNEL_CONSTANT
            rt = np.where(score > KERNEL_MAX_SCORE, 0, 1. / score)
            err = np.linalg.norm(rnd - rnd_t, axis = 1) ** 2
            err_mean = np.mean(err)
            err_std = max(np.std(err), 1e-9)
            rnd_alpha = 1. + (err - err_mean) / err_std
            irs = rt * np.clip(rnd_alpha, 1., MAX_RND) / INTRINSIC_REWARD_NORM
            irs = np.clip(irs, -MAX_INTRINSIC_REWARD, MAX_INTRINSIC_REWARD)
            irs = np.append(irs, 0.)
            internal_rewards[i] = irs[idx - start]

            dn = min(idx + N_STEP, end)
            dis = DISCOUNT ** (self.discounts[ep - 1] * np.arange(dn - idx))
            n_external_rewards[i] = np.sum(dis * np.array(self.rewards[idx:dn]))
            n_internal_rewards[i] = np.sum(dis * irs[(idx - start):(dn - start)])

            if idx + 1 != end:
                next_hash_states[i, :, :] = self.states[idx + 1].copy().astype(np.float32) / 255.

            i += 1
        self.network_lock.release()

        discounts = np.array([self.discounts[i] for i in eps], dtype = np.float32)
        betas = np.array([self.betas[i] for i in eps], dtype = np.float32)
        dones = ((indices + 1) == ends).astype(np.float32)
        n_dones = ((indices + N_STEP) >= ends).astype(np.float32)
        return (current_states, past_actions, actions, external_rewards, internal_rewards, next_states,
                next_past_actions, dones, discounts, betas, n_states, n_past_actions, n_external_rewards,
                n_internal_rewards, n_dones, hash_states, next_hash_states)

    def train(self):
        while len(self.actions) < WARM_UP:
            time.sleep(5)
        print('Training Started')
        while True:
            self.memory_lock.acquire()
            td_array = np.array(self.tds)
            td_prob = td_array / np.sum(td_array)
            num_td = td_prob.shape[0]
            batch_indices = np.random.choice(num_td, BATCH_SIZE, p = td_prob, replace = False)
            current_state_tensor, past_action_tensor, a_tensor, re_tensor, ri_tensor, next_state_tensor, \
            next_past_action_tensor, done_tensor, discount_tensor, beta_tensor, n_state_tensor, n_past_action_tensor, \
            n_re_tensor, n_ri_tensor, n_done_tensor, hash_tensor, next_hash_tensor = self.get_batch(batch_indices)
            self.memory_lock.release()

            weights = (1. - (1. - (1. / num_td)) ** BATCH_SIZE) / (1. - (1. - td_prob[batch_indices]) ** BATCH_SIZE)
            softmax_tensor = np.zeros((BATCH_SIZE, NUM_ACTIONS), dtype = np.float32)

            self.network_lock.acquire()
            qe, qi, _, _ = self.predictor([current_state_tensor, past_action_tensor, discount_tensor, beta_tensor])
            qe2, qi2, _, _ = self.predictor([next_state_tensor, next_past_action_tensor, discount_tensor, beta_tensor])
            qen, qin, _, _ = self.predictor([n_state_tensor, n_past_action_tensor, discount_tensor, beta_tensor])
            qt2 = self.target([next_state_tensor, next_past_action_tensor, discount_tensor, beta_tensor]).numpy()
            qtn = self.target([n_state_tensor, n_past_action_tensor, discount_tensor, beta_tensor]).numpy()
            rnd_target = self.rnd_target(hash_tensor).numpy()
            self.network_lock.release()

            qe = qe.numpy()
            qi = qi.numpy()
            qe2 = qe2.numpy()
            qi2 = qi2.numpy()
            qen = qen.numpy()
            qin = qin.numpy()

            training_qe2 = qe.copy()
            training_qi2 = qi.copy()
            training_qen = qe.copy()
            training_qin = qi.copy()
            max_q2 = np.argmax(qt2, axis = 1)
            max_qn = np.argmax(qtn, axis = 1)

            training_qe2[range(BATCH_SIZE), a_tensor] = h(re_tensor + done_tensor * (DISCOUNT ** discount_tensor) * h_inv(qe2[range(BATCH_SIZE), max_q2]))
            training_qi2[range(BATCH_SIZE), a_tensor] = h(ri_tensor + done_tensor * (DISCOUNT ** discount_tensor) * h_inv(qi2[range(BATCH_SIZE), max_q2]))
            training_qen[range(BATCH_SIZE), a_tensor] = h(n_re_tensor + n_done_tensor * (DISCOUNT ** ((N_STEP + 1) * discount_tensor)) * h_inv(qen[range(BATCH_SIZE), max_qn]))
            training_qin[range(BATCH_SIZE), a_tensor] = h(n_ri_tensor + n_done_tensor * (DISCOUNT ** ((N_STEP + 1) * discount_tensor)) * h_inv(qin[range(BATCH_SIZE), max_qn]))
            softmax_tensor[range(BATCH_SIZE), a_tensor] = 1.

            self.network_lock.acquire()
            self.predictor.fit([current_state_tensor, past_action_tensor, discount_tensor, beta_tensor],
                               [training_qe2, training_qi2, training_qen, training_qin],
                               batch_size = BATCH_SIZE,
                               verbose = 0,
                               sample_weight = weights)
            self.hasher.fit([hash_tensor, next_hash_tensor],
                            [softmax_tensor],
                            batch_size = BATCH_SIZE,
                            verbose = 0,
                            sample_weight = weights)
            self.rnd_net.fit([hash_tensor],
                             [rnd_target],
                             batch_size = BATCH_SIZE,
                             verbose = 0,
                             sample_weight = weights)
            self.player_hasher.set_weights(self.hasher.get_weights()[:8])
            qe, qi, _, _ = self.predictor([current_state_tensor, past_action_tensor, discount_tensor, beta_tensor])
            qe2, qi2, _, _ = self.predictor([next_state_tensor, next_past_action_tensor, discount_tensor, beta_tensor])
            self.network_lock.release()

            qe = qe.numpy()
            qi = qi.numpy()
            qe2 = qe2.numpy()
            qi2 = qi2.numpy()
            r = re_tensor + beta_tensor * ri_tensor
            q1 = qe[range(BATCH_SIZE), a_tensor] + beta_tensor * qi[range(BATCH_SIZE), a_tensor]
            q2 = qe2 + np.expand_dims(beta_tensor, axis = -1) * qi2
            td = h(r + done_tensor * (DISCOUNT ** discount_tensor) * h_inv(np.max(q2, axis = 1))) - q1
            td = np.power(np.absolute(td), PER_ALPHA) + PER_EPSILON

            self.memory_lock.acquire()
            for i in range(BATCH_SIZE):
                self.tds[batch_indices[i]] = td[i]
            while len(self.actions) > MAX_MEMORY:
                self.cum_steps = self.cum_steps[1:]
                step = self.cum_steps[0]
                for i in range(len(self.cum_steps)):
                    self.cum_steps[i] -= step
                self.states = self.states[step:]
                self.actions = self.actions[step:]
                self.rewards = self.rewards[step:]
                self.discounts = self.discounts[1:]
                self.betas = self.betas[1:]
                self.tds = self.tds[step:]
            self.memory_lock.release()

            self.num_training_episodes += 1
            if (self.num_training_episodes % UPDATE_TARGET_PERIOD) == 0:
                self.update_target()


if __name__ == '__main__':
    players = []
    connections = []
    for i in range(NUM_PLAYERS):
        parent_con, child_con = mp.Pipe()
        epsilon = get_epsilon(i)
        process = mp.Process(target = player_process, args = (child_con, i, epsilon))
        players.append(process)
        connections.append(parent_con)

    for p in players:
        p.start()
        time.sleep(1)

    trainer = Trainer(connections)
    listen_thread = threading.Thread(target = trainer.listen)
    # train_thread = threading.Thread(target = trainer.train)
    listen_thread.start()
    # train_thread.start()
    listen_thread.join()