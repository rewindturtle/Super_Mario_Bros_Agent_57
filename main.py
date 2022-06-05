from Hyperparameters import *
import Neural_Networks as nn
from player_process import player_process
import multiprocessing as mp
import threading
import numpy as np
import time
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
        states, actions, rewards, tds, discount, beta = batch
        self.memory_lock.acquire()
        sz = len(actions)
        self.states = self.states + states
        self.actions = self.actions + actions
        self.rewards = self.rewards + rewards
        self.tds = self.tds + tds
        self.discounts.append(discount)
        self.betas.append(beta)
        self.cum_steps.append(self.cum_steps[-1] + sz)
        print(len(self.actions))
        self.memory_lock.release()

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
            i = (i + 1) % NUM_PLAYERS

    def get_batch(self, idx):
        for ep in range(1, len(self.cum_steps)):
            if idx < self.cum_steps[ep]:
                idx_start = self.cum_steps[ep - 1]
                idx_end = self.cum_steps[ep]
                break

        episode_states = np.array(self.states[idx_start:idx_end]).astype(np.float32) / 255.
        episode_actions = np.array(self.actions[idx_start:idx_end])

        diff = idx + 1 - idx_start
        current_state = episode_states[:diff, :, :].copy()
        next_state = episode_states[:min(diff + 1, idx_end - idx_start), :, :].copy()
        n_state = episode_states[:min(diff + N_STEP, idx_end - idx_start), :, :].copy()
        state_len = current_state.shape[0]
        next_state_len = next_state.shape[0]
        n_state_len = n_state.shape[0]

        past_actions = np.zeros((state_len, NUM_ACTIONS), dtype = np.float32)
        past_actions[range(1, state_len), episode_actions[:state_len - 1]] = 1.
        next_past_actions = np.zeros((next_state_len, NUM_ACTIONS), dtype = np.float32)
        next_past_actions[range(1, next_state_len), episode_actions[:next_state_len - 1]] = 1.
        n_past_actions = np.zeros((n_state_len, NUM_ACTIONS), dtype = np.float32)
        n_past_actions[range(1, n_state_len), episode_actions[:n_state_len - 1]] = 1.

        action = self.actions[idx]
        done = ((idx + 1) == idx_end)
        discount = self.discounts[ep - 1]
        beta = self.betas[ep - 1]
        hash_state = self.states[idx].copy().astype(np.float32) / 255.
        if (idx + 1) == idx_end:
            next_hash_state = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype = np.float32)
        else:
            next_hash_state = self.states[idx + 1].copy().astype(np.float32) / 255.

        self.network_lock.acquire()
        hash = self.player_hasher(episode_states).numpy()
        rnd = self.rnd_net(episode_states[1:, :, :]).numpy()
        rnd_t = self.rnd_target(episode_states[1:, :, :]).numpy()
        self.network_lock.release()

        h1 = np.repeat(hash[1:, :].copy(), hash.shape[0], axis = 0)
        h2 = np.tile(hash.copy(), (hash.shape[0] - 1, 1))
        h3 = np.linalg.norm(h1 - h2, axis = 1) ** 2
        h3 = np.reshape(h3[np.newaxis, :], (hash.shape[0] - 1, hash.shape[0]))
        dists = np.sort(h3, axis = 1)[:, 1:(NEAREST_NEIGHBOURS + 1)]
        dists = dists / max(np.mean(dists), 1e-9)
        dists = np.clip(dists - KERNEL_CLUSTER, 0., np.inf)
        kernel = KERNEL_EPSILON / (dists + KERNEL_EPSILON)
        score = np.sqrt(np.sum(kernel, axis=1)) + KERNEL_CONSTANT
        rt = np.where(score > KERNEL_MAX_SCORE, 0, 1. / score)
        err = np.linalg.norm(rnd - rnd_t, axis=1) ** 2
        err_mean = np.mean(err)
        err_std = max(np.std(err), 1e-9)
        rnd_alpha = 1. + (err - err_mean) / err_std
        internal_rewards = rt * np.clip(rnd_alpha, 1., MAX_RND)
        internal_rewards = np.append(internal_rewards, 0.)

        external_reward = self.rewards[idx]
        internal_reward = internal_rewards[idx - idx_start]
        n_external_reward = 0.
        n_internal_reward = 0.
        n_done = False
        for n in range(N_STEP):
            n_external_reward += (DISCOUNT ** (n * discount)) * self.rewards[idx + n]
            n_internal_reward += (DISCOUNT ** (n * discount)) * internal_rewards[idx - idx_start + n]
            if (idx + n + 1) == idx_end:
                n_done = True
                break
        return [current_state, past_actions, action, external_reward, internal_reward, next_state, next_past_actions,
                done, discount, beta, n_state, n_past_actions, n_external_reward, n_internal_reward, n_done, n + 1,
                hash_state, next_hash_state]

    def train(self):
        while len(self.actions) < WARM_UP:
            time.sleep(5)
        print('Training Started')
        while True:
            current_states = []
            past_actions = []
            actions = []
            external_rewards = []
            internal_rewards = []
            next_states = []
            next_past_actions = []
            dones = []
            discounts = []
            betas = []
            n_states = []
            n_past_actions = []
            n_external_rewards = []
            n_internal_rewards = []
            n_dones = []
            n_exps = []
            hash_states = []
            next_hash_states = []

            self.memory_lock.acquire()
            td_array = np.array(self.tds)
            td_prob = td_array / np.sum(td_array)
            num_td = td_prob.shape[0]
            batch_indices = np.random.choice(num_td, BATCH_SIZE, p = td_prob, replace = False)
            for idx in batch_indices:
                batch = self.get_batch(idx)
                current_states.append(batch[0])
                past_actions.append(batch[1])
                actions.append(batch[2])
                external_rewards.append(batch[3])
                internal_rewards.append(batch[4])
                next_states.append(batch[5])
                next_past_actions.append(batch[6])
                dones.append(batch[7])
                discounts.append(batch[8])
                betas.append(batch[9])
                n_states.append(batch[10])
                n_past_actions.append(batch[11])
                n_external_rewards.append(batch[12])
                n_internal_rewards.append(batch[13])
                n_dones.append(batch[14])
                n_exps.append(batch[15])
                hash_states.append(batch[16])
                next_hash_states.append(batch[17])
            self.memory_lock.release()

            weights = (1. - (1. - (1. / num_td)) ** BATCH_SIZE) / (1. - (1. - td_prob) ** BATCH_SIZE)
            softmax_tensor = np.zeros((BATCH_SIZE, NUM_ACTIONS), dtype = np.float32)

            max_s1 = 0.
            max_s2 = 0.
            max_sn = 0.
            for i in range(BATCH_SIZE):
                softmax_tensor[i, actions[i]] = 1.
                max_s1 = max(current_states[i].shape[0], max_s1)
                max_s2 = max(next_states[i].shape[0], max_s2)
                max_sn = max(n_states[i].shape[0], max_sn)

            a_tensor = np.array(actions).astype(int)
            re_tensor = np.array(external_rewards).astype(np.float32)
            ri_tensor = np.array(internal_rewards).astype(np.float32)
            n_re_tensor = np.array(n_external_rewards).astype(np.float32)
            n_ri_tensor = np.array(n_internal_rewards).astype(np.float32)
            done_tensor = np.array(dones).astype(np.float32)
            n_done_tensor = np.array(n_dones).astype(np.float32)
            discount_tensor = np.array(discounts).astype(np.float32)
            beta_tensor = np.array(betas).astype(np.float32)
            n_tensor = np.array(n_exps).astype(np.float32)
            hash_tensor = np.array(hash_states).astype(np.float32)
            next_hash_tensor = np.array(next_hash_states).astype(np.float32)

            for i in range(BATCH_SIZE):
                diff_1 = max_s1 - current_states[i].shape[0]
                diff_2 = max_s2 - next_states[i].shape[0]
                diff_n = max_sn - n_states[i].shape[0]

                shape_1 = (diff_1, FRAME_HEIGHT, FRAME_WIDTH)
                shape_2 = (diff_2, FRAME_HEIGHT, FRAME_WIDTH)
                shape_n = (diff_n, FRAME_HEIGHT, FRAME_WIDTH)

                current_states[i] = np.concatenate((np.zeros(shape_1, dtype = np.float32), current_states[i]))
                next_states[i] = np.concatenate((np.zeros(shape_2, dtype = np.float32), next_states[i]))
                n_states[i] = np.concatenate((np.zeros(shape_n, dtype = np.float32), n_states[i]))

                past_actions[i] = np.concatenate((np.zeros((diff_1, NUM_ACTIONS), dtype = np.float32), past_actions[i]))
                next_past_actions[i] = np.concatenate((np.zeros((diff_2, NUM_ACTIONS), dtype = np.float32), next_past_actions[i]))
                n_past_actions[i] = np.concatenate((np.zeros((diff_n, NUM_ACTIONS), dtype = np.float32), n_past_actions[i]))
            current_state_tensor = np.array(current_states).astype(np.float32)
            next_state_tensor = np.array(next_states).astype(np.float32)
            n_state_tensor = np.array(n_states).astype(np.float32)
            past_action_tensor = np.array(past_actions).astype(np.float32)
            next_past_action_tensor = np.array(next_past_actions).astype(np.float32)
            n_past_action_tensor = np.array(n_past_actions).astype(np.float32)

            self.network_lock.acquire()
            qe, qi, _, _ = self.predictor([current_state_tensor, past_action_tensor, discount_tensor])
            qe2, qi2, _, _ = self.predictor([next_state_tensor, next_past_action_tensor, discount_tensor])
            qen, qin, _, _ = self.predictor([n_state_tensor, n_past_action_tensor, discount_tensor])
            qt2 = self.target([next_state_tensor, next_past_action_tensor, discount_tensor, beta_tensor]).numpy()
            qtn = self.target([n_state_tensor, n_past_action_tensor, discount_tensor, beta_tensor]).numpy()
            rnd_target = self.rnd_target(current_state_tensor).numpy()
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
            training_qen[range(BATCH_SIZE), a_tensor] = h(n_re_tensor + n_done_tensor * (DISCOUNT ** (n_tensor * discount_tensor)) * h_inv(qen[range(BATCH_SIZE), max_qn]))
            training_qin[range(BATCH_SIZE), a_tensor] = h(n_ri_tensor + n_done_tensor * (DISCOUNT ** (n_tensor * discount_tensor)) * h_inv(qin[range(BATCH_SIZE), max_qn]))

            self.network_lock.acquire()
            self.predictor.fit([current_state_tensor, past_action_tensor, discount_tensor],
                               [training_qe2, training_qi2, training_qen, training_qin],
                               batch_size = BATCH_SIZE,
                               verbose = 0,
                               sample_weight = weights)
            self.hasher.fit([hash_tensor, next_hash_tensor],
                            [softmax_tensor],
                            batch_size = BATCH_SIZE,
                            verbose = 0,
                            sample_weight = weights)
            self.rnd_net.fit([hash_states],
                             [rnd_target],
                             batch_size = BATCH_SIZE,
                             verbose = 0,
                             sample_weight = weights)
            self.player_hasher.set_weights(self.hasher.get_weights()[:8])
            qe, qi, _, _ = self.predictor([current_state_tensor, past_action_tensor, discount_tensor])
            qe2, qi2, _, _ = self.predictor([next_state_tensor, next_past_action_tensor, discount_tensor])
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
        print(epsilon)
        process = mp.Process(target = player_process, args = (child_con, epsilon))
        players.append(process)
        connections.append(parent_con)

    for p in players:
        p.start()
        time.sleep(1)

    trainer = Trainer(connections)
    listen_thread = threading.Thread(target = trainer.listen)
    train_thread = threading.Thread(target = trainer.train)
    listen_thread.start()
    train_thread.start()
    listen_thread.join()