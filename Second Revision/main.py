from Hyperparameters import *
import Neural_Networks as nn
from player_process import player_process
import multiprocessing as mp
import threading
import numpy as np
import time
from numba import jit
from timeit import default_timer as timer


def get_epsilon(num):
    e1 = np.log(MAX_EPSILON)
    e2 = np.log(MIN_EPSILON)
    e3 = (e1 - e2) * num / (NUM_PLAYERS - 1) + e2
    return np.exp(e3)


def h(x):
    return np.sign(x) * (np.sqrt(np.absolute(x) + 1.) - 1.) + SQUISH * x


def h_inv(x):
    x = np.clip(x, a_min = -Q_CLIP, a_max = Q_CLIP)
    a = 4 * SQUISH * (np.absolute(x) + SQUISH + 1.) + 1.
    f1 = (1. - np.sqrt(a)) / (2. * (SQUISH ** 2))
    f2 = (np.absolute(x) + 1) / SQUISH
    return np.sign(x) * (f1 + f2)


def get_y2(qte, qti, qte2, qti2, gamma_tensor, action_tensor, a_opt, e_reward_tensor, i_reward_tensor, done_tensor):
    ye2 = np.zeros((BATCH_SIZE, TRACE_LENGTH), dtype = np.float32)
    yi2 = np.zeros((BATCH_SIZE, TRACE_LENGTH), dtype = np.float32)
    for i in range(BATCH_SIZE):
        for j in range(TRACE_LENGTH):
            ye = 0.
            yi = 0.

            ### Trying to speed up nested for loop for retrace loss

            for n in range(j + 1, TRACE_LENGTH):
                c = np.prod(RETRACE * (action_tensor[i, (j + 1):n] == a_opt[i, (j + 1):n]))
                if c != 0.:
                    g = gamma_tensor[i] ** n
                    tde = e_reward_tensor[i, n] + gamma_tensor[i] * done_tensor[i, n] * h_inv(np.max(qte2[i, n])) - h_inv(
                        qte[i, n, action_tensor[i, n]])
                    tdi = i_reward_tensor[i, n] + gamma_tensor[i] * done_tensor[i, n] * h_inv(np.max(qti2[i, n])) - h_inv(
                        qti[i, n, action_tensor[i, n]])
                    ye += g * c * tde
                    yi += g * c * tdi
            ye2[i, j] = ye
            yi2[i, j] = yi
    return ye2, yi2


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
        self.next_states = []
        self.actions = []
        self.past_actions = []
        self.e_rewards = []
        self.i_rewards = []
        self.dones = []
        self.arms = []
        self.tds = []

        self.num_training_episodes = 0

    def update_memory(self, batch):
        states, next_states, actions, past_actions, e_rewards, i_rewards, dones, tds, arms, data = batch
        self.memory_lock.acquire()
        self.states = self.states + states
        self.next_states = self.next_states + next_states
        self.actions = self.actions + actions
        self.past_actions = self.past_actions + past_actions
        self.e_rewards = self.e_rewards + e_rewards
        self.i_rewards = self.i_rewards + i_rewards
        self.dones = self.dones + dones
        self.tds = self.tds + tds
        self.arms = self.arms + arms
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

    def train(self):
        while len(self.actions) < WARM_UP:
            time.sleep(5)
        print('Training Started')
        while True:
            self.memory_lock.acquire()
            td_tensor = np.array(self.tds, dtype = np.float32)
            td_prob = td_tensor / np.sum(td_tensor)
            num_td = td_prob.shape[0]
            batch_indices = np.random.choice(num_td, BATCH_SIZE, p = td_prob, replace = False).astype(np.int32)
            state_tensor = np.array([self.states[i] for i in batch_indices], dtype = np.float32) / 255.
            next_state_tensor = np.array([self.next_states[i] for i in batch_indices], dtype = np.float32) / 255.
            action_tensor = np.array([self.actions[i] for i in batch_indices], dtype = np.int32)
            past_action_tensor = np.array([self.past_actions[i] for i in batch_indices], dtype = np.int32)
            e_reward_tensor = np.array([self.e_rewards[i] for i in batch_indices], dtype = np.float32)
            i_reward_tensor = np.array([self.i_rewards[i] for i in batch_indices], dtype = np.float32)
            done_tensor = np.array([self.dones[i] for i in batch_indices], dtype = np.float32)
            arm_tensor = np.array([self.arms[i] for i in batch_indices], dtype = np.int32)
            self.memory_lock.release()

            weights = (1. - (1. - (1. / num_td)) ** BATCH_SIZE) / (1. - (1. - td_prob[batch_indices]) ** BATCH_SIZE)
            softmax_tensor = np.zeros((BATCH_SIZE * TRACE_LENGTH, NUM_ACTIONS), dtype = np.float32)
            beta_tensor = np.array([BETAS[i] for i in arm_tensor], dtype = np.float32)
            gamma_tensor = np.array([GAMMAS[i] for i in arm_tensor], dtype = np.float32)
            hash_tensor = np.reshape(state_tensor.copy(), (BATCH_SIZE * TRACE_LENGTH, FRAME_HEIGHT, FRAME_WIDTH))
            next_hash_tensor = np.reshape(next_state_tensor.copy(), (BATCH_SIZE * TRACE_LENGTH, FRAME_HEIGHT, FRAME_WIDTH))
            hash_a_tensor = np.reshape(action_tensor.copy(), (BATCH_SIZE * TRACE_LENGTH))
            hash_weights = np.repeat(weights, TRACE_LENGTH)

            self.network_lock.acquire()
            qe, qi = self.predictor([state_tensor, past_action_tensor, arm_tensor])
            qte, qti = self.target([state_tensor, past_action_tensor, arm_tensor])
            qte2, qti2 = self.target([next_state_tensor, action_tensor, arm_tensor])
            rnd_target = self.rnd_target(hash_tensor).numpy()
            self.network_lock.release()

            qe = qe.numpy()
            qi = qi.numpy()
            qte = qte.numpy()
            qti = qti.numpy()
            qte2 = qte2.numpy()
            qti2 = qti2.numpy()
            qt = qte + beta_tensor[:, None, None] * qti

            training_qe2 = qe.copy()
            training_qi2 = qi.copy()
            a_opt = np.argmax(qt, axis = 2)
            I, J = np.ogrid[:BATCH_SIZE, :TRACE_LENGTH]

            ye1 = e_reward_tensor + gamma_tensor[:, None] * done_tensor * h_inv(qte2[I, J, a_opt])
            yi1 = i_reward_tensor + gamma_tensor[:, None] * done_tensor * h_inv(qti2[I, J, a_opt])

            # ye2 = np.zeros((BATCH_SIZE, TRACE_LENGTH), dtype = np.float32)
            # yi2 = np.zeros((BATCH_SIZE, TRACE_LENGTH), dtype = np.float32)
            # for i in range(BATCH_SIZE):
            #     for j in range(TRACE_LENGTH):
            #         ye = 0.
            #         yi = 0.
            #         for n in range(j + 1, TRACE_LENGTH):
            #             g = gamma_tensor[i] ** n
            #             c = np.prod(RETRACE * (action_tensor[i, 1:n] == a_opt[i, 1:n]))
            #             tde = e_reward_tensor[i, n] + gamma_tensor[i] * done_tensor[i, n] * h_inv(np.max(qte2[i, n])) - h_inv(qte[i, n, action_tensor[i, n]])
            #             tdi = i_reward_tensor[i, n] + gamma_tensor[i] * done_tensor[i, n] * h_inv(np.max(qti2[i, n])) - h_inv(qti[i, n, action_tensor[i, n]])
            #             ye += g * c * tde
            #             yi += g * c * tdi
            #         ye2[i, j] = ye
            #         yi2[i, j] = yi
            ye2, yi2 = get_y2(qte, qti, qte2, qti2, gamma_tensor, action_tensor, a_opt, e_reward_tensor, i_reward_tensor, done_tensor)

            training_qe2[I, J, action_tensor] = h(ye1 + ye2)
            training_qi2[I, J, action_tensor] = h(yi1 + yi2)
            softmax_tensor[range(BATCH_SIZE * TRACE_LENGTH), hash_a_tensor] = 1.

            self.network_lock.acquire()
            self.predictor.fit([state_tensor, past_action_tensor, arm_tensor],
                               [training_qe2, training_qi2],
                               batch_size = BATCH_SIZE,
                               verbose = 0,
                               sample_weight = weights)
            self.hasher.fit([hash_tensor, next_hash_tensor],
                            [softmax_tensor],
                            batch_size = BATCH_SIZE,
                            verbose = 0,
                            sample_weight = hash_weights)
            self.rnd_net.fit([hash_tensor],
                             [rnd_target],
                             batch_size = BATCH_SIZE,
                             verbose = 0,
                             sample_weight = hash_weights)
            self.player_hasher.set_weights(self.hasher.get_weights()[:8])
            qe, qi = self.predictor([state_tensor, past_action_tensor, arm_tensor])
            qe2, qi2 = self.predictor([next_state_tensor, action_tensor, arm_tensor])
            self.network_lock.release()

            qe = qe.numpy()
            qi = qi.numpy()
            qe2 = qe2.numpy()
            qi2 = qi2.numpy()
            r = e_reward_tensor + beta_tensor[:, None] * i_reward_tensor
            q1 = qe[I, J, action_tensor] + beta_tensor[:, None] * qi[I, J, action_tensor]
            q2 = qe2 + beta_tensor[:, None, None] * qi2
            td = np.absolute(h(r + done_tensor * gamma_tensor[:, None] * h_inv(np.max(q2, axis = 2))) - q1)
            td = PER_ETA * np.max(td, axis = 1) + (1. - PER_ETA) * np.mean(td, axis = 1) + PER_EPSILON

            self.memory_lock.acquire()
            for i in range(BATCH_SIZE):
                self.tds[batch_indices[i]] = td[i]
            if len(self.actions) > MAX_MEMORY:
                diff = len(self.actions) - MAX_MEMORY

                self.states = self.states[diff:]
                self.next_states = self.next_states[diff:]
                self.actions = self.actions[diff:]
                self.past_actions = self.past_actions[diff:]
                self.e_rewards = self.e_rewards[diff:]
                self.i_rewards = self.i_rewards[diff:]
                self.dones = self.dones[diff:]
                self.tds = self.tds[diff:]
                self.arms = self.arms[diff:]
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
    train_thread = threading.Thread(target = trainer.train)
    listen_thread.start()
    train_thread.start()
    listen_thread.join()