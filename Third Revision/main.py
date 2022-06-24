from Hyperparameters import *
import Neural_Networks as nn
from player_process import player_process
import multiprocessing as mp
import threading
import numpy as np
import time
import gc
import tensorflow.keras.backend as K
from timeit import default_timer as timer


def get_epsilon(num):
    e1 = np.log(MAX_EPSILON)
    e2 = np.log(MIN_EPSILON)
    e3 = (e1 - e2) * num / (NUM_PLAYERS - 1) + e2
    return np.exp(e3)


def h(x):
    return np.sign(x) * (np.sqrt(np.absolute(x) + 1.) - 1.) + SQUISH * x


def h_inv(x):
    x = np.clip(x, -Q_CLIP, Q_CLIP)
    a = 4 * SQUISH * (np.absolute(x) + SQUISH + 1.) + 1.
    f1 = (1. - np.sqrt(a)) / (2. * SQUISH)
    f2 = np.absolute(x) + 1.
    return np.sign(x) * (f1 + f2) / SQUISH


class Trainer:
    def __init__(self, connections):
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        self.connections = connections
        self.memory_lock = threading.Lock()
        self.network_lock = threading.Lock()
        self.display_lock = threading.Lock()

        self.predictor = nn.create_trainer_predictor()
        self.target = nn.create_target_predictor()
        self.update_target()

        self.trainer_hasher = nn.create_trainer_hasher()
        self.trainer_forward = nn.create_trainer_forward_model()
        self.hasher = nn.create_hasher()
        self.intrinsic_calc = nn.create_intrinsic_reward_calc()
        self.update_intrinsic_calc_and_hasher()

        self.states = []
        self.actions = []
        self.e_rewards = []
        self.dones = []
        self.arms = []
        self.tds = []

        self.num_training_episodes = 0

    def update_intrinsic_calc_and_hasher(self):
        weights1 = self.trainer_hasher.get_weights()[:8]
        weights2 = self.trainer_forward.get_weights()
        weights = weights1 + weights1 + weights2
        self.intrinsic_calc.set_weights(weights)
        self.hasher.set_weights(weights1)

    def update_memory(self, batch):
        states, actions, e_rewards, dones, tds, arms, data = batch
        self.memory_lock.acquire()
        self.states = self.states + states
        self.actions = self.actions + actions
        self.e_rewards = self.e_rewards + e_rewards
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
        if PRINT_GAME_DATA:
            self.print(["Target Updated"])
        self.network_lock.release()

    def get_weights(self):
        self.network_lock.acquire()
        w0 = self.predictor.get_weights()
        w1 = self.intrinsic_calc.get_weights()
        self.network_lock.release()
        weights = [w0, w1]
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
            state_tensor = np.array([self.states[i][:TRACE_LENGTH, ...].copy() for i in batch_indices], dtype = np.float32) / 255.
            next_state_tensor = np.array([self.states[i][1:, ...].copy() for i in batch_indices], dtype = np.float32) / 255.
            action_tensor = np.array([self.actions[i][1:] for i in batch_indices], dtype = np.int32)
            past_action_tensor = np.array([self.actions[i][:TRACE_LENGTH] for i in batch_indices], dtype = np.int32)
            e_reward_tensor = np.array([self.e_rewards[i] for i in batch_indices], dtype = np.float32)
            done_tensor = np.array([self.dones[i] for i in batch_indices], dtype = np.float32)
            arm_tensor = np.array([self.arms[i] for i in batch_indices], dtype = np.int32)
            self.memory_lock.release()

            beta_tensor = np.array([BETAS[i] for i in arm_tensor], dtype = np.float32)
            gamma_tensor = np.array([GAMMAS[i] for i in arm_tensor], dtype = np.float32)

            weights = 1. - (1. - td_prob[batch_indices].astype(np.float64)) ** BATCH_SIZE
            weights = np.clip(weights, 1e-10, 1.)
            weights = np.min(weights) / weights

            r_state_tensor = np.reshape(state_tensor.copy(), (HASH_SIZE, FRAME_HEIGHT, FRAME_WIDTH))
            r_next_state_tensor = np.reshape(next_state_tensor.copy(), (HASH_SIZE, FRAME_HEIGHT, FRAME_WIDTH))
            r_action_tensor = np.reshape(action_tensor.copy(), HASH_SIZE)
            r_weights = np.repeat(weights, TRACE_LENGTH)
            softmax_tensor = np.zeros((HASH_SIZE, NUM_ACTIONS), dtype = np.float32)
            softmax_tensor[range(HASH_SIZE), r_action_tensor] = 1.

            self.network_lock.acquire()
            state_hash = self.hasher(r_state_tensor).numpy()
            next_state_hash = self.hasher(r_next_state_tensor).numpy()
            self.trainer_hasher.fit([r_state_tensor, r_next_state_tensor],
                                    [softmax_tensor],
                                    batch_size = BATCH_SIZE,
                                    verbose = 0,
                                    sample_weight = r_weights)
            K.clear_session()
            gc.collect()
            self.trainer_forward.fit([state_hash, r_action_tensor],
                                     [next_state_hash],
                                     batch_size = BATCH_SIZE,
                                     verbose = 0,
                                     sample_weight = r_weights)
            K.clear_session()
            gc.collect()
            self.update_intrinsic_calc_and_hasher()

            qe, qi = self.predictor([state_tensor, past_action_tensor, arm_tensor])
            qte, qti = self.target([state_tensor, past_action_tensor, arm_tensor])
            qte2, qti2 = self.target([next_state_tensor, action_tensor, arm_tensor])
            i_reward_tensor = self.intrinsic_calc([r_state_tensor, r_next_state_tensor, r_action_tensor]).numpy()
            self.network_lock.release()

            qe = qe.numpy()
            qi = qi.numpy()
            qte = qte.numpy()
            qti = qti.numpy()
            qte2 = qte2.numpy()
            qti2 = qti2.numpy()
            qt2 = qte2 + beta_tensor[:, None, None] * qti2

            i_reward_tensor = np.reshape(i_reward_tensor, (BATCH_SIZE, TRACE_LENGTH))

            training_qe2 = qe.copy()
            training_qi2 = qi.copy()
            a_opt = np.argmax(qt2, axis = 2)

            ye1 = e_reward_tensor + gamma_tensor[:, None] * done_tensor * h_inv(qte2[I, J, a_opt])
            yi1 = i_reward_tensor + gamma_tensor[:, None] * done_tensor * h_inv(qti2[I, J, a_opt])

            ye2 = np.zeros((BATCH_SIZE, TRACE_LENGTH), dtype = np.float32)
            yi2 = np.zeros((BATCH_SIZE, TRACE_LENGTH), dtype = np.float32)
            c = RETRACE * gamma_tensor[:, None] * (a_opt[:, :-1] == action_tensor[:, :-1]).astype(np.float32)
            c = MASK_1 * np.repeat(c[:, None, :], TRACE_LENGTH - 1, axis = 1)
            c = np.cumprod(c + MASK_2, axis = 2) - MASK_2

            tde1 = e_reward_tensor[range(BATCH_SIZE), 1:] + gamma_tensor[:, None] * h_inv(qte2[I, J[:, :-1], a_opt[:, 1:]])
            tde2 = h_inv(qte[I, J[:, :-1], action_tensor[:, :-1]])
            tde = tde1 - tde2
            tde = np.repeat(tde[:, None, :], TRACE_LENGTH - 1, axis = 1)

            tdi1 = i_reward_tensor[range(BATCH_SIZE), 1:] + gamma_tensor[:, None] * h_inv(qti2[I, J[:, :-1], a_opt[:, 1:]])
            tdi2 = h_inv(qti[I, J[:, :-1], action_tensor[:, :-1]])
            tdi = tdi1 - tdi2
            tdi = np.repeat(tdi[:, None, :], TRACE_LENGTH - 1, axis=1)

            ye2[:, :-1] = np.sum(c * tde, axis = 2)
            yi2[:, :-1] = np.sum(c * tdi, axis = 2)

            training_qe2[I, J, action_tensor] = h(ye1 + ye2)
            training_qi2[I, J, action_tensor] = h(yi1 + yi2)

            self.network_lock.acquire()
            self.predictor.fit([state_tensor, past_action_tensor, arm_tensor],
                               [training_qe2, training_qi2],
                               batch_size = BATCH_SIZE,
                               verbose = 0,
                               sample_weight = weights)
            K.clear_session()
            gc.collect()
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
                self.actions = self.actions[diff:]
                self.e_rewards = self.e_rewards[diff:]
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