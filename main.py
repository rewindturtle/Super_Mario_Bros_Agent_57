from Hyperparameters import *
import Neural_Networks as nn
from player_process import player_process
import multiprocessing as mp
import threading
import numpy as np
import time


def get_epsilon(num):
    e1 = np.log(MAX_EPSILON)
    e2 = np.log(MIN_EPSILON)
    e3 = (e1 - e2) * num / (NUM_PLAYERS - 1) + e2
    return np.exp(e3)


class Trainer:
    def __init__(self, connections):
        self.connections = connections
        self.memory_lock = threading.Lock()
        self.network_lock = threading.Lock()

        self.predictor = nn.create_trainer_predictor()
        self.target = nn.create_target_predictor()

        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.tds = []
        self.discounts = []
        self.betas = []
        self.steps = [0]
        self.cum_steps = [0]

    def update_memory(self, batch):
        states, actions, rewards, dones, tds, discount, beta = batch
        self.memory_lock.acquire()
        sz = len(actions)
        self.states = self.states + states
        self.actions = self.actions + actions
        self.rewards = self.rewards + rewards
        self.dones = self.dones + dones
        self.tds = self.tds + tds
        self.discounts.append(discount)
        self.betas.append(beta)
        self.steps.append(sz)
        self.cum_steps.append(self.cum_steps[-1] + sz)
        self.memory_lock.release()

    def update_target(self):
        self.network_lock.acquire()
        self.target.set_weights(self.predictor.get_weights())
        self.network_lock.release()

    def get_weights(self):
        self.network_lock.acquire()
        weights = self.predictor.get_weights()
        self.network_lock.release()
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
            i = (i + 1) % NUM_PLAYERS


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
    t = threading.Thread(target = trainer.listen)
    t.start()
    t.join()