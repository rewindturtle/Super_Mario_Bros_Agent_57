from Hyperparameters import *
import super_mario_bros_env
import ray
import numpy as np
import os


ray.init(num_cpus = os.cpu_count())


@ray.remote
class Player():
    def __init__(self, epsilon, level = 0):
        from Neural_Networks import create_player_predictor
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        self.env = super_mario_bros_env.make(level)
        self.epsilon = epsilon
        self.predictor = create_player_predictor()
        self.discount = 0.9
        self.beta = 0.3

    def _choose_action(self, state, past_action):
        s = np.expand_dims(state.copy().astype(np.float32) / 255., axis=0)
        pa = np.zeros((1, NUM_ACTIONS), dtype=np.float32)
        pa[0][past_action] = 1.
        d = np.array([[self.discount]])
        b = np.array([[self.beta]])
        a = self.predictor([s, pa, d, b]).numpy()[0]
        print(a)
        if (np.random.random() < self.epsilon):
            a = np.random.randint(NUM_ACTIONS)
        return a

    def play(self):
        self.predictor.reset_states()
        state = self.env.reset()
        a = 0
        if RENDER:
            self.env.render()
        while True:
            a = self._choose_action(state, a)
            state, x, done, info = self.env.step(a)
            if RENDER:
                self.env.render()
            if done:
                break
        return x


if __name__ == '__main__':
    players = []
    for i in range(4):
        p = Player.remote(0.1)
        players.append(p)
    x = []
    for p in players:
        x.append(p.play.remote())
    print(ray.get(x))