import super_mario_bros_env
import ray
import os


ray.init(num_cpus = os.cpu_count())


@ray.remote
class Player():
    def __init__(self, level=0):
        self.env = super_mario_bros_env.make(level)

    def play(self):
        state = self.env.reset()
        while True:
            a = self.env.action_space.sample()
            state, x, done, info = self.env.step(a)
            if done:
                break
        return x


if __name__ == '__main__':
    players = [Player.remote() for i in range(4)]
    x = []
    for p in players:
        x.append(p.play.remote())
    print(ray.get(x))