import numpy as np
import random

class TDAgent:

    def __init__(self, env, gamma=0.7, alpha = 0.2, eps = 0.1, num_eps = 1000):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.num_eps = num_eps
        self.Q = np.zeros([env.observation_space.n, env.action_space.n])

    def train(self, algo):

        for episode in range(self.num_eps):
            done = False
            total_reward = 0
            st, _ = self.env.reset()

            if random.uniform(0, 1) < self.eps:
                at = self.env.action_space.sample()
            else:
                at = np.argmax(self.Q[st])

            while not done:
                stp1, r, terminated, truncated, info = self.env.step(at)

                done = terminated or truncated

                if random.uniform(0, 1) < self.eps:
                    atp1 = self.env.action_space.sample()
                else:
                    atp1 = np.argmax(self.Q[stp1])

                if algo == "Q-learning":
                    valtp1 = np.max(self.Q[stp1])
                else:
                    valtp1 = self.Q[stp1][atp1]
                valt = self.Q[st][at]

                self.Q[st][at] = valt + self.alpha * (r + self.gamma*valtp1 - valt)

                total_reward += r
                st = stp1
                at = atp1

            if episode % 100 == 0:
                print("Episode {} Total Reward: {}".format(episode, total_reward))

    def determine_action(self, state):
        return np.argmax(self.Q[state])