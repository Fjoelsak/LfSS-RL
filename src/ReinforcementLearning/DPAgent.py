import numpy as np

class DPAgent:

    def __init__(self, env, gamma=.99, theta = 0.01):
        self.state = 0
        self.gamma = gamma
        self.theta = theta
        self.env = env.unwrapped                    # get rid of timeLimit since P is not available
        self.V = np.zeros(env.observation_space.n)

    def value_iteration(self):
        obs, _ = self.env.reset()

        while True:
            delta = 0
            for i in range(self.env.observation_space.n):
                v = self.V[i]
                max_v = float('-inf')
                for a in range(self.env.action_space.n):
                    self.state = i
                    prob, inext, reward, _ = self.env.P[i][a][0]
                    max_v = max(max_v, prob*(reward + self.gamma * self.V[inext]))
                self.V[i] = max_v
                delta = max(delta, abs(v - self.V[i]))

            if delta < self.theta:
                break

        return self.V

    def determine_action(self, state):
        vmax = 0
        best_a = 0
        for a in range(self.env.action_space.n):
            prob, inext, reward, _ = self.env.P[state][a][0]
            if prob*(reward+self.gamma*self.V[inext]) > vmax:
                best_a = a
                vmax = prob*(reward+self.gamma*self.V[inext])

        return best_a
