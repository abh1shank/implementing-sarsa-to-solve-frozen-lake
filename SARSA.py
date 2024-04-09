import numpy as np

class Sarsa:
    def __init__(self, env, alpha, gamma, epsilon, num_eps):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.state_num = env.observation_space.n
        self.action_num = env.action_space.n
        self.num_eps = num_eps
        self.learned_policy = np.zeros(env.observation_space.n)
        self.Qmatrix = np.zeros((self.state_num, self.action_num))

    def select_action(self, state, index):
        if index < 100:
            return np.random.choice(self.action_num)
        random_num = np.random.random()
        if index > 1000:
            self.epsilon = 0.9 * self.epsilon

        if random_num < self.epsilon:
            return np.random.choice(self.action_num)
        else:
            return np.random.choice(np.where(self.Qmatrix[state, :] == np.max(self.Qmatrix[state, :]))[0])

    def simulate_eps(self):
        for index in range(self.num_eps):
            (S,prob) = self.env.reset()
            A = self.select_action(S, index)
            print(f"episode number : {index+1}")
            terminal = False

            while not terminal:
                S_, reward, terminal, _, _ = self.env.step(A)
                A_ = self.select_action(S_, index)
                if not terminal:
                    error = reward + self.gamma * self.Qmatrix[S_, A_] - self.Qmatrix[S, A]
                else:
                    error = reward - self.Qmatrix[S, A]
                self.Qmatrix[S, A] += self.alpha * error
                S, A = S_, A_

    def compute_final_policy(self):
        for indexS in range(self.state_num):
            self.learned_policy[indexS] = np.random.choice(
                np.where(self.Qmatrix[indexS] == np.max(self.Qmatrix[indexS]))[0])
