import gym
import matplotlib.pyplot as plt
import numpy as np
import time
from SARSA import Sarsa
env=gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False)
env.reset()
env.observation_space
env.action_space
alpha = 0.1
gamma = 0.9
epsilon = 0.2
number_episodes = 100000
sarsa_agent = Sarsa(env, alpha, gamma, epsilon, number_episodes)
sarsa_agent.simulate_eps()
sarsa_agent.compute_final_policy()
final_learned_policy = sarsa_agent.learned_policy
matrix=sarsa_agent.Qmatrix
aspect_ratio = matrix.shape[1] / matrix.shape[0]
plt.imshow(matrix, cmap='viridis', interpolation='nearest', aspect=aspect_ratio)
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        plt.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', color='white')
plt.colorbar()
plt.title('Q-Matrix')
plt.show()
while True:
    env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode='human')
    current_state, _ = env.reset()
    env.render()
    time.sleep(2)

    terminal_state = False
    for _ in range(100):
        if not terminal_state:
            current_state, _, terminal_state, _, _ = env.step(int(final_learned_policy[current_state]))
            time.sleep(1)
        else:
            break
    time.sleep(0.5)
env.close()
