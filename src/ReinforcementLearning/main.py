import gymnasium as gym
import pygame
from DPAgent import DPAgent
from TDAgent import TDAgent

"""
The algorithms value iteration, q-learning and SARSA are implemented
"""
algorithm = 'Q-learning'

env = gym.make('Taxi-v3')

if algorithm == 'Value_iteration':
    agent = DPAgent(env)
    agent.value_iteration()
else:
    agent = TDAgent(env)
    agent.train(algorithm)


env = gym.make('Taxi-v3', render_mode='human')
obs, _ = env.reset()

done = False

while not done:
    action = agent.determine_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated
    env.render()

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                pygame.quit()
                done = True

env.close()