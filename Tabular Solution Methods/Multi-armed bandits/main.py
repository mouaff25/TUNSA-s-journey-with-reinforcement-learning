from multi_armed_bandits import KArmedBandit
from agents import EpsilonGreedyAgent, OptimisticAgent, UpperConfidenceBoundAgent, GradientBanditAgent
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

np.random.seed(0)

n_simulations = 2000
env1 = KArmedBandit()
agent11 = EpsilonGreedyAgent(name='realistic, epsilon-greedy (eps=0.1)')
agent12 = OptimisticAgent(name='optimistic, greedy')
agent13 = UpperConfidenceBoundAgent(name='Upper-Confidence-Bound (c=2)')
agent14 = GradientBanditAgent(name='Gradient Bandit with baseline (alpha=0.1)')
agents1 = [agent11, agent12, agent13, agent14]
n_agents1 = len(agents1)
rewards1 = np.zeros((1, n_agents1, 1000))

env2 = KArmedBandit(action_values_mean=4)
agent21 = GradientBanditAgent(
    name='alpha=0.1, with baseline', step_size=0.1, with_baseline=True)
agent22 = GradientBanditAgent(
    name='alpha=0.1, without baseline', step_size=0.1, with_baseline=False)
agent23 = GradientBanditAgent(
    name='alpha=0.4, with baseline', step_size=0.4, with_baseline=True)
agent24 = GradientBanditAgent(
    name='alpha=0.4, without baseline', step_size=0.4, with_baseline=False)
agents2 = [agent21, agent22, agent23, agent24]
n_agents2 = len(agents2)
rewards2 = np.zeros((1, n_agents2, 1000))


time_steps = [i for i in range(1000)]


for _ in tqdm(range(n_simulations)):
    env1.init()
    env2.init()
    for agent in agents1:
        agent.init()
    for agent in agents2:
        agent.init()
    env1.set_agents(agents1)
    env2.set_agents(agents2)
    env1.run()
    env2.run()

    current_rewards1 = np.array(
        env1.get_rewards()).reshape((1, n_agents1, 1000))
    current_rewards2 = np.array(
        env2.get_rewards()).reshape((1, n_agents2, 1000))

    rewards1 = np.concatenate((rewards1, current_rewards1), axis=0)
    rewards2 = np.concatenate((rewards2, current_rewards2), axis=0)


rewards1 = np.mean(rewards1[1:, :, :], axis=0)
rewards2 = np.mean(rewards2[1:, :, :], axis=0)

plt.figure(1)
for i, agent in enumerate(agents1):
    plt.plot(time_steps, rewards1[i], label=agent.get_name())
plt.title(f'Agents average rewards over {n_simulations} runs')
plt.xlabel('steps')
plt.ylabel('Average reward')
plt.legend()
plt.show()

plt.figure(2)
for i, agent in enumerate(agents2):
    plt.plot(time_steps, rewards2[i], label=agent.get_name())
plt.title(f'Average performance of the gradient-bandit algorithm with and without baseline on the 10-armed testbed (action_values_mean=4)')
plt.xlabel('steps')
plt.ylabel('Average reward')
plt.legend()
plt.show()
