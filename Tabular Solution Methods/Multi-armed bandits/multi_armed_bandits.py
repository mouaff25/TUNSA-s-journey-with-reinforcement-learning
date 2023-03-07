import numpy as np


class KArmedBandit:
    """
    A k-armed bandit problem environment.

    In the k-armed bandit problem, you are presented with k different slot machines, or "one-armed bandits", each with a different payout distribution. Each time you play one of the slot machines, you receive a reward from its payout distribution (normal distributions with different means). The goal is to find the machine with the highest expected payout by playing the machines repeatedly.

    Parameters
    ----------
    k : int, default=10
        The number of different actions.

    max_timesteps : int, default=1000
        The maximum number of timesteps (length of the episode)

    action_values_mean: float, default=0
        The true action value of each of the k actions is selected according to a normal distribution with mean action_values_mean.
    
    action_values_std: float, default=1
        The value of the standard deviation of the distribution of action values.

    rewards_std: float, default=1
        The rewards are selected from a normal distribution with mean q(A) and standard deviation rewards_std

    save_rewards: bool, default=True
        If save_rewards is set to True, each agent's rewards are saved.


    """

    def __init__(self, k=10, max_timesteps=1000, action_values_mean=0, action_values_std=1, rewards_std=1, save_rewards=True):
        self.k = k
        self.max_timesteps = max_timesteps
        self.action_values_std = action_values_std
        self.action_values_mean = action_values_mean
        self.rewards_std = rewards_std
        self.save_rewards = save_rewards
        self.timestep = 0
        self.init()

    def init(self):
        self.timestep = 0
        self.action_values = None
        self.agents = None
        self.rewards = None
        self.init_action_values(
            mean=self.action_values_mean, std=self.action_values_std)

    def init_action_values(self, mean=0, std=1):
        self.action_values = np.random.normal(loc=mean,
                                              scale=std,
                                              size=(self.k,))

    def set_action_values(self, action_values):
        self.action_values = action_values

    def get_action_values(self):
        return self.action_values

    def set_agents(self, agents):
        self.agents = agents
        self.init_rewards()

    def init_rewards(self):
        if self.save_rewards:
            self.rewards = [[] for _ in self.agents]

    def generate_reward(self, action):
        action_value = self.action_values[action]
        return np.random.normal(action_value, self.rewards_std)

    def get_rewards(self):
        return self.rewards

    def next_timestep(self):
        self.timestep += 1
        for i, agent in enumerate(self.agents):
            action = agent.make_action()
            reward = self.generate_reward(action)
            agent.give_reward(reward)
            if self.save_rewards:
                self.rewards[i].append(reward)

    def run(self):
        if self.max_timesteps is not None:
            while self.timestep < self.max_timesteps:
                self.next_timestep()
