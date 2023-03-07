import numpy as np


class EpsilonGreedyAgent:
    """
    An epsilon-greedy agent for the k-armed bandit problem.

    In the epsilon-greedy algorithm, the agent selects the action with the highest estimated reward (i.e., the exploitation step) with probability 1-ε, where ε is a small value (e.g., 0.1). However, with probability ε, the agent selects a random action instead (i.e., the exploration step).

    Parameters
    ----------
    name: string, default=None
        Name of the agent. If name=None, the name attribute of the agent becomes agent_{i} with i the current number of agents.

    k: string, default=10
        The number of actions.

    eps: float, default=0.1
        The parameter ε of the algorithm, must be a float between 0 and 1, with higher values prioritizing exploration.
    """

    n_agents = 0

    def __init__(self, name=None, k=10, eps=0.1):

        EpsilonGreedyAgent.n_agents += 1
        if name is None:
            name = 'agent_'+str(EpsilonGreedyAgent.n_agents)
        self.name = name
        self.k = k
        self.eps = eps
        self.action_space = [i for i in range(k)]
        self.init()

    def init(self):
        self.action_value_estimates = None
        self.action_timesteps = None
        self.action = None
        self.timestep = 0
        self.init_action_value_estimates()

    def get_name(self):
        return self.name

    def init_action_value_estimates(self):
        self.action_value_estimates = np.zeros((self.k,))
        self.action_timesteps = [0 for _ in range(self.k)]

    def get_action_value_estimates(self):
        return self.action_value_estimates

    def get_policy_action(self):
        self.action = np.argmax(self.action_value_estimates)

    def make_action(self):
        """
        A method that choses the action with best estimated action value (with probability 1-ε), or choses a random action (with probability ε), then returns it.
        """
        self.timestep += 1
        eps = np.random.uniform()
        if eps >= self.eps:
            self.get_policy_action()
        else:
            self.action = np.random.choice(self.action_space)
        self.action_timesteps[self.action] += 1
        return self.action

    def give_reward(self, reward):
        """
        A method that takes the current timestep's reward as a parameter and updates the estimated action value for the action selected earlier in the timestep.
        """
        action_value_estimate = self.action_value_estimates[self.action]
        action_timestep = self.action_timesteps[self.action]
        action_value_estimate += (reward -
                                  action_value_estimate) / action_timestep
        self.action_value_estimates[self.action] = action_value_estimate


class OptimisticAgent(EpsilonGreedyAgent):
    """
    An agent that implements the optimistic initial values approach.

    An optimistic initial values agent is a type of reinforcement learning agent that uses optimistic estimates of the expected rewards for different actions. In this approach, the agent initially assumes that all actions have a high expected reward, which encourages exploration of the environment. As the agent takes actions and receives feedback, it updates its estimates of the expected rewards for each action and gradually becomes more accurate.

    Parameters
    ----------
    name: string, default=None
        Name of the agent. If name=None, the name attribute of the agent becomes agent_{i} with i the current number of agents.

    k: string, default=10
        The number of actions.
    """
    def __init__(self, name=None, k=10):
        super().__init__(name, k, 0)

    def init_action_value_estimates(self):
        self.action_value_estimates = np.ones((self.k,)) * 5
        self.action_timesteps = [0 for _ in range(self.k)]


class UpperConfidenceBoundAgent(EpsilonGreedyAgent):
    """
    An Upper Confidence Bound agent.

    An Upper Confidence Bound (UCB) agent is a type of reinforcement learning agent that uses a strategy based on balancing exploration and exploitation. The UCB algorithm selects actions by assigning an upper confidence bound to each action's expected reward based on previous experience.

    Parameters
    ----------
    name: string, default=None
        Name of the agent. If name=None, the name attribute of the agent becomes agent_{i} with i the current number of agents.

    k: string, default=10
        The number of actions.
    """
    def __init__(self, name=None, k=10, c=2):
        self.c = c
        super().__init__(name, k, 0)

    def get_policy_action(self):
        if 0 in self.action_timesteps:
            self.action = np.random.choice(
                np.where(np.array(self.action_timesteps) == 0)[0])
        else:
            self.action = np.argmax(self.action_value_estimates + self.c*np.sqrt(
                np.log(self.timestep) / np.array(self.action_timesteps)))


class GradientBanditAgent(EpsilonGreedyAgent):
    """
    The Gradient Bandit Algorithm selects actions based on the probabilities assigned to each arm, with the probability of selecting each arm being proportional to the exponential of its estimated action-value, with a normalization factor to ensure that the probabilities sum to one.

    Parameters
    ----------
    name: string, default=None
        Name of the agent. If name=None, the name attribute of the agent becomes agent_{i} with i the current number of agents.
    
    k: string, default=10
        The number of actions.
    """
    def __init__(self, name=None, k=10, step_size=0.1, with_baseline=True):
        self.with_baseline = with_baseline
        self.step_size = step_size
        super().__init__(name, k, 0)

    def init(self):
        self.preferences = None
        self.policy = None
        self.action_timesteps = None
        self.action = None
        self.timestep = 0
        self.expected_reward = 0
        self.init_preferences()
        self.update_policy()

    def init_preferences(self):
        self.preferences = np.zeros((self.k,))
        self.action_timesteps = [0 for _ in range(self.k)]

    def update_policy(self):
        self.policy = np.exp(self.preferences) / \
            np.sum(np.exp(self.preferences))

    def get_policy_action(self):
        self.action = np.random.choice(self.action_space, p=self.policy)

    def give_reward(self, reward):
        # updates preferences and policy
        if self.with_baseline:
            self.expected_reward += (reward -
                                     self.expected_reward) / self.timestep
        for i, preference in enumerate(self.preferences):
            if i == self.action:
                preference += self.step_size * \
                    (reward - self.expected_reward)*(1 - self.policy[i])
            else:
                preference -= self.step_size * \
                    (reward - self.expected_reward)*self.policy[i]
            self.preferences[i] = preference
        self.update_policy()
