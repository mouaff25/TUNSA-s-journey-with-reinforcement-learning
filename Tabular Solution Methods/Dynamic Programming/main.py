import gym_examples
from gym_examples.envs.grid_world import GridWorldEnv
import gym
import numpy as np
import time

SIZE = 4
N_TARGETS = 2
DISCOUNT_TERM = 1.0
THETA = 1e-3
DECIMAL_DIGITS = 2
ALGORITHM = 'policy_iteration'   # DP algorithm (policy_iteration or value_iteration)


class StateSpace:
    def __iter__(self):
        self.i = 0
        self.j = -1
        return self
    
    def __next__(self):
        if self.j >= SIZE - 1:
            self.j = -1
            if self.i >= SIZE - 1:
                self.i = 0
                raise StopIteration
            else:
                self.i += 1
        self.j += 1
        return (self.i, self.j)

def iterative_policy_evaluation(obs, policy_dict, state_value_dict, discount_term, theta):
    state_space = iter(StateSpace())
    delta = float('inf')
    while delta >= theta:
        # Iterate over all states (agent_location)
        delta = 0
        for state in state_space:
            obs['agent'] = np.array(state)
            if GridWorldEnv.goal_state(obs):
                continue
            value = state_value_dict[state]
            new_state_value = 0
            for action, probability in policy_dict[state].items():
                result_obs, reward, _, _ = GridWorldEnv.transition_model(obs, action, SIZE)
                result_state = tuple(result_obs['agent'])
                new_state_value += probability * (reward + discount_term*state_value_dict[result_state])
            
            delta = max(delta, abs(value - new_state_value))
            state_value_dict[state] = new_state_value
    
    # Reduce float precision
    for state in state_space:
        state_value_dict[state] = round(state_value_dict[state], DECIMAL_DIGITS)
    return delta

def policy(policy_dict, state):
    max_probability = float('-inf')
    max_action = None
    for action, probability in policy_dict[state].items():
        if probability > max_probability:
            max_probability = probability
            max_action = action
    return max_action

def policy_improvement(obs, policy_dict, state_value_dict, discount_term):
    policy_stable = True
    state_space = iter(StateSpace())
    for state in state_space:
        obs['agent'] = np.array(state)
        if GridWorldEnv.goal_state(obs):
            continue
        policy_action = policy(policy_dict, state)
        max_action_value = float('-inf')
        max_actions = []

        for action in policy_dict[state].keys():
            result_obs, reward, _, _ = GridWorldEnv.transition_model(obs, action, SIZE)
            result_state = tuple(result_obs['agent'])
            action_value = reward + discount_term*state_value_dict[result_state]
            if action_value > max_action_value:
                max_action_value = action_value
                max_actions.clear()
            if action_value == max_action_value:
                max_actions.append(action)

        if policy_action not in max_actions:
            policy_stable = False

        for action in policy_dict[state].keys():
            if action not in max_actions:
                policy_dict[state][action] = 0
            else:
                policy_dict[state][action] = 1 / len(max_actions)
            
        
    return policy_stable
            

def policy_iteration(obs, discount_term):
    obs = obs.copy()
    equiprobable_policy = {i: 0.25 for i in  range(4)}  # action: probability dictionary
    state_space = iter(StateSpace())
    policy_dict = {state: equiprobable_policy.copy() for state in state_space}
    state_value_dict = {state: 0 for state in state_space}

    policy_stable = False
    while not policy_stable:
        iterative_policy_evaluation(obs, policy_dict, state_value_dict, discount_term, THETA)
        policy_stable = policy_improvement(obs, policy_dict, state_value_dict, discount_term)

    return policy_dict


def value_iteration(obs, discount_term):
    obs = obs.copy()
    equiprobable_policy = {i: 0.25 for i in  range(4)}  # action: probability dictionary
    state_space = iter(StateSpace())
    policy_dict = {state: equiprobable_policy.copy() for state in state_space}
    state_value_dict = {state: 0 for state in state_space}

    iterative_policy_evaluation(obs, policy_dict, state_value_dict, discount_term, THETA)
    _ = policy_improvement(obs, policy_dict, state_value_dict, discount_term)

    return policy_dict

def agent_fn(policy_dict, obs):
    state = tuple(obs['agent'])
    return policy(policy_dict, state)


def main():
    assert ALGORITHM in ['value_iteration', 'policy_iteration']
    assert 0 < THETA <= 1

    env = gym.make('gym_examples/GridWorld-v1', size=SIZE, n_targets=N_TARGETS, render_mode='human')
    num_episodes = 1

    for _ in range(num_episodes):
        obs, _ = env.reset()
        if ALGORITHM == 'value_iteration':
            policy_dict = value_iteration(obs, discount_term=DISCOUNT_TERM)
        else:
            policy_dict = policy_iteration(obs, discount_term=DISCOUNT_TERM)
            
        done = False
        while not done:
            action = agent_fn(policy_dict, obs)
            obs, reward, done, truncated, info = env.step(action)
            # Render the env
            env.render()
            # Wait a bit before the next frame unless you want to see a crazy fast video
            time.sleep(0.1)

    # Close the env
    env.close()


if __name__ == "__main__":
    main()


