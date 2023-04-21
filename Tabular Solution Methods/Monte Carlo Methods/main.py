from time import sleep
import random
import gym
from tqdm import tqdm
from collections import defaultdict
import numpy as np

RANDOM_SEED = 42
DISCOUNT_FACTOR = .9

def create_random_policy(env):
    p = {action: 1 / env.action_space.n for action in range(env.action_space.n)}
    policy = {obs: p.copy() for obs in range(env.observation_space.n)}
    return policy

def create_state_action_dictionary(env):
    q = {a: 0.0 for a in range(env.action_space.n)}
    Q = {obs: q.copy() for obs in range(env.observation_space.n)}
    return Q


def run_game(env, policy, exploring_start=False, display=False):
    env.reset()
    episode = []
    finished = False

    while not finished:
        s = env.env.s
        if display:
            env.render()
            sleep(1)

        timestep = []
        timestep.append(s)
        if exploring_start and not episode:
            action = random.randrange(env.action_space.n)
        else:
            chose_action = False
            n = random.uniform(0, sum(policy[s].values()))
            top_range = 0
            for prob in policy[s].items():
                top_range += prob[1]
                if n < top_range:
                    chose_action = True
                    action = prob[0]
                    break
            if not chose_action:
                action = env.action_space.n - 1
        state, reward, finished, truncated, info = env.step(action)
        timestep.append(action)
        timestep.append(reward)

        episode.append(timestep)

    if display:
        env.render()
        sleep(1)
    return episode



def test_policy(policy, env):
    wins = 0
    r = 100
    for _ in range(r):
        w = run_game(env, policy, display=False)[-1][-1]
        if w == 1:
            wins += 1
    return wins / r




def monte_carlo_es(env, episodes=100, policy=None):
    if policy is None:
        policy = create_random_policy(env)  # Create an empty dictionary to store state action values    
    Q = create_state_action_dictionary(env) # Empty dictionary for storing rewards for each state-action pair
    returns = {}
    
    for _ in tqdm(range(episodes)): # Looping through episodes
        G = 0 # Store cumulative reward in G (initialized at 0)
        episode = run_game(env=env, policy=policy, exploring_start=True, display=False) # Store state, action and value respectively 
        
        # for loop through reversed indices of episode array. 
        # The logic behind it being reversed is that the eventual reward would be at the end. 
        # So we have to go back from the last timestep to the first one propagating result from the future.
        for i in reversed(range(0, len(episode))):
            s_t, a_t, r_t = episode[i] 
            state_action = (s_t, a_t)
            G = G*DISCOUNT_FACTOR + r_t # Increment total reward by reward on current timestep
            
            if not state_action in [(x[0], x[1]) for x in episode[0:i]]: # 
                if returns.get(state_action):
                    returns[state_action].append(G)
                else:
                    returns[state_action] = [G]   
                    
                Q[s_t][a_t] = sum(returns[state_action]) / len(returns[state_action]) # Average reward across episodes
                
                Q_list = list(map(lambda x: x[1], Q[s_t].items())) # Finding the action with maximum value
                indices = [i for i, x in enumerate(Q_list) if x == max(Q_list)]
                
                for a in policy[s_t].items(): # Update action probability for s_t in policy
                    if a[0] in indices:
                        policy[s_t][a[0]] = 1 / len(indices)
                    else:
                        policy[s_t][a[0]] = 0
    return policy


def monte_carlo_e_soft(env, episodes=100, policy=None, epsilon=0.01):
    if policy is None:
        policy = create_random_policy(env)  # Create an empty dictionary to store state action values    
    Q = create_state_action_dictionary(env) # Empty dictionary for storing rewards for each state-action pair
    returns = {}
    
    for _ in tqdm(range(episodes)): # Looping through episodes
        G = 0 # Store cumulative reward in G (initialized at 0)

        episode = run_game(env=env, policy=policy, exploring_start=False, display=False) # Store state, action and value respectively 
        
        # for loop through reversed indices of episode array. 
        # The logic behind it being reversed is that the eventual reward would be at the end. 
        # So we have to go back from the last timestep to the first one propagating result from the future.
        
        for i in reversed(range(0, len(episode))):   
            s_t, a_t, r_t = episode[i] 
            state_action = (s_t, a_t)
            G = G*DISCOUNT_FACTOR + r_t # Increment total reward by reward on current timestep

            
            if not state_action in [(x[0], x[1]) for x in episode[0:i]]: # 
                if returns.get(state_action):
                    returns[state_action].append(G)
                else:
                    returns[state_action] = [G]   
                    
                Q[s_t][a_t] = sum(returns[state_action]) / len(returns[state_action]) # Average reward across episodes
                
                Q_list = list(map(lambda x: x[1], Q[s_t].items())) # Finding the action with maximum value
                indices = [i for i, x in enumerate(Q_list) if x == max(Q_list)]
                max_Q = random.choice(indices)
                
                A_star = max_Q
                
                for a in policy[s_t].items(): # Update action probability for s_t in policy
                    if a[0] == A_star:
                        policy[s_t][a[0]] = 1 - epsilon + (epsilon / abs(sum(policy[s_t].values())))
                    else:
                        policy[s_t][a[0]] = (epsilon / abs(sum(policy[s_t].values())))
    return policy


#####################################################################################################

def random_policy(nA):
    A = np.ones(nA, dtype=float) / nA
    def policy_fn(observation):
        return A
    return policy_fn

def greedy_policy(Q):
    def policy_fn(state):
        A = np.zeros_like(Q[state], dtype=float)
        best_action = np.argmax(Q[state])
        A[best_action] = 1.0
        return A
    return policy_fn

def monte_carlo_off_policy(env, behavior_policy=None, episodes=100):
    if behavior_policy is None:
        behavior_policy = create_random_policy(env)
    Q = defaultdict(lambda:np.ones(env.action_space.n) / env.action_space.n)
    C = defaultdict(lambda:np.zeros(env.action_space.n))

    target_policy = greedy_policy(Q)

    for _ in tqdm(range(episodes)):
        episode = run_game(env=env, policy=behavior_policy, display=False)
        
        
        G = 0.0
        W = 1.0
        for t in range(len(episode))[::-1]:
            state, action, reward = episode[t]
            G = DISCOUNT_FACTOR * G + reward
            C[state][action] += W
            Q[state][action] += (W / C[state][action]) * (G - Q[state][action])
            if action !=  np.argmax(target_policy(state)):
                break
            W = W * 1./behavior_policy[state][action]
    target_policy = {}
    for state in range(env.observation_space.n):
        target_policy[state] = {a: prob for a, prob in enumerate(Q[state])}
    return target_policy


def main():
    random.seed(RANDOM_SEED)
    env = gym.make('FrozenLake-v1', is_slippery=False)
    policy = monte_carlo_off_policy(env, episodes=5000)
    # env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='human')
    win_percentage = test_policy(policy, env) * 100
    print(f'Winning Percentage: {win_percentage:.2f}%')

if __name__ == '__main__':
    main()