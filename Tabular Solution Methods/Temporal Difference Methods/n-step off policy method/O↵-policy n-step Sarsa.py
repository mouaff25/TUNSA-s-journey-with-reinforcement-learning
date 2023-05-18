#!/usr/bin/env python
# coding: utf-8

# This notebook provides an explanation of off policy n-step sarsa algorithm on a simple example. We will work on the environment "Frozen Lake" avalaible on open AI Gym
# 
# Let's begin by initializing the environment and importing necessary packages.

# In[117]:


import sys
import gym
import numpy as np
import random
import math
from collections import defaultdict, deque
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import check_test
from plot_utils import plot_values


# In[118]:


env= gym.make('CliffWalking-v0')


# In[119]:


print(env.action_space)
print(env.observation_space)


# In[120]:


env.reset()


# In[121]:


# define the optimal state-value function
V_opt = np.zeros((4,12))
V_opt[0][0:13] = -np.arange(3, 15)[::-1]
V_opt[1][0:13] = -np.arange(3, 15)[::-1] + 1
V_opt[2][0:13] = -np.arange(3, 15)[::-1] + 2
V_opt[3][0] = -13

plot_values(V_opt)


# <img src="offpolicy_algo.png" alt="Alternative text" />

# In[187]:


from numpy.random import choice
from random import random
def update_Q_sarsa_off_policy(T,taux,alpha, gamma, Q,state,action, states, actions, rewards,b_policy,pi_policy,number):
    rho=1
    n=number
    new_value=Q[state][action]   if state is not None else 0
    if taux >=0 :
        for m in range(taux+1,min(taux+n-1,T-1)+1):
            rho=rho*pi_policy[states[(m%(n+1))]][actions[m%(n+1)]]/(b_policy[states[m%(n+1)]][actions[m%(n+1)]])
        reward=0
        for m in range(taux+1,min(taux+n-1,T-1)+1):
            reward+=gamma**(m-taux-1)*rewards[m%(n+1)]
        if taux+n<T:
            reward+=(gamma**n)*Q[states[(taux+n)%(n+1)]][actions[(taux+n)%(n+1)]] if states[(taux+n)%(n+1)] is not None else 0
            
    
        current = Q[state][action] if state is not None else 0  # estimate in Q-table (for current state, action pair)
        # get value of state, action pair at next time step   
        target = reward                # construct TD target
        new_value = current + (alpha *rho* (target - current)) # get updated value
    return new_value

def epsilon_greedy(Q, state, nA, eps):
    """Selects epsilon-greedy action for supplied state.
    
    Params
    ======
        Q (dictionary): action-value function
        state (int): current state
        nA (int): number actions in the environment
        eps (float): epsilon
    """
    if random() > eps: # select greedy action with probability epsilon
        return np.argmax(Q[state])
    else:                     # otherwise, select an action randomly
        return choice(np.arange(env.action_space.n))
def epsilon_greedy_policy(Q,nA,eps):
    b_policy= defaultdict(lambda: np.zeros(nA))
    for state in range(48):
        for action in [0,1,2,3]:
            if action==np.argmax(Q[state]):
                b_policy[state][action]=(1-eps)
            else:
                b_policy[state][action]=eps/3
    return b_policy
def random_policy(Q,nA):
    b_policy= defaultdict(lambda: np.zeros(nA))
    for state in range(48):
        for action in [0,1,2,3]:
                b_policy[state][action]=1/nA
    return b_policy
def unif(nA):
    return choice(np.arange(nA))


def sarsa(env, num_episodes, alpha,number, gamma=0.9,plot_every=100):
    nA = env.action_space.n                # number of actions
    Q = defaultdict(lambda: np.zeros(nA))  # initialize empty dictionary of arrays
    eps = 0.005  
    b_policy = random_policy(Q,nA)
    pi_policy=epsilon_greedy_policy(Q,nA,eps)# set value of epsilon
    # monitor performance
    tmp_scores = deque(maxlen=plot_every)     # deque for keeping track of scores
    avg_scores = deque(maxlen=num_episodes)   # average scores over every plot_every episodes
    states=list(range(0,number+1))
    actions=list(range(0,number+1))
    for m in range(number+1):
        actions[m]=0
    rewards=list(range(0,number+1))
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()   
        score = 0                                             # initialize score
        state,prob = env.reset()                                   # start episode
        t=0
        T=50
        
        action = unif(nA)            # epsilon-greedy action selection
        
        while True:
            if t<=T-1:
                states[(t%(number+1))]=state
                actions[t%(number+1)]=action
                next_state, reward, done, info,info1 = env.step(action) # take action A, observe R, S'
                score += reward   # add reward to agent's score
                rewards[t%(number+1)]=reward
                if done:
                    T=t+1
                    tmp_scores.append(score)    # append score
                    break
                if not done:
                    next_action = epsilon_greedy(Q, next_state, nA, eps) # epsilon-greedy action
            taux=t-number+1
            state_updated=states[taux%(number+1)]
            action_updated=actions[taux%(number+1)]
            Q[state][action] = update_Q_sarsa_off_policy(T,taux,alpha, gamma, Q,state_updated,action_updated, \
                                                                 states, actions, rewards,b_policy,pi_policy,number)
            state = next_state     # S <- S'
            action = next_action   # A <- A'
            t=t+1
            pi_policy=epsilon_greedy_policy(Q,nA,eps)# set value of epsilon
            if taux==T-1:
                break
        if (i_episode % plot_every == 0):
            avg_scores.append(np.mean(tmp_scores))
           

    # plot performance
    plt.plot(np.linspace(0,num_episodes,len(avg_scores),endpoint=False), np.asarray(avg_scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()
    # print best 100-episode performance
    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(avg_scores))    
    return Q


# In[188]:


# obtain the estimated optimal policy and corresponding action-value function
Q_sarsa = sarsa(env,5000,.01,2)
# print the estimated optimal policy
policy_sarsa = np.array([np.argmax(Q_sarsa[key]) if key in Q_sarsa else -1 for key in np.arange(48)]).reshape(4,12)
check_test.run_check('td_control_check', policy_sarsa)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_sarsa)

# plot the estimated optimal state-value function
V_sarsa = ([np.max(Q_sarsa[key]) if key in Q_sarsa else 0 for key in np.arange(48)])
plot_values(V_sarsa)


# In[ ]:





# In[ ]:




