#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from Environment import StochasticWindyGridworld
from Agent import BaseAgent

class QLearningAgent(BaseAgent):

    def update(self,s,a,r,s_next,done):
        if done:
            target = r
        else:
            target = r + self.gamma * np.max(self.Q_sa[s_next])

        self.Q_sa[s, a] += self.learning_rate * (target - self.Q_sa[s, a])

# def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
#     ''' runs a single repetition of q_learning
#     Return: rewards, a vector with the observed rewards at each timestep '''
#
#     env = StochasticWindyGridworld(initialize_model=False)
#     eval_env = StochasticWindyGridworld(initialize_model=False)
#     agent = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
#     eval_timesteps = []
#     eval_returns = []
#
#
#
#     # TO DO: Write your Q-learning algorithm here!
#
#     # if plot:
#     #    env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Q-learning execution
#
#
#     return np.array(eval_returns), np.array(eval_timesteps)

def q_learning(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True, eval_interval=500):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep '''
    # print("Running Q-learning with policy {}, epsilon {}, temp {}".format(policy, epsilon, temp))

    env = StochasticWindyGridworld(initialize_model=False)
    eval_env = StochasticWindyGridworld(initialize_model=False)
    agent = QLearningAgent(env.n_states, env.n_actions, learning_rate, gamma)
    eval_timesteps = []
    eval_returns = []

    s = env.reset()

    for timestep in range(0, n_timesteps):
        # Sample action using exploration policy
        a = agent.select_action(s, policy=policy, epsilon=epsilon, temp=temp)

        # Simulate environment
        s_next, r, done = env.step(a)

        # Q-learning update
        agent.update(s, a, r, s_next, done)

        # Evaluate policy at eval_interval steps
        if timestep % eval_interval == 0:
            mean_return = agent.evaluate(eval_env, n_eval_episodes=30, max_episode_length=100)
            eval_returns.append(mean_return)
            eval_timesteps.append(timestep)

        # Reset environment if episode is done
        if done:
            s = env.reset()
        else:
            s = s_next

        if plot:
            env.render(Q_sa=agent.Q_sa, plot_optimal_policy=True, step_pause=0.001)
    # try:
    #     print("Rendering finished — close the figure window to exit.")
    #     plt.show(block=True)
    # except Exception:
    #     input("Rendering finished. Press Enter to exit.")

    return np.array(eval_returns), np.array(eval_timesteps)


def test():
    
    n_timesteps = 50001
    eval_interval= 1000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax'
    epsilon = 0.03
    temp = 0.01
    
    # Plotting parameters
    plot = True

    eval_returns, eval_timesteps = q_learning(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot, eval_interval)
    # print(eval_returns,eval_timesteps)

if __name__ == '__main__':
    test()
