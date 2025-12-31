import matplotlib
import numpy as np
import sys
import time
from collections import defaultdict

from envs import BlackjackEnv
import plotting
matplotlib.style.use('ggplot')

env = BlackjackEnv()

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA  # Initialize the action probabilities
        best_action = np.argmax(Q[observation])      # Find the best action
        A[best_action] += (1.0 - epsilon)            # Add (1 - epsilon) probability to the best action
        return A
    return policy_fn

def mc_first_visit(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        epsilon: Chance to sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities.
    """
    
    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)  # 计数用 int 更合理
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        #########################Implement your code here#########################
        raise NotImplementedError("Not implemented")
        # Step 1: Generate an episode: an array of (state, action, reward) tuples

        # Step 2: Find first-visit index for each (state, action) pair

        # Step 3: Calculate returns backward, update only at first-visit time step
        #########################Implement your code end#########################
    return Q, policy


def mc_every_visit(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.
    """
    
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)  # 计数用 int 更合理
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(1, num_episodes + 1):
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        #########################Implement your code here#########################
        raise NotImplementedError("Not implemented")
        # Step 1: Generate an episode
        
        # Step 2: Calculate returns for each (state, action) pair (every-visit)
        
        #########################Implement your code end#########################

    return Q, policy

if __name__ == "__main__":
    # First-Visit Monte Carlo
    Q, policy = mc_first_visit(env, num_episodes=10000, epsilon=0.1)
    V = defaultdict(float)
    for state, actions in Q.items():
        V[state] = np.max(actions)
    plotting.plot_value_function(V, title="Optimal Value Function", 
        file_name="First_Visit_Value_Function_Episodes_10000")
    
    # Every-Visit Monte Carlo
    # Q, policy = mc_every_visit(env, num_episodes=10000, epsilon=0.1)
    # V = defaultdict(float)
    # for state, actions in Q.items():
    #     V[state] = np.max(actions)
    # plotting.plot_value_function(V, title="Optimal Value Function", 
    #     file_name="Every_Visit_Value_Function_Episodes_10000")