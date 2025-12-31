import matplotlib
import numpy as np
import sys

from collections import defaultdict
from envs import CliffWalkingEnv
import plotting

matplotlib.style.use('ggplot')

env = CliffWalkingEnv()

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
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def make_epsilon_greedy_policy_double(Q1, Q2, epsilon, nA):
    """
    Creates an epsilon-greedy policy for Double Q-learning based on Q1+Q2.
    
    Args:
        Q1: First Q-function dictionary.
        Q2: Second Q-function dictionary.
        epsilon: The probability to select a random action. Float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes an observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    """
    def policy_fn(observation):
        # Use Q1 + Q2 for action selection (more stable and standard)
        combined_Q = Q1[observation] + Q2[observation]
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(combined_Q)
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn


def q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1, max_steps=10000):
    """
    Q-Learning algorithm: Off-policy TD control. Finds the optimal greedy policy
    while following an epsilon-greedy policy
    
    Args:
        env: OpenAI environment.
        num_episodes: Number of episodes to run for.
        discount_factor: Gamma discount factor.
        alpha: TD learning rate.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
        max_steps: Maximum number of steps per episode (safety limit).
    
    Returns:
        A tuple (Q, episode_lengths).
        Q is the optimal action-value function, a dictionary mapping state -> action values.
        stats is an EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes)
    )    
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(num_episodes):
        # Print out which episode we're on, useful for debugging.
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        
        # Reset the environment
        state = env.reset()

        # One step in the environment (with max_steps safety limit)
        for t in range(max_steps):
            #########################Implement your code here#########################
            raise NotImplementedError("Not implemented")
            # step 1 : Take a step

            # step 2 : TD Update (with terminal handling)

            # step 3 : Move to next state and handle episode end

            #########################Implement your code end#########################
    return Q, stats


def double_q_learning(env, num_episodes, discount_factor=1.0, alpha=0.5, epsilon=0.1, max_steps=10000):
    Q1 = defaultdict(lambda: np.zeros(env.action_space.n))
    Q2 = defaultdict(lambda: np.zeros(env.action_space.n))
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    
    # Use combined Q1+Q2 for behavior policy (standard Double Q-learning)
    policy = make_epsilon_greedy_policy_double(Q1, Q2, epsilon, env.action_space.n)

    for i_episode in range(num_episodes):
        if (i_episode + 1) % 100 == 0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()

        state = env.reset()

        for t in range(max_steps):
            #########################Implement your code here#########################       
            raise NotImplementedError("Not implemented")
            # step 1 : Take a step using combined Q1+Q2 policy

            # step 2 : Double Q-learning update

            # step 3 : Move to next state and handle episode end
            #########################Implement your code end#########################
                
    return Q1, Q2, stats


if __name__ == '__main__':
    # Q-Learning
    Q, stats = q_learning(env, 1000)
    plotting.plot_episode_stats(stats, file_name='episode_stats_q_learning')
    
    # Double Q-Learning
    # Q1, Q2, stats = double_q_learning(env, 1000)
    # plotting.plot_episode_stats(stats, file_name='episode_stats_double_q_learning')
