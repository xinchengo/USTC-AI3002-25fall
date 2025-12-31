import matplotlib
import numpy as np
import pandas as pd
import os
from collections import namedtuple
from matplotlib import pyplot as plt

# Named tuple for episode statistics
EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])

def plot_cost_to_go_mountain_car(env, estimator, num_tiles=20):
    """
    Plot cost-to-go function for Mountain Car environment.
    """
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)

    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Value')
    ax.set_title("Mountain \"Cost To Go\" Function")
    fig.colorbar(surf)
    plt.show()


def plot_value_function(V, title="Value Function", file_name="value_function"):
    """
    Plot value function surface for Blackjack.
    """
    os.makedirs('./figures', exist_ok=True)
    
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title, file_name):
        """Helper function to plot 3D surface."""
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.savefig(f"./figures/{file_name}", bbox_inches='tight')
        plt.close(fig)

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title), f"{file_name}_no_ace.png")
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title), f"{file_name}_ace.png")


def plot_episode_stats(stats, smoothing_window=10, file_name='episode_stats', noshow=True):
    """
    Plot episode statistics: lengths, rewards, and timesteps.
    """
    os.makedirs('./figures', exist_ok=True)
    
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    plt.savefig(f"./figures/{file_name}_length.png", bbox_inches='tight')
    if noshow:
        plt.close(fig1)

    fig2 = plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    plt.savefig(f"./figures/{file_name}_reward.png", bbox_inches='tight')
    if noshow:
        plt.close(fig2)

    fig3 = plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    plt.savefig(f"./figures/{file_name}_timesteps.png", bbox_inches='tight')
    if noshow:
        plt.close(fig3)
    
    return fig1, fig2, fig3

def plot_multiple_episode_stats(stats_dict, smoothing_window=10, file_name='episode_stats_compare', noshow=True):
    """
    Plot comparison of multiple algorithms' episode statistics.
    """
    os.makedirs('./figures', exist_ok=True)
    
    colors = ['blue', 'orange', 'purple', 'red', 'green', 'brown']

    fig1 = plt.figure(figsize=(10, 5))
    for i, (algo_name, stats) in enumerate(stats_dict.items()):
        plt.plot(stats.episode_lengths, color=colors[i], label=algo_name, alpha=0.7)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    plt.legend()
    plt.savefig(f"./figures/{file_name}_length.png", bbox_inches='tight')
    if noshow:
        plt.close(fig1)

    fig2 = plt.figure(figsize=(10, 5))
    for i, (algo_name, stats) in enumerate(stats_dict.items()):
        rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
        plt.plot(rewards_smoothed, color=colors[i], label=algo_name, alpha=0.7)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title(f"Episode Reward over Time (Smoothed over window size {smoothing_window})")
    plt.legend()
    plt.savefig(f"./figures/{file_name}_reward.png", bbox_inches='tight')
    if noshow:
        plt.close(fig2)

    fig3 = plt.figure(figsize=(10, 5))
    for i, (algo_name, stats) in enumerate(stats_dict.items()):
        plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)), color=colors[i], label=algo_name, alpha=0.7)
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    plt.legend()
    plt.savefig(f"./figures/{file_name}_timesteps.png", bbox_inches='tight')
    if noshow:
        plt.close(fig3)

    return fig1, fig2, fig3