"""
https://gymnasium.farama.org/environments/toy_text/blackjack/
"""
import gymnasium as gym

class BlackjackEnv:
    """Minimal wrapper for gymnasium's Blackjack-v1 with old Gym API."""
    def __init__(self, natural: bool = False, sab: bool = False):
        """
        Args:
            natural: whether to give 1.5 reward for a natural blackjack.
            sab: use Sutton & Barto's book rules if True.
        """
        self._env = gym.make("Blackjack-v1", natural=natural, sab=sab)
        # expose spaces for compatibility
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    def reset(self):
        # only return observation for compatibility
        obs, info = self._env.reset()
        return obs

    def step(self, action):
        # convert new gym output (terminated+truncated) to old (done)
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        return obs, reward, done, info
