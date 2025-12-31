"""
https://gymnasium.farama.org/environments/toy_text/cliff_walking/
"""
import gymnasium as gym

class CliffWalkingEnv:
    """Thin wrapper around gymnasium's CliffWalking-v1 with old Gym-style API."""
    def __init__(self, is_slippery: bool = False):
        """
        Args:
            is_slippery: whether to use slippery cliff (official param, default False).
        Sets render_mode="ansi" to print text grid in terminal.
        """
        self._env = gym.make(
            "CliffWalking-v1",
            is_slippery=is_slippery,
            render_mode="ansi"
        )
        # expose action and observation space for compatibility
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    def reset(self):
        # only return observation for compatibility
        obs, info = self._env.reset()
        return obs

    def step(self, action):
        # convert new gym output to old form (done, not terminated+truncated)
        obs, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

    def render(self, mode: str = "human"):
        # simple text render: returns grid string (from gymnasium render)
        out = self._env.render()
        if mode == "human":
            # print grid to stdout if output is string
            if isinstance(out, str):
                print(out, end="")
        return out
