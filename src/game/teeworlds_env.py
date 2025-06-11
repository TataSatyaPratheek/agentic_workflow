# src/game/teeworlds_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2

from src.io.vision import ScreenCapture
from src.game.controller import GameController
from src.config import ACTIONS, OBS_SHAPE

class TeeworldsEnv(gym.Env):
    """A custom Gymnasium environment for Teeworlds."""
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super().__init__()
        self.vision = ScreenCapture()
        self.controller = GameController()
        
        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(*OBS_SHAPE, 1), dtype=np.uint8
        )

    def _get_obs(self):
        """Get and preprocess the game frame for the agent."""
        frame = self.vision.get_frame()
        # Convert to grayscale and resize to save resources
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized_frame = cv2.resize(gray_frame, OBS_SHAPE, interpolation=cv2.INTER_AREA)
        # Add a channel dimension for the CNN policy
        return np.expand_dims(resized_frame, axis=-1)

    def step(self, action: int):
        """Execute one time step within the environment."""
        action_name = ACTIONS[action]
        self.controller.perform_action(action_name)
        
        obs = self._get_obs()
        
        # For this MVP, reward is handled externally by the main loop.
        # Terminated/truncated are always False for a continuous task.
        reward = 0.0
        terminated = False
        truncated = False
        info = {} # Can be used to pass debug info
        
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state."""
        super().reset(seed=seed)
        return self._get_obs(), {}

    def render(self, mode='human'):
        # The game is its own renderer, so this is a no-op.
        pass

    def close(self):
        self.vision.close()
