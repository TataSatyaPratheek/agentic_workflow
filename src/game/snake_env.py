# src/game/snake_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from src.game.snake_game import SnakeGame

class SnakeEnv(gym.Env):
    """A custom Gymnasium environment for the Snake game."""
    metadata = {'render_modes': ['human'], 'render_fps': 20}

    def __init__(self, render_mode='human'):
        super().__init__()
        self.game = SnakeGame()
        self.render_mode = render_mode
        
        # Action space: [Straight, Right Turn, Left Turn]
        self.action_space = spaces.Discrete(3)
        # Observation space: Use a simple 11-element vector representing state
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.float32)

    def _get_obs(self):
        """Return a vector observation of the game state."""
        head = self.game.head
        # ... (State vector logic, e.g., danger straight/right/left, food direction) ...
        # This is a standard approach to avoid vision-based models for speed.
        # [See PyTorch Snake tutorial for a common implementation of this state vector]
        # For now, a placeholder:
        return np.random.rand(11).astype(np.float32)

    def step(self, action: int):
        # Convert discrete action to one-hot for the game
        action_one_hot = [0, 0, 0]
        action_one_hot[action] = 1
        
        reward, game_over, score = self.game.play_step(action_one_hot)
        obs = self._get_obs()
        
        if self.render_mode == 'human':
            self.render()
            
        return obs, reward, game_over, False, {"score": score}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        return self._get_obs(), {}

    def render(self):
        # The game's _update_ui method already handles rendering
        pass

    def close(self):
        pygame.quit()
