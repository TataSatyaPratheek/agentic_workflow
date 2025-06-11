# src/game/snake_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from src.game.snake_game import SnakeGame, Point, BLOCK_SIZE
from src.config import OBS_SPACE_SIZE

class SnakeEnv(gym.Env):
    """
    A canonical, parallel-safe Gymnasium environment for the Snake game,
    designed for Ray RLlib.

    - It is stateless: It does not hold external information like voice commands.
    - It is headless-first: It only initializes Pygame and renders if specifically
      configured with `render_mode='human'`.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, config: dict = None):
        super().__init__()
        # Config is passed by RLlib runners. Default to render_mode=None.
        config = config or {}
        self.render_mode = config.get("render_mode")
        
        self.game = SnakeGame()
        self.action_space = spaces.Discrete(3)  # 0: straight, 1: right, 2: left
        
        # The observation space is now purely about the game state.
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(OBS_SPACE_SIZE,), dtype=np.float32
        )

        # Only initialize Pygame if we are in human-render mode.
        # This prevents headless workers from loading graphics.
        if self.render_mode == 'human':
            self.game._init_pygame()

    def _get_obs(self) -> np.ndarray:
        """Generates an observation purely from the current game state."""
        head = self.game.head
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        dir_l = self.game.direction == self.game.direction_map['LEFT']
        dir_r = self.game.direction == self.game.direction_map['RIGHT']
        dir_u = self.game.direction == self.game.direction_map['UP']
        dir_d = self.game.direction == self.game.direction_map['DOWN']

        # Danger ahead, danger right, danger left
        danger_state = [
            (dir_r and self.game.is_collision(point_r)) or (dir_l and self.game.is_collision(point_l)) or (dir_u and self.game.is_collision(point_u)) or (dir_d and self.game.is_collision(point_d)),
            (dir_u and self.game.is_collision(point_r)) or (dir_d and self.game.is_collision(point_l)) or (dir_l and self.game.is_collision(point_u)) or (dir_r and self.game.is_collision(point_d)),
            (dir_d and self.game.is_collision(point_r)) or (dir_u and self.game.is_collision(point_l)) or (dir_r and self.game.is_collision(point_u)) or (dir_l and self.game.is_collision(point_d)),
        ]

        # Current direction one-hot
        direction_state = [dir_l, dir_r, dir_u, dir_d]

        # Food location relative to head
        nearest_food = self.game.find_nearest_food()
        food_state = [0, 0, 0, 0] # Default: no food perceived
        if nearest_food:
            food_state = [
                nearest_food.x < head.x,  # Food left
                nearest_food.x > head.x,  # Food right
                nearest_food.y < head.y,  # Food up
                nearest_food.y > head.y   # Food down
            ]
        
        obs = danger_state + direction_state + food_state
        return np.array(obs, dtype=np.float32)

    def step(self, action: int):
        action_one_hot = [0] * 3
        action_one_hot[action] = 1
        
        # Use the new, correct return values from the game engine
        reward, terminated, truncated, score = self.game.play_step(action_one_hot)
        obs = self._get_obs()

        self.render()
        
        # Pass terminated and truncated directly to RLlib
        return obs, reward, terminated, truncated, {"score": score}


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        obs = self._get_obs()

        # We now call render here, similar to step().
        # self.game._update_ui() handles headless mode gracefully.
        self.render()
        return obs, {"score": 0}

    def render(self):
        # This single call now handles rendering safely.
        # self.game._update_ui() checks if Pygame components (like self.game.display)
        # are initialized and does nothing if they aren't.
        self.game._update_ui()

    # The close() method remains unchanged as it correctly handles Pygame quitting
    # only when in 'human' render mode.
