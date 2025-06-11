# src/game/snake_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from src.game.snake_game import SnakeGame, Point, BLOCK_SIZE # Point and BLOCK_SIZE are used in _get_obs
from src.config import OBS_SPACE_SIZE, GAME_WIDTH, GAME_HEIGHT

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

        config = config or {}
        self.render_mode = config.get("render_mode")
        
        self.game = SnakeGame(w=GAME_WIDTH, h=GAME_HEIGHT)
        
        self.action_space = spaces.Discrete(3)  # 0: straight, 1: right, 2: left
        
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(OBS_SPACE_SIZE,), dtype=np.float32
        )

        if self.render_mode == 'human':
            self.game._init_pygame()

    def _get_obs(self) -> np.ndarray:
        """Generates an observation purely from the current game state."""
        head = self.game.head
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)
        
        # Corrected direction checking:
        # self.game.direction is a string (e.g., "LEFT", "RIGHT").
        # Compare directly with string literals.
        dir_l = self.game.direction == "LEFT"
        dir_r = self.game.direction == "RIGHT"
        dir_u = self.game.direction == "UP"
        dir_d = self.game.direction == "DOWN"

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
        
        reward, terminated, truncated, score = self.game.play_step(action_one_hot)
        obs = self._get_obs()

        self.render()
        
        return obs, reward, terminated, truncated, {"score": score}


    def reset(self, *, seed=None, options=None):
        # This line is CRITICAL. It seeds self.np_random which is used by the game engine.
        super().reset(seed=seed)
        
        # Now that the RNG is seeded, reset the game state
        self.game.reset()
        
        # Get the initial observation and info
        obs = self._get_obs()
        info = {"score": 0} # info should always be a dict

        self.render()
        return obs, info

    def render(self):
        self.game._update_ui()
