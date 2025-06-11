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

    def __init__(self, render_mode: str = None, config: dict = None):
        super().__init__()

        # Determine the effective render_mode
        # Priority:
        # 1. Direct render_mode argument (used by gym.make and check_env)
        # 2. 'render_mode' key in the config dictionary (used by your RLlib env_creator)
        # 3. None (default)
        if render_mode is not None:
            self.render_mode = render_mode
        elif config and "render_mode" in config:
            self.render_mode = config.get("render_mode")
        else:
            self.render_mode = None

        # Store the rest of the config if provided, for other parameters
        self.env_config = config or {}

        # --- THE FINAL FIX: Defer game creation ---
        # The game instance will be created in reset(), where the seeded
        # RNG is guaranteed to be available.
        self.game = None
        # --- END OF FIX ---

        self.action_space = spaces.Discrete(3)  # 0: straight, 1: right, 2: left
        
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(OBS_SPACE_SIZE,), dtype=np.float32
        )

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
        super().reset(seed=seed)
        
        # --- THE FINAL FIX: Create the game instance here ---
        # Now, the game is always created with the correctly seeded self.np_random.
        self.game = SnakeGame(w=GAME_WIDTH, h=GAME_HEIGHT, np_random=self.np_random)
        # --- END OF FIX ---
        
        # This call to self.game.reset() is redundant because SnakeGame's __init__
        # already calls its own reset method. Consider removing it for conciseness.
        self.game.reset() # This now resets the new game instance

        # Get the initial observation and info
        obs = self._get_obs()
        info = {"score": 0} # info should always be a dict

        self.render()
        return obs, info

    # def render(self):
    #     # This is the original render method.
    #     # For full check_env compliance, it's good to ensure it only renders
    #     # when render_mode is 'human'.
    #     self.game._update_ui()

    def render(self):
        if self.render_mode == 'human':
            if self.game is None: # Should be initialized by reset
                return
            
            self.game._init_pygame() # Ensure Pygame is initialized for human mode
            if self.game.display: # Proceed only if display was successfully initialized
                self.game._update_ui()
