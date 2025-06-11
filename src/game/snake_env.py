# src/game/snake_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from src.game.snake_game import SnakeGame, Point, BLOCK_SIZE
from src.config import OBS_SPACE_SIZE

class SnakeEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 40}

    def __init__(self, render_mode='human'):
        super().__init__()
        self.game = SnakeGame()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(OBS_SPACE_SIZE,), dtype=np.float32)
        self.last_voice_command_idx = -1 # Initialize voice command state

    def _get_obs(self, voice_command_idx):
        head = self.game.head
        point_l, point_r, point_u, point_d = Point(head.x-BLOCK_SIZE, head.y), Point(head.x+BLOCK_SIZE, head.y), Point(head.x, head.y-BLOCK_SIZE), Point(head.x, head.y+BLOCK_SIZE)
        dir_l, dir_r, dir_u, dir_d = self.game.direction == 1, self.game.direction == 0, self.game.direction == 2, self.game.direction == 3

        # Collision states (danger straight, right, left) and current direction
        collision_dir_state = [
            (dir_r and self.game.is_collision(point_r)) or (dir_l and self.game.is_collision(point_l)) or (dir_u and self.game.is_collision(point_u)) or (dir_d and self.game.is_collision(point_d)),
            (dir_u and self.game.is_collision(point_r)) or (dir_d and self.game.is_collision(point_l)) or (dir_l and self.game.is_collision(point_u)) or (dir_r and self.game.is_collision(point_d)),
            (dir_d and self.game.is_collision(point_r)) or (dir_u and self.game.is_collision(point_l)) or (dir_r and self.game.is_collision(point_u)) or (dir_l and self.game.is_collision(point_d)),
            dir_l, dir_r, dir_u, dir_d
        ]
        
        # Food observation - relative to nearest food
        food_obs = [False, False, False, False] # Default: no food perceived
        
        # 'head' is self.game.head, defined at the start of _get_obs
        nearest_food_item = self.game.find_nearest_food() # Call the new method in SnakeGame
        
        if nearest_food_item:
            food_obs = [
                nearest_food_item.x < head.x,  # Food left
                nearest_food_item.x > head.x,  # Food right
                nearest_food_item.y < head.y,  # Food up
                nearest_food_item.y > head.y   # Food down
            ]
        
        game_state = collision_dir_state + food_obs
        voice_state = [0] * 3
        if voice_command_idx != -1: voice_state[voice_command_idx] = 1
        return np.array(game_state + voice_state, dtype=np.float32)

    def step(self, action: int):
        action_one_hot = [0] * 3; action_one_hot[action] = 1
        
        # Core game step
        reward, game_over, score = self.game.play_step(action_one_hot)

        # Observation is based on the voice command state set externally
        obs = self._get_obs(self.last_voice_command_idx)
        
        return obs, reward, game_over, False, {"score": score}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        self.last_voice_command_idx = -1 # Reset voice command state
        return self._get_obs(self.last_voice_command_idx), {}

    def render(self, score=0, reward=0.0, status=""):
        self.game._update_ui(score, reward, status)
    def close(self): pygame.quit()
