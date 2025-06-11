# src/game/snake_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from src.game.snake_game import SnakeGame, Point, BLOCK_SIZE
from src.config import OBS_SPACE_SIZE, REWARD_CLOSER, REWARD_FARTHER

class SnakeEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 40}

    def __init__(self, render_mode='human'):
        super().__init__()
        self.game = SnakeGame()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(OBS_SPACE_SIZE,), dtype=np.float32)

    def _get_obs(self, voice_command_idx):
        head = self.game.head
        point_l, point_r, point_u, point_d = Point(head.x-BLOCK_SIZE, head.y), Point(head.x+BLOCK_SIZE, head.y), Point(head.x, head.y-BLOCK_SIZE), Point(head.x, head.y+BLOCK_SIZE)
        dir_l, dir_r, dir_u, dir_d = self.game.direction == 1, self.game.direction == 0, self.game.direction == 2, self.game.direction == 3

        game_state = [
            (dir_r and self.game.is_collision(point_r)) or (dir_l and self.game.is_collision(point_l)) or (dir_u and self.game.is_collision(point_u)) or (dir_d and self.game.is_collision(point_d)),
            (dir_u and self.game.is_collision(point_r)) or (dir_d and self.game.is_collision(point_l)) or (dir_l and self.game.is_collision(point_u)) or (dir_r and self.game.is_collision(point_d)),
            (dir_d and self.game.is_collision(point_r)) or (dir_u and self.game.is_collision(point_l)) or (dir_r and self.game.is_collision(point_u)) or (dir_l and self.game.is_collision(point_d)),
            dir_l, dir_r, dir_u, dir_d,
            self.game.food.x < self.game.head.x, self.game.food.x > self.game.head.x,
            self.game.food.y < self.game.head.y, self.game.food.y > self.game.head.y
        ]
        
        voice_state = [0] * 3
        if voice_command_idx != -1: voice_state[voice_command_idx] = 1
        return np.array(game_state + voice_state, dtype=np.float32)

    def step(self, action: int, voice_command_idx: int = -1):
        action_one_hot = [0] * 3; action_one_hot[action] = 1
        
        dist_before = np.linalg.norm(np.array([self.game.head.x, self.game.head.y]) - np.array([self.game.food.x, self.game.food.y]))
        reward, game_over, score = self.game.play_step(action_one_hot)
        dist_after = np.linalg.norm(np.array([self.game.head.x, self.game.head.y]) - np.array([self.game.food.x, self.game.food.y]))

        if not game_over and reward == 0:
            reward = REWARD_CLOSER if dist_after < dist_before else REWARD_FARTHER
            
        obs = self._get_obs(voice_command_idx)
        if self.render_mode == 'human': self.render()
        return obs, reward, game_over, False, {"score": score}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        return self._get_obs(voice_command_idx=-1), {}

    def render(self): pass
    def close(self): pygame.quit()
