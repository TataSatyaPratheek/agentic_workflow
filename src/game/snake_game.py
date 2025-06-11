# src/game/snake_game.py
import pygame
import random
import numpy as np
from collections import namedtuple
from src.config import REWARD_FOOD, REWARD_DEATH # Import rewards

pygame.init()

Point = namedtuple('Point', 'x, y')
BLOCK_SIZE = 20
SPEED = 40 # Increase speed for more challenging gameplay

class SnakeGame:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake RL')
        self.clock = pygame.time.Clock()
        self.obstacles = []
        self._place_obstacles()
        self.reset()

    def _place_obstacles(self):
        for i in range(5, 15): self.obstacles.append(Point(i * BLOCK_SIZE, 10 * BLOCK_SIZE))
        for i in range(20, 30): self.obstacles.append(Point(i * BLOCK_SIZE, 15 * BLOCK_SIZE))

    def reset(self):
        self.direction = 3 # 0:Right, 1:Left, 2:Up, 3:Down
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, Point(self.head.x, self.head.y-BLOCK_SIZE)]
        self.score = 0
        self.food = None
        self._place_food()
        if self.is_collision(): self.reset() # Ensure initial state is not game over
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake or self.food in self.obstacles:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); quit()
        
        self._move(action)
        self.snake.insert(0, self.head)
        
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = REWARD_DEATH
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = REWARD_FOOD
            self._place_food()
        else:
            self.snake.pop()
        
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None: pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0: return True
        if pt in self.snake[1:]: return True
        if pt in self.obstacles: return True
        return False

    def _update_ui(self):
        self.display.fill((0,0,0))
        for pt in self.snake: pygame.draw.rect(self.display, (0, 200, 50), pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, (200,0,0), pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        for obs in self.obstacles: pygame.draw.rect(self.display, (100, 100, 100), pygame.Rect(obs.x, obs.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.display.flip()

    def _move(self, action):
        # --- REVISED TURNING LOGIC ---
        # Action: [Straight, Right Turn, Left Turn]
        # This logic is clearer and less error-prone.
        dirs = [0, 1, 2, 3] # R, L, U, D
        current_dir_idx = dirs.index(self.direction)

        if np.array_equal(action, [1, 0, 0]): # Straight
            new_dir = self.direction
        elif np.array_equal(action, [0, 1, 0]): # Right Turn
            turn_map = {0: 3, 3: 1, 1: 2, 2: 0} # R->D, D->L, L->U, U->R
            new_dir = turn_map[self.direction]
        else: # Left Turn
            turn_map = {0: 2, 2: 1, 1: 3, 3: 0} # R->U, U->L, L->D, D->R
            new_dir = turn_map[self.direction]
        self.direction = new_dir
        
        x, y = self.head.x, self.head.y
        if self.direction == 0: x += BLOCK_SIZE
        elif self.direction == 1: x -= BLOCK_SIZE
        elif self.direction == 2: y -= BLOCK_SIZE
        elif self.direction == 3: y += BLOCK_SIZE
        self.head = Point(x, y)
