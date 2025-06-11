# src/game/snake_game.py
import pygame
import random
import numpy as np
from collections import namedtuple, deque
from src.config import * # Import all new settings

pygame.init()
Point = namedtuple('Point', 'x, y')
BLOCK_SIZE = 20
TEXT_COLOR = (255, 255, 255)

class SnakeGame:
    def __init__(self, w=640, h=480):
        self.w, self.h = w, h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption(GAME_WINDOW_TITLE)
        self.font = pygame.font.SysFont('arial', 25)
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.level = 1
        self.speed = BASE_SPEED
        self.direction = 3 # Start moving down
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, Point(self.head.x, self.head.y-BLOCK_SIZE)]
        self.score = 0
        self.food = None
        self.food_move_timer = 0
        self._generate_maze()
        self._place_food()
        if self.is_collision(): self.reset() # Ensure initial state is not game over
        self.frame_iteration = 0

    def _generate_maze(self):
        """Generates a random maze using a simple randomized algorithm."""
        self.obstacles = []
        density = min(0.3, self.level * MAZE_DENSITY_INCREMENT) # Cap density at 30%
        for x in range(0, self.w, BLOCK_SIZE):
            for y in range(0, self.h, BLOCK_SIZE):
                if random.random() < density:
                    # Avoid creating obstacles near the center spawn point
                    if np.linalg.norm(np.array([x, y]) - np.array([self.w/2, self.h/2])) > BLOCK_SIZE * 5:
                        self.obstacles.append(Point(x, y))

    def _is_path_available(self, start, end):
        """Checks if a path exists from start to end using Breadth-First Search (BFS)."""
        q = deque([start])
        visited = {start}
        while q:
            current = q.popleft()
            if current == end:
                return True
            for dx, dy in [(0, BLOCK_SIZE), (0, -BLOCK_SIZE), (BLOCK_SIZE, 0), (-BLOCK_SIZE, 0)]:
                neighbor = Point(current.x + dx, current.y + dy)
                if 0 <= neighbor.x < self.w and 0 <= neighbor.y < self.h and \
                   neighbor not in visited and \
                   not self.is_collision(neighbor):
                    visited.add(neighbor)
                    q.append(neighbor)
        return False

    def _place_food(self):
        """Places food in a random location that is reachable by the snake."""
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food = Point(x, y)
            if self.food not in self.snake and self.food not in self.obstacles:
                # IMPORTANT: Ensure the food is reachable
                if self._is_path_available(self.head, self.food):
                    break
    
    def _move_food(self):
        """Moves the food to a new random valid position."""
        if self.level >= FOOD_MOVE_THRESHOLD:
            self.food_move_timer += 1
            if self.food_move_timer >= FOOD_MOVE_INTERVAL:
                self.food_move_timer = 0
                self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); quit()
        
        self._move(action)
        self.snake.insert(0, self.head)
        self._move_food() # Check if food should move this frame
        
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
            # Level Up Logic
            if self.score % LEVEL_UP_SCORE == 0:
                self.level += 1
                self.speed += SPEED_INCREMENT
                self._generate_maze() # Generate a new, harder maze
                # TTS should announce this in the main loop
        else:
            self.snake.pop()
        
        # self._update_ui() # Rendering will be handled by the agent script
        self.clock.tick(self.speed)
        return reward, game_over, self.score


    def is_collision(self, pt=None):
        if pt is None: pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0: return True
        if pt in self.snake[1:]: return True
        if pt in self.obstacles: return True
        return False

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

    def _draw_stats(self, text, x, y):
        """Helper to draw text on the screen."""
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        text_surface = font.render(text, True, (255, 255, 255)) # White text
        self.display.blit(text_surface, (x, y))

    def _update_ui(self, score=0, reward=0.0, status="Starting..."):
        """Updated to include drawing live stats."""
        self.display.fill((0,0,0))
        for pt in self.snake:
            pygame.draw.rect(self.display, (0, 200, 50), pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, (200,0,0), pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        for obs in self.obstacles:
            pygame.draw.rect(self.display, (100, 100, 100), pygame.Rect(obs.x, obs.y, BLOCK_SIZE, BLOCK_SIZE))

        self._draw_stats(f"Score: {score}", 10, 10)
        self._draw_stats(f"Last Reward: {reward:.2f}", 10, 30)
        self._draw_stats(f"Status: {status}", 10, 50)
        
        pygame.display.flip()
