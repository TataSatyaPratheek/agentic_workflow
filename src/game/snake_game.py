# src/game/snake_game.py
import pygame
import random
import numpy as np
from collections import namedtuple

from src.config import * # Import all game settings

# Point is a simple data structure, independent of Pygame
Point = namedtuple('Point', 'x, y')

class SnakeGame:
    """
    A pure, headless-first simulation engine for the Snake game.
    It has zero Pygame dependencies in its core logic (`play_step`, `reset`).
    Graphics are initialized lazily only when explicitly required for rendering.
    """
    def __init__(self, w=640, h=480):
        self.w, self.h = w, h
        # --- CRITICAL: Graphics are NOT initialized here ---
        self.display = None
        self.font = None
        self.clock = None
        # ---
        self.direction_map = {'RIGHT': 0, 'LEFT': 1, 'UP': 2, 'DOWN': 3}
        self.reset()

    def _init_pygame(self):
        """
        Lazily initializes all Pygame components. This is only ever called by
        the SnakeEnv when render_mode is 'human'.
        """
        if self.display is None:
            print("Initializing Pygame for rendering...")
            pygame.init()
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption(GAME_WINDOW_TITLE)
            self.font = pygame.font.SysFont('arial', 25)
            self.clock = pygame.time.Clock()

    def reset(self):
        self.level = 1
        self.speed = BASE_SPEED
        self.direction = "RIGHT"
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head, Point(self.head.x - BLOCK_SIZE, self.head.y)]
        self.score = 0
        self.food_list = []
        self._generate_maze()
        for _ in range(NUM_FOOD_ITEMS):
            self._place_food()
        if self.is_collision(): # Ensure initial state is valid
            self.reset()
        self.frame_iteration = 0
    
    def play_step(self, action: list) -> tuple[int, bool, int]:
        """
        Runs one step of the simulation. This is the high-performance core.
        It has NO Pygame calls.
        """
        self.frame_iteration += 1
        self._move(action)
        self.snake.insert(0, self.head)
        
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = REWARD_DEATH
            return reward, game_over, self.score
        
        food_eaten = False
        for food_item in list(self.food_list):
            if self.head == food_item:
                self.score += 1
                reward = REWARD_FOOD
                self.food_list.remove(food_item)
                self._place_food()
                food_eaten = True
                if self.score > 0 and self.score % LEVEL_UP_SCORE == 0:
                    self.level += 1
                    self.speed += SPEED_INCREMENT
                    self._generate_maze()
                break
        
        if not food_eaten:
            self.snake.pop()
            
        return reward, game_over, self.score

    def _update_ui(self):
        """
        Renders the current game state. This version uses "dirty rect" updating
        for significant performance gains, only redrawing what has changed.
        """
        if self.display is None:
            return  # Headless instance

        # --- OPTIMIZATION: Dirty Rect Rendering ---
        self.display.fill((0, 0, 0)) # Clear the screen once
        dirty_rects = [] # A list to hold rectangles that need updating

        # Draw all components and add their bounding boxes to the dirty list
        for pt in self.snake:
            rect = pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(self.display, (0, 200, 50), rect)
            dirty_rects.append(rect)

        for obs in self.obstacles:
            rect = pygame.Rect(obs.x, obs.y, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(self.display, (100, 100, 100), rect)
            dirty_rects.append(rect)

        for food_item in self.food_list:
            rect = pygame.Rect(food_item.x, food_item.y, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(self.display, (200, 0, 0), rect)
            dirty_rects.append(rect)
        
        score_text = self.font.render(f"Score: {self.score}", True, TEXT_COLOR)
        # We must also update the area where the score is drawn
        score_rect = self.display.blit(score_text, [10, 10])
        dirty_rects.append(score_rect)
        
        # Instead of flip(), update only the changed parts of the screen.
        pygame.display.update(dirty_rects)
        # --- END OF OPTIMIZATION ---

        self.clock.tick(self.speed)

    # --- All other helper methods (_generate_maze, _is_path_available, etc.) remain unchanged ---
    # --- They are pure logic and do not depend on Pygame display. ---
    def _generate_maze(self):
        self.obstacles = []
        num_obstacles_to_place = min(MAX_OBSTACLES, (self.level - 1))
        potential_obstacle_points = [Point(x, y) for x in range(0, self.w, BLOCK_SIZE) for y in range(0, self.h, BLOCK_SIZE) if np.linalg.norm(np.array([x, y]) - np.array([self.w/2, self.h/2])) > BLOCK_SIZE * 3 and Point(x,y) not in self.snake]
        if potential_obstacle_points and num_obstacles_to_place > 0:
            self.obstacles = random.sample(potential_obstacle_points, min(num_obstacles_to_place, len(potential_obstacle_points)))

    def _place_food(self):
        if len(self.food_list) >= NUM_FOOD_ITEMS: return
        max_attempts = (self.w // BLOCK_SIZE) * (self.h // BLOCK_SIZE)
        for _ in range(max_attempts):
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            candidate_food = Point(x, y)
            if candidate_food not in self.snake and candidate_food not in self.obstacles and candidate_food not in self.food_list:
                self.food_list.append(candidate_food)
                return

    def find_nearest_food(self):
        if not self.food_list: return None
        return min(self.food_list, key=lambda f: np.linalg.norm(np.array([f.x, f.y]) - np.array([self.head.x, self.head.y])))

    def is_collision(self, pt=None):
        if pt is None: pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0: return True
        if pt in self.snake[1:]: return True
        if pt in self.obstacles: return True
        return False

    def _move(self, action):
        if np.array_equal(action, [1, 0, 0]): pass
        elif np.array_equal(action, [0, 1, 0]):
            turn_map = {"UP": "RIGHT", "RIGHT": "DOWN", "DOWN": "LEFT", "LEFT": "UP"}
            self.direction = turn_map[self.direction]
        else:
            turn_map = {"UP": "LEFT", "LEFT": "DOWN", "DOWN": "RIGHT", "RIGHT": "UP"}
            self.direction = turn_map[self.direction]
        x, y = self.head.x, self.head.y
        move_map = {"RIGHT": (x + BLOCK_SIZE, y), "LEFT": (x - BLOCK_SIZE, y), "UP": (x, y - BLOCK_SIZE), "DOWN": (x, y + BLOCK_SIZE)}
        next_x, next_y = move_map[self.direction]
        self.head = Point(next_x % self.w, next_y % self.h)
