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
        self.foods = [] # Changed from self.food to self.foods (a list)
        self._generate_maze()
        self._place_initial_foods() # New method to place multiple food items
        if self.is_collision(): self.reset() # Ensure initial state is not game over
        self.frame_iteration = 0

    def _generate_maze(self):
        """Generates a random maze. Obstacles increase with level, capped by MAX_OBSTACLES."""
        self.obstacles = []
        
        # Number of obstacles increases with level, e.g., level 1 = 0, level 2 = 1, ..., up to MAX_OBSTACLES
        num_obstacles_to_place = min(MAX_OBSTACLES, (self.level - 1))

        potential_obstacle_points = []
        for x_coord in range(0, self.w, BLOCK_SIZE):
            for y_coord in range(0, self.h, BLOCK_SIZE):
                # Avoid center spawn area and snake's initial position
                if np.linalg.norm(np.array([x_coord, y_coord]) - np.array([self.w/2, self.h/2])) > BLOCK_SIZE * 3:
                    if Point(x_coord, y_coord) not in self.snake: # Check against current snake state during reset
                         potential_obstacle_points.append(Point(x_coord, y_coord))
        
        if potential_obstacle_points and num_obstacles_to_place > 0:
            num_to_actually_place = min(num_obstacles_to_place, len(potential_obstacle_points))
            self.obstacles = random.sample(potential_obstacle_points, num_to_actually_place)

    def _is_path_available(self, start, end):
        """Checks if a path exists from start to end using Breadth-First Search (BFS)."""
        q = deque([start])
        visited = {start}
        while q:
            current_node = q.popleft()
            if current_node == end:
                return True
            for dx, dy in [(0, BLOCK_SIZE), (0, -BLOCK_SIZE), (BLOCK_SIZE, 0), (-BLOCK_SIZE, 0)]:
                neighbor = Point(current_node.x + dx, current_node.y + dy)
                if 0 <= neighbor.x < self.w and 0 <= neighbor.y < self.h and \
                   neighbor not in visited and \
                   not self.is_collision(neighbor) and \
                   neighbor not in self.obstacles: # BFS should consider obstacles
                    visited.add(neighbor)
                    q.append(neighbor)
        return False

    def _place_initial_foods(self):
        """Places the initial set of food items."""
        self.foods = []
        for _ in range(NUM_FOOD_ITEMS):
            self._add_new_food()

    def _add_new_food(self):
        """Adds a single new food item to a random, reachable, and valid location."""
        if len(self.foods) >= NUM_FOOD_ITEMS:
            return # Max food items reached

        max_attempts = (self.w // BLOCK_SIZE) * (self.h // BLOCK_SIZE)
        for _ in range(max_attempts):
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            candidate_food = Point(x, y)
            if candidate_food not in self.snake and \
               candidate_food not in self.obstacles and \
               candidate_food not in self.foods: # Check against other food items
                if self._is_path_available(self.head, candidate_food):
                    self.foods.append(candidate_food)
                    return
        print("Warning: Could not place a new reachable food item after max attempts.")

    def find_nearest_food(self):
        """Finds the food item closest to the snake's head."""
        if not self.foods:
            return None
        
        # self.head is Point(x,y)
        # food items in self.foods are also Point(x,y)
        # Calculate distance from self.head to each food item
        nearest = min(self.foods, 
                      key=lambda food_item: np.linalg.norm(
                          np.array([food_item.x, food_item.y]) - np.array([self.head.x, self.head.y])
                      ))
        return nearest

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
        
        food_eaten = False
        for food_item in list(self.foods): # Iterate over a copy for safe removal
            if self.head == food_item:
                self.score += 1
                reward = REWARD_FOOD
                self.foods.remove(food_item)
                self._add_new_food() # Replenish food
                food_eaten = True

                # Level Up Logic
                if self.score > 0 and self.score % LEVEL_UP_SCORE == 0: # Ensure level up only on score multiples
                    self.level += 1
                    self.speed += SPEED_INCREMENT
                    self._generate_maze() # Generate a new, harder maze
                break # Eat one food per step
        
        if not food_eaten:
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

        # Screen wrapping logic
        if x < 0:
            x = self.w - BLOCK_SIZE  # Wrap to right edge
        elif x >= self.w:
            x = 0  # Wrap to left edge
        
        if y < 0:
            y = self.h - BLOCK_SIZE  # Wrap to bottom edge
        elif y >= self.h:
            y = 0  # Wrap to top edge
            
        self.head = Point(x,y)

    def _draw_stats(self, text, x, y):
        """Helper to draw text on the screen."""
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        text_surface = font.render(text, True, (255, 255, 255)) # White text
        self.display.blit(text_surface, (x, y))

    def _draw_input_box(self, rect, text, font, is_active):
        """Helper to draw the text input box."""
        color = INPUT_BOX_COLOR_ACTIVE if is_active else INPUT_BOX_COLOR_INACTIVE
        pygame.draw.rect(self.display, color, rect, 2)
        text_surface = font.render(text, True, (255, 255, 255))
        # Render text inside the box, with a small padding
        self.display.blit(text_surface, (rect.x + 5, rect.y + 5))

    def _update_ui(self, score=0, reward=0.0, status="Starting...", input_text="", input_active=False):
        """Updated to include drawing live stats and the text input box."""
        self.display.fill((0,0,0))
        for pt in self.snake:
            pygame.draw.rect(self.display, (0, 200, 50), pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        for obs in self.obstacles:
            pygame.draw.rect(self.display, (100, 100, 100), pygame.Rect(obs.x, obs.y, BLOCK_SIZE, BLOCK_SIZE))
        for food_item in self.foods: # Draw all food items
            pygame.draw.rect(self.display, (200,0,0), pygame.Rect(food_item.x, food_item.y, BLOCK_SIZE, BLOCK_SIZE))

        self._draw_stats(f"Score: {score}", 10, 10)
        self._draw_stats(f"Last Reward: {reward:.2f}", 10, 30)
        self._draw_stats(f"Status: {status}", 10, 50)
        
        # --- Draw the new input box ---
        input_box_pygame_rect = pygame.Rect(INPUT_BOX_RECT)
        # Use the same font as _draw_stats or define a new one if needed
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        self._draw_input_box(input_box_pygame_rect, input_text, font, input_active)
        
        pygame.display.flip()
