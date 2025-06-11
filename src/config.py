# src/config.py

# --- Dynamic Game Settings ---
GAME_WINDOW_TITLE = "Dynamic Snake RL"
BASE_SPEED = 20
SPEED_INCREMENT = 5         # How much speed increases per level
LEVEL_UP_SCORE = 5          # Score needed to advance to the next level
MAZE_DENSITY_INCREMENT = 0.02 # How much denser the maze gets per level (can be kept or removed if direct obstacle count is preferred)
MAX_OBSTACLES = 10          # Maximum number of obstacles in the maze
BLOCK_SIZE = 20             # Size of each block (snake segment, food, obstacle) in pixels
NUM_FOOD_ITEMS = 3          # Number of food items on screen at a time

# --- Core RL & AI Settings ---
AGENT_ACTIONS = ["straight", "right", "left"]
DISTRACTION_ACTIONS = ["up", "down", "left", "right", "straight"]
OBS_SPACE_SIZE = 14
MODEL_PATH = "dynamic_snake_ppo_agent.zip"

# Rewards & Penalties
REWARD_FOOD = 25.0
REWARD_DEATH = -25.0
REWARD_CLOSER = 0.2
REWARD_FARTHER = -0.3
DISTRACTION_PENALTY = -2.0

# AI Model Settings
ASR_MODEL = "tiny.en"
LLM_CLASSIFIER = "Recognai/zeroshot_selectra_small"

# --- NEW: UI Settings for Text Input ---
INPUT_BOX_RECT = (10, 80, 400, 32) # Position and size (x, y, width, height)
INPUT_BOX_COLOR_INACTIVE = (100, 100, 100)
INPUT_BOX_COLOR_ACTIVE = (200, 200, 200)
TEXT_COLOR = (255, 255, 255) # Default color for text rendering (e.g., score)
