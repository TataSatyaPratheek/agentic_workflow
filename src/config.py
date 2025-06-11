# src/config.py

# --- Dynamic Game Settings ---
GAME_WINDOW_TITLE = "Dynamic Snake RL"
BASE_SPEED = 20
SPEED_INCREMENT = 5         # How much speed increases per level
LEVEL_UP_SCORE = 5          # Score needed to advance to the next level
MAZE_DENSITY_INCREMENT = 0.02 # How much denser the maze gets per level (can be kept or removed if direct obstacle count is preferred)
MAX_OBSTACLES = 10          # Maximum number of obstacles in the maze
NUM_FOOD_ITEMS = 3          # Number of food items on screen at a time

# --- Core RL & AI Settings ---
ACTIONS = ["straight", "right", "left"]
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
