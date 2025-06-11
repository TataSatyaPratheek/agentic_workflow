# src/config.py

# --- Dynamic Game Settings ---
GAME_WINDOW_TITLE = "Dynamic Snake RL"
BASE_SPEED = 20
SPEED_INCREMENT = 5         # How much speed increases per level
LEVEL_UP_SCORE = 5          # Score needed to advance to the next level
MAZE_DENSITY_INCREMENT = 0.02 # How much denser the maze gets per level
FOOD_MOVE_THRESHOLD = 3     # The level at which food starts moving
FOOD_MOVE_INTERVAL = 50     # Food moves every 50 frames

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
