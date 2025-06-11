# src/config.py

# --- Dynamic Game Settings ---
GAME_WINDOW_TITLE = "Optimized Agentic Snake"
# --- OPTIMIZATION: Reduce rendering FPS to a reasonable limit ---
# The human eye won't notice much above 60. This frees up CPU cycles.
BASE_SPEED = 600
SPEED_INCREMENT = 50
LEVEL_UP_SCORE = 5
MAX_OBSTACLES = 0
BLOCK_SIZE = 20
NUM_FOOD_ITEMS = 1

# --- Core RL & AI Settings ---
AGENT_ACTIONS = ["straight", "right", "left"]
OBS_SPACE_SIZE = 11
MODEL_PATH = "rllib_snake_model_checkpoints" # Directory for model checkpoints

# --- OPTIMIZATION: Rewards tuned slightly for faster learning ---
REWARD_FOOD = 10.0
REWARD_DEATH = -10.0
DISTRACTION_PENALTY = -2.0

# AI Model Settings
ASR_MODEL = "distil-whisper/distil-small.en" # Explicitly define the model used
LLM_CLASSIFIER = "Recognai/zeroshot_selectra_small"

# UI Settings
INPUT_BOX_RECT = (10, 80, 400, 32)
INPUT_BOX_COLOR_INACTIVE = (100, 100, 100)
INPUT_BOX_COLOR_ACTIVE = (200, 200, 200)
TEXT_COLOR = (255, 255, 255)
GAME_WIDTH, GAME_HEIGHT = 640, 480
