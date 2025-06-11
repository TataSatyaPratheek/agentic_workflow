# src/config.py

# --- Snake Game Settings ---
GAME_WINDOW_TITLE = "Snake RL"
# Actions: 0 = Straight, 1 = Right Turn, 2 = Left Turn
ACTIONS = ["straight", "right", "left"]
# Observation space size (11 for game state + 3 for voice command)
OBS_SPACE_SIZE = 14

# --- RL & Reward Settings ---
MODEL_PATH = "snake_ppo_agent.zip"
REWARD_FOOD = 20.0       # Large reward for eating food
REWARD_DEATH = -20.0     # Large penalty for dying
REWARD_CLOSER = 0.1      # Small reward for moving closer to food
REWARD_FARTHER = -0.2    # Small penalty for moving farther away
DISTRACTION_PENALTY = -1.0 # Penalty for obeying a distracting command

# --- AI Model Settings ---
ASR_MODEL = "tiny.en"
# This is a lightweight, working classifier for our task.
LLM_CLASSIFIER = "Recognai/zeroshot_selectra_small"
