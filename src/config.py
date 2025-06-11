# src/config.py

# Model Configurations (using small, fast models)
ASR_MODEL = "tiny.en"  # Model for faster-whisper
LLM_TOKENIZER = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # A small model that uses tokenizers
VLM_MODEL = None  # Placeholder for a small vision model

# Game Control Settings
GAME_WINDOW_TITLE = "Teeworlds" # The exact window title of the game
KEY_PRESS_DURATION = 0.1 # How long to press a key

# RL Settings
REWARD_GOOD_ACTION = 1.0
REWARD_BAD_ACTION = -1.0
