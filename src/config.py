# src/config.py

# Model Configurations (using small, fast models)
ASR_MODEL = "tiny.en"  # Model for faster-whisper
LLM_TOKENIZER = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # A small model that uses tokenizers
LLM_CLASSIFIER = "distilbert-base-uncased-finetuned-sst-2-english" # A very small text classifier for intent
VLM_MODEL = None  # Placeholder for a small vision model

# Game Control Settings
GAME_WINDOW_TITLE = "Teeworlds" # The exact window title of the game
KEY_PRESS_DURATION = 0.1 # How long to press a key

# RL Settings
# Define the discrete actions the agent can take
ACTIONS = ["left", "right", "jump", "shoot", "idle"]
REWARD_IMITATION_SUCCESS = 1.0
REWARD_IMITATION_FAILURE = -1.0
# Observation shape for the RL model (small and grayscale)
OBS_SHAPE = (84, 84)
