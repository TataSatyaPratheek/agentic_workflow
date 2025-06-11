# check_env.py
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import sys, os

# --- FIX: Add project root to sys.path ---
# Since check_env.py is in the project root, os.path.dirname(os.path.abspath(__file__)) gives the project root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# --- END FIX ---
from src.game.snake_env import SnakeEnv

if __name__ == "__main__":
    print("Creating and checking the SnakeEnv...")
    try:
        env = gym.make("Snake-v1")
        snake = SnakeEnv(config={"render_mode": None})
        check_env(env.unwrapped)
        print("\n✅ Success! The environment passed all checks and is fully compliant.")

        print("You can now proceed with RLlib training.")
    
    except Exception as e:
        print(f"\n❌ Error: The environment failed the check.")
        print(f"Details: {e}")
