# check_env.py
import sys
import os
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.game.snake_env import SnakeEnv

if __name__ == "__main__":
    print("Creating and checking the SnakeEnv...")
    try:
        # --- THE FIX: Instantiate using gymnasium.make() ---
        env = gym.make("Snake-v1")
        # --- END OF FIX ---
        
        check_env(env.unwrapped)
        print("\n✅ Success! The environment passed all checks and is fully compliant.")

        print("You can now proceed with RLlib training.")
    
    except Exception as e:
        print(f"\n❌ Error: The environment failed the check.")
        print(f"Details: {e}")
