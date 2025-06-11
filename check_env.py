# check_env.py
import sys
import os
from gymnasium.utils.env_checker import check_env

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.game.snake_env import SnakeEnv

if __name__ == "__main__":
    print("Creating and checking the SnakeEnv...")
    try:
        # Instantiate your custom environment
        env = SnakeEnv()
        
        # This is the official Gymnasium environment checker
        check_env(env.unwrapped)
        
        print("\n✅ Success! The environment passed all checks and is fully compliant.")
        print("You can now proceed with RLlib training.")
    
    except Exception as e:
        print(f"\n❌ Error: The environment failed the check.")
        print(f"Details: {e}")
