# scripts/run_agent.py (Simplified for Self-Play)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game.snake_env import SnakeEnv
from stable_baselines3 import PPO

def train_agent():
    env = SnakeEnv(render_mode='human')
    # Use MlpPolicy for vector-based observations
    model = PPO("MlpPolicy", env, verbose=1)
    
    print("Starting self-play training...")
    model.learn(total_timesteps=20000)
    
    print("Training complete. Saving model.")
    model.save("snake_ppo_agent")
    env.close()

if __name__ == "__main__":
    train_agent()
