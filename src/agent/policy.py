# src/agent/policy.py
from stable_baselines3 import PPO
import os

class RLAgentPolicy:
    """Manages the Stable-Baselines3 RL agent."""
    def __init__(self, env):
        self.model_path = "teeworlds_ppo_agent.zip"
        # "CnnPolicy" is designed for image-based observations
        if os.path.exists(self.model_path):
            print(f"Loading existing model from {self.model_path}")
            self.model = PPO.load(self.model_path, env=env)
        else:
            print("Creating a new PPO model.")
            self.model = PPO("CnnPolicy", env, verbose=1)

    def predict(self, observation):
        """Get the agent's action for a given observation."""
        action, _states = self.model.predict(observation, deterministic=True)
        return action

    def learn(self, total_timesteps: int = 1000):
        """Train the model."""
        self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
        self.model.save(self.model_path)
