# src/agent/policy.py
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.vec_env import DummyVecEnv
import os

class RLAgentPolicy:
    """Manages the Stable-Baselines3 RL agent."""
    def __init__(self, env):
        self.model_path = "teeworlds_ppo_agent.zip"
        
        # --- REVISED LOGIC ---
        # If the environment is not already vectorized, wrap it in a DummyVecEnv.
        # This gives us explicit control and avoids hidden wrappers.
        # Crucially, we are creating a VecEnv with n_envs=1.
        if not is_wrapped(env, DummyVecEnv):
            self.env = DummyVecEnv([lambda: env])
        else:
            self.env = env
        # --- END REVISED LOGIC ---

        if os.path.exists(self.model_path):
            print(f"Loading existing model from {self.model_path}")
            # Pass the explicitly wrapped env to the model
            self.model = PPO.load(self.model_path, env=self.env)
        else:
            print("Creating a new PPO model.")
            # Pass the explicitly wrapped env to the model
            self.model = PPO("CnnPolicy", self.env, verbose=1)

    def predict(self, observation):
        """Get the agent's action for a given observation."""
        # The observation from a single env needs to be wrapped in an array for the model
        action, _states = self.model.predict(observation, deterministic=True)
        return action

    def learn(self, total_timesteps: int = 1000):
        """Train the model."""
        self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False)
        self.model.save(self.model_path)
