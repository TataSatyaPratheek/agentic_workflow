# scripts/run_agent.py
import sys, os
import pygame
import ray
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game.snake_env import SnakeEnv
from src.config import *

def env_creator(env_config):
    return SnakeEnv(**env_config)

def run_training():
    ray.init(ignore_reinit_error=True)
    print("Ray initialized.")
    register_env("SnakeEnv-v1", env_creator)

    # Use N-1 CPU cores for data collection to leave one for the main process
    num_workers = os.cpu_count() - 1 if os.cpu_count() > 1 else 1
    print(f"Using {num_workers} parallel workers for data collection.")

    # --- THE DEFINITIVE FIX: Use the correct, modern hyperparameter names ---
    config = (
        PPOConfig()
        .environment(env="SnakeEnv-v1", env_config={"render_mode": None})
        .framework("torch")
        .env_runners(num_env_runners=num_workers)
        .training(
            # General training parameters
            gamma=0.99,
            lr=5e-5,
            train_batch_size=4000,
            model={"fcnet_hiddens": [256, 256]},
            
            # --- CORRECTED: Use the modern parameter names for PPO ---
            minibatch_size=128,      # Renamed from sgd_minibatch_size
            num_epochs=10,           # Renamed from num_sgd_iter
            use_critic=True,
            use_gae=True,
            lambda_=0.95,
            kl_coeff=0.2,
            clip_param=0.2,
            vf_loss_coeff=1.0,
            entropy_coeff=0.01  # A small amount of entropy helps exploration
        )
        .evaluation(
            evaluation_interval=1,
            evaluation_num_env_runners=1,
            # We must pass the full class path for env_config overrides in evaluation
            evaluation_config=PPOConfig.overrides(
                explore=False,
                env_config={"render_mode": "human"}
            ),
        )
    )
    # --- END OF FIX ---
    
    algo = config.build_algo()
    
    checkpoint_dir = os.path.join(MODEL_PATH, "checkpoint")
    if os.path.exists(checkpoint_dir):
        try:
            algo.restore(checkpoint_dir)
            print(f"Restored model from checkpoint: {checkpoint_dir}")
        except Exception as e:
            print(f"Could not restore model: {e}. Starting fresh.")
    else:
        print("Creating new model.")
    
    print("\nStarting training. The interactive window will show the agent's progress.")
    
    for i in range(1, 1001):
        results = algo.train()
        
        # This provides a clean, single-line status update
        print(f"\r--- Iteration {i}/1000 --- "
              f"Training Reward: {results.get('episode_reward_mean', 0):.2f} --- "
              f"Interactive Reward: {results.get('evaluation', {}).get('episode_reward_mean', 0):.2f}", end="")

        if i % 25 == 0:
            save_dir = os.path.join(MODEL_PATH, "checkpoint")
            checkpoint_result = algo.save(save_dir)
            # Print on a new line to not interfere with the status line
            print(f"\nCheckpoint saved in {checkpoint_result.checkpoint.path}")
            
    print("\nTraining complete. Model saved.")
    algo.save(os.path.join(MODEL_PATH, "checkpoint"))
    ray.shutdown()

if __name__ == "__main__":
    run_training()
