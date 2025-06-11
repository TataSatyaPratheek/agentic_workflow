# scripts/run_agent.py
import sys, os
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game.snake_env import SnakeEnv
from src.config import MODEL_PATH

def env_creator(env_config: dict):
    return SnakeEnv(config=env_config)

def run_training():
    ray.init(ignore_reinit_error=True)
    print("Ray initialized.")
    register_env("SnakeEnv-v1", env_creator)

    # --- OPTIMIZATION 1: Reduce Resource Contention ---
    # For an 8-core M1 Air, 4 workers is a sweet spot that leaves
    # resources for the learner, evaluation, and OS.
    num_workers = 4
    print(f"Using {num_workers} parallel workers for data collection (optimized for M1 Air).")

    # First, instantiate the PPOConfig object
    ppo_config_obj = PPOConfig()

    # --- OPTIMIZATION 2: Tune Hyperparameters for Speed ---
    # Set common RL training hyperparameters as attributes on the config object
    ppo_config_obj.gamma = 0.99
    ppo_config_obj.lr = 5e-5
    ppo_config_obj.train_batch_size = 2000
    ppo_config_obj.sgd_minibatch_size = 128
    ppo_config_obj.num_sgd_iter = 5
    ppo_config_obj.entropy_coeff = 0.01

    # Configure the model using the new API stack's rl_module method
    ppo_config_obj = ppo_config_obj.rl_module(
        model_config={"fcnet_hiddens": [128, 128]}
    )

    # Now, chain the rest of the configuration methods
    config = (
        ppo_config_obj
        .environment(env="SnakeEnv-v1", env_config={"render_mode": None})
        .framework("torch")
        .env_runners(num_env_runners=num_workers)
        # --- OPTIMIZATION 3: Fix the Rendering Bottleneck ---
        .evaluation(
            # Render an episode every 10 training iterations, not every single one.
            # This is the single most important change for speed.
            evaluation_interval=10,
            evaluation_num_env_runners=1, # Only one worker should ever render
            evaluation_config=PPOConfig.overrides(
                env_config={"render_mode": "human"}
            ),
        )
    )
    
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
    
    print("\nStarting optimized training...")
    start_time = time.time()
    
    for i in range(1, 1001):
        results = algo.train()
        
        # Provide a more detailed and clean status update
        log_line = (
            f"\r--- Iteration {i}/1000 --- "
            f"Time: {time.time() - start_time:.0f}s --- "
            f"Training Reward: {results.get('episode_reward_mean', 0):.2f} --- "
            f"Timesteps: {results.get('num_agent_steps_trained', 0)} --- "
        )
        # Only add evaluation stats when they are available (every 10 iterations)
        if 'evaluation' in results:
            log_line += f"Interactive Reward: {results['evaluation']['episode_reward_mean']:.2f}"
        
        print(log_line, end="")

        if i % 50 == 0:
            save_dir = os.path.join(MODEL_PATH, "checkpoint")
            checkpoint_result = algo.save(save_dir)
            print(f"\nCheckpoint saved in {checkpoint_result.checkpoint.path}")
            
    end_time = time.time()
    print(f"\n\nTraining complete in {end_time - start_time:.2f} seconds.")
    algo.save(os.path.join(MODEL_PATH, "checkpoint"))
    ray.shutdown()

if __name__ == "__main__":
    run_training()
