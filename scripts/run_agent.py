# scripts/run_agent.py
import sys, os
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game.snake_env import SnakeEnv
from src.config import MODEL_PATH

def env_creator(env_config: dict):
    # --- THE PRIMARY FIX ---
    # Pass the config dictionary directly to the environment constructor
    # instead of unpacking it.
    return SnakeEnv(config=env_config)
    # --- END OF FIX ---

def run_training():
    ray.init(ignore_reinit_error=True)
    print("Ray initialized.")
    register_env("SnakeEnv-v1", env_creator)

    num_workers = os.cpu_count() - 1 if os.cpu_count() > 1 else 1
    print(f"Using {num_workers} parallel workers for data collection.")

    config = (
        PPOConfig()
        .environment(env="SnakeEnv-v1", env_config={"render_mode": None})
        .framework("torch")
        .env_runners(num_env_runners=num_workers)
        .training(
            gamma=0.99,
            lr=5e-5,
            train_batch_size=4000,
            model={"fcnet_hiddens": [256, 256]},
            num_sgd_iter=10,
            use_critic=True,
            use_gae=True,
            lambda_=0.95,
            kl_coeff=0.2,
            clip_param=0.2,
            vf_loss_coeff=1.0,
            entropy_coeff=0.01
        )
        .evaluation(
            evaluation_interval=1,
            evaluation_num_env_runners=1,
            evaluation_config=PPOConfig.overrides(
                env_config={"render_mode": "human"}
            ),
        )
    )

    # Set algorithm-specific hyperparameters directly on the config object
    config.sgd_minibatch_size = 128
    
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
        
        print(f"\r--- Iteration {i}/1000 --- "
              f"Training Reward: {results.get('episode_reward_mean', 0):.2f} --- "
              f"Interactive Reward: {results.get('evaluation', {}).get('episode_reward_mean', 0):.2f}", end="")

        if i % 25 == 0:
            save_dir = os.path.join(MODEL_PATH, "checkpoint")
            checkpoint_result = algo.save(save_dir)
            print(f"\nCheckpoint saved in {checkpoint_result.checkpoint.path}")
            
    print("\nTraining complete. Model saved.")
    algo.save(os.path.join(MODEL_PATH, "checkpoint"))
    ray.shutdown()

if __name__ == "__main__":
    run_training()
