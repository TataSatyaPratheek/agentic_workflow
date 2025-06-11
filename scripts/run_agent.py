# scripts/run_agent.py
import sys, os
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.tune.registry import register_env
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game.snake_env import SnakeEnv
from src.config import MODEL_PATH as MODEL_BASE_DIR_NAME # Renamed for clarity

def env_creator(env_config: dict):
    return SnakeEnv(config=env_config)

# scripts/run_agent.py

def run_training():
    ray.init(ignore_reinit_error=True)
    print("Ray initialized.")

    # Determine project root to make paths absolute
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Define the main directory where all model artifacts will be stored
    main_model_artifacts_path = os.path.join(PROJECT_ROOT, MODEL_BASE_DIR_NAME)
    os.makedirs(main_model_artifacts_path, exist_ok=True) # Ensure this base directory exists

    register_env("SnakeEnv-v1", env_creator)

    num_workers = 4
    print(f"Using {num_workers} parallel workers for data collection (optimized for M1 Air).")

    # --- THE CANONICAL FIX: Adopt the Modern RLlib API Stack ---
    # This new structure eliminates all deprecation warnings by explicitly
    # configuring the new components (RLModule, Learners, etc.).
    # All configuration is done via chained methods on the PPOConfig object.
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .framework("torch")
        .environment(env="SnakeEnv-v1", env_config={"render_mode": None})
        .env_runners(
            num_env_runners=num_workers,
            # This is a new required setting for the new API stack
            rollout_fragment_length=64
        )
        # --- CORRECT MODEL CONFIGURATION FOR DEFAULT MODULE ---
        # Configure the model inside the .rl_module() method using DefaultModelConfig
        .rl_module(
            model_config=DefaultModelConfig(
                fcnet_hiddens=[128, 128],
            )
        )
        # --- END OF CORRECTION ---
        # Configure training parameters inside the .training() method
        .training(
            gamma=0.99,
            lr=5e-5,
            train_batch_size_per_learner=256, # Renamed for clarity in new API
            # sgd_minibatch_size and num_sgd_iter are PPO-specific and might not be
            # accepted by .training() in all Ray versions. They will be set directly below.
            entropy_coeff=0.01,
        )
        .evaluation(
            evaluation_interval=10,
            evaluation_num_env_runners=1,
            evaluation_config=PPOConfig.overrides(
                env_config={"render_mode": "human"}
            ),
        )
    )

    # Set PPO-specific training parameters directly on the config object.
    # This is a robust way if .training() doesn't accept them as kwargs in your Ray version.
    config.sgd_minibatch_size = 64
    config.num_sgd_iter = 5

    
    # Use .build_algo() as .build() is deprecated
    algo = config.build_algo()
    
    # Checkpoints will be saved in a subdirectory, e.g., .../rllib_snake_model_checkpoints/checkpoint
    checkpoint_save_restore_path = os.path.join(main_model_artifacts_path, "checkpoint")

    if os.path.exists(checkpoint_save_restore_path):
        try:
            algo.restore(checkpoint_save_restore_path)
            print(f"Restored model from checkpoint: {checkpoint_save_restore_path}")
        except Exception as e:
            print(f"Could not restore model: {e}. Starting fresh.")
    else:
        print("Creating new model.")

    print("\nStarting optimized training...")
    start_time = time.time()
    
    for i in range(1, 1001):
        results = algo.train()
        
        # --- KEYERROR FIX: Use the new metric keys ---
        # The new API provides metrics in a more structured way.
        training_reward = results.get("sampler_results", {}).get("episode_reward_mean", 0)
        
        log_line = (
            f"\r--- Iteration {i}/1000 --- "
            f"Time: {time.time() - start_time:.0f}s --- "
            f"Training Reward: {training_reward:.2f} --- "
            f"Timesteps: {results.get('num_agent_steps_trained', 0)} --- "
        )

        # Safely access evaluation metrics
        # results.get('evaluation', {}) ensures that if 'evaluation' key is missing or None,
        # we operate on an empty dict, preventing an error on the subsequent .get().
        evaluation_sampler_results = results.get('evaluation', {}).get('sampler_results')

        if evaluation_sampler_results:  # Check if sampler_results were found within evaluation
            eval_reward = evaluation_sampler_results.get('episode_reward_mean')
            if eval_reward is not None:  # Check if the specific metric exists
                log_line += f"Interactive Reward: {eval_reward:.2f}"

        print(log_line, end="")

        if i % 50 == 0:
            checkpoint_result = algo.save(checkpoint_save_restore_path)
            print(f"\nCheckpoint saved in {checkpoint_result.checkpoint.path}")
            
    end_time = time.time()
    print(f"\n\nTraining complete in {end_time - start_time:.2f} seconds.")
    final_checkpoint_info = algo.save(checkpoint_save_restore_path)
    print(f"Final model checkpoint saved in {final_checkpoint_info.checkpoint.path}")
    ray.shutdown()

if __name__ == "__main__":
    run_training()
