# scripts/run_agent.py
import sys, os
import pygame
import ray
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game.snake_env import SnakeEnv
from src.agent.llm_parser import LLMCommandParser
from src.io.tts import TTS
from src.io.asr import ASR
from src.config import *

def env_creator(env_config):
    # This creator is now used by background workers as well
    return SnakeEnv(**env_config)

def run_training():
    ray.init(ignore_reinit_error=True)
    print("Ray initialized.")
    register_env("SnakeEnv-v1", env_creator)

    # These are only for the interactive, visible agent
    parser = LLMCommandParser()
    tts = TTS()
    asr = ASR()

    # --- OPTIMIZATION 1: Configure for Parallelism and Batching ---
    config = (
        PPOConfig()
        .environment(env="SnakeEnv-v1", env_config={"render_mode": "human"})
        .framework("torch")
        # Use 3 background workers for data collection. Adjust based on your CPU cores.
        .env_runners(num_env_runners=3, num_envs_per_env_runner=1)
        .training(
            model={"fcnet_hiddens": [256, 256]},
            # Collect 4000 steps of experience before each training update.
            train_batch_size=4000,
            # sgd_minibatch_size and num_sgd_iter are PPO-specific and will be set directly on the config object.
        )
    )
    # Set PPO-specific training parameters directly on the config object
    config.sgd_minibatch_size = 128  # Break the batch into smaller pieces for SGD.
    config.num_sgd_iter = 10
    
    algo = config.build_algo()
    module = algo.get_module()


    checkpoint_dir = os.path.join(MODEL_PATH, "checkpoint")
    if os.path.exists(checkpoint_dir):
        try:
            algo.restore(checkpoint_dir)
            print(f"Restored model from checkpoint: {checkpoint_dir}")
        except Exception as e:
            print(f"Could not restore model: {e}. Starting fresh.")
    else:
        print("Creating new model.")

    tts.speak("Agent online. Press V for voice command.")
    
    # The main environment is only for rendering and interaction
    env = SnakeEnv(render_mode='human')
    clock = pygame.time.Clock()
    
    # --- OPTIMIZATION 2: Decoupled Training and Interaction Loop ---
    # The game loop no longer calls algo.train(). It only handles interaction.
    # Training happens on the river of data from the background workers.
    for i in range(1000): # Run for 1000 training iterations
        print(f"--- Training Iteration {i+1}/1000 ---")
        
        # This is now non-blocking for interaction; it processes data from workers
        train_results = algo.train()
        
        print(f"Episode Reward Mean: {train_results.get('episode_reward_mean', 'N/A')}")

        # Play one interactive episode to see the agent's progress
        obs, info = env.reset()
        terminated, truncated = False, False
        # Your event handling logic for keyboard/voice is unchanged...
        # Initialize these for the interactive loop
        score, last_level = 0, 1 
        agent_status, input_text, input_active = "Interactive Mode", "", False
        while not terminated and not truncated:
            for event in pygame.event.get():
                # ... (event handling code is unchanged) ...
                if event.type == pygame.QUIT:
                    ray.shutdown()
                    return
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if pygame.Rect(INPUT_BOX_RECT).collidepoint(event.pos): input_active = not input_active
                    else: input_active = False
                if input_active and event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        if input_text:
                            # In interactive mode, commands don't directly affect reward
                            # but we can still parse and announce them.
                            parsed_command = parser.parse_command(input_text)
                            tts.speak(f"Text command: {input_text}")
                        input_text = ""
                    elif event.key == pygame.K_BACKSPACE: input_text = input_text[:-1] # type: ignore
                    else: input_text += event.unicode
                elif not input_active and event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT: distraction_command = "left"
                    elif event.key == pygame.K_RIGHT: distraction_command = "right"
                    elif event.key == pygame.K_UP: distraction_command = "up"
                    elif event.key == pygame.K_DOWN: distraction_command = "down"
                    elif event.key == pygame.K_v:
                        agent_status = "Listening..."
                        env.render(score, 0, agent_status, input_text, input_active) # Reward not relevant here
                        voice_command = asr.listen_and_transcribe()
                        if voice_command:
                            parsed_command = parser.parse_command(voice_command)
                            tts.speak(f"Voice command: {voice_command}")
                        else:
                            tts.speak("Could not hear you.")
                        agent_status = "Interactive Mode" # Reset status
            
            # Get action from the latest trained model
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            fwd_out = module.forward_inference({"obs": obs_tensor})
            action_logits = fwd_out['action_dist_inputs']
            action = torch.argmax(action_logits, dim=1).squeeze().item()
            
            obs, reward, terminated, truncated, info = env.step(action)
            score = info.get('score', score)

            # Render the interactive game window
            env.render(score, reward, agent_status, input_text, input_active)

            if env.game.level > last_level:
                tts.speak(f"Level {env.game.level} reached!")
                last_level = env.game.level

            clock.tick(30) # Run at 30 frames per second
            
        tts.speak(f"Interactive episode finished. Score: {score}")

        if (i + 1) % 10 == 0:
            save_dir = os.path.join(MODEL_PATH, "checkpoint")
            checkpoint_result = algo.save(save_dir)
            print(f"Checkpoint saved in {checkpoint_result.checkpoint.path}")

    algo.save(os.path.join(MODEL_PATH, "checkpoint"))
    tts.speak("Model saved. Agent offline.")
    ray.shutdown()
    env.close()

if __name__ == "__main__":
    run_training()
