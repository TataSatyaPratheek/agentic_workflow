# scripts/run_agent.py
import sys, os
import pygame
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game.snake_env import SnakeEnv
from src.agent.llm_parser import LLMCommandParser
from src.io.tts import TTS
from src.io.asr import ASR
from src.config import *

# RLlib specific: Environment Creator Function
def env_creator(env_config):
    return SnakeEnv(**env_config)

def run_training():
    # RLlib specific: Initialize Ray
    ray.init(ignore_reinit_error=True)
    print("Ray initialized.")

    # RLlib specific: Register the custom environment
    register_env("SnakeEnv-v1", env_creator)

    # Initialize our helper modules
    parser = LLMCommandParser()
    tts = TTS()
    asr = ASR()

    # --- RLlib specific: Configure the PPO Algorithm ---
    config = (
        PPOConfig()
        .environment(
            env="SnakeEnv-v1",
            env_config={"render_mode": "human"}
        )
        # --- FIXED: Use the new .env_runners() API instead of .rollouts() ---
        .env_runners(num_env_runners=0)
        .framework("torch")
        .training(model={"fcnet_hiddens": [256, 256]})
    )
    
    # RLlib specific: Build the Algorithm object
    algo = config.build()

    # Load Checkpoint if it exists
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

    # Manual Interaction and Training Loop
    env = SnakeEnv(render_mode='human')
    score, last_level = 0, 1
    agent_status, input_text, input_active = "Self-Playing", "", False
    
    for episode in range(1, 10001):
        obs, info = env.reset()
        terminated, truncated = False, False
        episode_reward = 0
        
        print(f"--- Starting Episode {episode} ---")

        while not terminated and not truncated:
            distraction_command = "idle"
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    ray.shutdown()
                    return
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if pygame.Rect(INPUT_BOX_RECT).collidepoint(event.pos): input_active = not input_active
                    else: input_active = False
                if input_active and event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        if input_text:
                            distraction_command = parser.parse_command(input_text)
                            tts.speak(f"Text command: {input_text}")
                        input_text = ""
                    elif event.key == pygame.K_BACKSPACE: input_text = input_text[:-1]
                    else: input_text += event.unicode
                elif not input_active and event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT: distraction_command = "left"
                    elif event.key == pygame.K_RIGHT: distraction_command = "right"
                    elif event.key == pygame.K_UP: distraction_command = "up"
                    elif event.key == pygame.K_DOWN: distraction_command = "down"
                    elif event.key == pygame.K_v:
                        agent_status = "Listening..."
                        env.render(score, episode_reward, agent_status, input_text, input_active)
                        voice_command = asr.listen_and_transcribe()
                        if voice_command:
                            distraction_command = parser.parse_command(voice_command)
                            tts.speak(f"Voice command: {voice_command}")
                        else:
                            tts.speak("Could not hear you.")
            
            action = algo.compute_single_action(obs)

            agent_dir = env.game.direction.lower()
            reward_modifier = 0
            if distraction_command != "idle" and distraction_command == agent_dir:
                reward_modifier = DISTRACTION_PENALTY
                agent_status = f"Distracted! Obeyed {distraction_command}"
            else:
                agent_status = "Self-Playing"

            obs, reward, terminated, truncated, info = env.step(action)
            reward += reward_modifier
            episode_reward += reward
            score = info.get('score', score)

            env.render(score, reward, agent_status, input_text, input_active)

            if env.game.level > last_level:
                tts.speak(f"Level {env.game.level} reached!")
                last_level = env.game.level

        print("Episode finished. Training...")
        train_results = algo.train()
        print(f"Train results: {train_results['info']['learner']['default_policy']['learner_stats']}")

        tts.speak(f"Game over! Score was {score}. Resetting.")
        score, last_level = 0, 1

        if episode % 10 == 0:
            save_dir = os.path.join(MODEL_PATH, "checkpoint")
            checkpoint_result = algo.save(save_dir)
            print(f"Checkpoint saved in {checkpoint_result.checkpoint.path}")

    save_dir = os.path.join(MODEL_PATH, "checkpoint")
    algo.save(save_dir)
    tts.speak("Model saved. Agent offline.")
    ray.shutdown()
    env.close()

if __name__ == "__main__":
    run_training()
