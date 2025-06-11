# scripts/run_agent.py
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game.snake_env import SnakeEnv
from src.agent.llm_parser import LLMCommandParser
from src.io.tts import TTS
from src.config import *
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def run_interactive_training():
    env = make_vec_env(lambda: SnakeEnv(render_mode='human'), n_envs=1)
    parser = LLMCommandParser()
    tts = TTS()

    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}"); model = PPO.load(MODEL_PATH, env=env)
    else:
        print("Creating new PPO model."); model = PPO("MlpPolicy", env, verbose=0)

    tts.speak("Agent is online. Starting training.")
    obs = env.reset()
    total_reward, episode_steps = 0, 0
    
    try:
        while True:
            episode_steps += 1
            user_input = input("Enter a command to distract (or press Enter): ")
            parsed_action_str = parser.parse_command(user_input) if user_input else "idle"
            
            action, _ = model.predict(obs, deterministic=False)
            
            voice_cmd_idx = ACTIONS.index(parsed_action_str) if parsed_action_str != "idle" else -1
            
            # We need to manually call the env's step method to pass the voice command
            # This is a small hack to inject voice command into the next observation
            next_obs, reward, done, info = env.envs[0].step(action[0], voice_command_idx=voice_cmd_idx)
            
            final_reward = reward
            if parsed_action_str != "idle" and ACTIONS[action[0]] == parsed_action_str:
                final_reward += DISTRACTION_PENALTY
                tts.speak(f"Distracted! Following your command: {parsed_action_str}.")

            total_reward += final_reward
            
            model.replay_buffer.add(obs[0], next_obs, action[0], final_reward, done, info[0])
            if episode_steps % model.n_steps == 0:
                model.train()

            obs = np.array([next_obs])
            
            if done:
                score = info[0]['score']
                tts.speak(f"Game over! Score: {score}. Resetting.")
                obs = env.reset()
                total_reward, episode_steps = 0, 0

    except KeyboardInterrupt:
        print("\nTraining interrupted.")
    finally:
        model.save(MODEL_PATH)
        tts.speak(f"Model saved to {MODEL_PATH}. Agent offline.")
        env.close()

if __name__ == "__main__":
    run_interactive_training()
