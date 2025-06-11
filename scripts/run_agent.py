# scripts/run_agent.py
import sys
import os
import time
import numpy as np

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from src.io.tts import TTS
from src.game.teeworlds_env import TeeworldsEnv
from src.agent.llm_parser import LLMCommandParser
from src.agent.policy import RLAgentPolicy
from src.config import ACTIONS, REWARD_IMITATION_SUCCESS, REWARD_IMITATION_FAILURE, GAME_WINDOW_TITLE

def main_loop():
    # --- Initialization ---
    env = TeeworldsEnv()
    tts = TTS()
    llm_parser = LLMCommandParser()
    policy = RLAgentPolicy(env)

    print("Agent initialized. Bringing game to front...")
    # --- REVISED CODE BLOCK ---
    # Use AppleScript to activate the game window, which is the standard macOS way.
    # This replaces the non-functional .activate() method.
    os.system(f'osascript -e \'tell application "{GAME_WINDOW_TITLE}" to activate\'')
    # --- END REVISED CODE BLOCK ---
    time.sleep(1)

    print("Starting main loop in 3 seconds...")
    time.sleep(3)
    tts.speak("Agent online. I am ready to learn.")
    
    # ... the rest of your main_loop remains exactly the same ...
    obs, _ = env.reset()
    try:
        while True:
            user_command_text = input("Enter command (or 'quit'): ")
            if user_command_text.lower() == 'quit':
                break

            parsed_action_str = llm_parser.parse_command(user_command_text)
            predicted_action_idx = policy.predict(obs)
            predicted_action_str = ACTIONS[predicted_action_idx]

            obs, _, _, _, _ = env.step(predicted_action_idx)
            
            if predicted_action_str == parsed_action_str:
                reward = REWARD_IMITATION_SUCCESS
                feedback = f"I chose to {predicted_action_str}, as you suggested."
            else:
                reward = REWARD_IMITATION_FAILURE
                feedback = f"You suggested {parsed_action_str}, but I chose to {predicted_action_str}."
            
            print(f"Reward: {reward}")
            tts.speak(feedback)

            policy.model.replay_buffer.add(
                obs=np.array([obs]), action=np.array([predicted_action_idx]),
                reward=np.array([reward]), next_obs=np.array([obs]),
                done=np.array([False]), infos=[{}]
            )
            policy.learn(total_timesteps=128)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
    finally:
        policy.model.save(policy.model_path)
        print(f"Model saved to {policy.model_path}")
        tts.speak("Agent offline.")
        env.close()

if __name__ == "__main__":
    main_loop()

