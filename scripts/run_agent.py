# scripts/run_agent.py
import time
import numpy as np

from src.io.tts import TTS
from src.game.teeworlds_env import TeeworldsEnv
from src.agent.llm_parser import LLMCommandParser
from src.agent.policy import RLAgentPolicy
from src.config import ACTIONS, REWARD_IMITATION_SUCCESS, REWARD_IMITATION_FAILURE

def main_loop():
    # --- Initialization ---
    env = TeeworldsEnv()
    tts = TTS()
    llm_parser = LLMCommandParser()
    policy = RLAgentPolicy(env)

    print("Agent initialized. Starting main loop in 3 seconds...")
    time.sleep(3)
    tts.speak("Agent online. I am ready to learn.")

    # --- Main Online Learning Loop ---
    obs, _ = env.reset()
    try:
        while True:
            # 1. Perceive: Get user command
            user_command_text = input("Enter command (or 'quit'): ")
            if user_command_text.lower() == 'quit':
                break

            # 2. Think: Parse command and get RL agent's prediction
            parsed_action_str = llm_parser.parse_command(user_command_text)
            predicted_action_idx = policy.predict(obs)
            predicted_action_str = ACTIONS[predicted_action_idx]

            # 3. Act: The RL agent performs its chosen action
            obs, _, _, _, _ = env.step(predicted_action_idx)
            
            # 4. Reward & Learn: Reward the agent for imitating the parsed command
            # This is the first step towards learning when to trust the user.
            if predicted_action_str == parsed_action_str:
                reward = REWARD_IMITATION_SUCCESS
                feedback = f"I chose to {predicted_action_str}, as you suggested."
            else:
                reward = REWARD_IMITATION_FAILURE
                feedback = f"You suggested {parsed_action_str}, but I chose to {predicted_action_str}."
            
            print(f"Reward: {reward}")
            tts.speak(feedback)

            # Update the policy's replay buffer and learn from the experience
            # This is a simplified online learning step.
            policy.model.replay_buffer.add(
                obs=np.array([obs]),
                action=np.array([predicted_action_idx]),
                reward=np.array([reward]),
                next_obs=np.array([obs]), # In this simple setup, next_obs is the same
                done=np.array([False]),
                infos=[{}]
            )
            # Train for a small number of steps
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

