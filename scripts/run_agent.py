# scripts/run_agent.py
import sys, os, time, torch as th
import numpy as np
import pygame
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game.snake_env import SnakeEnv
from src.agent.llm_parser import LLMCommandParser
from src.io.tts import TTS
from src.config import *
from stable_baselines3 import PPO

def run_interactive_training():
    """
    A stable, interactive training loop using a standard Pygame event handler
    and the correct on-policy training method for PPO.
    """
    # --- Initialization ---
    pygame.init()
    env = SnakeEnv(render_mode='human') # We use the single, direct environment
    parser = LLMCommandParser()
    tts = TTS()

    # n_steps must match the size of the rollout buffer
    n_steps = 512
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        model = PPO.load(MODEL_PATH, env=env, n_steps=n_steps)
    else:
        print("Creating new PPO model.")
        model = PPO("MlpPolicy", env, verbose=0, n_steps=n_steps)

    # --- Game Loop Variables ---
    obs, _ = env.reset()
    last_reward, score, last_level = 0.0, 0, 1
    # prev_done indicates if the 'obs' is the start of a new episode.
    # True for the initial state after reset.
    prev_done = True
    agent_status = "Self-Playing"
    running = True

    tts.speak("Agent is online. Use keyboard L, R, S to distract me.")
    
    # --- The Stable Game Loop ---
    while running:
        # 1. Handle Non-Blocking User Input via Pygame Events
        distraction_command = "idle"
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_l: distraction_command = "left"
                elif event.key == pygame.K_r: distraction_command = "right"
                elif event.key == pygame.K_s: distraction_command = "straight"
                
                if distraction_command != "idle":
                    agent_status = f"Distracted: {distraction_command}"
                    tts.speak(f"Distraction received: {distraction_command}")

        # 2. Agent Decides Action, gets value and log_prob for the current 'obs'
        # Convert current observation 'obs' to a PyTorch tensor
        obs_tensor = th.as_tensor(obs, device=model.device).unsqueeze(0)
        with th.no_grad():
            # model.policy(obs_tensor) returns action, value, log_probability
            action_tensor, value_tensor, log_prob_tensor = model.policy(obs_tensor)

        action_np = action_tensor.cpu().numpy() # np.array([int_action]) for discrete
        action_for_env = action_np[0] # Integer action for env.step()

        # 3. Environment Steps Forward with the chosen action
        voice_cmd_idx = ACTIONS.index(distraction_command) if distraction_command != "idle" else -1
        # Manually set the voice command state on the env before stepping
        env.last_voice_command_idx = voice_cmd_idx
        next_obs, reward, done_after_step, _, info = env.step(action_for_env)
        
        last_reward = reward
        score = info.get('score', score)
        
        # 4. Calculate Final Reward (with distraction penalty)
        final_reward = reward
        if distraction_command != "idle" and ACTIONS[action_for_env] == distraction_command:
            final_reward += DISTRACTION_PENALTY
            agent_status = f"Distracted! Obeyed {distraction_command}"

        # 5. Add experience to PPO's rollout_buffer
        # obs: current observation (s_t)
        # action_np: action taken (a_t)
        # reward_for_buffer: reward received (r_t)
        # episode_start_for_buffer: whether 'obs' was the start of an episode (d_{t-1})
        # value_tensor: V(s_t)
        # log_prob_tensor: log_prob(a_t | s_t)
        reward_for_buffer = np.array([final_reward], dtype=np.float32)
        episode_start_for_buffer = np.array([prev_done], dtype=np.float32)

        model.rollout_buffer.add(obs, action_np, reward_for_buffer, episode_start_for_buffer, value_tensor, log_prob_tensor)
        
        # Update obs to next_obs and prev_done to done_after_step for the next iteration
        obs = next_obs
        prev_done = done_after_step
        
        # 6. Train if the buffer is full (The Correct On-Policy Way)
        if model.rollout_buffer.full:
            agent_status = "Learning..."
            # Compute GAE and returns before training
            with th.no_grad():
                # Value of the last observation (current 'obs', which is s_{t+N})
                next_value_for_gae = model.policy.predict_values(th.as_tensor(obs, device=model.device).unsqueeze(0))
            # 'dones' for compute_returns_and_advantage is the 'done' status of the last state in the rollout (prev_done)
            model.rollout_buffer.compute_returns_and_advantage(last_values=next_value_for_gae, dones=np.array([prev_done]))
            model.train()
            model.rollout_buffer.reset()
        elif "Distracted" not in agent_status:
             agent_status = "Self-Playing"
        
        # 7. Render the Game and Stats
        env.render(score, last_reward, agent_status)
        
        # 8. Handle Game Over and Level Ups
        if done_after_step:
            tts.speak(f"Game over! Final score was {score}. Resetting.")
            obs, _ = env.reset()
            score, last_level = 0, 1
            prev_done = True # After reset, the new 'obs' is an episode start
        
        current_level = env.game.level
        if current_level > last_level:
            tts.speak(f"Level {current_level} reached!")
            last_level = current_level

    # --- Shutdown ---
    print("\nShutting down. Saving model...")
    model.save(MODEL_PATH)
    tts.speak("Model saved. Agent offline.")
    env.close()

if __name__ == "__main__":
    run_interactive_training()
