# scripts/run_agent.py
import sys, os, time
import numpy as np
import pygame
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game.snake_env import SnakeEnv
from src.agent.llm_parser import LLMCommandParser
from src.io.tts import TTS
from src.config import *
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class InteractiveCallback(BaseCallback):
    """
    A custom callback to handle user input and UI updates during training.
    """
    def __init__(self, tts: TTS, parser: LLMCommandParser, verbose=0):
        super(InteractiveCallback, self).__init__(verbose)
        self.tts = tts
        self.parser = parser
        self.score = 0
        self.last_reward = 0.0
        self.agent_status = "Self-Playing"

    def _on_step(self) -> bool:
        # Handle Pygame events for non-blocking input
        distraction_command = "idle"
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False # Stops training
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_l: distraction_command = "left"
                elif event.key == pygame.K_r: distraction_command = "right"
                elif event.key == pygame.K_s: distraction_command = "straight"
        
        # Update the environment with the latest voice command
        env = self.training_env.envs[0]
        parsed_action_str = self.parser.parse_command(distraction_command)
        env.last_voice_command_idx = ACTIONS.index(parsed_action_str) if parsed_action_str != "idle" else -1

        # Check for distraction and apply penalty
        # The `self.locals` dict gives us access to internal RL variables
        last_action = self.locals['actions'][0]
        if parsed_action_str != "idle" and ACTIONS[last_action] == parsed_action_str:
            self.locals['rewards'][0] += DISTRACTION_PENALTY
            self.agent_status = f"Distracted! Obeyed {parsed_action_str}"
            self.tts.speak(self.agent_status)
        else:
            self.agent_status = "Self-Playing"

        self.last_reward = self.locals['rewards'][0]
        self.score = self.locals['infos'][0].get('score', self.score)
        
        # Render the game with live stats
        env.render(self.score, self.last_reward, self.agent_status)
        
        if self.locals['dones'][0]:
            self.tts.speak(f"Game over! Score was {self.score}. Resetting.")
            self.score = 0
        
        return True # Continue training

def run_training():
    # --- Initialization ---
    pygame.init()
    env = SnakeEnv(render_mode='human')
    parser = LLMCommandParser()
    tts = TTS()
    callback = InteractiveCallback(tts=tts, parser=parser)

    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        model = PPO.load(MODEL_PATH, env=env)
    else:
        print("Creating new PPO model.")
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./snake_tensorboard/")

    tts.speak("Agent is online. Use keyboard L, R, S to distract me.")
    
    try:
        # This is the stable, correct way to run the agent.
        # The callback handles all interaction inside this blocking call.
        model.learn(total_timesteps=1_000_000, callback=callback)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        model.save(MODEL_PATH)
        tts.speak("Model saved. Agent offline.")
        env.close()

if __name__ == "__main__":
    run_training()
