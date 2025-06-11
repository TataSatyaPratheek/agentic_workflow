# scripts/run_agent.py
import sys, os
import pygame
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.game.snake_env import SnakeEnv
from src.agent.llm_parser import LLMCommandParser # We are using this again!
from src.io.tts import TTS
from src.config import *
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class InteractiveCallback(BaseCallback):
    """A custom callback for stable, hybrid (keyboard + text) interaction."""
    def __init__(self, tts: TTS, parser: LLMCommandParser, verbose=0):
        super().__init__(verbose)
        self.tts = tts
        self.parser = parser
        self.score, self.last_level = 0, 1
        self.agent_status = "Self-Playing"
        self.input_text = ""
        self.input_active = False

    def _on_step(self) -> bool:
        # 1. Handle Hybrid Pygame Events
        distraction_command = "idle"
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return False
            
            # Text input handling
            if event.type == pygame.MOUSEBUTTONDOWN:
                if pygame.Rect(INPUT_BOX_RECT).collidepoint(event.pos):
                    self.input_active = not self.input_active
                else:
                    self.input_active = False
            
            if self.input_active:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        # Parse the text with the LLM when Enter is pressed
                        if self.input_text:
                            distraction_command = self.parser.parse_command(self.input_text)
                            self.tts.speak(f"Command received: {self.input_text}")
                        self.input_text = "" # Clear the box
                    elif event.key == pygame.K_BACKSPACE:
                        self.input_text = self.input_text[:-1]
                    else:
                        self.input_text += event.unicode
            
            # Arrow key handling (always active)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT: distraction_command = "left"
                elif event.key == pygame.K_RIGHT: distraction_command = "right"
                elif event.key == pygame.K_UP: distraction_command = "up"
                elif event.key == pygame.K_DOWN: distraction_command = "down"

        # 2. Apply Distraction Penalty
        last_action = self.locals['actions'][0]
        agent_dir = self.training_env.envs[0].game.direction.lower()
        
        # Check if agent's absolute direction matches a distraction command
        if distraction_command != "idle" and distraction_command == agent_dir:
            self.locals['rewards'][0] += DISTRACTION_PENALTY
            self.agent_status = f"Distracted! Obeyed {distraction_command}"
        else:
            self.agent_status = "Self-Playing"

        # 3. Render the Game with Live Stats and Input Box
        last_reward = self.locals['rewards'][0]
        self.score = self.locals['infos'][0].get('score', self.score)
        env = self.training_env.envs[0]
        env.render(self.score, last_reward, self.agent_status, self.input_text, self.input_active)
        
        if self.locals['dones'][0]:
            self.tts.speak(f"Game over! Score was {self.score}. Resetting.")
            self.score = 0
            self.last_level = 1
        
        if env.game.level > self.last_level:
            self.tts.speak(f"Level {env.game.level} reached!")
            self.last_level = env.game.level
            
        return True

def run_training():
    pygame.init()
    env = SnakeEnv(render_mode='human')
    parser = LLMCommandParser() # Your LLM parser is back!
    tts = TTS()
    callback = InteractiveCallback(tts=tts, parser=parser)

    if os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        model = PPO.load(MODEL_PATH, env=env)
    else:
        print("Creating new PPO model.")
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./snake_tensorboard/")

    tts.speak("Agent is online. Use ARROW KEYS or CLICK THE BOX to type commands.")
    
    try:
        model.learn(total_timesteps=1_000_000, callback=callback)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        model.save(MODEL_PATH)
        tts.speak("Model saved. Agent offline.")
        env.close()

if __name__ == "__main__":
    run_training()
