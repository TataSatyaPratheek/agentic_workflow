# scripts/run_agent.py
import time
from src.io.vision import ScreenCapture
from src.io.asr import ASR
from src.io.tts import TTS
from src.game.controller import GameController
# from src.agent.policy import RLAgentPolicy # Will be imported later

def main_loop():
    # --- Initialization ---
    vision = ScreenCapture()
    asr = ASR()
    tts = TTS()
    controller = GameController()
    # policy = RLAgentPolicy() # TODO: Initialize the actual RL policy

    print("Agent initialized. Starting main loop in 3 seconds...")
    time.sleep(3)
    tts.speak("Agent online.")

    # --- Main Loop ---
    try:
        while True:
            # 1. Perceive: Get game state and user command
            current_frame = vision.get_frame()
            # For MVP, we'll use a placeholder for voice input
            # user_command = asr.transcribe_from_file("path/to/command.wav")
            user_command_text = input("Enter command (e.g., jump, left, right): ")

            # 2. Think: LLM parsing and Policy decision
            # TODO: Add LLM call here to interpret user_command_text
            # TODO: Get action from the RL policy based on frame and command
            action_to_perform = user_command_text # Simple passthrough for now

            # 3. Act: Perform the action in the game
            controller.perform_action(action_to_perform)
            tts.speak(f"Performing action: {action_to_perform}")

            # Prevent overwhelming the CPU
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("Agent shutting down.")
        tts.speak("Agent offline.")
        vision.close()

if __name__ == "__main__":
    main_loop()

