# src/game/controller.py
import pyautogui
from src.config import KEY_PRESS_DURATION

class GameController:
    """Sends keyboard commands to control the game."""
    def _press(self, key: str):
        pyautogui.keyDown(key)
        pyautogui.sleep(KEY_PRESS_DURATION)
        pyautogui.keyUp(key)

    def move_left(self):
        self._press('a')

    def move_right(self):
        self._press('d')

    def jump(self):
        self._press('space')

    def shoot(self):
        pyautogui.click() # Assumes mouse click is shoot

    def perform_action(self, action_name: str):
        """Performs a named action."""
        action_map = {
            "left": self.move_left,
            "right": self.move_right,
            "jump": self.jump,
            "shoot": self.shoot,
            "idle": lambda: None, # Do nothing
        }
        if action_name in action_map:
            action_map[action_name]()
        else:
            print(f"Warning: Unknown action '{action_name}'")

