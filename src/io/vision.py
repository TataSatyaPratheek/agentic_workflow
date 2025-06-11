# src/io/vision.py
import numpy as np
import cv2
import mss
import pygetwindow as gw
from src.config import GAME_WINDOW_TITLE

class ScreenCapture:
    """A class to capture a specific game window."""
    def __init__(self):
        self.sct = mss.mss()
        try:
            # Find the window by title
            self.game_window = gw.getWindowsWithTitle(GAME_WINDOW_TITLE)[0]
            print(f"Successfully attached to game window: '{self.game_window.title}'")
        except IndexError:
            raise Exception(f"Game window '{GAME_WINDOW_TITLE}' not found. Is the game running?")
        
        # Define the monitor region based on the game window's geometry
        self.monitor = {
            "top": self.game_window.top,
            "left": self.game_window.left,
            "width": self.game_window.width,
            "height": self.game_window.height,
        }

    def get_frame(self) -> np.ndarray:
        """Captures a frame from the game window and returns it as a NumPy array (BGR)."""
        sct_img = self.sct.grab(self.monitor)
        frame = np.array(sct_img)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def close(self):
        self.sct.close()

