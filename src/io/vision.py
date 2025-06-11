# src/io/vision.py
import numpy as np
import cv2
import mss

class ScreenCapture:
    """A simple class to capture the game screen."""
    def __init__(self):
        self.sct = mss.mss()
        # For MVP, we'll capture the whole screen. Later, target the game window.
        self.monitor = self.sct.monitors[1]

    def get_frame(self) -> np.ndarray:
        """Captures a frame and returns it as a NumPy array (BGR)."""
        sct_img = self.sct.grab(self.monitor)
        frame = np.array(sct_img)
        # MSS captures in BGRA, convert to BGR for OpenCV
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def close(self):
        self.sct.close()
