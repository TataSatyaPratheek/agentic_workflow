# src/io/vision.py
import numpy as np
import cv2
import mss
import Quartz # Use the native macOS Quartz API
from src.config import GAME_WINDOW_TITLE

class ScreenCapture:
    """A class to capture a specific game window using native macOS APIs."""
    def __init__(self):
        self.sct = mss.mss()
        self.game_window_info = self._find_window_by_title(GAME_WINDOW_TITLE)
        
        if not self.game_window_info:
            # Add a helpful debug print to show what windows were found
            print("Debug: Could not find game window. Here are the windows that were found:")
            self._print_all_windows()
            raise Exception(f"Game window '{GAME_WINDOW_TITLE}' not found. Is the game running and are screen recording permissions granted?")

        print(f"Successfully attached to game window: '{self.game_window_info.get('kCGWindowOwnerName', '')}'")
        
        window_bounds = self.game_window_info['kCGWindowBounds']
        self.monitor = {
            "top": int(window_bounds['Y']), "left": int(window_bounds['X']),
            "width": int(window_bounds['Width']), "height": int(window_bounds['Height']),
        }

    def _get_window_list(self):
        # --- REVISED LINE ---
        # Use kCGWindowListOptionAll to search all windows, not just "on-screen" ones.
        # This is more reliable across different Spaces and fullscreen apps.
        return Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionAll | Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID
        )

    def _find_window_by_title(self, title: str) -> dict | None:
        """Helper to find a window's info using Quartz."""
        for window in self._get_window_list():
            window_title = window.get(Quartz.kCGWindowName, None)
            if window_title and title in window_title:
                return window
        return None

    def _print_all_windows(self):
        """Debug helper to print all visible window titles."""
        for window in self._get_window_list():
            owner = window.get(Quartz.kCGWindowOwnerName, "Unknown")
            title = window.get(Quartz.kCGWindowName, "Unknown")
            print(f"  - Owner: {owner}, Title: {title}")

    def get_frame(self) -> np.ndarray:
        sct_img = self.sct.grab(self.monitor)
        frame = np.array(sct_img)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def close(self):
        self.sct.close()
