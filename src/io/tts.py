# src/io/tts.py
import os
from gtts import gTTS

class TTS:
    """Handles text-to-speech for agent feedback."""
    def speak(self, text: str):
        """Converts text to speech and plays it."""
        print(f"Agent says: {text}")
        try:
            tts = gTTS(text=text, lang='en')
            # Using a temp file for playback is a simple MVP approach
            tts.save("feedback.mp3")
            os.system("afplay feedback.mp3") # afplay is a macOS command
            os.remove("feedback.mp3")
        except Exception as e:
            print(f"TTS failed: {e}")

