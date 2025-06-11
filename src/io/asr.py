# src/io/asr.py
import numpy as np
import sounddevice as sd
import time
from faster_whisper import WhisperModel
from src.config import ASR_MODEL

class ASR:
    """Handles real-time audio transcription using faster-whisper."""
    def __init__(self, sample_rate=16000, record_duration=3):
        """
        Initializes the ASR system.

        Args:
            sample_rate (int): The sample rate for audio recording. Whisper expects 16kHz.
            record_duration (int): The duration in seconds to listen for a command.
        """
        self.model = WhisperModel(ASR_MODEL, device="cpu", compute_type="int8")
        self.sample_rate = sample_rate
        self.record_duration = record_duration
        print(f"ASR model '{ASR_MODEL}' loaded. Ready to listen.")

    def listen_and_transcribe(self) -> str:
        """
        Listens to the microphone for a fixed duration and transcribes the speech.
        """
        print(f"Listening for {self.record_duration} seconds...")
        
        # Record audio from the default microphone.
        audio_data = sd.rec(
            int(self.record_duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,  # Use mono audio
            dtype='float32' # Whisper expects float32
        )
        sd.wait()  # Wait for the recording to complete.
        print("Processing...")

        # The recorded audio_data is already a NumPy array, which faster-whisper can process directly.
        # We flatten it to ensure it's a 1D array.
        segments, _ = self.model.transcribe(audio_data.flatten(), beam_size=5)

        transcription = "".join(segment.text for segment in segments)
        
        if transcription:
            print(f"Heard: '{transcription.strip()}'")
        else:
            print("No speech detected.")
            
        return transcription.strip()

    def transcribe_from_file(self, audio_path: str) -> str:
        """
        Transcribes audio from a file path. Kept for debugging or alternative inputs.
        """
        segments, _ = self.model.transcribe(audio_path, beam_size=5)
        transcription = "".join(segment.text for segment in segments)
        return transcription.strip()
