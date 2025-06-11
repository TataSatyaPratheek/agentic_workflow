# src/io/asr.py
from faster_whisper import WhisperModel
from src.config import ASR_MODEL

class ASR:
    """Handles audio transcription using faster-whisper."""
    def __init__(self):
        # Using a tiny model for speed. device="cpu" and compute_type="int8"
        # are great for M1 Air performance.
        self.model = WhisperModel(ASR_MODEL, device="cpu", compute_type="int8")
        print(f"ASR model '{ASR_MODEL}' loaded.")

    def transcribe_from_file(self, audio_path: str) -> str:
        """Transcribes audio from a file path."""
        # TODO: Replace with real-time microphone input later
        segments, _ = self.model.transcribe(audio_path, beam_size=5)
        transcription = "".join(segment.text for segment in segments)
        return transcription.strip()
