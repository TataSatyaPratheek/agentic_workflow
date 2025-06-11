# src/io/asr.py
import numpy as np
import sounddevice as sd
import torch
from transformers import pipeline

class ASR:
    """
    Handles real-time audio transcription using the highly optimized 'distil-whisper'
    model from Hugging Face. This model is significantly faster and smaller than
    the original Whisper, making it ideal for real-time agentic applications.
    """
    def __init__(self, sample_rate=16000, record_duration=3):
        """Initializes the ASR system."""
        
        # --- The M1 Fix: Detect and prepare for Apple Silicon GPU ---
        if torch.backends.mps.is_available():
            device = "mps"
            # Use float16 for a significant performance boost on MPS devices
            torch_dtype = torch.float16
            print("ASR (distil-whisper): MPS device found, using Apple Silicon GPU with float16.")
        else:
            device = "cpu"
            torch_dtype = torch.float32
            print("ASR (distil-whisper): MPS not found, using CPU.")
        # --- End of Fix ---

        # --- NEW: Initialize the Hugging Face ASR pipeline ---
        # This pipeline handles all the complexity of tokenization, model inference,
        # and decoding the final text.
        self.pipeline = pipeline(
            "automatic-speech-recognition",
            model="distil-whisper/distil-large-v2", # A robust and performant distilled model
            torch_dtype=torch_dtype,
            device=device,
        )

        self.sample_rate = sample_rate
        self.record_duration = record_duration

    def listen_and_transcribe(self) -> str:
        """Listens to the microphone and transcribes the speech."""
        print(f"Listening for {self.record_duration} seconds...")
        
        audio_data = sd.rec(
            int(self.record_duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        print("Processing...")

        # The pipeline expects a dictionary with raw audio data and the sample rate.
        result = self.pipeline({
            "raw": audio_data.flatten(),
            "sampling_rate": self.sample_rate
        })
        
        transcription = result['text']
        
        if transcription:
            print(f"Heard: '{transcription.strip()}'")
        else:
            print("No speech detected.")
            
        return transcription.strip()
