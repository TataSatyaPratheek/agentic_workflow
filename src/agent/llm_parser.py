# src/agent/llm_parser.py
import torch
from transformers import pipeline

class LLMCommandParser:
    """Uses a zero-shot classification model to parse natural language commands."""
    def __init__(self):
        # --- THE M1 FIX ---
        # Check if MPS (Apple Silicon GPU) is available and use it.
        # Fallback to CPU if not. This makes the code portable.
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("LLMCommandParser: MPS device found, using Apple Silicon GPU.")
        else:
            self.device = torch.device("cpu")
            print("LLMCommandParser: MPS not found, using CPU.")
        # --- END OF FIX ---

        # Pass the selected device to the pipeline.
        # The model will be automatically moved to the GPU.
        self.classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=self.device
        )
        self.candidate_labels = ['up', 'down', 'left', 'right']
        print("LLM Command Parser initialized.")

    def parse_command(self, command: str) -> str:
        """
        Parses the text command and returns the most likely direction.
        """
        if not command:
            return "idle"
            
        result = self.classifier(command, self.candidate_labels)
        # The highest score is the most likely command
        return result['labels'][0]
