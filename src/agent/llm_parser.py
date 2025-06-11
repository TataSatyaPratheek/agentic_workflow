# src/agent/llm_parser.py
from transformers import pipeline
from src.config import ACTIONS, LLM_CLASSIFIER

class LLMCommandParser:
    """Uses a zero-shot pipeline to classify user commands into game actions."""
    def __init__(self):
        # --- REVISED LINE ---
        # Load the model directly using the identifier from config.
        self.classifier = pipeline(
            "zero-shot-classification",
            model=LLM_CLASSIFIER,
            device="cpu"
        )
        # --- END REVISED LINE ---
        print("LLM Command Parser initialized.")

    def parse_command(self, text: str) -> str:
        """
        Parses the user's text command to determine the most likely action.
        Returns one of the actions from the global ACTIONS list.
        """
        if not text:
            return "idle"
        result = self.classifier(text, candidate_labels=ACTIONS)
        return result['labels'][0]
