# src/agent/llm_parser.py
from transformers import pipeline
from src.config import ACTIONS, LLM_CLASSIFIER

class LLMCommandParser:
    """Uses a zero-shot pipeline to classify user commands into game actions."""
    def __init__(self):
        # Use a very small, fast model for classification.
        # This is much more efficient than a full generative LLM.
        self.classifier = pipeline(
            "zero-shot-classification",
            model=f"valhalla/{LLM_CLASSIFIER}", # Using a community finetuned distilbert
            device="cpu"
        )
        print("LLM Command Parser initialized.")

    def parse_command(self, text: str) -> str:
        """
        Parses the user's text command to determine the most likely action.
        Returns one of the actions from the global ACTIONS list.
        """
        if not text:
            return "idle"
        # The pipeline returns the label with the highest score.
        result = self.classifier(text, candidate_labels=ACTIONS)
        return result['labels'][0]

