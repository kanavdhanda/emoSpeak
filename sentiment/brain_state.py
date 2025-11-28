import json
import os
import numpy as np

class NeuroState:
    def __init__(self, state_dict=None):
        # Always start with default state to ensure all keys exist
        self.state = self._get_default_state()
        # If state is provided, update the default with it
        if state_dict:
            self.state.update(state_dict)

    def _get_default_state(self):
        return {
            "baseline_energy": 0.5,       # Neuroplasticity: Moving average of volume
            "baseline_pitch_var": 0.5,    # Neuroplasticity: Moving average of expression
            "emotional_memories": [],     # Neurogenesis: Storing new complex patterns
            "conversation_history": [],   # Context: Full history of turns
            "interaction_count": 0
        }

    def add_interaction(self, user_text, user_emotion, ai_response):
        """Stores the full turn in history."""
        self.state["conversation_history"].append({
            "user": user_text,
            "user_emotion": user_emotion,
            "ai": ai_response
        })
        # Keep history manageable (last 20 turns)
        if len(self.state["conversation_history"]) > 20:
            self.state["conversation_history"].pop(0)

    def update_neuroplasticity(self, current_energy, current_pitch):
        """
        Updates the brain's 'physical structure' (baselines) based on repeated exposure.
        Alpha = learning rate (0.1 means slow, stable adaptation).
        """
        alpha = 0.1 
        self.state["baseline_energy"] = float((1 - alpha) * self.state["baseline_energy"] + (alpha * current_energy))
        self.state["baseline_pitch_var"] = float((1 - alpha) * self.state["baseline_pitch_var"] + (alpha * current_pitch))
        self.state["interaction_count"] += 1
        # No saving to file anymore

    def detect_neurogenesis_event(self, text, emotion, confidence):
        """
        If a highly specific/weird emotional mix happens (High Confidence), 
        store it as a new 'concept' (Neurogenesis).
        """
        if confidence > 0.90:
            new_memory = {"trigger": text[:20], "state": emotion, "timestamp": self.state["interaction_count"]}
            self.state["emotional_memories"].append(new_memory)

    def get_context(self):
        return self.state
