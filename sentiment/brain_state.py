import json
import os
import numpy as np

class NeuroState:
    def __init__(self, user_id="default_user"):
        self.filename = f"{user_id}_amygdala.json"
        self.state = self._load_state()

    def _load_state(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, ValueError):
                print(f"⚠️ Warning: Brain state file {self.filename} was corrupt. Resetting memory.")
                return self._get_default_state()
        return self._get_default_state()

    def _get_default_state(self):
        return {
            "baseline_energy": 0.5,       # Neuroplasticity: Moving average of volume
            "baseline_pitch_var": 0.5,    # Neuroplasticity: Moving average of expression
            "emotional_memories": [],     # Neurogenesis: Storing new complex patterns
            "interaction_count": 0
        }

    def update_neuroplasticity(self, current_energy, current_pitch):
        """
        Updates the brain's 'physical structure' (baselines) based on repeated exposure.
        Alpha = learning rate (0.1 means slow, stable adaptation).
        """
        alpha = 0.1 
        self.state["baseline_energy"] = float((1 - alpha) * self.state["baseline_energy"] + (alpha * current_energy))
        self.state["baseline_pitch_var"] = float((1 - alpha) * self.state["baseline_pitch_var"] + (alpha * current_pitch))
        self.state["interaction_count"] += 1
        self._save_state()

    def detect_neurogenesis_event(self, text, emotion, confidence):
        """
        If a highly specific/weird emotional mix happens (High Confidence), 
        store it as a new 'concept' (Neurogenesis).
        """
        if confidence > 0.90:
            # Check if we already have a memory similar to this (simplified)
            # In a real app, use Vector Search here.
            new_memory = {"trigger": text[:20], "state": emotion, "timestamp": self.state["interaction_count"]}
            self.state["emotional_memories"].append(new_memory)
            self._save_state()

    def _save_state(self):
        with open(self.filename, 'w') as f:
            json.dump(self.state, f, indent=4)
            
    def get_context(self):
        return self.state
