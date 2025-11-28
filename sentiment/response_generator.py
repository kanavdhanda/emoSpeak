import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class EmotionalGenerator:
    def __init__(self):
        # Ensure API key is set. 
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("⚠️ WARNING: GOOGLE_API_KEY not found in environment variables or .env file.")
        else:
            genai.configure(api_key=api_key)
            
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def generate_response(self, user_text, user_emotion, brain_state):
        # 1. Construct the Context
        prompt_context = f"""
        User Input: "{user_text}"
        Detected Emotion: {user_emotion}
        
        USER PROFILE (Neuroplasticity Data):
        - Typical Energy Level: {brain_state.get('baseline_energy', 0.5)}
        - Total Interactions: {brain_state.get('interaction_count', 0)}
        - Past Emotional Anchors: {str(brain_state.get('emotional_memories', [])[-3:])} 
        """

        # 2. The System Directive
        system_prompt = """
        You are an empathetic AI with an evolving digital brain. 
        Analyze the user's input and their emotional profile.
        
        Your Goal:
        1. Generate a verbal response text.
        2. Analyze your own emotional state for the response.
        
        OUTPUT FORMAT (Strict JSON):
        {
          "text": "The actual words to speak.",
          "emotion": ["Primary Emotion", "Secondary Emotion"],
          "percentage": ["XX%", "YY%"]
        }
        """

        try:
            # 3. Call Gemini
            response = self.model.generate_content(
                system_prompt + "\n\n" + prompt_context,
                generation_config={"response_mime_type": "application/json"}
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"Error generating response: {e}")
            return {
                "text": "I'm having trouble thinking right now. Please check my API key.",
                "emotion": ["Confused", "Neutral"],
                "percentage": ["100%", "0%"]
            }
