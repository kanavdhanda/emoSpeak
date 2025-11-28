import time
import queue
import threading
import numpy as np
import sounddevice as sd
import whisper
import librosa
from transformers import pipeline
from brain_state import NeuroState
from response_generator import EmotionalGenerator

class RealTimeEmotionPipeline:
    def __init__(self, model_size="base", chunk_duration=5):
        print("⏳ Loading models... (This might take a moment)")
        
        # 1. Load Whisper (Speech-to-Text)
        # We use the standard OpenAI Whisper model
        self.asr_model = whisper.load_model(model_size)
        
        # 2. Load Audio Emotion Model (Tonality)
        # device=0 uses GPU if available. Remove 'device=0' if you only have CPU.
        self.emotion_classifier = pipeline(
            "audio-classification", 
            model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        )
        
        # Audio Settings
        self.fs = 16000  # Sample rate (Required by both models)
        self.chunk_duration = chunk_duration # How many seconds to record at a time
        self.chunk_samples = int(self.fs * self.chunk_duration)
        
        # Queue for thread safety
        self.audio_queue = queue.Queue()
        self.running = False
        
        # Initialize Brain and Generator
        self.brain = NeuroState()
        self.generator = EmotionalGenerator()
        
        print("✅ Models loaded. Ready to start.")

    def audio_callback(self, indata, frames, time, status):
        """Producer: This runs in a background thread by sounddevice"""
        if status:
            print(status)
        # Add a copy of the current audio chunk to the queue
        self.audio_queue.put(indata.copy())

    def process_audio(self):
        """Consumer: Processes audio chunks as they arrive"""
        while self.running:
            try:
                # Get audio chunk from queue (wait up to 1s)
                audio_data = self.audio_queue.get(timeout=1)
                
                # Flatten to 1D array (mono) for models
                audio_flat = audio_data.flatten().astype(np.float32)

                # --- PIPELINE STEP 1: TONALITY (Emotion) ---
                # We pass the raw numpy array directly to the pipeline
                # The pipeline expects a dict with 'array' and 'sampling_rate'
                emotion_result = self.emotion_classifier(
                    {'array': audio_flat, 'sampling_rate': self.fs}, 
                    top_k=1
                )
                voice_mood = emotion_result[0]['label']
                voice_score = emotion_result[0]['score']

                # --- PIPELINE STEP 2: CONTENT (Whisper) ---
                # Whisper expects float32 audio. We can pass the array directly.
                # Use fp16=False if on CPU to avoid warnings
                transcription_result = self.asr_model.transcribe(
                    audio_flat, 
                    fp16=False, 
                    language='english'
                )
                text = transcription_result['text'].strip()

                # --- OUTPUT ---
                if text: # Only print if someone actually spoke
                    # 2. Extract Features for Adaptation
                    rms = np.mean(librosa.feature.rms(y=audio_flat)) # Energy
                    
                    # 3. Neuroplasticity Check
                    # Compare current energy to user's historical baseline
                    baseline = self.brain.state["baseline_energy"]
                    
                    if voice_mood == "anger" and rms < (baseline * 1.2):
                        # If model says "Anger" but volume is normal for this user,
                        # Downgrade it to "Annoyance" or "Serious"
                        adjusted_emotion = "frustrated_calm" 
                    else:
                        adjusted_emotion = voice_mood

                    # 4. Update the Brain
                    self.brain.update_neuroplasticity(rms, 0.5) # 0.5 is placeholder for pitch
                    
                    # 5. Generate Response
                    response = self.generator.generate_response(text, adjusted_emotion, self.brain.get_context())
                    
                    print(f"\n🗣️  User Said: \"{text}\"")
                    print(f"🎭 Voice Tone: {voice_mood.upper()} (Confidence: {voice_score:.2f})")
                    print(f"🧠 Adjusted Emotion: {adjusted_emotion} (Baseline: {baseline:.2f})")
                    print(f"🤖 AI Thought: {response.get('thought_process')}")
                    print(f"🗣️ AI Says: {response.get('text_response')}")
                    print(f"🎛️ TTS Settings: {response.get('tts_directive')}")
                    print("-" * 40)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing chunk: {e}")

    def logic_gate(self, text, voice_mood):
        """
        Decides the final sentiment based on conflict between text and tone.
        """
        # Example Logic: Tone overrides Text for "Sarcasm" or "Hidden Distress"
        
        # If text is positive but voice is angry -> Potential Argument/Sarcasm
        if "good" in text.lower() and voice_mood == "angry":
            return "DEFENSIVE/SARCASTIC"
        
        # If text is neutral but voice is fearful -> Distress
        if voice_mood == "fear":
            return "DISTRESS (Prioritize Audio)"
            
        # Default: Trust the audio mood to color the text
        return f"{voice_mood} interaction"

    def start(self):
        self.running = True
        
        # Start the Consumer thread (Processing)
        processing_thread = threading.Thread(target=self.process_audio)
        processing_thread.start()
        
        print(f"\n🎤 Listening... (Chunks of {self.chunk_duration}s). Press Ctrl+C to stop.\n")
        
        # Start the Producer (Recording)
        with sd.InputStream(callback=self.audio_callback, 
                            channels=1, 
                            samplerate=self.fs, 
                            blocksize=self.chunk_samples):
            try:
                while self.running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n🛑 Stopping...")
                self.running = False
                processing_thread.join()

if __name__ == "__main__":
    # Create and start the pipeline
    pipeline_obj = RealTimeEmotionPipeline(model_size="base", chunk_duration=5)
    pipeline_obj.start()