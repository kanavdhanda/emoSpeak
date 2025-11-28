import os
import shutil
import numpy as np
import librosa
import whisper
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from brain_state import NeuroState
from response_generator import EmotionalGenerator
from collections import defaultdict
import uvicorn

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Models
asr_model = None
audio_classifier = None
text_classifier = None
generator = None

@app.on_event("startup")
async def startup_event():
    global asr_model, audio_classifier, text_classifier, generator
    print("⏳ Loading models...")
    
    # 1. Speech to Text
    asr_model = whisper.load_model("medium")
    
    # 2. Audio Emotion (Voice Tone)
    audio_classifier = pipeline(
        "audio-classification", 
        model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    )
    
    # 3. Text Emotion (Semantic Content) - NEW
    print("   Loading Text Emotion Model...")
    text_classifier = pipeline(
        "text-classification", 
        model="SamLowe/roberta-base-go_emotions", 
        top_k=None
    )
    
    generator = EmotionalGenerator()
    print("✅ All Models loaded.")

@app.post("/process_audio")
async def process_audio(
    file: UploadFile = File(...),
    brain_state: str = Form(...) # Expecting JSON string
):
    # Save temp file
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Load Brain from Frontend State
        import json
        try:
            state_dict = json.loads(brain_state)
        except:
            state_dict = None
            
        brain = NeuroState(state_dict=state_dict)

        # 1. Load Audio
        audio, sr = librosa.load(temp_filename, sr=16000)
        
        # 2. Speech to Text (Content)
        transcription_result = asr_model.transcribe(audio, fp16=False, language='english')
        text = transcription_result['text'].strip()
        
        if not text:
            return {"error": "No speech detected"}

        # 3. Multi-Modal Emotion Detection
        
        # A. Audio Analysis (Tone)
        audio_results = audio_classifier({'array': audio, 'sampling_rate': 16000}, top_k=None)
        audio_map = {
            'angry': 'anger', 'calm': 'neutral', 'disgust': 'disgust', 
            'fearful': 'fear', 'happy': 'joy', 'neutral': 'neutral', 
            'sad': 'sadness', 'surprised': 'surprise'
        }
        audio_scores = defaultdict(float)
        for res in audio_results:
            label = audio_map.get(res['label'], res['label'])
            audio_scores[label] += res['score']

        # B. Text Analysis (Semantics)
        text_results = text_classifier(text)[0]
        text_scores = defaultdict(float)
        for res in text_results:
            text_scores[res['label']] = res['score']

        # C. Adaptive Weighting
        interaction_count = brain.state["interaction_count"]
        voice_weight = min(0.5, 0.1 + (interaction_count * 0.02)) 
        text_weight = 1.0 - voice_weight
        
        # D. Fusion
        final_scores = {}
        all_labels = set(audio_scores.keys()) | set(text_scores.keys())
        
        for label in all_labels:
            s_a = audio_scores.get(label, 0.0)
            s_t = text_scores.get(label, 0.0)
            final_scores[label] = (s_a * voice_weight) + (s_t * text_weight)

        # Get Top Emotions
        sorted_emotions = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_emotions[:3]
        
        detected_emotion_str = ", ".join([f"{e[0]} ({e[1]:.0%})" for e in top_3])
        primary_emotion = top_3[0][0]
        confidence = top_3[0][1]

        # 4. Neuroplasticity Update
        rms = float(np.mean(librosa.feature.rms(y=audio)))
        brain.update_neuroplasticity(rms, 0.5)
        
        print(f"🧠 Fusion: Text({text_weight:.2f}) + Voice({voice_weight:.2f}) | Result: {detected_emotion_str}")

        # 5. Generate Response
        response_data = generator.generate_response(text, detected_emotion_str, brain.get_context())
        
        # 6. Update History
        brain.add_interaction(text, detected_emotion_str, response_data.get("text", ""))
        
        # Construct final response
        return {
            "user_input": {
                "text": text,
                "detected_emotion": detected_emotion_str, # Return full string "Joy (60%), Surprise (30%)"
                "adjusted_emotion": detected_emotion_str,
                "confidence": float(confidence)
            },
            "ai_response": response_data,
            "brain_state": brain.get_context() # Return updated state to frontend
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
