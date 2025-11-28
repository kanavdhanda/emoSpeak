import os
import shutil
import numpy as np
import librosa
import whisper
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
from brain_state import NeuroState
from response_generator import EmotionalGenerator
from textblob import TextBlob
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
emotion_classifier = None
brain = None
generator = None

@app.on_event("startup")
async def startup_event():
    global asr_model, emotion_classifier, brain, generator
    print("⏳ Loading models...")
    asr_model = whisper.load_model("base")
    emotion_classifier = pipeline(
        "audio-classification", 
        model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    )
    brain = NeuroState()
    generator = EmotionalGenerator()
    print("✅ Models loaded.")

@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    global brain
    
    # Save temp file
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 1. Load Audio
        # librosa loads as float32, mono by default, sr=22050. We want 16000 for models.
        audio, sr = librosa.load(temp_filename, sr=16000)
        
        # 2. Emotion Analysis (Tonality)
        emotion_result = emotion_classifier(
            {'array': audio, 'sampling_rate': 16000}, 
            top_k=1
        )
        voice_mood = emotion_result[0]['label']
        voice_score = emotion_result[0]['score']
        
        # 3. Speech to Text (Content)
        # Whisper expects raw audio
        transcription_result = asr_model.transcribe(
            audio, 
            fp16=False, 
            language='english'
        )
        text = transcription_result['text'].strip()
        
        if not text:
            return {"error": "No speech detected"}

        # 4. Neuroplasticity & Adaptation
        rms = float(np.mean(librosa.feature.rms(y=audio)))
        baseline = brain.state["baseline_energy"]
        
        # Hybrid Logic: Audio + Text Sentiment
        blob = TextBlob(text)
        text_sentiment = blob.sentiment.polarity # -1.0 (Negative) to 1.0 (Positive)
        
        print(f"🔍 Analysis: Audio='{voice_mood}' | Text Sentiment={text_sentiment:.2f} | Energy={rms:.4f} (Base: {baseline:.4f})")

        if voice_mood == "anger":
            if text_sentiment > 0.3:
                adjusted_emotion = "passionate_excitement" # High energy + Positive text
            elif rms < (baseline * 1.2):
                adjusted_emotion = "frustrated_calm" # Low energy anger
            else:
                adjusted_emotion = "anger"
        elif voice_mood == "neutral" and text_sentiment > 0.5:
            adjusted_emotion = "happy"
        else:
            adjusted_emotion = voice_mood
            
        brain.update_neuroplasticity(rms, 0.5)
        
        # 5. Generate Response
        response_data = generator.generate_response(text, adjusted_emotion, brain.get_context())
        
        # Construct final response
        return {
            "user_input": {
                "text": text,
                "detected_emotion": voice_mood,
                "adjusted_emotion": adjusted_emotion,
                "confidence": float(voice_score)
            },
            "ai_response": response_data,
            "brain_state": {
                "baseline_energy": brain.state["baseline_energy"],
                "interaction_count": brain.state["interaction_count"]
            }
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
