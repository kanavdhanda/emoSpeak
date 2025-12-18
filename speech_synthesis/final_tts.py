import sys
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import torchaudio

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav


# ============================================================
# Emotion Mapping (Human → Model)
# ============================================================
EMOTION_MAP = {
    "Cheerful": [
        "Contentment_best", "Elation_best", "Pleasure_Ecstasy_best",
        "Hope_Enthusiasm_Optimism_best", "Warm_vs._Cold_best", "Affection_best"
    ],
    "Warm": [
        "Warm_vs._Cold_best", "Affection_best", "Authenticity_best",
        "Soft_vs._Harsh_best"
    ],
    "Sad": [
        "Sadness_best", "Distress_best", "Vulnerable_vs._Emotionally_Detached_best",
        "Longing_best"
    ],
    "Angry": [
        "Anger_best", "Malevolence_Malice_best", "Impatience_and_Irritability_best",
        "Contempt_best"
    ],
    "Fear": [
        "Fear_best", "Helplessness_best", "Distress_best"
    ],
    "Pain": [
        "Pain_best", "Distress_best", "Sadness_best"
    ],
    "Proud": [
        "Pride_best", "Triumph_best"
    ],
    "Confident": [
        "Confident_vs._Hesitant_best", "Serious_vs._Humorous_best"
    ],
    "Tired": [
        "Fatigue_Exhaustion_best", "Emotional_Numbness_best"
    ],
    "Surprised": [
        "Astonishment_Surprise_best", "Awe_best"
    ]
}


# ============================================================
# Load emotion index once
# ============================================================
df = pd.read_csv("./emilia_emotion_time_index.csv")
EMO_COLS = [c for c in df.columns if c.endswith("_best")]

# Precompute normalized emotion matrix
MAT = df[EMO_COLS].values
MAT_NORM = normalize(MAT, axis=1)


# ============================================================
# Select best emotional prompt (PURE FUNCTION)
# ============================================================
def select_best_mapped(emotions, percentages, top_k=1):
    # Convert percentages → numeric weights
    weights = np.array([float(p.strip('%')) / 100 for p in percentages], dtype=float)
    weights = weights / (weights.sum() + 1e-12)

    # Build target vector
    target_vec = np.zeros(len(EMO_COLS), dtype=float)
    print(target_vec)

    for emo_name, w in zip(emotions, weights):
        mapped_cols = EMOTION_MAP.get(emo_name, [])
        for col in mapped_cols:
            if col in EMO_COLS:
                idx = EMO_COLS.index(col)
                target_vec[idx] += w
    print(target_vec)

    # Normalize target
    target_norm = normalize(target_vec.reshape(1, -1))[0]
    print(target_norm)

    # Cosine similarity = dot product (vectors already normalized)
    sims = MAT_NORM @ target_norm
    print(sims)

    # Top K indices
    top_idx = np.argsort(sims)[-top_k:][::-1]
    print(top_idx)

    # Return file paths
    return df.iloc[top_idx]["wav"].tolist()


# ============================================================
# Load CosyVoice only once
# ============================================================
MODEL_DIR = "/Users/kanavdhanda/.cache/modelscope/hub/models/iic/CosyVoice2-0.5B"

print("🔄 Loading CosyVoice2 model...")
cosyvoice = CosyVoice2(MODEL_DIR)
print("✅ CosyVoice2 loaded!")


# ============================================================
# Main API function
# ============================================================
def generate_emotional_tts(text, emotions, percentages, top_k=1):

    # Select emotional sample
    best_wav = select_best_mapped(emotions, percentages, top_k)[0]
    print(f"🎯 Selected prompt: {best_wav}")

    prompt_speech_16k = load_wav(best_wav, 16000)
    PROMPT_TEXT = df[df.wav == best_wav].iloc[0]["text"]



    output_path = "/data/CosyVoice/api_output.wav"
    
    # Generate voice
    for out in cosyvoice.inference_zero_shot(
        text,
        PROMPT_TEXT,
        prompt_speech_16k,
        stream=False
    ):
        torchaudio.save(output_path, out["tts_speech"], 22050)
        print(f"💾 Saved output → {output_path}")

    return output_path
