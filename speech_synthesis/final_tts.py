import sys
import os

PROJECT_ROOT = "/data/CosyVoice"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# ================================
# Load emotion index only once
# ================================
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("/data/emilia_emotion_index.csv")

HUMAN_TO_MODEL = {
    "Cheerful": ["Contentment_best", "Elation_best", "Pleasure_Ecstasy_best"],
    "Warm": ["Warm_vs._Cold_best", "Affection_best"],
    "Sad": ["Sadness_best"],
    "Angry": ["Anger_best", "Malevolence_Malice_best"],
    "Fear": ["Fear_best"],
    "Pain": ["Pain_best"],
}

def select_best_blended(emotions, percentages, top_k=1):

    # Convert percentages to 0–1 weights
    weights = [float(p.strip('%')) / 100 for p in percentages]

    # Build target vector
    target = {}
    for emotion_name, w in zip(emotions, weights):
        for emo_col in HUMAN_TO_MODEL[emotion_name]:
            target[emo_col] = target.get(emo_col, 0.0) + w

    # Dataset emotion columns
    emo_cols = [c for c in df.columns if c.endswith("_best")]

    # Convert target dict → fixed-length vector
    target_vec = np.array([target.get(col, 0.0) for col in emo_cols]).reshape(1, -1)

    # Matrix of all sample embeddings
    mat = df[emo_cols].values

    # Cosine similarity
    sims = cosine_similarity(target_vec, mat)[0]

    df["similarity"] = sims

    # Pick best matching samples
    best = df.sort_values("similarity", ascending=False).head(top_k)

    return best["wav"].tolist()

# ================================
# Load CosyVoice only once (fast)
# ================================
MODEL_DIR = "/data/CosyVoice/pretrained_models/CosyVoice2-0.5B"

print("🔄 Loading CosyVoice2 model once...")
cosyvoice = CosyVoice2(MODEL_DIR)
print("✅ Model loaded!")


# ============================================================
# Main function used by the API
# ============================================================
def generate_emotional_tts(text, emotions, percentages, top_k=1):
    output_path = f"/data/CosyVoice/api_output.wav"

    # Pick best prompt WAV
    best_wav = select_best_blended(emotions, percentages, top_k)[0]
    print(best_wav)

    prompt_speech_16k = load_wav(best_wav, 16000)

    PROMPT_TEXT = "This is the emotional tone of the speaker."


    # Run TTS
    i = 0

    for out in cosyvoice.inference_zero_shot(
        text,
        PROMPT_TEXT,
        prompt_speech_16k,
        stream=False
    ):
        torchaudio.save(output_path, out["tts_speech"], 22050)
        print(f"Saved: {output_path}")
        i += 1

    return output_path
