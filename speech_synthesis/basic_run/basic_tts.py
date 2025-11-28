import sys
import torchaudio
# IMPORTANT: We now import the 'CosyVoice2' class
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# --- 1. Fix the PYTHONPATH ---
# This is still correct and necessary
sys.path.append('third_party/Matcha-TTS')

# --- 2. Define Your Inputs ---
TEXT_TO_SPEAK = "Shut the box, Jack! Let's play a game."
MODEL_DIR = 'pretrained_models/CosyVoice2-0.5B'
OUTPUT_WAV = 'basic_output_v2.wav'

# --- 3. Define the Prompt (REQUIRED for CosyVoice2) ---
# We will use the default prompt file included in the GitHub repo
PROMPT_AUDIO_PATH = './asset/rahul_sighn.wav'
PROMPT_TEXT = 'My name is Rahul Singh.'

# --- 4. Load the Model ---
print("Loading CosyVoice2 model... (This may take a moment)")
try:
    # Use the correct CosyVoice2 class
    cosyvoice = CosyVoice2(MODEL_DIR)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please make sure you have run the download script first.")
    sys.exit(1)

# --- 5. Load the Prompt Audio ---
# The model expects prompts at a 16000Hz sample rate
prompt_speech_16k = load_wav(PROMPT_AUDIO_PATH, 16000)
print(f"Using prompt: {PROMPT_AUDIO_PATH}")

# --- 6. Generate Speech ---
# We now use the 'inference_zero_shot' method
print(f"Generating speech for: '{TEXT_TO_SPEAK}'")
for i, j in enumerate(
    cosyvoice.inference_zero_shot(
        TEXT_TO_SPEAK,
        PROMPT_TEXT,
        prompt_speech_16k,
        stream=False
    )
):
    # The output sample rate is 22050Hz
    torchaudio.save(
        f"output_v2_chunk_{i}.wav", # Save each generated chunk
        j['tts_speech'],
        22050
    )
    print(f"\nSuccess! Audio chunk {i} saved to: output_v2_chunk_{i}.wav")

print("Generation complete.")