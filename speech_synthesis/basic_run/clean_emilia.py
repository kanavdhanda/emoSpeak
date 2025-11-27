import os, json, glob
from tqdm import tqdm
import webdataset as wds
from io import BytesIO
from pydub import AudioSegment
import numpy as np
import torch
import torchaudio

RAW_SHARDS = sorted(glob.glob("/data/emilia_raw/*.tar.gz"))
OUT_DIR = "/data/emilia_clean"
os.makedirs(OUT_DIR, exist_ok=True)

MIN_REC_QUALITY = 1.0
MAX_BACKGROUND_NOISE = 1.5

# ------------ MP3 decoder using FFmpeg (pydub) ------------
def decode_mp3(mp3_bytes):
    audio = AudioSegment.from_file(BytesIO(mp3_bytes), format="mp3")
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)

    if audio.channels > 1:
        samples = samples.reshape(-1, audio.channels).mean(axis=1)

    samples /= (1 << (8 * audio.sample_width - 1))  # normalize
    wav = torch.from_numpy(samples).unsqueeze(0)
    return wav, audio.frame_rate

# ------------ Load dataset WITHOUT decode() ---------------
dataset = (
    wds.WebDataset(RAW_SHARDS)
       .to_tuple("mp3", "json", "__key__")
)

count = 0

for mp3_bytes, meta_bytes, key in tqdm(dataset, desc="Cleaning"):

    # Parse metadata
    try:
        meta = json.loads(meta_bytes.decode("utf-8"))
    except:
        continue

    emo = meta.get("emotion_annotation", {})
    if not emo:
        continue

    # Quality filters
    rec_q = emo.get("Recording_Quality_best", -1)
    noise = emo.get("Background_Noise_best", 99)
    if rec_q < MIN_REC_QUALITY or noise > MAX_BACKGROUND_NOISE:
        continue

    # Decode mp3 safely
    try:
        wav, sr = decode_mp3(mp3_bytes)
    except:
        continue

    # Resample to 16k
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    # save
    out_name = key.replace("/", "_") + ".wav"
    wav_path = os.path.join(OUT_DIR, out_name)
    json_path = wav_path.replace(".wav", ".json")

    torchaudio.save(wav_path, wav, 16000)
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    count += 1

print(f"✔ DONE — saved {count} clean English samples → {OUT_DIR}")
