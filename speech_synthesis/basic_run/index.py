import os, json, glob
import pandas as pd
from tqdm import tqdm

CLEAN_DIR = "/data/emilia_clean"

records = []

for json_path in tqdm(glob.glob(os.path.join(CLEAN_DIR, "*.json"))):
    wav_path = json_path.replace(".json", ".wav")

    with open(json_path) as f:
        meta = json.load(f)

    emo = meta.get("emotion_annotation", {})
    entry = {"wav": wav_path}

    for k, v in emo.items():
        entry[k] = v

    records.append(entry)

df = pd.DataFrame(records)
df.to_csv("/data/emilia_emotion_index.csv", index=False)

print("Index saved: /data/emilia_emotion_index.csv")
