import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

tqdm.pandas()

# ---------- CONFIG ---------- #
CSV_PATH = "data.csv"          # <-- dataset file
TEXT_COLUMN = "text"           # <-- column name with text
N_CLUSTERS = 3                 # <-- number of sentiment-like clusters
MODEL_PATH = "kmeans_sentiment_model.joblib"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMB_CACHE = "embeddings.npy"   # optional cache for faster startup


# ---------- TRAINING ---------- #
def train_model():
    print("⚙️ Training KMeans model as no saved model found...")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"{CSV_PATH} not found. Please provide your dataset.")

    df = pd.read_csv(CSV_PATH)
    if TEXT_COLUMN not in df.columns:
        raise ValueError(f"Missing required column: {TEXT_COLUMN}")

    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).fillna("")

    print(f"📊 Loaded {len(df)} samples. Generating embeddings using {EMBED_MODEL_NAME} ...")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = np.vstack(df[TEXT_COLUMN].progress_apply(lambda x: model.encode(x, show_progress_bar=False)).values)

    print(f"🧠 Training KMeans (n_clusters={N_CLUSTERS}) ...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    kmeans.fit(embeddings)

    # Save model + optional embedding cache
    joblib.dump(kmeans, MODEL_PATH)
    np.save(EMB_CACHE, embeddings)
    print(f"✅ Saved model to {MODEL_PATH} and cached embeddings to {EMB_CACHE}")

    # Attach cluster IDs for inspection
    df["cluster"] = kmeans.predict(embeddings)
    df.to_csv("sentiment_clusters.csv", index=False)
    print("📁 Saved clustered dataset → sentiment_clusters.csv")


# ---------- FASTAPI SERVER ---------- #
app = FastAPI(title="Sentiment Clustering Service", description="KMeans-based sentiment grouping API")

class SentimentRequest(BaseModel):
    text: str


def load_resources():
    """Load model and embedding model."""
    print("🔍 Loading model and resources...")
    kmeans = joblib.load(MODEL_PATH)
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    print("✅ Resources loaded successfully.")
    return kmeans, embedder


@app.on_event("startup")
def startup_event():
    global kmeans_model, embedder

    # Train if model missing
    if not os.path.exists(MODEL_PATH):
        train_model()

    kmeans_model, embedder = load_resources()


@app.post("/predict")
def predict_cluster(req: SentimentRequest):
    try:
        embedding = embedder.encode(req.text)
        cluster_id = int(kmeans_model.predict([embedding])[0])
        return {"cluster": cluster_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- MAIN ENTRY ---------- #
if __name__ == "__main__":
    uvicorn.run("sentiment_kmeans_service:app", host="0.0.0.0", port=8000, reload=False)
