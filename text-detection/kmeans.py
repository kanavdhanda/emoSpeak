import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

tqdm.pandas()

# ---------- CONFIG ---------- #
CSV_PATH = "data.csv"          # <-- dataset file
TEXT_COLUMN = "text"           # <-- column name with text
LABEL_COLUMN = "sentiment"       # <-- column name with label
N_CLUSTERS = 2                 # <-- number of sentiment-like clusters
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
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str) # Ensure labels are strings

    print(f"📊 Loaded {len(df)} samples. Generating embeddings using {EMBED_MODEL_NAME} ...")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = model.encode(df[TEXT_COLUMN].tolist(), show_progress_bar=True)
    labels = df[LABEL_COLUMN].values

    # 5-Fold Cross Validation
    print(f"\n🔄 Starting 5-Fold Cross-Validation (n_clusters={N_CLUSTERS})...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []

    fold = 1
    for train_index, test_index in skf.split(embeddings, labels):
        X_train, X_test = embeddings[train_index], embeddings[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        kmeans_fold = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
        kmeans_fold.fit(X_train)
        
        # Map clusters to labels
        train_clusters = kmeans_fold.labels_
        cluster_map = {}
        for i in range(N_CLUSTERS):
            mask = (train_clusters == i)
            if np.any(mask):
                # Find most frequent label in this cluster
                cluster_labels = y_train[mask]
                most_common = pd.Series(cluster_labels).mode()[0]
                cluster_map[i] = most_common
            else:
                cluster_map[i] = "unknown" # Should not happen usually

        # Predict on test
        test_clusters = kmeans_fold.predict(X_test)
        preds = np.array([cluster_map[c] for c in test_clusters])
        
        acc = accuracy_score(y_test, preds)
        accuracies.append(acc)
        print(f"Fold {fold}: Accuracy = {acc:.4f}")
        fold += 1

    mean_acc = np.mean(accuracies)
    print(f"\n📈 Average Accuracy over 5 folds: {mean_acc:.4f}")

    print(f"🧠 Training KMeans (n_clusters={N_CLUSTERS}) on all data...")
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
    # uvicorn.run("sentiment_kmeans_service:app", host="0.0.0.0", port=8000, reload=False)
    train_model()
