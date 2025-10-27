import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

tqdm.pandas()

# ---------- CONFIG ---------- #
CSV_PATH = "data.csv"               # <-- your dataset
TEXT_COLUMN = "text"                # <-- column name with user text
LABEL_COLUMN = "sentiment"          # <-- column name with sentiment label
K_NEIGHBORS = 5
MODEL_PATH = "knn_sentiment_model.joblib"
ENCODER_PATH = "label_encoder.joblib"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---------- TRAINING ---------- #
def train_model():
    print("⚙️ Training model as no saved model found...")

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"{CSV_PATH} not found. Please provide your dataset.")

    df = pd.read_csv(CSV_PATH)
    if TEXT_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
        raise ValueError(f"Missing required columns: {TEXT_COLUMN}, {LABEL_COLUMN}")

    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).fillna("")
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str)

    le = LabelEncoder()
    y = le.fit_transform(df[LABEL_COLUMN])

    print(f"📊 Loaded {len(df)} samples. Generating embeddings using {EMBED_MODEL_NAME} ...")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = np.vstack(df[TEXT_COLUMN].progress_apply(lambda x: model.encode(x, show_progress_bar=False)).values)

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"🧠 Training KNN (k={K_NEIGHBORS}) ...")
    knn = KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
    knn.fit(X_train, y_train)

    preds = knn.predict(X_test)
    print("\n--- Model Evaluation ---")
    print(classification_report(y_test, preds))
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")

    joblib.dump(knn, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"✅ Saved model to {MODEL_PATH} and encoder to {ENCODER_PATH}")


# ---------- FASTAPI SERVER ---------- #
app = FastAPI(title="Sentiment Classifier", description="KNN-based Sentiment Analysis API")

class SentimentRequest(BaseModel):
    texts: list[str]  # <-- multiple sentences now


def load_resources():
    """Load model, encoder, and embedding model."""
    print("🔍 Loading model and resources...")
    knn = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    print("✅ Resources loaded successfully.")
    return knn, le, embedder


@app.on_event("startup")
def startup_event():
    global knn_model, label_encoder, embedder

    # Train if not saved yet
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        train_model()

    knn_model, label_encoder, embedder = load_resources()


@app.post("/predict")
def predict_sentiment(req: SentimentRequest):
    try:
        if not req.texts:
            raise HTTPException(status_code=400, detail="No texts provided.")

        # Encode all sentences
        embeddings = embedder.encode(req.texts, show_progress_bar=False)
        preds = knn_model.predict(embeddings)
        sentiments = label_encoder.inverse_transform(preds)

        # Count sentiments
        sentiment_counts = pd.Series(sentiments).value_counts()
        dominant_sentiment = sentiment_counts.idxmax().lower()

        # Simplify to positive or negative (based on dominant sentiment)
        if "neg" in dominant_sentiment:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "positive"

        return {
            "individual_sentiments": sentiments.tolist(),
            "overall_sentiment": overall_sentiment,
            "distribution": sentiment_counts.to_dict()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- MAIN ENTRY ---------- #
if __name__ == "__main__":
    uvicorn.run("sentiment_service:app", host="0.0.0.0", port=8000, reload=False)
