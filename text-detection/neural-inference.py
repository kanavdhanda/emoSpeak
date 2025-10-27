import torch
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from train_sentiment_nn import SentimentNN, HIDDEN_DIM, MODEL_PATH, ENCODER_PATH, EMBED_MODEL_NAME, DEVICE

# Load model + encoder
embedder = SentenceTransformer(EMBED_MODEL_NAME)
le = joblib.load(ENCODER_PATH)
input_dim = 384  # for MiniLM
num_classes = len(le.classes_)
model = SentimentNN(input_dim, HIDDEN_DIM, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

text = "I absolutely loved this product!"
embedding = embedder.encode(text)
embedding = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0).to(DEVICE)
pred = model(embedding)
label = le.inverse_transform([torch.argmax(pred, dim=1).item()])[0]
print(label)
