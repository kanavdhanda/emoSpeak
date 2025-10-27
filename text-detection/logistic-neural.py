import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sentence_transformers import SentenceTransformer

tqdm.pandas()

# ---------------- CONFIG ---------------- #
CSV_PATH = "data.csv"
TEXT_COLUMN = "text"
LABEL_COLUMN = "sentiment"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_PATH = "sentiment_nn.pt"
ENCODER_PATH = "label_encoder.joblib"
EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4
HIDDEN_DIM = 256
DROPOUT = 0.3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- MODEL ---------------- #
class SentimentNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.3):
        super(SentimentNN, self).__init__()
        self.num_classes = num_classes
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Logistic regression head
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

        # Activation: sigmoid for binary, softmax for multi-class
        if num_classes == 2:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.feature_extractor(x)
        logits = self.classifier(x)
        probs = self.activation(logits)
        return probs


# ---------------- DATA PREP ---------------- #
def load_and_prepare_data():
    df = pd.read_csv(CSV_PATH)
    if TEXT_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
        raise ValueError(f"Columns {TEXT_COLUMN} or {LABEL_COLUMN} not found in CSV")

    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).fillna("")
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(str)

    print(f"📊 Loaded {len(df)} samples. Generating embeddings using {EMBED_MODEL_NAME} ...")
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = np.vstack(df[TEXT_COLUMN].progress_apply(lambda x: embedder.encode(x, show_progress_bar=False)).values)

    le = LabelEncoder()
    labels = le.fit_transform(df[LABEL_COLUMN])
    joblib.dump(le, ENCODER_PATH)
    print(f"✅ Label encoder saved → {ENCODER_PATH}")

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )
    return X_train, X_test, y_train, y_test, len(le.classes_), embeddings.shape[1]


def create_dataloader(X, y, batch_size=32):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32 if len(np.unique(y)) == 2 else torch.long)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ---------------- TRAIN LOOP ---------------- #
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)

            if model.num_classes == 2:
                loss = criterion(outputs.squeeze(), y_batch)
            else:
                loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f}")


def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        preds = model(X_test_tensor)

        if model.num_classes == 2:
            preds = (preds.squeeze().cpu().numpy() > 0.5).astype(int)
        else:
            preds = torch.argmax(preds, dim=1).cpu().numpy()

        print("\n--- Evaluation ---")
        print(classification_report(y_test, preds))
        print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")


# ---------------- MAIN ---------------- #
def main():
    X_train, X_test, y_train, y_test, num_classes, input_dim = load_and_prepare_data()

    model = SentimentNN(input_dim, HIDDEN_DIM, num_classes, DROPOUT).to(DEVICE)
    criterion = nn.BCELoss() if num_classes == 2 else nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    train_loader = create_dataloader(X_train, y_train, BATCH_SIZE)

    print("🚀 Starting training...")
    train_model(model, train_loader, criterion, optimizer, EPOCHS)

    print("🧠 Evaluating model...")
    evaluate_model(model, X_test, y_test)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"✅ Model saved → {MODEL_PATH}")


if __name__ == "__main__":
    main()
