import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Config
MODEL_NAME = "distilbert-base-uncased"
CSV_PATH = "data.csv"
TEXT_COLUMN = "text"
LABEL_COLUMN = "sentiment"
BATCH_SIZE = 32
EPOCHS = 1
LR = 2e-5
MAX_LEN = 128
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def train_bert():
    print(f"🚀 Training BERT ({MODEL_NAME}) with 5-Fold CV on {DEVICE}...")
    
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"{CSV_PATH} not found.")

    # Load Data
    df = pd.read_csv(CSV_PATH)
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).fillna("")
    texts = df[TEXT_COLUMN].values
    labels = df[LABEL_COLUMN].values.astype(int)

    # Tokenize All
    print("Tokenizing data...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoded = tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    
    # 5-Fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    
    fold = 1
    for train_idx, test_idx in skf.split(input_ids, labels):
        print(f"\n--- Fold {fold} ---")
        
        # Split
        train_inputs, test_inputs = input_ids[train_idx], input_ids[test_idx]
        train_masks, test_masks = attention_mask[train_idx], attention_mask[test_idx]
        train_labels, test_labels = torch.tensor(labels[train_idx]), torch.tensor(labels[test_idx])
        
        # Datasets
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        test_data = TensorDataset(test_inputs, test_masks, test_labels)
        
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)
        
        # Model
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
        model.to(DEVICE)
        
        optimizer = AdamW(model.parameters(), lr=LR)
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        
        # Train
        model.train()
        for epoch in range(EPOCHS):
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for b_ids, b_mask, b_labels in loop:
                b_ids, b_mask, b_labels = b_ids.to(DEVICE), b_mask.to(DEVICE), b_labels.to(DEVICE)
                
                model.zero_grad()
                outputs = model(b_ids, attention_mask=b_mask, labels=b_labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                loop.set_postfix(loss=loss.item())
        
        # Eval
        model.eval()
        preds = []
        true_labels = []
        with torch.no_grad():
            for b_ids, b_mask, b_labels in test_loader:
                b_ids, b_mask = b_ids.to(DEVICE), b_mask.to(DEVICE)
                outputs = model(b_ids, attention_mask=b_mask)
                logits = outputs.logits
                batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
                preds.extend(batch_preds)
                true_labels.extend(b_labels.numpy())
        
        acc = accuracy_score(true_labels, preds)
        accuracies.append(acc)
        print(f"Fold {fold} Accuracy: {acc:.4f}")
        
        # Cleanup to save memory
        del model
        del optimizer
        del scheduler
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        fold += 1
        
    print(f"\n📈 Average BERT Accuracy: {np.mean(accuracies):.4f}")

if __name__ == "__main__":
    train_bert()
