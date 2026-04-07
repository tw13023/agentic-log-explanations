"""
BGL Anomaly Detection Screener
-------------------------------
Loads the pre-trained AllLinLog model and runs inference on the BGL test set.

Usage:
    python BGL_screener.py

Requirements: torch, numpy, pandas, scikit-learn, tqdm, tiktoken, linformer
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score,
)
from tqdm import tqdm
import tiktoken
from linformer import Linformer
import time
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Resolve repo root so the script works from any working directory
REPO_ROOT = Path(__file__).resolve().parent.parent

LOG_FILE   = str(REPO_ROOT / 'logs' / 'BGL.log')
MODEL_PATH = str(REPO_ROOT / 'best_model' / 'best_model_20250724_072857.pth')

WINDOWS_SIZE = 10
STEP_SIZE    = 10
TRAIN_RATIO  = 0.7
SEED         = 42
BATCH_SIZE   = 8
MAX_TOKEN_LENGTH = 4096   # updated after data loading

# Model hyperparameters (must match training checkpoint)
CL100K_VOCAB_SIZE = 100264   # GPT-4 BPE
EMBEDDING_DIM     = 128
FF_HIDDEN_DIM     = 128
NUM_LAYERS        = 1
NUM_HEADS         = 4
K                 = 32       # Linformer projection dimension
DROPOUT           = 0.5


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class LogDataset(Dataset):
    def __init__(self, sessions):
        self.sessions = sessions

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        return self.sessions[idx]


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------
def load_gpt4_tokenizer():
    print("Loading cl100k_base (GPT-4) tokenizer...")
    return tiktoken.get_encoding("cl100k_base")


def tokenize_and_construct_input(log_sequence, tokenizer, max_len=4096):
    input_ids  = []
    segment_ids = []

    allowed_special = {"<|startoftext|>", "<|endoftext|>"}
    bos_token = tokenizer.encode("<|startoftext|>", allowed_special=allowed_special)[0]
    eos_token = tokenizer.encode("<|endoftext|>",   allowed_special=allowed_special)[0]

    for i, log in enumerate(log_sequence):
        tokens = tokenizer.encode(log, allowed_special=allowed_special)
        if i == 0:
            tokens = [bos_token] + tokens
        tokens = tokens + [eos_token]
        input_ids.extend(tokens)
        segment_ids.extend([i] * len(tokens))
        input_ids   = input_ids[:max_len]
        segment_ids = segment_ids[:max_len]

    return input_ids, segment_ids


def create_sessions_with_segment_ids(log_data, tokenizer, windows_size, step_size):
    sessions = []
    print("Creating sessions...")
    for i in tqdm(range(0, len(log_data) - windows_size, step_size), desc="Processing Sessions", mininterval=1.0, dynamic_ncols=True):
        logs_in_session = []
        label = 0
        for j in range(i, i + windows_size):
            content = log_data[j]
            if content[0] != "-":
                label = 1
            content = content[content.find(' ') + 1:]
            logs_in_session.append(content)

        input_ids, segment_ids = tokenize_and_construct_input(logs_in_session, tokenizer)
        sessions.append({
            "input_ids":     input_ids,
            "segment_ids":   segment_ids,
            "session_label": label,
        })
    return sessions


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, max_seq_len, segment_vocab_size, embedding_dim=128):
        super().__init__()
        self.token_embedding    = nn.Embedding(vocab_size,         embedding_dim)
        self.segment_embedding  = nn.Embedding(segment_vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len,        embedding_dim)

    def forward(self, input_ids, segment_ids, position_ids=None):
        if position_ids is None:
            position_ids = (
                torch.arange(input_ids.size(1), device=input_ids.device)
                .unsqueeze(0)
                .repeat(input_ids.size(0), 1)
            )
        return (
            self.token_embedding(input_ids)
            + self.segment_embedding(segment_ids)
            + self.position_embedding(position_ids)
        )


class LinformerEncoderLayer(nn.Module):
    def __init__(self, embedding_dim, max_seq_len, num_heads=2, ff_hidden_dim=128, k=128, dropout=0.1):
        super().__init__()
        self.self_attention = Linformer(
            dim=embedding_dim,
            seq_len=max_seq_len,
            depth=1,
            heads=num_heads,
            k=k,
            one_kv_head=True,
            share_kv=True,
        )
        self.norm1   = nn.LayerNorm(embedding_dim)
        self.ffn     = nn.Sequential(
            nn.Linear(embedding_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embedding_dim),
        )
        self.norm2   = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x + self.dropout(self.self_attention(x)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class LinformerTransformerEncoder(nn.Module):
    def __init__(self, num_layers, embedding_dim, max_seq_len, num_heads=2, ff_hidden_dim=128, k=128, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            LinformerEncoderLayer(embedding_dim, max_seq_len, num_heads, ff_hidden_dim, k, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class AllLinLog(nn.Module):
    def __init__(self, vocab_size, max_seq_len, segment_vocab_size, embedding_dim=128,
                 num_layers=1, num_heads=2, ff_hidden_dim=128, k=128, num_classes=2, dropout=0.1):
        super().__init__()
        self.embedding_layer = EmbeddingLayer(vocab_size, max_seq_len, segment_vocab_size, embedding_dim)
        self.encoder         = LinformerTransformerEncoder(num_layers, embedding_dim, max_seq_len, num_heads, ff_hidden_dim, k, dropout)
        self.fc              = nn.Linear(embedding_dim, num_classes)

    def forward(self, input_ids, segment_ids, position_ids, attention_mask=None):
        embeddings     = self.embedding_layer(input_ids, segment_ids, position_ids)
        encoder_output = self.encoder(embeddings)
        pooled_output  = torch.mean(encoder_output, dim=1)
        return self.fc(pooled_output)


# ---------------------------------------------------------------------------
# DataLoader collate
# ---------------------------------------------------------------------------
def collate_fn(batch):
    input_ids     = [torch.tensor(item["input_ids"],   dtype=torch.long) for item in batch]
    segment_ids   = [torch.tensor(item["segment_ids"], dtype=torch.long) for item in batch]
    session_labels = torch.tensor([item["session_label"] for item in batch], dtype=torch.long)

    padded_input_ids   = pad_sequence(input_ids,   batch_first=True, padding_value=0)[:, :MAX_TOKEN_LENGTH]
    padded_segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value=0)[:, :MAX_TOKEN_LENGTH]
    attention_masks    = (padded_input_ids != 0).long()

    return padded_input_ids, padded_segment_ids, attention_masks, session_labels


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def evaluate_test_set(model, test_loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Running Inference", mininterval=1.0, dynamic_ncols=True):
            input_ids, segment_ids, attention_masks, labels = [b.to(device) for b in batch]
            position_ids = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0).expand(input_ids.size(0), -1)
            logits = model(input_ids, segment_ids, position_ids, attention_masks)
            probs  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def screen_logs(log_messages, model, tokenizer, device, windows_size=10):
    """Screen a fixed-size window of BGL log messages for anomalies."""
    model.eval()

    if len(log_messages) < windows_size:
        log_messages = log_messages + [""] * (windows_size - len(log_messages))
    log_messages = log_messages[:windows_size]

    input_ids, segment_ids = tokenize_and_construct_input(log_messages, tokenizer, MAX_TOKEN_LENGTH)

    input_ids_tensor   = torch.tensor([input_ids],   dtype=torch.long).to(device)
    segment_ids_tensor = torch.tensor([segment_ids], dtype=torch.long).to(device)

    if input_ids_tensor.size(1) < MAX_TOKEN_LENGTH:
        pad_size = MAX_TOKEN_LENGTH - input_ids_tensor.size(1)
        input_ids_tensor   = torch.nn.functional.pad(input_ids_tensor,   (0, pad_size), value=0)
        segment_ids_tensor = torch.nn.functional.pad(segment_ids_tensor, (0, pad_size), value=0)

    attention_mask = (input_ids_tensor != 0).long()

    with torch.no_grad():
        position_ids = torch.arange(input_ids_tensor.size(1), device=input_ids_tensor.device).unsqueeze(0).expand(input_ids_tensor.size(0), -1)
        logits = model(input_ids_tensor, segment_ids_tensor, position_ids, attention_mask)
        probs  = torch.softmax(logits, dim=1)
        pred   = logits.argmax(dim=1).item()

    return {
        "prediction":          "Anomalous" if pred == 1 else "Normal",
        "anomaly_probability": probs[0, 1].item(),
        "normal_probability":  probs[0, 0].item(),
        "confidence":          probs[0, pred].item(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    global MAX_TOKEN_LENGTH

    set_seed(SEED)
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    print(f"Log file:     {LOG_FILE}")
    print(f"Model path:   {MODEL_PATH}")

    # --- Load data ---
    print("\nLoading logs from:", LOG_FILE)
    start_time = time.time()
    with open(LOG_FILE, mode="r", encoding="utf8") as f:
        logs = [x.strip() for x in tqdm(f, desc="Reading Logs", mininterval=1.0, dynamic_ncols=True)]
    print(f"Loaded {len(logs)} logs in {time.time() - start_time:.2f} seconds.")

    tokenizer   = load_gpt4_tokenizer()
    all_sessions = create_sessions_with_segment_ids(logs, tokenizer, WINDOWS_SIZE, STEP_SIZE)

    token_lengths    = [len(s["input_ids"]) for s in all_sessions]
    MAX_TOKEN_LENGTH = max(token_lengths)
    print(f"Max tokens in sessions: {MAX_TOKEN_LENGTH}")

    # Stratified split (same as training)
    session_labels = [s["session_label"] for s in all_sessions]
    train_sessions, temp_sessions = train_test_split(
        all_sessions, test_size=(1 - TRAIN_RATIO),
        stratify=session_labels, random_state=42,
    )
    val_sessions, test_sessions = train_test_split(
        temp_sessions, test_size=0.5,
        stratify=[s["session_label"] for s in temp_sessions], random_state=42,
    )
    print(f"\nTrain: {len(train_sessions)} | Val: {len(val_sessions)} | Test: {len(test_sessions)}")
    test_normal    = sum(s["session_label"] == 0 for s in test_sessions)
    test_anomalous = sum(s["session_label"] == 1 for s in test_sessions)
    print(f"Test set => Normal: {test_normal} | Anomalous: {test_anomalous}")
    print(f"Anomalous ratio: {test_anomalous / (test_normal + test_anomalous):.2%}")

    # DataLoader
    test_loader = DataLoader(
        LogDataset(test_sessions),
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    print(f"Test DataLoader created with {len(test_loader)} batches")

    # --- Load model ---
    print(f"\nLoading model from: {MODEL_PATH}")
    model = AllLinLog(
        vocab_size=CL100K_VOCAB_SIZE,
        max_seq_len=MAX_TOKEN_LENGTH,
        segment_vocab_size=WINDOWS_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        ff_hidden_dim=FF_HIDDEN_DIM,
        k=K,
        num_classes=2,
        dropout=DROPOUT,
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / (1024**2):.2f} MB")

    # --- Evaluate ---
    print("\n" + "=" * 60)
    print("RUNNING INFERENCE ON TEST SET")
    print("=" * 60)
    predictions, labels, probabilities = evaluate_test_set(model, test_loader, device)

    # --- Results ---
    target_names = ["Normal", "Anomalous"]
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(labels, predictions, target_names=target_names, digits=5))

    print("=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    cm     = confusion_matrix(labels, predictions)
    cm_df  = pd.DataFrame(cm, index=target_names, columns=[f"Pred_{n}" for n in target_names])
    print(cm_df)

    accuracy  = (predictions == labels).mean()
    tn, fp, fn, tp = cm.ravel()
    precision = precision_score(labels, predictions, average="macro")
    recall    = recall_score(labels,    predictions, average="macro")
    f1        = f1_score(labels,        predictions, average="macro")

    print("\n" + "=" * 60)
    print("SUMMARY METRICS")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.5f}")
    print(f"Precision: {precision:.5f}  (macro avg)")
    print(f"Recall:    {recall:.5f}  (macro avg)")
    print(f"F1-Score:  {f1:.5f}  (macro avg)")
    print(f"True Positives (Anomalies detected):       {tp}")
    print(f"True Negatives (Normal correctly identified): {tn}")
    print(f"False Positives (False alarms):            {fp}")
    print(f"False Negatives (Missed anomalies):        {fn}")

    # --- Demo ---
    print("\n" + "=" * 60)
    print("DEMO: Screening sample sessions from test set")
    print("=" * 60)
    normal_samples    = [s for s in test_sessions if s["session_label"] == 0][:2]
    anomalous_samples = [s for s in test_sessions if s["session_label"] == 1][:2]

    for i, sample in enumerate(normal_samples + anomalous_samples):
        actual = "Normal" if sample["session_label"] == 0 else "Anomalous"
        input_ids_t   = torch.tensor([sample["input_ids"]],   dtype=torch.long).to(device)
        segment_ids_t = torch.tensor([sample["segment_ids"]], dtype=torch.long).to(device)

        if input_ids_t.size(1) < MAX_TOKEN_LENGTH:
            pad_size = MAX_TOKEN_LENGTH - input_ids_t.size(1)
            input_ids_t   = torch.nn.functional.pad(input_ids_t,   (0, pad_size), value=0)
            segment_ids_t = torch.nn.functional.pad(segment_ids_t, (0, pad_size), value=0)

        attention_mask = (input_ids_t != 0).long()
        with torch.no_grad():
            position_ids = torch.arange(input_ids_t.size(1), device=input_ids_t.device).unsqueeze(0).expand(input_ids_t.size(0), -1)
            logits = model(input_ids_t, segment_ids_t, position_ids, attention_mask)
            probs  = torch.softmax(logits, dim=1)
            pred   = "Anomalous" if logits.argmax(dim=1).item() == 1 else "Normal"

        status = "[OK]" if pred == actual else "[WRONG]"
        print(f"\nSample {i+1}: Actual={actual}, Predicted={pred} {status}")
        print(f"  Anomaly probability: {probs[0, 1].item():.4f}")


if __name__ == "__main__":
    main()
