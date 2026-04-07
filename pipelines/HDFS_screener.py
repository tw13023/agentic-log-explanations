"""
HDFS Anomaly Detection Screener
---------------------------------
Loads the pre-trained AllLinLog model and runs inference on the HDFS test set.

Usage:
    python HDFS_screener.py

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
from datetime import datetime
import time
import random
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Resolve repo root so the script works from any working directory
REPO_ROOT = Path(__file__).resolve().parent.parent

LOG_FILE   = str(REPO_ROOT / 'logs' / 'HDFS.log')
LABEL_FILE = str(REPO_ROOT / 'logs' / 'anomaly_label_HDFS.csv')
MODEL_PATH = str(REPO_ROOT / 'best_model_HDFS' / 'best_model_HDFS20250804_201746.pth')

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
SEED        = 42
BATCH_SIZE  = 8
MAX_TOKEN_LENGTH = 18000   # updated after data loading

# Model hyperparameters (must match training checkpoint)
CL100K_VOCAB_SIZE  = 100264   # GPT-4 BPE
EMBEDDING_DIM      = 128
FF_HIDDEN_DIM      = 128
NUM_LAYERS         = 1
NUM_HEADS          = 4
K                  = 32       # Linformer projection dimension
DROPOUT            = 0.5
MAX_SEGMENT_LENGTHS = 298     # must match trained model checkpoint


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
        s = self.sessions[idx]
        return {
            "input_ids":     s["input_ids"],
            "segment_ids":   s["segment_ids"],
            "session_label": s["session_label"],
        }


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------
def load_gpt4_tokenizer():
    print("Loading cl100k_base (GPT-4) tokenizer...")
    return tiktoken.get_encoding("cl100k_base")


def tokenize_and_construct_input(log_sequence, tokenizer, max_len=18000):
    input_ids   = []
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

    if len(input_ids) > max_len:
        input_ids   = input_ids[:max_len]
        segment_ids = segment_ids[:max_len]

    return input_ids, segment_ids


def create_sessions_with_segment_ids(log_data, tokenizer, label_file=None, max_len=18000):
    """Group HDFS logs by block ID and tokenise each session."""
    session_dict = {}
    for line in tqdm(log_data, desc="Grouping logs by session", mininterval=1.0, dynamic_ncols=True):
        tokens = line.split()
        if len(tokens) < 2:
            continue
        try:
            timestamp = datetime.strptime(" ".join(tokens[:2]), '%y%m%d %H%M%S').timestamp()
        except Exception:
            continue

        blk_ids = list(set(re.findall(r'(blk_-?\d+)', line)))
        if len(blk_ids) != 1:
            continue
        blk_id = blk_ids[0]
        session_dict.setdefault(blk_id, []).append((timestamp, line))

    label_mapping = {}
    if label_file:
        label_df = pd.read_csv(label_file, engine='c', na_filter=False)
        label_mapping = label_df.set_index("BlockId")["Label"].to_dict()

    sessions = []
    for blk_id, events in tqdm(session_dict.items(), desc="Processing sessions", mininterval=1.0, dynamic_ncols=True):
        events.sort(key=lambda x: x[0])
        log_sequence   = [msg for (_, msg) in events]
        session_label  = (1 if label_mapping.get(blk_id, "Normal") == "Anomaly" else 0) if label_file else 0
        input_ids, segment_ids = tokenize_and_construct_input(log_sequence, tokenizer, max_len)
        sessions.append({
            "block_id":      blk_id,
            "input_ids":     input_ids,
            "segment_ids":   segment_ids,
            "session_label": session_label,
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
    def __init__(self, embedding_dim, seq_len, num_heads=2, ff_hidden_dim=128, k=128, dropout=0.1):
        super().__init__()
        self.self_attention = Linformer(
            dim=embedding_dim,
            seq_len=int(seq_len),
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
    def __init__(self, num_layers, embedding_dim, seq_len, num_heads=2, ff_hidden_dim=128, k=128, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            LinformerEncoderLayer(embedding_dim, seq_len, num_heads, ff_hidden_dim, k, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class AllLinLog(nn.Module):
    def __init__(self, vocab_size, max_seq_len, segment_vocab_size, embedding_dim=128,
                 num_layers=1, num_heads=2, ff_hidden_dim=128, k=128,
                 num_classes=2, dropout=0.1, max_segment_lengths=100):
        super().__init__()
        self.embedding_layer = EmbeddingLayer(
            vocab_size, max_seq_len,
            segment_vocab_size=max_segment_lengths,
            embedding_dim=embedding_dim,
        )
        self.encoder = LinformerTransformerEncoder(
            num_layers, embedding_dim, max_seq_len, num_heads, ff_hidden_dim, k, dropout
        )
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, input_ids, segment_ids, position_ids, attention_mask=None):
        embeddings     = self.embedding_layer(input_ids, segment_ids, position_ids)
        encoder_output = self.encoder(embeddings)
        pooled_output  = torch.mean(encoder_output, dim=1)
        return self.fc(pooled_output)


# ---------------------------------------------------------------------------
# DataLoader collate
# ---------------------------------------------------------------------------
def collate_fn(batch):
    input_ids      = [torch.tensor(item["input_ids"],   dtype=torch.long) for item in batch]
    segment_ids    = [torch.tensor(item["segment_ids"], dtype=torch.long) for item in batch]
    session_labels = torch.tensor([item["session_label"] for item in batch], dtype=torch.long)

    padded_input_ids   = pad_sequence(input_ids,   batch_first=True, padding_value=0)[:, :MAX_TOKEN_LENGTH]
    padded_segment_ids = pad_sequence(segment_ids, batch_first=True, padding_value=0)[:, :MAX_TOKEN_LENGTH]
    padded_segment_ids = torch.clamp(padded_segment_ids, 0, MAX_SEGMENT_LENGTHS - 1)
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


def screen_hdfs_logs(log_messages, model, tokenizer, device, max_len=18000, max_segment=298):
    """Screen a single HDFS block session for anomalies."""
    model.eval()
    input_ids, segment_ids = tokenize_and_construct_input(log_messages, tokenizer, max_len)
    segment_ids = [min(s, max_segment - 1) for s in segment_ids]

    input_ids_tensor   = torch.tensor([input_ids],   dtype=torch.long).to(device)
    segment_ids_tensor = torch.tensor([segment_ids], dtype=torch.long).to(device)
    attention_mask     = (input_ids_tensor != 0).long()

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
        "sequence_length":     len(input_ids),
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
    print(f"Label file:   {LABEL_FILE}")
    print(f"Model path:   {MODEL_PATH}")

    # --- Load data ---
    print("\nLoading logs from:", LOG_FILE)
    start_time = time.time()
    with open(LOG_FILE, mode="r", encoding="utf8") as f:
        logs = [x.strip() for x in tqdm(f, desc="Reading Logs", mininterval=1.0, dynamic_ncols=True)]
    print(f"Loaded {len(logs)} logs in {time.time() - start_time:.2f} seconds.")

    tokenizer    = load_gpt4_tokenizer()
    all_sessions = create_sessions_with_segment_ids(logs, tokenizer, label_file=LABEL_FILE, max_len=MAX_TOKEN_LENGTH)

    token_lengths    = [len(s["input_ids"]) for s in all_sessions]
    MAX_TOKEN_LENGTH = max(token_lengths)
    print(f"Max tokens in sessions: {MAX_TOKEN_LENGTH}")
    print(f"Number of sessions: {len(all_sessions)}")

    # Stratified split (same as training)
    session_labels = [s["session_label"] for s in all_sessions]
    train_sessions, temp_sessions, _, temp_labels = train_test_split(
        all_sessions, session_labels,
        test_size=(1 - TRAIN_RATIO), stratify=session_labels, random_state=42,
    )
    val_relative = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    temp_labels  = [s["session_label"] for s in temp_sessions]
    val_sessions, test_sessions, _, _ = train_test_split(
        temp_sessions, temp_labels,
        test_size=(1 - val_relative), stratify=temp_labels, random_state=42,
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
        segment_vocab_size=MAX_SEGMENT_LENGTHS,
        embedding_dim=EMBEDDING_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        ff_hidden_dim=FF_HIDDEN_DIM,
        k=K,
        num_classes=2,
        dropout=DROPOUT,
        max_segment_lengths=MAX_SEGMENT_LENGTHS,
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
    cm    = confusion_matrix(labels, predictions)
    cm_df = pd.DataFrame(cm, index=target_names, columns=[f"Pred_{n}" for n in target_names])
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
    print(f"True Positives (Anomalies detected):          {tp}")
    print(f"True Negatives (Normal correctly identified): {tn}")
    print(f"False Positives (False alarms):               {fp}")
    print(f"False Negatives (Missed anomalies):           {fn}")

    # --- Demo ---
    print("\n" + "=" * 60)
    print("DEMO: Screening sample sessions from test set")
    print("=" * 60)
    normal_samples    = [s for s in test_sessions if s["session_label"] == 0][:2]
    anomalous_samples = [s for s in test_sessions if s["session_label"] == 1][11:13]

    for i, sample in enumerate(normal_samples + anomalous_samples):
        actual   = "Normal" if sample["session_label"] == 0 else "Anomalous"
        block_id = sample.get("block_id", "Unknown")

        input_ids_t   = torch.tensor([sample["input_ids"]],   dtype=torch.long).to(device)
        segment_ids_t = torch.tensor([sample["segment_ids"]], dtype=torch.long).to(device)
        segment_ids_t = torch.clamp(segment_ids_t, 0, MAX_SEGMENT_LENGTHS - 1)
        attention_mask = (input_ids_t != 0).long()

        with torch.no_grad():
            position_ids = torch.arange(input_ids_t.size(1), device=input_ids_t.device).unsqueeze(0).expand(input_ids_t.size(0), -1)
            logits = model(input_ids_t, segment_ids_t, position_ids, attention_mask)
            probs  = torch.softmax(logits, dim=1)
            pred   = "Anomalous" if logits.argmax(dim=1).item() == 1 else "Normal"

        status = "[OK]" if pred == actual else "[WRONG]"
        print(f"\nSample {i+1} (Block: {block_id}): Actual={actual}, Predicted={pred} {status}")
        print(f"  Anomaly probability: {probs[0, 1].item():.4f}")
        print(f"  Sequence length: {len(sample['input_ids'])} tokens")


if __name__ == "__main__":
    main()
