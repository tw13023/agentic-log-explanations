"""
Screener Wrapper for AllLinLog model inference.

Wraps the trained AllLinLog model to provide a clean interface for:
- Running inference on Session objects
- Outputting prediction, logits, probabilities, and margin
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from tqdm import tqdm
import tiktoken

# Import Linformer
try:
    from linformer import Linformer
except ImportError:
    raise ImportError("Please install linformer: pip install linformer")

from .data_loader import Session


@dataclass
class ScreenerOutput:
    """Output from the Screener model."""
    session_id: str
    pred: int  # 0=normal, 1=anomaly
    logits: List[float]  # [l_norm, l_anom]
    prob: List[float]  # [p_norm, p_anom]
    margin: float  # abs(p_anom - p_norm) - lower = less confident
    
    @property
    def is_anomaly(self) -> bool:
        return self.pred == 1
    
    @property
    def confidence(self) -> float:
        """Confidence in the prediction (probability of predicted class)."""
        return self.prob[self.pred]
    
    @property
    def anomaly_prob(self) -> float:
        return self.prob[1]
    
    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "pred": self.pred,
            "logits": self.logits,
            "prob": self.prob,
            "margin": self.margin,
            "is_anomaly": self.is_anomaly,
            "confidence": self.confidence
        }


# ============================================================
# Model Architecture (must match training)
# ============================================================

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, max_seq_len, segment_vocab_size, embedding_dim=128):
        super(EmbeddingLayer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.segment_embedding = nn.Embedding(segment_vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

    def forward(self, input_ids, segment_ids, position_ids=None):
        if position_ids is None:
            position_ids = torch.arange(
                input_ids.size(1), device=input_ids.device
            ).unsqueeze(0).repeat(input_ids.size(0), 1)
        E_token = self.token_embedding(input_ids)
        E_segment = self.segment_embedding(segment_ids)
        E_position = self.position_embedding(position_ids)
        return E_token + E_segment + E_position


class LinformerEncoderLayer(nn.Module):
    def __init__(self, embedding_dim, max_seq_len, num_heads=2, ff_hidden_dim=128, k=128, dropout=0.1):
        super(LinformerEncoderLayer, self).__init__()
        self.self_attention = Linformer(
            dim=embedding_dim,
            seq_len=max_seq_len,
            depth=1,
            heads=num_heads,
            k=k,
            one_kv_head=True,
            share_kv=True
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embedding_dim)
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention_output = self.self_attention(x)
        x = self.norm1(x + self.dropout(attention_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class LinformerTransformerEncoder(nn.Module):
    def __init__(self, num_layers, embedding_dim, max_seq_len, num_heads=2, ff_hidden_dim=128, k=128, dropout=0.1):
        super(LinformerTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            LinformerEncoderLayer(embedding_dim, max_seq_len, num_heads, ff_hidden_dim, k, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class AllLinLog(nn.Module):
    """AllLinLog: Linear Self-Attention based Log Anomaly Detector."""
    
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        segment_vocab_size: int,
        embedding_dim: int = 128,
        num_layers: int = 1,
        num_heads: int = 2,
        ff_hidden_dim: int = 128,
        k: int = 128,
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        super(AllLinLog, self).__init__()
        self.embedding_layer = EmbeddingLayer(
            vocab_size, max_seq_len, segment_vocab_size, embedding_dim
        )
        self.encoder = LinformerTransformerEncoder(
            num_layers, embedding_dim, max_seq_len, num_heads, ff_hidden_dim, k, dropout
        )
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, input_ids, segment_ids, position_ids, attention_mask=None):
        embeddings = self.embedding_layer(input_ids, segment_ids, position_ids)
        encoder_output = self.encoder(embeddings)
        pooled_output = torch.mean(encoder_output, dim=1)
        logits = self.fc(pooled_output)
        return logits


# ============================================================
# Screener Wrapper
# ============================================================

class Screener:
    """
    Wrapper around AllLinLog model for easy inference.
    
    Usage:
        screener = Screener.from_pretrained("BGL", model_path, device)
        results = screener.screen_sessions(sessions)
        
        # Or simpler:
        screener = Screener(model_path="path/to/model.pth", dataset="BGL")
        screener.load()
    """
    
    # Default model configs
    CONFIGS = {
        "BGL": {
            "vocab_size": 100264,  # GPT4 BPE cl100k_base
            "embedding_dim": 128,
            "ff_hidden_dim": 128,
            "num_layers": 1,
            "num_heads": 4,
            "k": 32,
            "dropout": 0.5,
            "segment_vocab_size": 10,  # windows_size
            "max_seq_len": 2549,  # Actual model dimension
        },
        "HDFS": {
            "vocab_size": 100264,
            "embedding_dim": 128,
            "ff_hidden_dim": 128,
            "num_layers": 1,
            "num_heads": 4,
            "k": 32,
            "dropout": 0.5,
            "segment_vocab_size": 50,  # Max logs per block
            "max_seq_len": 2549,  # Adjust if HDFS model differs
        }
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        dataset: str = "BGL",
        device: Optional[torch.device] = None,
        max_seq_len: Optional[int] = None,
        # Internal use - when loading from_pretrained
        model: Optional[AllLinLog] = None,
        tokenizer = None,
        windows_size: int = 10
    ):
        """
        Initialize Screener.
        
        For simple usage:
            screener = Screener(model_path="path/to/model.pth", dataset="BGL")
            screener.load()
        """
        self.model_path = model_path
        self.dataset = dataset.upper()
        self.windows_size = windows_size
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device
        
        # Get config
        config = self.CONFIGS.get(self.dataset)
        if config is None:
            raise ValueError(f"Unknown dataset: {dataset}. Supported: {list(self.CONFIGS.keys())}")
        
        self.max_seq_len = max_seq_len or config["max_seq_len"]
        self._config = config
        
        # Model and tokenizer (lazy loaded or passed in)
        self.model = model
        self.tokenizer = tokenizer
        self._loaded = model is not None
    
    def load(self) -> "Screener":
        """Load the model and tokenizer."""
        if self._loaded:
            return self
        
        if not self.model_path:
            raise ValueError("model_path is required when not using from_pretrained()")
        
        print(f"Loading Screener for {self.dataset} on {self.device}")
        
        # Load tokenizer
        print("Loading cl100k_base (GPT-4) tokenizer...")
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Create model
        self.model = AllLinLog(
            vocab_size=self._config["vocab_size"],
            max_seq_len=self.max_seq_len,
            segment_vocab_size=self._config["segment_vocab_size"],
            embedding_dim=self._config["embedding_dim"],
            num_layers=self._config["num_layers"],
            num_heads=self._config["num_heads"],
            ff_hidden_dim=self._config["ff_hidden_dim"],
            k=self._config["k"],
            num_classes=2,
            dropout=self._config["dropout"]
        ).to(self.device)
        
        # Load weights
        print(f"Loading model weights from: {self.model_path}")
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device, weights_only=False)
        )
        self.model.eval()
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model loaded! Parameters: {total_params:,}")
        
        self._loaded = True
        return self
    
    # Alias for backward compatibility
    def predict(self, session: Session) -> ScreenerOutput:
        """Alias for screen_session."""
        return self.screen_session(session)
    
    @classmethod
    def from_pretrained(
        cls,
        dataset: str,
        model_path: str,
        device: Optional[torch.device] = None,
        max_seq_len: Optional[int] = None
    ) -> "Screener":
        """
        Load a pretrained Screener model.
        
        Args:
            dataset: "BGL" or "HDFS"
            model_path: Path to the .pth model file
            device: torch device (auto-detected if None)
            max_seq_len: Override max sequence length
        """
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        
        print(f"Loading Screener for {dataset} on {device}")
        
        # Get config
        config = cls.CONFIGS.get(dataset.upper())
        if config is None:
            raise ValueError(f"Unknown dataset: {dataset}. Supported: {list(cls.CONFIGS.keys())}")
        
        if max_seq_len:
            config["max_seq_len"] = max_seq_len
        
        # Load tokenizer
        print("Loading cl100k_base (GPT-4) tokenizer...")
        tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Create model
        model = AllLinLog(
            vocab_size=config["vocab_size"],
            max_seq_len=config["max_seq_len"],
            segment_vocab_size=config["segment_vocab_size"],
            embedding_dim=config["embedding_dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            ff_hidden_dim=config["ff_hidden_dim"],
            k=config["k"],
            num_classes=2,
            dropout=config["dropout"]
        ).to(device)
        
        # Load weights
        print(f"Loading model weights from: {model_path}")
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded! Parameters: {total_params:,}")
        
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_seq_len=config["max_seq_len"],
            dataset=dataset
        )
    
    def _tokenize_session(self, lines: List[str]) -> Tuple[List[int], List[int]]:
        """Tokenize log lines into input_ids and segment_ids."""
        input_ids = []
        segment_ids = []
        
        allowed_special = {"<|startoftext|>", "<|endoftext|>"}
        bos_token = self.tokenizer.encode("<|startoftext|>", allowed_special=allowed_special)[0]
        eos_token = self.tokenizer.encode("<|endoftext|>", allowed_special=allowed_special)[0]
        
        for i, log in enumerate(lines):
            tokens = self.tokenizer.encode(log, allowed_special=allowed_special)
            if i == 0:
                tokens = [bos_token] + tokens
            tokens = tokens + [eos_token]
            input_ids.extend(tokens)
            segment_ids.extend([i] * len(tokens))
            
            # Truncate if too long
            if len(input_ids) >= self.max_seq_len:
                input_ids = input_ids[:self.max_seq_len]
                segment_ids = segment_ids[:self.max_seq_len]
                break
        
        return input_ids, segment_ids
    
    def screen_session(self, session: Session) -> ScreenerOutput:
        """
        Screen a single session for anomalies.
        
        Args:
            session: Session object
            
        Returns:
            ScreenerOutput with prediction details
        """
        # Tokenize
        input_ids, segment_ids = self._tokenize_session(session.lines)
        
        # Convert to tensors
        input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        segment_ids_tensor = torch.tensor([segment_ids], dtype=torch.long).to(self.device)
        
        # Pad to max length
        if input_ids_tensor.size(1) < self.max_seq_len:
            pad_size = self.max_seq_len - input_ids_tensor.size(1)
            input_ids_tensor = torch.nn.functional.pad(input_ids_tensor, (0, pad_size), value=0)
            segment_ids_tensor = torch.nn.functional.pad(segment_ids_tensor, (0, pad_size), value=0)
        
        attention_mask = (input_ids_tensor != 0).long()
        
        # Inference
        with torch.no_grad():
            logits = self.model(input_ids_tensor, segment_ids_tensor, attention_mask)
            probs = torch.softmax(logits, dim=1)
            pred = logits.argmax(dim=1).item()
        
        logits_list = logits[0].cpu().numpy().tolist()
        probs_list = probs[0].cpu().numpy().tolist()
        margin = abs(probs_list[1] - probs_list[0])
        
        return ScreenerOutput(
            session_id=session.session_id,
            pred=pred,
            logits=logits_list,
            prob=probs_list,
            margin=margin
        )
    
    def screen_sessions(
        self,
        sessions: List[Session],
        batch_size: int = 8,
        show_progress: bool = True
    ) -> List[ScreenerOutput]:
        """
        Screen multiple sessions for anomalies.
        
        Args:
            sessions: List of Session objects
            batch_size: Batch size for inference
            show_progress: Show progress bar
            
        Returns:
            List of ScreenerOutput
        """
        results = []
        
        # Process in batches
        iterator = range(0, len(sessions), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Screening sessions")
        
        for i in iterator:
            batch_sessions = sessions[i:i + batch_size]
            
            # Tokenize batch
            batch_input_ids = []
            batch_segment_ids = []
            
            for session in batch_sessions:
                input_ids, segment_ids = self._tokenize_session(session.lines)
                batch_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
                batch_segment_ids.append(torch.tensor(segment_ids, dtype=torch.long))
            
            # Pad batch
            padded_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=0)
            padded_segment_ids = pad_sequence(batch_segment_ids, batch_first=True, padding_value=0)
            
            # Truncate/pad to max_seq_len
            if padded_input_ids.size(1) < self.max_seq_len:
                pad_size = self.max_seq_len - padded_input_ids.size(1)
                padded_input_ids = torch.nn.functional.pad(padded_input_ids, (0, pad_size), value=0)
                padded_segment_ids = torch.nn.functional.pad(padded_segment_ids, (0, pad_size), value=0)
            else:
                padded_input_ids = padded_input_ids[:, :self.max_seq_len]
                padded_segment_ids = padded_segment_ids[:, :self.max_seq_len]
            
            padded_input_ids = padded_input_ids.to(self.device)
            padded_segment_ids = padded_segment_ids.to(self.device)
            attention_masks = (padded_input_ids != 0).long()
            
            # Inference
            with torch.no_grad():
                logits = self.model(padded_input_ids, padded_segment_ids, attention_masks)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
            
            # Collect results
            for j, session in enumerate(batch_sessions):
                logits_list = logits[j].cpu().numpy().tolist()
                probs_list = probs[j].cpu().numpy().tolist()
                margin = abs(probs_list[1] - probs_list[0])
                
                results.append(ScreenerOutput(
                    session_id=session.session_id,
                    pred=preds[j].item(),
                    logits=logits_list,
                    prob=probs_list,
                    margin=margin
                ))
        
        return results
    
    def get_anomalies(
        self,
        sessions: List[Session],
        batch_size: int = 8,
        show_progress: bool = True
    ) -> Tuple[List[Session], List[ScreenerOutput]]:
        """
        Screen sessions and return only the anomalies.
        
        Returns:
            Tuple of (anomaly_sessions, anomaly_outputs)
        """
        outputs = self.screen_sessions(sessions, batch_size, show_progress)
        
        anomaly_sessions = []
        anomaly_outputs = []
        
        for session, output in zip(sessions, outputs):
            if output.is_anomaly:
                anomaly_sessions.append(session)
                anomaly_outputs.append(output)
        
        return anomaly_sessions, anomaly_outputs
    
    def evaluate(
        self,
        sessions: List[Session],
        batch_size: int = 8
    ) -> Dict:
        """
        Evaluate screener on labeled sessions.
        
        Returns:
            Dict with metrics (accuracy, precision, recall, f1)
        """
        from sklearn.metrics import classification_report, confusion_matrix
        
        outputs = self.screen_sessions(sessions, batch_size, show_progress=True)
        
        y_true = [s.label for s in sessions]
        y_pred = [o.pred for o in outputs]
        
        # Calculate metrics
        report = classification_report(
            y_true, y_pred, 
            target_names=["Normal", "Anomaly"],
            output_dict=True
        )
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        return {
            "accuracy": (tp + tn) / len(sessions),
            "precision": report["Anomaly"]["precision"],
            "recall": report["Anomaly"]["recall"],
            "f1": report["Anomaly"]["f1-score"],
            "confusion_matrix": {
                "tn": int(tn), "fp": int(fp),
                "fn": int(fn), "tp": int(tp)
            },
            "margin_stats": {
                "mean": np.mean([o.margin for o in outputs]),
                "std": np.std([o.margin for o in outputs]),
                "min": min(o.margin for o in outputs),
                "max": max(o.margin for o in outputs)
            }
        }


# Quick test
if __name__ == "__main__":
    from .data_loader import BGLDataLoader
    
    # Load data
    loader = BGLDataLoader(log_file="./logs/BGL.log")
    loader.load()
    
    # Load screener
    screener = Screener.from_pretrained(
        dataset="BGL",
        model_path="./best_model/best_model_20250724_072857.pth"
    )
    
    # Test on a few sessions
    test_sessions = loader.get_test()[:10]
    results = screener.screen_sessions(test_sessions)
    
    print("\nScreening results:")
    for session, result in zip(test_sessions, results):
        status = "✓" if result.pred == session.label else "✗"
        print(f"{session.session_id}: pred={result.pred}, actual={session.label} {status}")
        print(f"  prob={result.prob}, margin={result.margin:.4f}")
