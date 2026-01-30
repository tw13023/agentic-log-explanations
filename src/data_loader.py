"""
Session Data Loader for BGL and HDFS datasets.

Provides unified Session objects for all downstream modules.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Literal, Iterator
from pathlib import Path
import random
from tqdm import tqdm


@dataclass
class Session:
    """Unified session object for log anomaly detection."""
    session_id: str
    split: Literal["train", "val", "test"]
    label: int  # 0=normal, 1=anomaly (only for analysis, not given to LLM)
    lines: List[str]  # Original log lines
    metadata: Dict = field(default_factory=dict)  # For dataset-specific info
    
    def __len__(self) -> int:
        return len(self.lines)
    
    def __repr__(self) -> str:
        return f"Session(id={self.session_id}, split={self.split}, label={self.label}, lines={len(self.lines)})"


class BGLDataLoader:
    """
    BGL Dataset Loader using sliding window approach.
    
    BGL log format: each line starts with label indicator:
    - Lines starting with "-" are normal
    - Lines starting with other characters are anomalous
    """
    
    def __init__(
        self,
        log_file: str,
        windows_size: int = 10,
        step_size: int = 10,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,  # test_ratio = 1 - train_ratio - val_ratio
        seed: int = 42
    ):
        self.log_file = Path(log_file)
        self.windows_size = windows_size
        self.step_size = step_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        
        self._sessions: Dict[str, List[Session]] = {
            "train": [], "val": [], "test": []
        }
        self._loaded = False
    
    def load(self) -> "BGLDataLoader":
        """Load and split the BGL dataset."""
        if self._loaded:
            return self
            
        print(f"Loading BGL logs from: {self.log_file}")
        
        # Read all logs
        with open(self.log_file, mode="r", encoding="utf8") as f:
            logs = [x.strip() for x in tqdm(f, desc="Reading BGL logs")]
        
        print(f"Loaded {len(logs)} log lines")
        
        # Create sessions using sliding window
        all_sessions = []
        for i in tqdm(
            range(0, len(logs) - self.windows_size, self.step_size),
            desc="Creating sessions"
        ):
            lines = []
            label = 0
            
            for j in range(i, i + self.windows_size):
                content = logs[j]
                # Check if anomalous (line not starting with "-")
                if content[0] != "-":
                    label = 1
                # Remove the label prefix for the actual content
                content = content[content.find(' ') + 1:]
                lines.append(content)
            
            session = Session(
                session_id=f"BGL_{i:08d}",
                split="train",  # Will be updated after split
                label=label,
                lines=lines,
                metadata={"start_idx": i, "dataset": "BGL"}
            )
            all_sessions.append(session)
        
        # Stratified split
        self._split_sessions(all_sessions)
        self._loaded = True
        
        return self
    
    def _split_sessions(self, sessions: List[Session]) -> None:
        """Perform stratified train/val/test split."""
        random.seed(self.seed)
        
        # Separate by label for stratification
        normal = [s for s in sessions if s.label == 0]
        anomaly = [s for s in sessions if s.label == 1]
        
        # Shuffle
        random.shuffle(normal)
        random.shuffle(anomaly)
        
        # Calculate split indices
        def split_list(lst: List, train_r: float, val_r: float):
            n = len(lst)
            train_end = int(n * train_r)
            val_end = int(n * (train_r + val_r))
            return lst[:train_end], lst[train_end:val_end], lst[val_end:]
        
        normal_train, normal_val, normal_test = split_list(
            normal, self.train_ratio, self.val_ratio
        )
        anomaly_train, anomaly_val, anomaly_test = split_list(
            anomaly, self.train_ratio, self.val_ratio
        )
        
        # Assign splits and update session objects
        for s in normal_train + anomaly_train:
            s.split = "train"
            self._sessions["train"].append(s)
        
        for s in normal_val + anomaly_val:
            s.split = "val"
            self._sessions["val"].append(s)
        
        for s in normal_test + anomaly_test:
            s.split = "test"
            self._sessions["test"].append(s)
        
        # Shuffle each split
        for split in self._sessions.values():
            random.shuffle(split)
    
    def get_sessions(self, split: Optional[str] = None) -> List[Session]:
        """Get sessions for a specific split or all sessions."""
        if not self._loaded:
            self.load()
        
        if split is None:
            return (
                self._sessions["train"] + 
                self._sessions["val"] + 
                self._sessions["test"]
            )
        return self._sessions[split]
    
    def get_train(self) -> List[Session]:
        return self.get_sessions("train")
    
    def get_val(self) -> List[Session]:
        return self.get_sessions("val")
    
    def get_test(self) -> List[Session]:
        return self.get_sessions("test")
    
    def stats(self) -> Dict:
        """Return dataset statistics."""
        if not self._loaded:
            self.load()
        
        stats = {}
        for split_name, sessions in self._sessions.items():
            normal = sum(1 for s in sessions if s.label == 0)
            anomaly = sum(1 for s in sessions if s.label == 1)
            avg_lines = sum(len(s) for s in sessions) / len(sessions) if sessions else 0
            
            stats[split_name] = {
                "total": len(sessions),
                "normal": normal,
                "anomaly": anomaly,
                "anomaly_ratio": anomaly / len(sessions) if sessions else 0,
                "avg_lines": avg_lines
            }
        
        return stats
    
    def print_stats(self) -> None:
        """Pretty print dataset statistics."""
        stats = self.stats()
        print("\n" + "="*60)
        print("BGL Dataset Statistics")
        print("="*60)
        
        for split_name, s in stats.items():
            print(f"\n{split_name.upper()}:")
            print(f"  Total sessions: {s['total']:,}")
            print(f"  Normal: {s['normal']:,} | Anomaly: {s['anomaly']:,}")
            print(f"  Anomaly ratio: {s['anomaly_ratio']:.2%}")
            print(f"  Avg lines/session: {s['avg_lines']:.1f}")


class HDFSDataLoader:
    """
    HDFS Dataset Loader using block_id based sessions.
    
    HDFS uses block_id to group log lines into sessions.
    Labels are provided in a separate CSV file.
    """
    
    def __init__(
        self,
        log_file: str,
        label_file: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        seed: int = 42
    ):
        self.log_file = Path(log_file)
        self.label_file = Path(label_file)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        
        self._sessions: Dict[str, List[Session]] = {
            "train": [], "val": [], "test": []
        }
        self._loaded = False
    
    def load(self) -> "HDFSDataLoader":
        """Load and split the HDFS dataset."""
        if self._loaded:
            return self
        
        import pandas as pd
        import re
        
        print(f"Loading HDFS logs from: {self.log_file}")
        print(f"Loading labels from: {self.label_file}")
        
        # Load labels
        labels_df = pd.read_csv(self.label_file)
        block_labels = dict(zip(labels_df["BlockId"], labels_df["Label"]))
        
        # Read logs and group by block_id
        block_logs: Dict[str, List[str]] = {}
        block_pattern = re.compile(r"(blk_-?\d+)")
        
        with open(self.log_file, mode="r", encoding="utf8") as f:
            for line in tqdm(f, desc="Reading HDFS logs"):
                line = line.strip()
                match = block_pattern.search(line)
                if match:
                    block_id = match.group(1)
                    if block_id not in block_logs:
                        block_logs[block_id] = []
                    block_logs[block_id].append(line)
        
        print(f"Found {len(block_logs)} unique blocks")
        
        # Create sessions
        all_sessions = []
        for block_id, lines in block_logs.items():
            label = block_labels.get(block_id, 0)
            # Convert label: "Normal" -> 0, "Anomaly" -> 1
            if isinstance(label, str):
                label = 1 if label.lower() == "anomaly" else 0
            
            session = Session(
                session_id=f"HDFS_{block_id}",
                split="train",
                label=label,
                lines=lines,
                metadata={"block_id": block_id, "dataset": "HDFS"}
            )
            all_sessions.append(session)
        
        # Stratified split
        self._split_sessions(all_sessions)
        self._loaded = True
        
        return self
    
    def _split_sessions(self, sessions: List[Session]) -> None:
        """Perform stratified train/val/test split (same as BGL)."""
        random.seed(self.seed)
        
        normal = [s for s in sessions if s.label == 0]
        anomaly = [s for s in sessions if s.label == 1]
        
        random.shuffle(normal)
        random.shuffle(anomaly)
        
        def split_list(lst: List, train_r: float, val_r: float):
            n = len(lst)
            train_end = int(n * train_r)
            val_end = int(n * (train_r + val_r))
            return lst[:train_end], lst[train_end:val_end], lst[val_end:]
        
        normal_train, normal_val, normal_test = split_list(
            normal, self.train_ratio, self.val_ratio
        )
        anomaly_train, anomaly_val, anomaly_test = split_list(
            anomaly, self.train_ratio, self.val_ratio
        )
        
        for s in normal_train + anomaly_train:
            s.split = "train"
            self._sessions["train"].append(s)
        
        for s in normal_val + anomaly_val:
            s.split = "val"
            self._sessions["val"].append(s)
        
        for s in normal_test + anomaly_test:
            s.split = "test"
            self._sessions["test"].append(s)
        
        for split in self._sessions.values():
            random.shuffle(split)
    
    def get_sessions(self, split: Optional[str] = None) -> List[Session]:
        if not self._loaded:
            self.load()
        
        if split is None:
            return (
                self._sessions["train"] + 
                self._sessions["val"] + 
                self._sessions["test"]
            )
        return self._sessions[split]
    
    def get_train(self) -> List[Session]:
        return self.get_sessions("train")
    
    def get_val(self) -> List[Session]:
        return self.get_sessions("val")
    
    def get_test(self) -> List[Session]:
        return self.get_sessions("test")
    
    def stats(self) -> Dict:
        if not self._loaded:
            self.load()
        
        stats = {}
        for split_name, sessions in self._sessions.items():
            normal = sum(1 for s in sessions if s.label == 0)
            anomaly = sum(1 for s in sessions if s.label == 1)
            avg_lines = sum(len(s) for s in sessions) / len(sessions) if sessions else 0
            
            stats[split_name] = {
                "total": len(sessions),
                "normal": normal,
                "anomaly": anomaly,
                "anomaly_ratio": anomaly / len(sessions) if sessions else 0,
                "avg_lines": avg_lines
            }
        
        return stats
    
    def print_stats(self) -> None:
        stats = self.stats()
        print("\n" + "="*60)
        print("HDFS Dataset Statistics")
        print("="*60)
        
        for split_name, s in stats.items():
            print(f"\n{split_name.upper()}:")
            print(f"  Total sessions: {s['total']:,}")
            print(f"  Normal: {s['normal']:,} | Anomaly: {s['anomaly']:,}")
            print(f"  Anomaly ratio: {s['anomaly_ratio']:.2%}")
            print(f"  Avg lines/session: {s['avg_lines']:.1f}")


def get_data_loader(dataset: str, **kwargs):
    """Factory function to get the appropriate data loader."""
    if dataset.upper() == "BGL":
        return BGLDataLoader(**kwargs)
    elif dataset.upper() == "HDFS":
        return HDFSDataLoader(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# Quick test
if __name__ == "__main__":
    # Test BGL loader
    loader = BGLDataLoader(
        log_file="./logs/BGL.log",
        windows_size=10,
        step_size=10
    )
    loader.load()
    loader.print_stats()
    
    # Sample sessions
    print("\nSample sessions:")
    for session in loader.get_test()[:3]:
        print(f"\n{session}")
        print(f"  First line: {session.lines[0][:80]}...")
