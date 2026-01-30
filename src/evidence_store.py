"""
Evidence Store Builder for RAG retrieval.

Builds an evidence corpus from training sessions for use in retrieval-augmented
explanation generation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path
import json
import pickle
from tqdm import tqdm

from .data_loader import Session
from .normalizer import LogNormalizer, get_normalizer


@dataclass
class EvidenceDoc:
    """A single evidence document in the corpus."""
    evidence_id: str
    session_id: str
    text: str  # Normalized text
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "evidence_id": self.evidence_id,
            "session_id": self.session_id,
            "text": self.text,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "EvidenceDoc":
        return cls(
            evidence_id=d["evidence_id"],
            session_id=d["session_id"],
            text=d["text"],
            metadata=d.get("metadata", {})
        )


class EvidenceStore:
    """
    Evidence store for RAG-based explanation.
    
    Stores normalized training sessions as evidence documents that can be
    retrieved to support anomaly explanations.
    
    IMPORTANT: Only train split sessions are used to avoid data leakage.
    """
    
    def __init__(self, dataset: str = "BGL"):
        self.dataset = dataset
        self.normalizer = get_normalizer(dataset)
        self.documents: List[EvidenceDoc] = []
        self._id_to_doc: Dict[str, EvidenceDoc] = {}
        self._built = False
    
    def build_from_sessions(
        self,
        sessions: List[Session],
        show_progress: bool = True
    ) -> "EvidenceStore":
        """
        Build evidence store from training sessions.
        
        Args:
            sessions: List of Session objects (should be train split only!)
            show_progress: Show progress bar
            
        Returns:
            self for chaining
        """
        self.documents = []
        self._id_to_doc = {}
        
        iterator = sessions
        if show_progress:
            iterator = tqdm(sessions, desc="Building evidence store")
        
        for session in iterator:
            # Normalize the session text
            norm_result = self.normalizer.normalize_session(session)
            
            # Create evidence document
            evidence_id = f"E_{session.session_id}"
            doc = EvidenceDoc(
                evidence_id=evidence_id,
                session_id=session.session_id,
                text=norm_result.normalized_text,
                metadata={
                    "label": session.label,  # For analysis only, not given to LLM
                    "dataset": self.dataset,
                    "num_lines": len(session.lines),
                    "original_length": norm_result.original_length,
                    "normalized_length": norm_result.normalized_length,
                    "param_stats": norm_result.param_stats
                }
            )
            
            self.documents.append(doc)
            self._id_to_doc[evidence_id] = doc
        
        self._built = True
        print(f"Evidence store built with {len(self.documents)} documents")
        
        return self
    
    def get_document(self, evidence_id: str) -> Optional[EvidenceDoc]:
        """Get a document by its evidence_id."""
        return self._id_to_doc.get(evidence_id)
    
    def get_documents_by_label(self, label: int) -> List[EvidenceDoc]:
        """Get all documents with a specific label."""
        return [doc for doc in self.documents if doc.metadata.get("label") == label]
    
    def get_all_texts(self) -> List[str]:
        """Get all normalized texts (for building retriever index)."""
        return [doc.text for doc in self.documents]
    
    def get_all_ids(self) -> List[str]:
        """Get all evidence IDs."""
        return [doc.evidence_id for doc in self.documents]
    
    def stats(self) -> Dict:
        """Return evidence store statistics."""
        if not self._built:
            return {"status": "not built"}
        
        normal_docs = self.get_documents_by_label(0)
        anomaly_docs = self.get_documents_by_label(1)
        
        all_lengths = [doc.metadata.get("normalized_length", 0) for doc in self.documents]
        
        return {
            "total_documents": len(self.documents),
            "normal_documents": len(normal_docs),
            "anomaly_documents": len(anomaly_docs),
            "avg_text_length": sum(all_lengths) / len(all_lengths) if all_lengths else 0,
            "min_text_length": min(all_lengths) if all_lengths else 0,
            "max_text_length": max(all_lengths) if all_lengths else 0,
        }
    
    def save(self, path: str) -> None:
        """Save evidence store to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "dataset": self.dataset,
            "documents": [doc.to_dict() for doc in self.documents]
        }
        
        # Save as JSON for readability
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Evidence store saved to {path}")
    
    def load(self, path: str) -> "EvidenceStore":
        """Load evidence store from disk."""
        path = Path(path)
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.dataset = data["dataset"]
        self.normalizer = get_normalizer(self.dataset)
        self.documents = [EvidenceDoc.from_dict(d) for d in data["documents"]]
        self._id_to_doc = {doc.evidence_id: doc for doc in self.documents}
        self._built = True
        
        print(f"Evidence store loaded from {path} ({len(self.documents)} documents)")
        
        return self
    
    def __len__(self) -> int:
        return len(self.documents)
    
    def __getitem__(self, idx: int) -> EvidenceDoc:
        return self.documents[idx]


def build_evidence_store(
    data_loader,
    dataset: str,
    save_path: Optional[str] = None
) -> EvidenceStore:
    """
    Convenience function to build evidence store from a data loader.
    
    Args:
        data_loader: BGLDataLoader or HDFSDataLoader
        dataset: "BGL" or "HDFS"
        save_path: Optional path to save the evidence store
        
    Returns:
        Built EvidenceStore
    """
    # Get train sessions only
    train_sessions = data_loader.get_train()
    
    print(f"\nBuilding evidence store from {len(train_sessions)} training sessions")
    
    store = EvidenceStore(dataset=dataset)
    store.build_from_sessions(train_sessions)
    
    # Print stats
    stats = store.stats()
    print(f"\nEvidence Store Statistics:")
    print(f"  Total documents: {stats['total_documents']:,}")
    print(f"  Normal: {stats['normal_documents']:,}")
    print(f"  Anomaly: {stats['anomaly_documents']:,}")
    print(f"  Avg text length: {stats['avg_text_length']:.0f} chars")
    
    if save_path:
        store.save(save_path)
    
    return store


# Quick test
if __name__ == "__main__":
    from .data_loader import BGLDataLoader
    
    # Load data
    loader = BGLDataLoader(log_file="./logs/BGL.log")
    loader.load()
    loader.print_stats()
    
    # Build evidence store
    store = build_evidence_store(
        data_loader=loader,
        dataset="BGL",
        save_path="./results/evidence_store_BGL.json"
    )
    
    # Sample documents
    print("\nSample evidence documents:")
    for doc in store.documents[:3]:
        print(f"\n{doc.evidence_id}:")
        print(f"  Label: {doc.metadata['label']}")
        print(f"  Text preview: {doc.text[:100]}...")
