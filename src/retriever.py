"""
Retriever for RAG-based explanation.

Implements BM25 retrieval (MVP) with optional dense/hybrid retrieval support.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from pathlib import Path
import pickle
import json

from .evidence_store import EvidenceStore, EvidenceDoc
from .normalizer import get_normalizer
from .data_loader import Session


@dataclass
class RetrievalHit:
    """A single retrieval result."""
    evidence_id: str
    score: float
    text: str
    rank: int
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        return {
            "evidence_id": self.evidence_id,
            "score": self.score,
            "text": self.text,
            "rank": self.rank,
            "metadata": self.metadata
        }


class BM25Retriever:
    """
    BM25-based retriever for evidence retrieval.
    
    Uses rank_bm25 library for efficient BM25 scoring.
    """
    
    def __init__(
        self,
        evidence_store: EvidenceStore,
        k1: float = 1.5,
        b: float = 0.75
    ):
        """
        Initialize BM25 retriever.
        
        Args:
            evidence_store: EvidenceStore with documents
            k1: BM25 term frequency saturation parameter
            b: BM25 document length normalization parameter
        """
        self.evidence_store = evidence_store
        self.k1 = k1
        self.b = b
        self.normalizer = get_normalizer(evidence_store.dataset)
        
        self._bm25 = None
        self._corpus = None
        self._built = False
    
    def build_index(self) -> "BM25Retriever":
        """Build the BM25 index from evidence store documents."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Please install rank_bm25: pip install rank-bm25")
        
        print("Building BM25 index...")
        
        # Tokenize all documents (simple whitespace tokenization)
        self._corpus = []
        for doc in self.evidence_store.documents:
            # Simple tokenization: lowercase and split on whitespace
            tokens = doc.text.lower().split()
            self._corpus.append(tokens)
        
        # Build BM25 index
        self._bm25 = BM25Okapi(self._corpus, k1=self.k1, b=self.b)
        self._built = True
        
        print(f"BM25 index built with {len(self._corpus)} documents")
        
        return self
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        exclude_session_ids: Optional[List[str]] = None,
        min_score: float = 0.0
    ) -> List[RetrievalHit]:
        """
        Retrieve top-k most similar evidence documents.
        
        Args:
            query: Query text (will be normalized)
            top_k: Number of results to return
            exclude_session_ids: Session IDs to exclude (e.g., the query session)
            min_score: Minimum score threshold
            
        Returns:
            List of RetrievalHit objects
        """
        if not self._built:
            self.build_index()
        
        # Normalize and tokenize query
        norm_result = self.normalizer.normalize_lines([query])
        query_tokens = norm_result.normalized_text.lower().split()
        
        # Get BM25 scores
        scores = self._bm25.get_scores(query_tokens)
        
        # Create (score, idx) pairs and sort
        scored_docs = list(enumerate(scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Build results, excluding specified sessions
        results = []
        rank = 1
        
        for idx, score in scored_docs:
            if score < min_score:
                continue
                
            doc = self.evidence_store.documents[idx]
            
            # Exclude if session_id matches
            if exclude_session_ids and doc.session_id in exclude_session_ids:
                continue
            
            results.append(RetrievalHit(
                evidence_id=doc.evidence_id,
                score=float(score),
                text=doc.text,
                rank=rank,
                metadata=doc.metadata
            ))
            
            rank += 1
            if len(results) >= top_k:
                break
        
        return results
    
    def retrieve_for_session(
        self,
        session: Session,
        top_k: int = 5,
        exclude_self: bool = True,
        min_score: float = 0.0
    ) -> List[RetrievalHit]:
        """
        Retrieve evidence for a session.
        
        Args:
            session: Session object to find evidence for
            top_k: Number of results
            exclude_self: Exclude the session itself from results
            min_score: Minimum score threshold
            
        Returns:
            List of RetrievalHit objects
        """
        # Join session lines as query
        query = "\n".join(session.lines)
        
        exclude_ids = [session.session_id] if exclude_self else None
        
        return self.retrieve(
            query=query,
            top_k=top_k,
            exclude_session_ids=exclude_ids,
            min_score=min_score
        )
    
    def batch_retrieve(
        self,
        sessions: List[Session],
        top_k: int = 5,
        exclude_self: bool = True,
        show_progress: bool = True
    ) -> Dict[str, List[RetrievalHit]]:
        """
        Retrieve evidence for multiple sessions.
        
        Args:
            sessions: List of Session objects
            top_k: Number of results per session
            exclude_self: Exclude each session from its own results
            show_progress: Show progress bar
            
        Returns:
            Dict mapping session_id to list of RetrievalHit
        """
        from tqdm import tqdm
        
        results = {}
        
        iterator = sessions
        if show_progress:
            iterator = tqdm(sessions, desc="Retrieving evidence")
        
        for session in iterator:
            hits = self.retrieve_for_session(
                session=session,
                top_k=top_k,
                exclude_self=exclude_self
            )
            results[session.session_id] = hits
        
        return results
    
    def analyze_retrieval_quality(
        self,
        sessions: List[Session],
        top_k: int = 5
    ) -> Dict:
        """
        Analyze retrieval quality on labeled sessions.
        
        Measures how often retrieved evidence has the same label as the query.
        (This is for analysis only - labels are never shown to the LLM)
        
        Returns:
            Dict with quality metrics
        """
        same_label_counts = []
        anomaly_in_top_k = []
        
        for session in sessions:
            hits = self.retrieve_for_session(session, top_k=top_k)
            
            same_label = sum(
                1 for h in hits 
                if h.metadata.get("label") == session.label
            )
            same_label_counts.append(same_label / len(hits) if hits else 0)
            
            has_anomaly = any(h.metadata.get("label") == 1 for h in hits)
            anomaly_in_top_k.append(has_anomaly)
        
        return {
            "avg_same_label_ratio": np.mean(same_label_counts),
            "std_same_label_ratio": np.std(same_label_counts),
            "anomaly_in_top_k_rate": np.mean(anomaly_in_top_k),
            "top_k": top_k
        }
    
    def save_index(self, path: str) -> None:
        """Save the BM25 index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "wb") as f:
            pickle.dump({
                "bm25": self._bm25,
                "corpus": self._corpus,
                "k1": self.k1,
                "b": self.b
            }, f)
        
        print(f"BM25 index saved to {path}")
    
    def load_index(self, path: str) -> "BM25Retriever":
        """Load the BM25 index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self._bm25 = data["bm25"]
        self._corpus = data["corpus"]
        self.k1 = data["k1"]
        self.b = data["b"]
        self._built = True
        
        print(f"BM25 index loaded from {path}")
        
        return self


class Retriever:
    """
    Unified retriever interface supporting multiple retrieval methods.
    
    Currently supports:
    - BM25 (default, fast, interpretable)
    - Dense (future: sentence-transformers)
    - Hybrid (future: BM25 + Dense fusion)
    """
    
    def __init__(
        self,
        evidence_store: EvidenceStore,
        method: str = "bm25",
        **kwargs
    ):
        """
        Initialize retriever.
        
        Args:
            evidence_store: EvidenceStore with documents
            method: "bm25", "dense", or "hybrid"
            **kwargs: Additional arguments for the specific retriever
        """
        self.evidence_store = evidence_store
        self.method = method
        
        if method == "bm25":
            self._retriever = BM25Retriever(evidence_store, **kwargs)
        elif method == "dense":
            raise NotImplementedError("Dense retrieval not yet implemented")
        elif method == "hybrid":
            raise NotImplementedError("Hybrid retrieval not yet implemented")
        else:
            raise ValueError(f"Unknown retrieval method: {method}")
    
    def build_index(self) -> "Retriever":
        """Build the retrieval index."""
        self._retriever.build_index()
        return self
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        **kwargs
    ) -> List[RetrievalHit]:
        """Retrieve top-k evidence for a query."""
        return self._retriever.retrieve(query, top_k, **kwargs)
    
    def retrieve_for_session(
        self,
        session: Session,
        top_k: int = 5,
        **kwargs
    ) -> List[RetrievalHit]:
        """Retrieve top-k evidence for a session."""
        return self._retriever.retrieve_for_session(session, top_k, **kwargs)
    
    def batch_retrieve(
        self,
        sessions: List[Session],
        top_k: int = 5,
        **kwargs
    ) -> Dict[str, List[RetrievalHit]]:
        """Retrieve evidence for multiple sessions."""
        return self._retriever.batch_retrieve(sessions, top_k, **kwargs)


# Quick test
if __name__ == "__main__":
    from .data_loader import BGLDataLoader
    from .evidence_store import build_evidence_store
    
    # Load data
    loader = BGLDataLoader(log_file="./logs/BGL.log")
    loader.load()
    
    # Build evidence store
    store = build_evidence_store(loader, "BGL")
    
    # Build retriever
    retriever = Retriever(store, method="bm25")
    retriever.build_index()
    
    # Test retrieval
    test_sessions = [s for s in loader.get_test() if s.label == 1][:5]  # Get anomalies
    
    print("\nTesting retrieval for anomalous sessions:")
    for session in test_sessions:
        hits = retriever.retrieve_for_session(session, top_k=3)
        print(f"\n{session.session_id} (label={session.label}):")
        for hit in hits:
            print(f"  {hit.evidence_id}: score={hit.score:.2f}, label={hit.metadata.get('label')}")
