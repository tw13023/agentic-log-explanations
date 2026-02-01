"""
Error Signature Card Generator.

Analyzes anomaly sessions from training data to generate structured "signature cards"
that capture common error patterns. These cards provide domain knowledge context
for the LLM reasoner, enabling better explanations.

A signature card captures:
- Pattern name and description
- Key error indicators (keywords, phrases)
- Typical log structure
- Frequency in training data
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter, defaultdict
import re
import json

from .data_loader import Session
from .evidence_store import EvidenceStore, EvidenceDoc
from .normalizer import get_normalizer


@dataclass
class ErrorSignature:
    """
    An error signature card capturing a common anomaly pattern.
    
    These are auto-generated from training anomalies and provide
    domain knowledge for the LLM reasoner.
    """
    signature_id: str
    name: str  # Human-readable name (e.g., "Memory Parity Error")
    description: str  # What this error pattern indicates
    
    # Key indicators
    keywords: List[str]  # Key terms that identify this error
    patterns: List[str]  # Regex patterns that match this error
    
    # Statistics from training data
    frequency: int  # How many training anomalies match this signature
    example_session_ids: List[str]  # Example sessions (for traceability)
    
    # Typical structure
    typical_lines: List[str]  # Normalized example lines
    
    def to_evidence_text(self) -> str:
        """
        Convert signature card to text format for RAG retrieval.
        
        Format optimized for LLM understanding.
        """
        lines = [
            f"ERROR SIGNATURE: {self.name}",
            f"Description: {self.description}",
            f"",
            f"Key Indicators: {', '.join(self.keywords)}",
            f"Frequency: {self.frequency} occurrences in training data",
            f"",
            "Typical Log Pattern:",
        ]
        for line in self.typical_lines[:5]:  # Limit to 5 example lines
            lines.append(f"  {line}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        return {
            "signature_id": self.signature_id,
            "name": self.name,
            "description": self.description,
            "keywords": self.keywords,
            "patterns": self.patterns,
            "frequency": self.frequency,
            "example_session_ids": self.example_session_ids,
            "typical_lines": self.typical_lines,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "ErrorSignature":
        return cls(
            signature_id=d["signature_id"],
            name=d["name"],
            description=d["description"],
            keywords=d["keywords"],
            patterns=d["patterns"],
            frequency=d["frequency"],
            example_session_ids=d["example_session_ids"],
            typical_lines=d["typical_lines"],
        )


# BGL-specific error patterns (domain knowledge)
# These are extracted from BGL documentation and common HPC error patterns
BGL_ERROR_PATTERNS = {
    "memory_parity": {
        "name": "Memory Parity Error",
        "description": "Hardware memory error detected, often indicating failing DIMM or memory controller issues",
        "keywords": ["parity", "memory", "error", "ecc", "correctable", "uncorrectable"],
        "patterns": [r"memory.*parity", r"ecc.*error", r"dimm.*fail"],
    },
    "machine_check": {
        "name": "Machine Check Exception",
        "description": "Serious hardware error detected by CPU, may indicate CPU, memory, or bus failures",
        "keywords": ["machine check", "mce", "exception", "fatal", "hardware"],
        "patterns": [r"machine\s*check", r"mce", r"hardware\s*error"],
    },
    "kernel_panic": {
        "name": "Kernel Panic",
        "description": "Unrecoverable system error causing kernel to halt, often due to hardware or driver issues",
        "keywords": ["panic", "kernel", "fatal", "crash", "oops"],
        "patterns": [r"kernel\s*panic", r"oops", r"fatal\s*error"],
    },
    "torus_error": {
        "name": "Torus Network Error",
        "description": "Error in BGL's 3D torus interconnect network, may indicate network hardware or link issues",
        "keywords": ["torus", "receiver", "sender", "network", "link", "pipe"],
        "patterns": [r"torus.*error", r"torus.*receiver", r"torus.*sender"],
    },
    "tree_network": {
        "name": "Tree Network Error",
        "description": "Error in BGL's collective tree network, used for broadcasts and reductions",
        "keywords": ["tree", "receiver", "sender", "collective", "sync"],
        "patterns": [r"tree.*receiver", r"tree.*sender", r"tree.*error"],
    },
    "dma_error": {
        "name": "DMA Error",
        "description": "Direct Memory Access error, may indicate memory or I/O subsystem issues",
        "keywords": ["dma", "transfer", "error", "fifo", "injection"],
        "patterns": [r"dma.*error", r"dma.*fifo", r"injection.*error"],
    },
    "link_error": {
        "name": "Link/Communication Error",
        "description": "Communication link error between compute nodes or I/O nodes",
        "keywords": ["link", "error", "timeout", "retransmit", "crc"],
        "patterns": [r"link.*error", r"crc.*error", r"retransmit"],
    },
    "app_fatal": {
        "name": "Application Fatal Error",
        "description": "Application-level fatal error, job terminated abnormally",
        "keywords": ["fatal", "terminated", "killed", "ciod", "abort"],
        "patterns": [r"app.*fatal", r"job.*killed", r"ciod.*error"],
    },
    "core_dump": {
        "name": "Core Dump Generated",
        "description": "Process crashed and core file was generated for debugging",
        "keywords": ["core", "generating", "dump", "signal", "segfault"],
        "patterns": [r"generating\s*core", r"core\s*dump", r"segmentation"],
    },
}


class SignatureGenerator:
    """
    Generates error signature cards from training anomaly sessions.
    
    Process:
    1. Collect all anomaly sessions from training data
    2. Cluster by error pattern (keyword matching)
    3. Generate signature card for each cluster
    4. Add to evidence store as type="signature"
    """
    
    def __init__(self, dataset: str = "BGL"):
        self.dataset = dataset
        self.normalizer = get_normalizer(dataset)
        self.signatures: List[ErrorSignature] = []
        
        # Load dataset-specific patterns
        if dataset == "BGL":
            self.error_patterns = BGL_ERROR_PATTERNS
        else:
            # Default patterns for other datasets
            self.error_patterns = {}
    
    def analyze_anomaly_sessions(
        self,
        sessions: List[Session],
        min_frequency: int = 3
    ) -> List[ErrorSignature]:
        """
        Analyze anomaly sessions and generate signature cards.
        
        Args:
            sessions: Training sessions (will filter to anomalies)
            min_frequency: Minimum occurrences to create a signature
            
        Returns:
            List of generated ErrorSignature cards
        """
        # Filter to anomaly sessions only
        anomaly_sessions = [s for s in sessions if s.label == 1]
        print(f"Analyzing {len(anomaly_sessions)} anomaly sessions...")
        
        # Track which sessions match which patterns
        pattern_matches: Dict[str, List[Session]] = defaultdict(list)
        unmatched_sessions: List[Session] = []
        
        for session in anomaly_sessions:
            # Combine all lines for pattern matching
            session_text = " ".join(session.lines).lower()
            
            matched = False
            for pattern_key, pattern_info in self.error_patterns.items():
                # Check keywords
                keyword_match = any(kw in session_text for kw in pattern_info["keywords"])
                
                # Check regex patterns
                regex_match = any(
                    re.search(pat, session_text, re.IGNORECASE)
                    for pat in pattern_info["patterns"]
                )
                
                if keyword_match or regex_match:
                    pattern_matches[pattern_key].append(session)
                    matched = True
                    break  # Only assign to first matching pattern
            
            if not matched:
                unmatched_sessions.append(session)
        
        # Generate signature cards for patterns with enough matches
        self.signatures = []
        sig_id = 1
        
        for pattern_key, matched_sessions in pattern_matches.items():
            if len(matched_sessions) >= min_frequency:
                pattern_info = self.error_patterns[pattern_key]
                
                # Get typical lines from matched sessions
                typical_lines = self._extract_typical_lines(matched_sessions)
                
                signature = ErrorSignature(
                    signature_id=f"SIG_{self.dataset}_{sig_id:03d}",
                    name=pattern_info["name"],
                    description=pattern_info["description"],
                    keywords=pattern_info["keywords"],
                    patterns=pattern_info["patterns"],
                    frequency=len(matched_sessions),
                    example_session_ids=[s.session_id for s in matched_sessions[:5]],
                    typical_lines=typical_lines,
                )
                
                self.signatures.append(signature)
                sig_id += 1
        
        # Report statistics
        print(f"\nSignature Generation Results:")
        print(f"  Signatures created: {len(self.signatures)}")
        for sig in self.signatures:
            print(f"    - {sig.name}: {sig.frequency} matches")
        print(f"  Unmatched anomalies: {len(unmatched_sessions)}")
        
        return self.signatures
    
    def _extract_typical_lines(
        self,
        sessions: List[Session],
        max_lines: int = 5
    ) -> List[str]:
        """
        Extract the most representative/common lines from sessions.
        
        Uses normalized lines to find common patterns.
        """
        # Normalize all lines
        line_counter: Counter = Counter()
        
        for session in sessions[:50]:  # Sample first 50 sessions
            norm_result = self.normalizer.normalize_session(session)
            for line in norm_result.normalized_text.split("\n"):
                line = line.strip()
                if line:
                    line_counter[line] += 1
        
        # Return most common lines
        return [line for line, count in line_counter.most_common(max_lines)]
    
    def add_signatures_to_evidence_store(
        self,
        evidence_store: EvidenceStore
    ) -> int:
        """
        Add generated signatures to the evidence store.
        
        Args:
            evidence_store: The evidence store to add signatures to
            
        Returns:
            Number of signatures added
        """
        added = 0
        
        for signature in self.signatures:
            # Convert signature to evidence document
            doc = EvidenceDoc(
                evidence_id=f"E_{signature.signature_id}",
                session_id=signature.signature_id,  # Use signature_id as session_id
                text=signature.to_evidence_text(),
                evidence_type="signature",
                metadata={
                    "label": 1,  # Signatures are always from anomalies
                    "dataset": self.dataset,
                    "signature_name": signature.name,
                    "frequency": signature.frequency,
                    "keywords": signature.keywords,
                    "example_session_ids": signature.example_session_ids,
                }
            )
            
            # Add to evidence store
            evidence_store.documents.append(doc)
            evidence_store._id_to_doc[doc.evidence_id] = doc
            added += 1
        
        print(f"Added {added} signature cards to evidence store")
        return added
    
    def save(self, path: str) -> None:
        """Save signatures to JSON file."""
        data = {
            "dataset": self.dataset,
            "signatures": [s.to_dict() for s in self.signatures]
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(self.signatures)} signatures to {path}")
    
    def load(self, path: str) -> "SignatureGenerator":
        """Load signatures from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        
        self.dataset = data["dataset"]
        self.signatures = [ErrorSignature.from_dict(s) for s in data["signatures"]]
        print(f"Loaded {len(self.signatures)} signatures from {path}")
        return self


def build_signatures_from_training(
    train_sessions: List[Session],
    evidence_store: EvidenceStore,
    dataset: str = "BGL",
    min_frequency: int = 3
) -> SignatureGenerator:
    """
    Convenience function to build signatures and add to evidence store.
    
    Args:
        train_sessions: Training sessions
        evidence_store: Evidence store to add signatures to
        dataset: Dataset name
        min_frequency: Minimum matches to create signature
        
    Returns:
        SignatureGenerator with generated signatures
    """
    generator = SignatureGenerator(dataset=dataset)
    generator.analyze_anomaly_sessions(train_sessions, min_frequency=min_frequency)
    generator.add_signatures_to_evidence_store(evidence_store)
    return generator
