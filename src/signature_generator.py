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


# BGL-specific error patterns — AUTO-GENERATED from training data
# Discovery notebook: notebooks/05_signature_audit.ipynb
# Source: 27,315 training anomalies → 34 patterns covering 99.8%
# DO NOT hand-edit — regenerate from notebook if needed
BGL_ERROR_PATTERNS = {
    "bgl_auto_01": {"name": "Data Error Fatal Error", "description": "Anomaly cluster characterised by: data, error, fatal, interrupt, tlb", "keywords": ["data", "error", "fatal", "interrupt", "tlb"], "patterns": [r"\bdata\b", r"\berror\b", r"\bfatal\b", r"\binterrupt\b", "tlb"]},
    "bgl_auto_02": {"name": "Data Fatal Interrupt Error", "description": "Anomaly cluster characterised by: data, fatal, interrupt, storage", "keywords": ["data", "fatal", "interrupt", "storage"], "patterns": [r"\bdata\b", r"\bfatal\b", r"\binterrupt\b", r"\bstorage\b"]},
    "bgl_auto_03": {"name": "App Been Ciod Error", "description": "Anomaly cluster characterised by: app, been, ciod, ciostream, error, fatal, has, message, prefix, reading, socket", "keywords": ["app", "been", "ciod", "ciostream", "error", "fatal", "has", "message", "prefix", "reading", "socket"], "patterns": ["app", r"\bbeen\b", r"\bciod\b", r"\bciostream\b", r"\berror\b"]},
    "bgl_auto_04": {"name": "Fatal Error", "description": "Anomaly cluster characterised by: fatal", "keywords": ["fatal"], "patterns": [r"\bfatal\b"]},
    "bgl_auto_05": {"name": "Error Fatal Error", "description": "Anomaly cluster characterised by: error, fatal", "keywords": ["error", "fatal"], "patterns": [r"\berror\b", r"\bfatal\b"]},
    "bgl_auto_06": {"name": "App Ciod Ciostream Error", "description": "Anomaly cluster characterised by: app, ciod, ciostream, fatal, message, prefix, socket", "keywords": ["app", "ciod", "ciostream", "fatal", "message", "prefix", "socket"], "patterns": ["app", r"\bciod\b", r"\bciostream\b", r"\bfatal\b", r"\bmessage\b"]},
    "bgl_auto_07": {"name": "App Ciod Error Error", "description": "Anomaly cluster characterised by: app, ciod, error, fatal", "keywords": ["app", "ciod", "error", "fatal"], "patterns": ["app", r"\bciod\b", r"\berror\b", r"\bfatal\b"]},
    "bgl_auto_08": {"name": "Fatal Message Error", "description": "Anomaly cluster characterised by: fatal, message", "keywords": ["fatal", "message"], "patterns": [r"\bfatal\b", r"\bmessage\b"]},
    "bgl_auto_09": {"name": "App Ciod Ciostream Error", "description": "Anomaly cluster characterised by: app, ciod, ciostream, error, fatal, message, prefix, reading, socket", "keywords": ["app", "ciod", "ciostream", "error", "fatal", "message", "prefix", "reading", "socket"], "patterns": ["app", r"\bciod\b", r"\bciostream\b", r"\berror\b", r"\bfatal\b"]},
    "bgl_auto_10": {"name": "Error Fatal Interrupt Error", "description": "Anomaly cluster characterised by: error, fatal, interrupt", "keywords": ["error", "fatal", "interrupt"], "patterns": [r"\berror\b", r"\bfatal\b", r"\binterrupt\b"]},
    "bgl_auto_11": {"name": "Data Error Fatal Error", "description": "Anomaly cluster characterised by: data, error, fatal, interrupt", "keywords": ["data", "error", "fatal", "interrupt"], "patterns": [r"\bdata\b", r"\berror\b", r"\bfatal\b", r"\binterrupt\b"]},
    "bgl_auto_12": {"name": "App Been Ciod Error", "description": "Anomaly cluster characterised by: app, been, ciod, ciostream, error, fatal, has, interrupt, message, prefix, reading, socket", "keywords": ["app", "been", "ciod", "ciostream", "error", "fatal", "has", "interrupt", "message", "prefix", "reading", "socket"], "patterns": ["app", r"\bbeen\b", r"\bciod\b", r"\bciostream\b", r"\berror\b"]},
    "bgl_auto_13": {"name": "Fatal Interrupt Error", "description": "Anomaly cluster characterised by: fatal, interrupt", "keywords": ["fatal", "interrupt"], "patterns": [r"\bfatal\b", r"\binterrupt\b"]},
    "bgl_auto_14": {"name": "Error Fatal Message Error", "description": "Anomaly cluster characterised by: error, fatal, message", "keywords": ["error", "fatal", "message"], "patterns": [r"\berror\b", r"\bfatal\b", r"\bmessage\b"]},
    "bgl_auto_15": {"name": "Data Error Fatal Error", "description": "Anomaly cluster characterised by: data, error, fatal, interrupt, message, tlb", "keywords": ["data", "error", "fatal", "interrupt", "message", "tlb"], "patterns": [r"\bdata\b", r"\berror\b", r"\bfatal\b", r"\binterrupt\b", r"\bmessage\b"]},
    "bgl_auto_16": {"name": "App Been Ciod Error", "description": "Anomaly cluster characterised by: app, been, ciod, ciostream, data, error, fatal, has, interrupt, message, prefix, reading, socket", "keywords": ["app", "been", "ciod", "ciostream", "data", "error", "fatal", "has", "interrupt", "message", "prefix", "reading", "socket"], "patterns": ["app", r"\bbeen\b", r"\bciod\b", r"\bciostream\b", r"\bdata\b"]},
    "bgl_auto_17": {"name": "Been Ciod Data Error", "description": "Anomaly cluster characterised by: been, ciod, data, error, fatal, has, socket", "keywords": ["been", "ciod", "data", "error", "fatal", "has", "socket"], "patterns": [r"\bbeen\b", r"\bciod\b", r"\bdata\b", r"\berror\b", r"\bfatal\b"]},
    "bgl_auto_18": {"name": "Error Fatal Socket Error", "description": "Anomaly cluster characterised by: error, fatal, socket", "keywords": ["error", "fatal", "socket"], "patterns": [r"\berror\b", r"\bfatal\b", r"\bsocket\b"]},
    "bgl_auto_19": {"name": "App Been Ciod Error", "description": "Anomaly cluster characterised by: app, been, ciod, ciostream, data, error, fatal, has, message, prefix, reading, socket", "keywords": ["app", "been", "ciod", "ciostream", "data", "error", "fatal", "has", "message", "prefix", "reading", "socket"], "patterns": ["app", r"\bbeen\b", r"\bciod\b", r"\bciostream\b", r"\bdata\b"]},
    "bgl_auto_20": {"name": "Ciod Fatal Error", "description": "Anomaly cluster characterised by: ciod, fatal", "keywords": ["ciod", "fatal"], "patterns": [r"\bciod\b", r"\bfatal\b"]},
    "bgl_auto_21": {"name": "Data Error Fatal Error", "description": "Anomaly cluster characterised by: data, error, fatal", "keywords": ["data", "error", "fatal"], "patterns": [r"\bdata\b", r"\berror\b", r"\bfatal\b"]},
    "bgl_auto_22": {"name": "App Ciod Fatal Error", "description": "Anomaly cluster characterised by: app, ciod, fatal", "keywords": ["app", "ciod", "fatal"], "patterns": ["app", r"\bciod\b", r"\bfatal\b"]},
    "bgl_auto_23": {"name": "Ciod Error Fatal Error", "description": "Anomaly cluster characterised by: ciod, error, fatal", "keywords": ["ciod", "error", "fatal"], "patterns": [r"\bciod\b", r"\berror\b", r"\bfatal\b"]},
    "bgl_auto_24": {"name": "Error Error", "description": "Anomaly cluster characterised by: error", "keywords": ["error"], "patterns": [r"\berror\b"]},
    "bgl_auto_25": {"name": "Been Error Fatal Error", "description": "Anomaly cluster characterised by: been, error, fatal, has, socket", "keywords": ["been", "error", "fatal", "has", "socket"], "patterns": [r"\bbeen\b", r"\berror\b", r"\bfatal\b", "has", r"\bsocket\b"]},
    "bgl_auto_26": {"name": "App Ciod Ciostream Error", "description": "Anomaly cluster characterised by: app, ciod, ciostream, error, fatal, message, prefix, socket", "keywords": ["app", "ciod", "ciostream", "error", "fatal", "message", "prefix", "socket"], "patterns": ["app", r"\bciod\b", r"\bciostream\b", r"\berror\b", r"\bfatal\b"]},
    "bgl_auto_27": {"name": "Ciod Data Error Error", "description": "Anomaly cluster characterised by: ciod, data, error, fatal, interrupt", "keywords": ["ciod", "data", "error", "fatal", "interrupt"], "patterns": [r"\bciod\b", r"\bdata\b", r"\berror\b", r"\bfatal\b", r"\binterrupt\b"]},
    "bgl_auto_28": {"name": "Ciod Fatal Message Error", "description": "Anomaly cluster characterised by: ciod, fatal, message", "keywords": ["ciod", "fatal", "message"], "patterns": [r"\bciod\b", r"\bfatal\b", r"\bmessage\b"]},
    "bgl_auto_29": {"name": "Been Ciod Data Error", "description": "Anomaly cluster characterised by: been, ciod, data, error, fatal, has", "keywords": ["been", "ciod", "data", "error", "fatal", "has"], "patterns": [r"\bbeen\b", r"\bciod\b", r"\bdata\b", r"\berror\b", r"\bfatal\b"]},
    "bgl_auto_30": {"name": "Been Data Error Error", "description": "Anomaly cluster characterised by: been, data, error, has", "keywords": ["been", "data", "error", "has"], "patterns": [r"\bbeen\b", r"\bdata\b", r"\berror\b", "has"]},
    "bgl_auto_31": {"name": "App Been Ciod Error", "description": "Anomaly cluster characterised by: app, been, ciod, ciostream, data, error, fatal, has, message, prefix, reading, socket, tlb", "keywords": ["app", "been", "ciod", "ciostream", "data", "error", "fatal", "has", "message", "prefix", "reading", "socket", "tlb"], "patterns": ["app", r"\bbeen\b", r"\bciod\b", r"\bciostream\b", r"\bdata\b"]},
    "bgl_auto_32": {"name": "Data Fatal Interrupt Error", "description": "Anomaly cluster characterised by: data, fatal, interrupt", "keywords": ["data", "fatal", "interrupt"], "patterns": [r"\bdata\b", r"\bfatal\b", r"\binterrupt\b"]},
    "bgl_auto_33": {"name": "Been Data Error Error", "description": "Anomaly cluster characterised by: been, data, error, fatal, has, socket", "keywords": ["been", "data", "error", "fatal", "has", "socket"], "patterns": [r"\bbeen\b", r"\bdata\b", r"\berror\b", r"\bfatal\b", "has"]},
    "bgl_auto_34": {"name": "Data Fatal Message Error", "description": "Anomaly cluster characterised by: data, fatal, message", "keywords": ["data", "fatal", "message"], "patterns": [r"\bdata\b", r"\bfatal\b", r"\bmessage\b"]},
}


# HDFS-specific error patterns — AUTO-GENERATED from training data
# Discovery notebook: notebooks/05_signature_audit.ipynb
# Source: 11,786 training anomalies → 26 patterns covering 99.8%
# DO NOT hand-edit — regenerate from notebook if needed
HDFS_ERROR_PATTERNS = {
    "hdfs_01": {"name": "Ask + Blockinfo + Error + Found + Not + Trying + Unexpected + Volumemap + Warn", "description": "Anomaly with tokens: ask+blockinfo+error+found+not+trying+unexpected+volumemap+warn", "keywords": ["ask", "blockinfo", "error", "found", "not", "trying", "unexpected", "volumemap", "warn"], "patterns": ["ask", "blockinfo", "error", "found", "not", "trying", "unexpected", "volumemap", "warn"]},
    "hdfs_02": {"name": "Ask", "description": "receiving-received gap=0.8", "keywords": ["ask"], "patterns": ["ask"]},
    "hdfs_03": {"name": "Write Pipeline + Exception + Ask + Could + Ioexception + Java + Not + Read + Stream", "description": "exception detected; writeBlock operation present; receiving-received gap=2.0", "keywords": ["ask", "could", "exception", "ioexception", "java", "not", "read", "stream", "writeblock"], "patterns": ["ask", "could", "exception", "ioexception", "java", "not", "read", "stream", "writeblock"]},
    "hdfs_04": {"name": "Replication + Ask + Datatransfer + Read + Starting + Thread + Transfer + Transmitted", "description": "replication activity", "keywords": ["ask", "datatransfer", "read", "replicate", "starting", "thread", "transfer", "transmitted"], "patterns": ["ask", "datatransfer", "read", "replicate", "starting", "thread", "transfer", "transmitted"]},
    "hdfs_05": {"name": "Exception + Replication + Ask + Datatransfer + Read + Starting + Thread + Transfer + Transmitted + Warn", "description": "exception detected; replication activity", "keywords": ["ask", "datatransfer", "exception", "read", "replicate", "starting", "thread", "transfer", "transmitted", "warn"], "patterns": ["ask", "datatransfer", "exception", "read", "replicate", "starting", "thread", "transfer", "transmitted", "warn"]},
    "hdfs_06": {"name": "Exception + Ask + Blockinfo + Error + Found + Not + Trying + Unexpected + Volumemap + Warn", "description": "exception detected", "keywords": ["ask", "blockinfo", "error", "exception", "found", "not", "trying", "unexpected", "volumemap", "warn"], "patterns": ["ask", "blockinfo", "error", "exception", "found", "not", "trying", "unexpected", "volumemap", "warn"]},
    "hdfs_07": {"name": "Exception + Any + Ask + Belong + But + Does + Not + Request + Warn", "description": "exception detected", "keywords": ["any", "ask", "belong", "but", "does", "exception", "not", "request", "warn"], "patterns": ["any", "ask", "belong", "but", "does", "exception", "not", "request", "warn"]},
    "hdfs_08": {"name": "Ask + Redundant + Request + Warn", "description": "Anomaly with tokens: ask+redundant+request+warn", "keywords": ["ask", "redundant", "request", "warn"], "patterns": ["ask", "redundant", "request", "warn"]},
    "hdfs_09": {"name": "Replication + Ask + Blockinfo + Datatransfer + Error + Found + Not + Read + Starting + Thread + Transfer + Transmitted + Trying + Unexpected + Volumemap + Warn", "description": "replication activity", "keywords": ["ask", "blockinfo", "datatransfer", "error", "found", "not", "read", "replicate", "starting", "thread", "transfer", "transmitted", "trying", "unexpected", "volumemap", "warn"], "patterns": ["ask", "blockinfo", "datatransfer", "error", "found", "not", "read", "replicate", "starting", "thread", "transfer", "transmitted", "trying", "unexpected", "volumemap", "warn"]},
    "hdfs_10": {"name": "Any + Ask + Belong + But + Does + Not + Request", "description": "Anomaly with tokens: any+ask+belong+but+does+not+request", "keywords": ["any", "ask", "belong", "but", "does", "not", "request"], "patterns": ["any", "ask", "belong", "but", "does", "not", "request"]},
    "hdfs_11": {"name": "Exception + Ask + Redundant + Request + Warn", "description": "exception detected", "keywords": ["ask", "exception", "redundant", "request", "warn"], "patterns": ["ask", "exception", "redundant", "request", "warn"]},
    "hdfs_12": {"name": "Replication + Ask + Datatransfer + Read + Redundant + Request + Starting + Thread + Transfer + Transmitted + Warn", "description": "replication activity", "keywords": ["ask", "datatransfer", "read", "redundant", "replicate", "request", "starting", "thread", "transfer", "transmitted", "warn"], "patterns": ["ask", "datatransfer", "read", "redundant", "replicate", "request", "starting", "thread", "transfer", "transmitted", "warn"]},
    "hdfs_13": {"name": "Write Pipeline + Exception + Replication + Ask + Datatransfer + Ioexception + Java + Read + Starting + Thread + Transfer + Transmitted", "description": "exception detected; writeBlock operation present; replication activity; receiving-received gap=3.3", "keywords": ["ask", "datatransfer", "exception", "ioexception", "java", "read", "replicate", "starting", "thread", "transfer", "transmitted", "writeblock"], "patterns": ["ask", "datatransfer", "exception", "ioexception", "java", "read", "replicate", "starting", "thread", "transfer", "transmitted", "writeblock"]},
    "hdfs_14": {"name": "Write Pipeline + Exception + Ask + Ioexception + Java", "description": "exception detected; writeBlock operation present; receiving-received gap=2.3", "keywords": ["ask", "exception", "ioexception", "java", "writeblock"], "patterns": ["ask", "exception", "ioexception", "java", "writeblock"]},
    "hdfs_15": {"name": "Replication + Ask + Blockinfo + Datatransfer + Error + Found + Not + Read + Redundant + Request + Starting + Thread + Transfer + Transmitted + Trying + Unexpected + Volumemap + Warn", "description": "replication activity", "keywords": ["ask", "blockinfo", "datatransfer", "error", "found", "not", "read", "redundant", "replicate", "request", "starting", "thread", "transfer", "transmitted", "trying", "unexpected", "volumemap", "warn"], "patterns": ["ask", "blockinfo", "datatransfer", "error", "found", "not", "read", "redundant", "replicate", "request", "starting", "thread", "transfer", "transmitted", "trying", "unexpected", "volumemap", "warn"]},
    "hdfs_16": {"name": "Replication + Ask + Datatransfer + Read + Starting + Thread + Transfer + Transmitted + Warn", "description": "replication activity", "keywords": ["ask", "datatransfer", "read", "replicate", "starting", "thread", "transfer", "transmitted", "warn"], "patterns": ["ask", "datatransfer", "read", "replicate", "starting", "thread", "transfer", "transmitted", "warn"]},
    "hdfs_17": {"name": "Write Pipeline + Exception + Replication + Ask + Datatransfer + Ioexception + Java + Not + Read + Starting + Thread + Transfer + Warn", "description": "exception detected; writeBlock operation present; replication activity; receiving-received gap=1.0", "keywords": ["ask", "datatransfer", "exception", "ioexception", "java", "not", "read", "replicate", "starting", "thread", "transfer", "warn", "writeblock"], "patterns": ["ask", "datatransfer", "exception", "ioexception", "java", "not", "read", "replicate", "starting", "thread", "transfer", "warn", "writeblock"]},
    "hdfs_18": {"name": "Write Pipeline + Exception + Ask + Ioexception + Java + Read", "description": "exception detected; writeBlock operation present; receiving-received gap=3.0", "keywords": ["ask", "exception", "ioexception", "java", "read", "writeblock"], "patterns": ["ask", "exception", "ioexception", "java", "read", "writeblock"]},
    "hdfs_19": {"name": "Replication + Ask + Read + Starting + Thread + Transfer", "description": "replication activity", "keywords": ["ask", "read", "replicate", "starting", "thread", "transfer"], "patterns": ["ask", "read", "replicate", "starting", "thread", "transfer"]},
    "hdfs_20": {"name": "Exception + Ask + Warn", "description": "exception detected", "keywords": ["ask", "exception", "warn"], "patterns": ["ask", "exception", "warn"]},
    "hdfs_21": {"name": "Write Pipeline + Exception + Ask + Java", "description": "exception detected; writeBlock operation present; receiving-received gap=1.0", "keywords": ["ask", "exception", "java", "writeblock"], "patterns": ["ask", "exception", "java", "writeblock"]},
    "hdfs_22": {"name": "Write Pipeline + Exception + Replication + Ask + Datatransfer + Ioexception + Java + Read + Starting + Thread + Transfer + Transmitted + Warn", "description": "exception detected; writeBlock operation present; replication activity; receiving-received gap=3.5", "keywords": ["ask", "datatransfer", "exception", "ioexception", "java", "read", "replicate", "starting", "thread", "transfer", "transmitted", "warn", "writeblock"], "patterns": ["ask", "datatransfer", "exception", "ioexception", "java", "read", "replicate", "starting", "thread", "transfer", "transmitted", "warn", "writeblock"]},
    "hdfs_23": {"name": "Exception + Replication + Ask + Blockinfo + Datatransfer + Error + Found + Not + Read + Starting + Thread + Transfer + Transmitted + Trying + Unexpected + Volumemap + Warn", "description": "exception detected; replication activity", "keywords": ["ask", "blockinfo", "datatransfer", "error", "exception", "found", "not", "read", "replicate", "starting", "thread", "transfer", "transmitted", "trying", "unexpected", "volumemap", "warn"], "patterns": ["ask", "blockinfo", "datatransfer", "error", "exception", "found", "not", "read", "replicate", "starting", "thread", "transfer", "transmitted", "trying", "unexpected", "volumemap", "warn"]},
    "hdfs_24": {"name": "Replication + Ask + Read + Starting + Thread + Transfer + Warn", "description": "replication activity", "keywords": ["ask", "read", "replicate", "starting", "thread", "transfer", "warn"], "patterns": ["ask", "read", "replicate", "starting", "thread", "transfer", "warn"]},
    "hdfs_25": {"name": "Exception + Ask + Blockinfo + Error + Found + Not + Redundant + Request + Trying + Unexpected + Volumemap + Warn", "description": "exception detected", "keywords": ["ask", "blockinfo", "error", "exception", "found", "not", "redundant", "request", "trying", "unexpected", "volumemap", "warn"], "patterns": ["ask", "blockinfo", "error", "exception", "found", "not", "redundant", "request", "trying", "unexpected", "volumemap", "warn"]},
    "hdfs_26": {"name": "Exception + Replication + Ask + Read + Starting + Thread + Transfer + Warn", "description": "exception detected; replication activity", "keywords": ["ask", "exception", "read", "replicate", "starting", "thread", "transfer", "warn"], "patterns": ["ask", "exception", "read", "replicate", "starting", "thread", "transfer", "warn"]},
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
        elif dataset == "HDFS":
            self.error_patterns = HDFS_ERROR_PATTERNS
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
