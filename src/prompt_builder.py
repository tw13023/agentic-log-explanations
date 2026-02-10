"""
Trace Schema and Prompt Builder for LLM-based explanation.

Defines the structured output format for explanations and builds prompts
that guide the LLM to produce traceable, evidence-grounded explanations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import json
from datetime import datetime

from .data_loader import Session
from .retriever import RetrievalHit
from .screener import ScreenerOutput


# ============================================================
# Trace Schema Definition
# ============================================================

# Claim types for forensic explanation
CLAIM_TYPES = {
    "observation": "Direct observation from query session (E0)",
    "pattern_match": "Pattern matches known anomaly exemplars",
    "contrast": "Differs from normal evidence (if available)",
}


@dataclass
class Claim:
    """A single claim in the explanation with type annotation and span references."""
    type: str  # "observation", "pattern_match", or "contrast"
    claim: str
    evidence_ids: List[str]  # Kept for backward compatibility
    evidence_spans: List[str] = field(default_factory=list)  # NEW: ["E0-L8", "E1-L3"]
    confidence: Optional[str] = None  # "high", "medium", "low"
    
    def to_dict(self) -> Dict:
        d = {
            "type": self.type,
            "claim": self.claim,
            "evidence_ids": self.evidence_ids,
            "evidence_spans": self.evidence_spans
        }
        if self.confidence:
            d["confidence"] = self.confidence
        return d
    
    @classmethod
    def from_dict(cls, d: Dict) -> "Claim":
        return cls(
            type=d.get("type", "observation"),
            claim=d.get("claim", ""),
            evidence_ids=d.get("evidence_ids", []),
            evidence_spans=d.get("evidence_spans", []),
            confidence=d.get("confidence")
        )


@dataclass
class Signature:
    """Anomaly signature for deduplication and clustering."""
    name: str  # e.g., "RAS_KERNEL_FATAL__DATA_STORAGE_INTERRUPT"
    matched_evidence_ids: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "matched_evidence_ids": self.matched_evidence_ids
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "Signature":
        if d is None:
            return cls(name="UNKNOWN", matched_evidence_ids=[])
        return cls(
            name=d.get("name", "UNKNOWN"),
            matched_evidence_ids=d.get("matched_evidence_ids", [])
        )


@dataclass
class TraceExplanation:
    """
    Structured explanation with traceable claims.
    
    This is the core output format that makes explanations verifiable.
    Each claim must reference specific evidence spans (line numbers).
    """
    prediction: str  # "anomaly" or "normal"
    summary: str  # Brief forensic summary
    signature: Optional[Signature] = None  # NEW: anomaly signature for clustering
    claims: List[Claim] = field(default_factory=list)
    insufficient_evidence: bool = False
    raw_response: str = ""
    
    def to_dict(self) -> Dict:
        result = {
            "prediction": self.prediction,
            "summary": self.summary,
            "claims": [c.to_dict() for c in self.claims],
            "insufficient_evidence": self.insufficient_evidence
        }
        if self.signature:
            result["signature"] = self.signature.to_dict()
        return result
    
    @classmethod
    def from_dict(cls, d: Dict) -> "TraceExplanation":
        signature = None
        if "signature" in d and d["signature"]:
            signature = Signature.from_dict(d["signature"])
        return cls(
            prediction=d.get("prediction", "anomaly"),
            summary=d.get("summary", ""),
            signature=signature,
            claims=[Claim.from_dict(c) for c in d.get("claims", [])],
            insufficient_evidence=d.get("insufficient_evidence", False),
            raw_response=d.get("raw_response", "")
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "TraceExplanation":
        """Parse from JSON string, handling potential errors."""
        # Clean up potential markdown formatting
        content = json_str.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        
        d = json.loads(content)
        result = cls.from_dict(d)
        result.raw_response = json_str
        return result
    
    @property
    def all_evidence_ids(self) -> List[str]:
        """Get all unique evidence IDs referenced in claims."""
        ids = set()
        for claim in self.claims:
            ids.update(claim.evidence_ids)
        return list(ids)
    
    @property
    def evidence_coverage(self) -> float:
        """Fraction of claims that have at least one evidence ID."""
        if not self.claims:
            return 0.0
        with_evidence = sum(1 for c in self.claims if c.evidence_ids)
        return with_evidence / len(self.claims)


# ============================================================
# JSON Schema for LLM
# ============================================================

TRACE_SCHEMA = {
    "type": "object",
    "required": ["prediction", "summary", "signature", "claims"],
    "properties": {
        "prediction": {
            "type": "string",
            "enum": ["anomaly", "normal"],
            "description": "The classification of this log session"
        },
        "summary": {
            "type": "string",
            "description": "A brief forensic summary (1-2 sentences) with specific details"
        },
        "signature": {
            "type": "object",
            "required": ["name", "matched_evidence_ids"],
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Anomaly signature name: COMPONENT_SEVERITY__ERROR_TYPE (e.g., RAS_KERNEL_FATAL__DATA_STORAGE_INTERRUPT)"
                },
                "matched_evidence_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Evidence IDs that match this signature"
                }
            },
            "description": "Anomaly signature for clustering and deduplication"
        },
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["type", "claim", "evidence_ids", "evidence_spans"],
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["observation", "pattern_match", "contrast"],
                        "description": "Claim type: observation (from E0), pattern_match (matches anomaly exemplars), contrast (differs from normal)"
                    },
                    "claim": {
                        "type": "string",
                        "description": "A specific, quantified claim about the anomaly"
                    },
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of evidence IDs: E0, E1, E2, etc."
                    },
                    "evidence_spans": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific line references: E0-L8, E1-L3, etc."
                    }
                }
            },
            "description": "List of typed claims, each backed by specific evidence spans"
        },
        "insufficient_evidence": {
            "type": "boolean",
            "description": "True if the provided evidence is insufficient to explain the anomaly"
        }
    }
}


# ============================================================
# Prompt Templates
# ============================================================

# Dataset-specific signature examples
SIGNATURE_EXAMPLES = {
    "BGL": {
        "examples": [
            "RAS_KERNEL_FATAL__DATA_STORAGE_INTERRUPT",
            "CIOD_APP_FATAL__CIOSTREAM_SOCKET_ERROR",
            "RAS_KERNEL_FATAL__MACHINE_CHECK",
        ],
        "description": "BlueGene/L supercomputer RAS (Reliability, Availability, Serviceability) logs",
        "components": "RAS, KERNEL, CIOD, APP, MMCS",
        "severities": "FATAL, FAILURE, SEVERE, WARNING",
    },
    "HDFS": {
        "examples": [
            "DATANODE_ERROR__BLOCK_VERIFICATION_FAILED",
            "NAMENODE_WARN__BLOCK_REPLICAS_MISSING",
            "DATANODE_ERROR__WRITE_PIPELINE_FAILED",
            "DATANODE_ERROR__PACKET_RESPONDER_EXCEPTION",
            "BLOCK_ERROR__REPLICATION_INCOMPLETE",
        ],
        "description": "Hadoop Distributed File System logs",
        "components": "DATANODE, NAMENODE, BLOCK, FSNamesystem, DataXceiver, PacketResponder",
        "severities": "ERROR, WARN, FATAL",
    },
}


def get_system_prompt(dataset: str = "BGL") -> str:
    """Get dataset-specific system prompt."""
    sig_info = SIGNATURE_EXAMPLES.get(dataset.upper(), SIGNATURE_EXAMPLES["BGL"])
    sig_examples = ", ".join(sig_info["examples"][:3])
    
    return f"""You are an expert log analyst producing forensic, evidence-grounded explanations.
Your task is to explain WHY a log session is anomalous based on the provided evidence.

DATASET: {sig_info['description']}
COMPONENTS IN LOGS: {sig_info['components']}
SEVERITY LEVELS: {sig_info['severities']}

EVIDENCE FORMAT:
- Each evidence block has LINE NUMBERS: E0-L1, E0-L2, E1-L1, E1-L2, etc.
- [E0] = The query session being analyzed
- [E1], [E2], ... = Retrieved historical evidence (may include anomaly or normal sessions)

CLAIM TYPES (you MUST produce at least one of each type when evidence allows):
- "observation": Direct observation from E0 - MUST include COUNT or POSITION
- "pattern_match": Pattern matches anomaly exemplars - MUST name the signature
- "contrast": Differs from normal evidence - MUST list "X has Y, Z lacks Y" explicitly

=== SIGNATURE NAMING ===
Create a signature from the ACTUAL log content you see:
- Format: COMPONENT_SEVERITY__ERROR_TYPE (double underscore)
- Component: Extract from the log (DataNode, NameNode, FSNamesystem, PacketResponder, etc.)
- Severity: Extract from the log (ERROR, WARN, FATAL, INFO with error context)
- ErrorType: Describe what went wrong (BLOCK_WRITE_FAILURE, REPLICATION_INCOMPLETE, etc.)

Example valid signatures for this dataset: {sig_examples}
NOTE: These are examples of the FORMAT. You MUST create YOUR OWN signature based on what you see in [E0].

=== OUTPUT JSON SCHEMA ===
{{
    "prediction": "anomaly",
    "summary": "<SIGNATURE>: <quantified observation> at <line range>",
    "signature": {{
        "name": "<COMPONENT_SEVERITY__ERROR_TYPE from actual logs>",
        "matched_evidence_ids": ["E1", "E2"]
    }},
    "claims": [
        {{
            "type": "observation",
            "claim": "E0 contains <COUNT> <what> at lines <range>.",
            "evidence_ids": ["E0"],
            "evidence_spans": ["E0-L<n>", "E0-L<m>"]
        }},
        {{
            "type": "pattern_match", 
            "claim": "The pattern <keywords from logs> matches signature <YOUR SIGNATURE>.",
            "evidence_ids": ["E0", "E1"],
            "evidence_spans": ["E0-L<n>", "E1-L<m>"]
        }},
        {{
            "type": "contrast",
            "claim": "E0 has <X> at E0-L<n>; E<k> shows <Y> at E<k>-L<m>.",
            "evidence_ids": ["E0", "E<k>"],
            "evidence_spans": ["E0-L<n>", "E<k>-L<m>"]
        }}
    ],
    "insufficient_evidence": false
}}

=== CRITICAL RULES ===
1. READ the actual [E0] log content - do NOT copy from examples
2. IDENTIFY component and severity FROM THE LOGS 
3. CREATE a unique signature that describes THIS specific anomaly
4. QUANTIFY: count errors, specify line ranges
5. CITE specific evidence_spans (E0-L8, E1-L3) for each claim
6. RESPECT LINE RANGES: Each evidence block shows its valid range (e.g., "10 lines: E0-L1 to E0-L10"). NEVER reference a line number beyond the stated maximum.

SCOPE: You produce forensic explanations only. Do NOT infer root causes or remediation."""


def get_explanation_prompt_template(dataset: str = "BGL") -> str:
    """Get dataset-specific explanation prompt template."""
    # No example JSON here - it's now in the system prompt
    return """Analyze this LOG SESSION that was flagged as ANOMALOUS by our detection model.

=== [E0] QUERY SESSION TO ANALYZE ===
Session ID: {session_id}
Anomaly Probability: {anomaly_prob:.2%}
Confidence Margin: {margin:.4f}

Log Content (with line numbers):
{log_content}

=== RETRIEVED EVIDENCE ===
The following evidence sessions were retrieved from our historical corpus for comparison.
Each line is prefixed with its span ID (e.g., E1-L3 = Evidence 1, Line 3).
{evidence_block}

=== YOUR TASK ===
Analyze [E0] and produce a forensic explanation:
1. READ the actual log content in E0 carefully
2. IDENTIFY the component (DataNode, NameNode, etc.) and severity (ERROR, WARN, INFO)
3. CREATE a signature from what you see: COMPONENT_SEVERITY__ERROR_TYPE
4. COUNT errors, note LINE NUMBERS (E0-L5, E0-L8, etc.)
5. COMPARE with retrieved evidence to support your analysis

Output ONLY a valid JSON object with no additional text."""


# Keep old constants for backward compatibility
SYSTEM_PROMPT = get_system_prompt("BGL")
EXPLANATION_PROMPT_TEMPLATE = get_explanation_prompt_template("BGL")


def format_evidence_block(hits: List[RetrievalHit], max_chars_per_evidence: int = 500) -> str:
    """Format retrieved evidence for the prompt with type, label, and LINE NUMBERS."""
    if not hits:
        return "No evidence retrieved."
    
    output_lines = []
    for i, hit in enumerate(hits, 1):
        evidence_id = f"E{i}"
        
        # Get evidence type and label from metadata
        evidence_type = hit.metadata.get("evidence_type", "session") if hit.metadata else "session"
        label = hit.metadata.get("label", None) if hit.metadata else None
        label_str = "anomaly" if label == 1 else "normal" if label == 0 else "unknown"
        
        # Format header with type, label, and total line count
        total_evidence_lines = len(text_lines)
        output_lines.append(f"[{evidence_id}] (type={evidence_type}, label={label_str}, score={hit.score:.2f}, {total_evidence_lines} lines: {evidence_id}-L1 to {evidence_id}-L{total_evidence_lines})")
        
        # Add line numbers to each line of evidence
        text_lines = hit.text.split("\n")
        char_count = 0
        for line_num, line in enumerate(text_lines, 1):
            if char_count + len(line) > max_chars_per_evidence:
                output_lines.append(f"{evidence_id}-L{line_num}: ... (truncated)")
                break
            output_lines.append(f"{evidence_id}-L{line_num}: {line}")
            char_count += len(line) + 1
        
        output_lines.append("")  # Empty line separator
    
    return "\n".join(output_lines)


def format_query_session_with_lines(session_lines: List[str], max_lines: int = 20) -> str:
    """Format the query session (E0) with line numbers and total count."""
    total_lines = len(session_lines)
    output = [f"({total_lines} lines total, valid span range: E0-L1 to E0-L{total_lines})"]
    
    lines_to_show = session_lines[:max_lines]
    for line_num, line in enumerate(lines_to_show, 1):
        output.append(f"E0-L{line_num}: {line}")
    
    if total_lines > max_lines:
        output.append(f"... ({total_lines - max_lines} more lines, up to E0-L{total_lines})")
    
    return "\n".join(output)


# ============================================================
# Prompt Builder
# ============================================================

class PromptBuilder:
    """
    Builds prompts for LLM explanation generation.
    
    Combines the anomalous session with retrieved evidence into a
    structured prompt that guides the LLM to produce traceable explanations.
    """
    
    def __init__(
        self,
        max_log_lines: int = 20,
        max_chars_per_evidence: int = 500,
        max_evidence_items: int = 5,
        dataset: str = "BGL"
    ):
        """
        Initialize prompt builder.
        
        Args:
            max_log_lines: Maximum log lines to include from the session
            max_chars_per_evidence: Max characters per evidence item
            max_evidence_items: Maximum number of evidence items to include
            dataset: Dataset type ("BGL" or "HDFS") for dataset-specific prompts
        """
        self.max_log_lines = max_log_lines
        self.max_chars_per_evidence = max_chars_per_evidence
        self.max_evidence_items = max_evidence_items
        self.dataset = dataset.upper()
        
        # Get dataset-specific prompts
        self._system_prompt = get_system_prompt(self.dataset)
        self._explanation_template = get_explanation_prompt_template(self.dataset)
    
    def build_prompt(
        self,
        session: Session,
        screener_output: ScreenerOutput,
        evidence_hits: List[RetrievalHit]
    ) -> tuple[str, str]:
        """
        Build the explanation prompt.
        
        Args:
            session: The anomalous session to explain
            screener_output: Output from the screener model
            evidence_hits: Retrieved evidence from RAG
            
        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        # Format log content WITH LINE NUMBERS
        log_content = format_query_session_with_lines(
            session.lines,
            self.max_log_lines
        )
        
        # Format evidence block
        evidence_to_use = evidence_hits[:self.max_evidence_items]
        evidence_block = format_evidence_block(
            evidence_to_use,
            self.max_chars_per_evidence
        )
        
        # Build user prompt using dataset-specific template
        user_prompt = self._explanation_template.format(
            session_id=session.session_id,
            anomaly_prob=screener_output.anomaly_prob,
            margin=screener_output.margin,
            log_content=log_content,
            evidence_block=evidence_block
        )
        
        return self._system_prompt, user_prompt
    
    def build_evidence_id_mapping(
        self,
        session: Session,
        evidence_hits: List[RetrievalHit]
    ) -> Dict[str, str]:
        """
        Build mapping from simple IDs (E0, E1, E2) to original evidence IDs.
        
        Args:
            session: The query session (mapped to E0)
            evidence_hits: Retrieved evidence (mapped to E1, E2, ...)
        
        Returns:
            Dict mapping "E0" -> query session_id, "E1" -> "E_BGL_00001234", etc.
        """
        mapping = {"E0": session.session_id}
        for i, hit in enumerate(evidence_hits[:self.max_evidence_items], 1):
            mapping[f"E{i}"] = hit.evidence_id
        return mapping


# ============================================================
# Explanation Result
# ============================================================

@dataclass
class ExplanationResult:
    """Complete explanation result with all metadata."""
    session_id: str
    session: Session
    screener_output: ScreenerOutput
    evidence_hits: List[RetrievalHit]
    explanation: TraceExplanation
    evidence_id_mapping: Dict[str, str]
    
    # Metrics
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    
    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "label": self.session.label,  # Ground truth (for analysis only)
            "screener": self.screener_output.to_dict(),
            "evidence_ids": [h.evidence_id for h in self.evidence_hits],
            "evidence_id_mapping": self.evidence_id_mapping,
            "explanation": self.explanation.to_dict(),
            "metrics": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "latency_ms": self.latency_ms
            },
            "created_at": self.created_at
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# Quick test
if __name__ == "__main__":
    # Test schema parsing
    sample_json = '''
    {
        "prediction": "anomaly",
        "summary": "This session shows memory errors typical of hardware failure.",
        "claims": [
            {"claim": "Multiple DDR errors detected in short succession", "evidence_ids": ["E1", "E2"]},
            {"claim": "Error pattern matches known memory failure signature", "evidence_ids": ["E1", "E3"]}
        ],
        "insufficient_evidence": false
    }
    '''
    
    explanation = TraceExplanation.from_json(sample_json)
    print("Parsed explanation:")
    print(f"  Prediction: {explanation.prediction}")
    print(f"  Summary: {explanation.summary}")
    print(f"  Claims: {len(explanation.claims)}")
    print(f"  Evidence IDs used: {explanation.all_evidence_ids}")
    print(f"  Evidence coverage: {explanation.evidence_coverage:.0%}")
    
    print("\nRe-serialized:")
    print(json.dumps(explanation.to_dict(), indent=2))
