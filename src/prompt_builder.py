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

@dataclass
class Claim:
    """A single claim in the explanation."""
    claim: str
    evidence_ids: List[str]
    confidence: Optional[str] = None  # "high", "medium", "low"
    
    def to_dict(self) -> Dict:
        d = {"claim": self.claim, "evidence_ids": self.evidence_ids}
        if self.confidence:
            d["confidence"] = self.confidence
        return d
    
    @classmethod
    def from_dict(cls, d: Dict) -> "Claim":
        return cls(
            claim=d.get("claim", ""),
            evidence_ids=d.get("evidence_ids", []),
            confidence=d.get("confidence")
        )


@dataclass
class TraceExplanation:
    """
    Structured explanation with traceable claims.
    
    This is the core output format that makes explanations verifiable.
    Each claim must reference specific evidence IDs.
    """
    prediction: str  # "anomaly" or "normal"
    summary: str  # Brief summary of why this is anomalous
    claims: List[Claim]
    insufficient_evidence: bool = False
    raw_response: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "prediction": self.prediction,
            "summary": self.summary,
            "claims": [c.to_dict() for c in self.claims],
            "insufficient_evidence": self.insufficient_evidence
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "TraceExplanation":
        return cls(
            prediction=d.get("prediction", "anomaly"),
            summary=d.get("summary", ""),
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
    "required": ["prediction", "summary", "claims"],
    "properties": {
        "prediction": {
            "type": "string",
            "enum": ["anomaly", "normal"],
            "description": "The classification of this log session"
        },
        "summary": {
            "type": "string",
            "description": "A brief summary (1-2 sentences) explaining why this session is anomalous"
        },
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["claim", "evidence_ids"],
                "properties": {
                    "claim": {
                        "type": "string",
                        "description": "A specific claim about the anomaly"
                    },
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of evidence IDs (E1, E2, etc.) that support this claim"
                    }
                }
            },
            "description": "List of specific claims, each backed by evidence"
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

SYSTEM_PROMPT = """You are an expert log analyst specializing in system anomaly detection.
Your task is to explain WHY a log session is anomalous based on the provided evidence.

CRITICAL RULES:
1. Every claim MUST reference specific evidence IDs (E1, E2, etc.)
2. Only use information from the provided evidence - do not hallucinate
3. If evidence is insufficient, set insufficient_evidence to true
4. Be specific and technical in your analysis
5. Output ONLY valid JSON matching the schema

You are explaining anomalies detected by a machine learning model. Your job is to provide
human-understandable explanations grounded in evidence."""


EXPLANATION_PROMPT_TEMPLATE = """Analyze this LOG SESSION that was flagged as ANOMALOUS by our detection model.

=== LOG SESSION TO EXPLAIN ===
Session ID: {session_id}
Anomaly Probability: {anomaly_prob:.2%}
Confidence Margin: {margin:.4f}

Log Content:
{log_content}

=== RETRIEVED EVIDENCE ===
The following evidence sessions were retrieved from our corpus. Use these to support your explanation.
{evidence_block}

=== YOUR TASK ===
Explain WHY this log session is anomalous. For EACH claim you make:
1. Reference specific evidence IDs (E1, E2, etc.) that support it
2. Be specific about what patterns or errors indicate the anomaly

Output your explanation as JSON with this structure:
{{
    "prediction": "anomaly",
    "summary": "Brief 1-2 sentence summary of the anomaly",
    "claims": [
        {{"claim": "Specific claim about the anomaly", "evidence_ids": ["E1", "E3"]}},
        {{"claim": "Another claim", "evidence_ids": ["E2"]}}
    ],
    "insufficient_evidence": false
}}

IMPORTANT: Output ONLY the JSON object, no other text."""


def format_evidence_block(hits: List[RetrievalHit], max_chars_per_evidence: int = 500) -> str:
    """Format retrieved evidence for the prompt."""
    if not hits:
        return "No evidence retrieved."
    
    lines = []
    for i, hit in enumerate(hits, 1):
        evidence_id = f"E{i}"
        # Truncate long evidence
        text = hit.text
        if len(text) > max_chars_per_evidence:
            text = text[:max_chars_per_evidence] + "..."
        
        lines.append(f"[{evidence_id}] (score: {hit.score:.2f})")
        lines.append(f"Original ID: {hit.evidence_id}")
        lines.append(text)
        lines.append("")  # Empty line separator
    
    return "\n".join(lines)


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
        max_evidence_items: int = 5
    ):
        """
        Initialize prompt builder.
        
        Args:
            max_log_lines: Maximum log lines to include from the session
            max_chars_per_evidence: Max characters per evidence item
            max_evidence_items: Maximum number of evidence items to include
        """
        self.max_log_lines = max_log_lines
        self.max_chars_per_evidence = max_chars_per_evidence
        self.max_evidence_items = max_evidence_items
    
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
        # Format log content (truncate if needed)
        log_lines = session.lines[:self.max_log_lines]
        if len(session.lines) > self.max_log_lines:
            log_lines.append(f"... ({len(session.lines) - self.max_log_lines} more lines)")
        log_content = "\n".join(log_lines)
        
        # Format evidence block
        evidence_to_use = evidence_hits[:self.max_evidence_items]
        evidence_block = format_evidence_block(
            evidence_to_use,
            self.max_chars_per_evidence
        )
        
        # Build user prompt
        user_prompt = EXPLANATION_PROMPT_TEMPLATE.format(
            session_id=session.session_id,
            anomaly_prob=screener_output.anomaly_prob,
            margin=screener_output.margin,
            log_content=log_content,
            evidence_block=evidence_block
        )
        
        return SYSTEM_PROMPT, user_prompt
    
    def build_evidence_id_mapping(
        self,
        evidence_hits: List[RetrievalHit]
    ) -> Dict[str, str]:
        """
        Build mapping from simple IDs (E1, E2) to original evidence IDs.
        
        Returns:
            Dict mapping "E1" -> "E_BGL_00001234", etc.
        """
        mapping = {}
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
