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
    def from_dict(cls, d) -> "Signature":
        if d is None:
            return cls(name="UNKNOWN", matched_evidence_ids=[])
        if isinstance(d, str):
            return cls(name=d, matched_evidence_ids=[])
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
        # Handle alternate LLM schemas (component/severity/count format)
        if "prediction" not in d and "component" in d:
            comp = d.get("component", "UNKNOWN")
            err_type = d.get("error_type", "unknown_error")
            count = d.get("count", "?")
            sig_name = d.get("signature", f"{comp}__{err_type}".upper().replace(" ", "_"))
            line_nums = d.get("line_numbers", [])
            span_str = f"{line_nums[0]} to {line_nums[-1]}" if len(line_nums) >= 2 else ", ".join(line_nums)
            return cls(
                prediction="anomaly",
                summary=f"{sig_name}: {count} occurrences at {span_str}",
                signature=Signature.from_dict(sig_name),
                claims=[Claim(
                    type="observation",
                    claim=f"E0 contains {count} {err_type} events at {span_str}.",
                    evidence_ids=["E0"],
                    evidence_spans=line_nums,
                )],
                insufficient_evidence=d.get("insufficient_evidence", False),
            )

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
                        "description": "Specific line references.  EACH element must be exactly E<n>-L<line> (e.g. E0-L8) or a range E<n>-L<start> to E<n>-L<end> (e.g. E0-L5 to E0-L10).  NO other format."
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
            "KERNEL__DATA_TLB_ERROR",
            "KERNEL__DATA_STORAGE_INTERRUPT",
            "APP__CIOD_STREAM_ERROR",
            "KERNEL__LUSTRE_MOUNT_FAILED",
            "KERNEL__KERNEL_TERMINATED",
        ],
        "description": "BlueGene/L supercomputer RAS (Reliability, Availability, Serviceability) logs",
        "components": "KERNEL, APP, MMCS, LINKCARD",
    },
    "HDFS": {
        "examples": [
            "DATANODE__BLOCK_WRITE_FAILURE",
            "NAMENODE__INCOMPLETE_PIPELINE",
            "DATANODE__BLOCK_RECEIVE_FAILURE",
            "NAMENODE__REDUNDANT_STORED_BLOCK",
            "DATANODE__SERVING_EXCEPTION",
        ],
        "description": "Hadoop Distributed File System logs",
        "components": "DATANODE, NAMENODE, FSDATASET, BLOCKSCANNER",
    },
}


def get_system_prompt(dataset: str = "BGL") -> str:
    """Get dataset-specific system prompt."""
    sig_info = SIGNATURE_EXAMPLES.get(dataset.upper(), SIGNATURE_EXAMPLES["BGL"])
    sig_examples = ", ".join(sig_info["examples"][:3])
    
    # --- HDFS-specific analysis guidance (Phase 2a) ---
    if dataset.upper() == "HDFS":
        analysis_guidance = """
=== HDFS ANOMALY ANALYSIS PROCEDURE ===
The ML screener that flagged this session has F1-score = 0.996 — it is almost
always correct.  TRUST the screener's anomaly verdict by default.

HDFS anomalies fall into two categories:

STEP 1 — Scan for EXPLICIT errors:
  Look for lines containing WARN, ERROR, FATAL, IOException, "exception",
  "Got exception while serving", "Receiving empty packet", "Redundant
  addStoredBlock", "BlockInfo not found", "does not belong to any file".
  If you find ANY such lines, cite THOSE lines as the anomaly evidence.
  Do NOT cite normal INFO lines (Receiving block, Received block,
  PacketResponder, writeBlock, allocateBlock) as errors — these are
  routine HDFS operations that appear in EVERY session.

STEP 2 — If NO explicit errors exist, this is a STRUCTURAL / SUBTLE anomaly:
  The screener detected a pattern that is statistically anomalous even though
  every individual log line looks normal.  These anomalies are real — they
  represent unusual operation sequences, timing, or counts that a neural
  network can detect but human eyes cannot easily see.

  In this case:
  - Set "prediction": "anomaly" (trust the screener)
  - If E0 has a "STRUCTURAL:" summary line, use the operation counts and
    tags to describe the anomaly (e.g. INCOMPLETE_PIPELINE, EXCESS_REPLICATION)
  - If no STRUCTURAL line, explain that the screener detected a subtle
    statistical anomaly in the operation sequence and cite the full E0
    log range as evidence
  - Compare with normal evidence (E5) to highlight any differences
    in operation counts or ordering, even if subtle
  - Set "insufficient_evidence": true ONLY if you truly cannot find any
    difference from normal sessions — but still predict "anomaly"

SCREENER FALSE POSITIVE — use ONLY with very high confidence:
  You may set "prediction": "normal" ONLY when ALL of these are true:
  - The anomaly probability is close to the threshold (< 0.7)
  - ALL log lines are completely routine
  - The session is IDENTICAL to the normal evidence session (E5)
  - You are confident the screener made an error
  The screener's F1=0.996 means false positives are rare (~0.4%).
  When in doubt, trust the screener.

CRITICAL: Normal HDFS operations you must NEVER call errors:
  - "Receiving block" (routine start of block transfer)
  - "Received block" (successful block receipt)
  - "PacketResponder ... for block ... terminating" (normal close)
  - "writeBlock ... received exception" is an ERROR — but plain
    "writeBlock" without exception is normal.
"""
    else:
        analysis_guidance = ""
    
    return f"""You are an expert log analyst producing forensic, evidence-grounded explanations.
Your task is to analyze a log session flagged as anomalous by an ML screener
(F1-score = 0.996) and provide an evidence-grounded explanation.
The screener is almost always correct — trust its verdict by default.
You may override it ONLY with very high confidence that the session is normal.

DATASET: {sig_info['description']}
COMPONENTS IN LOGS: {sig_info['components']}
{analysis_guidance}
EVIDENCE FORMAT:
- Each evidence block has LINE NUMBERS: E0-L1, E0-L2, E1-L1, E1-L2, etc.
- [E0] = The query session being analyzed
- [E1]-[E4] = Retrieved anomaly exemplars from historical corpus (for pattern matching)
- [E5] = A NORMAL (non-anomalous) session, provided specifically for contrast claims.
  Use E5 to show how E0 differs from normal behavior.

CLAIM TYPES (you MUST produce at least one of each type when evidence allows):
- "observation": Direct observation from E0 - MUST include COUNT or POSITION
- "pattern_match": Pattern matches anomaly exemplars (E1-E4) - MUST name the signature
- "contrast": Differs from normal evidence (E5) - MUST list "E0 has X, E5 lacks X" explicitly

=== SIGNATURE NAMING ===
Create a signature from the ACTUAL log content you see:
- Format: COMPONENT__ERROR_TYPE (double underscore separator, NO severity)
- Component: Extract the ACTUAL component from the log.
  For HDFS logs: DATANODE, NAMENODE, FSDATASET, BLOCKSCANNER (from dfs.* class names).
  For BGL logs: KERNEL, APP, MMCS, LINKCARD (from the RAS subsystem field).
- ErrorType: Describe what went wrong (e.g. DATA_TLB_ERROR, BLOCK_WRITE_FAILURE).
- Do NOT include severity (WARN, ERROR, FATAL, INFO) in the signature — all sessions are already confirmed anomalies.

Example valid signatures for this dataset: {sig_examples}
NOTE: These are examples of the FORMAT. You MUST create YOUR OWN signature based on what you see in [E0].

=== OUTPUT JSON SCHEMA ===
{{
    "prediction": "anomaly",
    "summary": "<SIGNATURE>: <quantified observation> at <line range>",
    "signature": {{
        "name": "<COMPONENT__ERROR_TYPE from actual logs>",
        "matched_evidence_ids": ["E1", "E2"]
    }},
    "claims": [
        {{
            "type": "observation",
            "claim": "E0 contains <COUNT> <what> at lines <range>.",
            "evidence_ids": ["E0"],
            "evidence_spans": ["E0-L3", "E0-L5 to E0-L9"]
        }},
        {{
            "type": "pattern_match", 
            "claim": "The pattern <keywords from logs> matches signature <YOUR SIGNATURE>.",
            "evidence_ids": ["E0", "E1"],
            "evidence_spans": ["E0-L4", "E1-L2"]
        }},
        {{
            "type": "contrast",
            "claim": "E0 has <X> at E0-L7; E5 (normal) shows no such errors.",
            "evidence_ids": ["E0", "E5"],
            "evidence_spans": ["E0-L7", "E5-L1 to E5-L20"]
        }}
    ],
    "insufficient_evidence": false
}}

=== CRITICAL RULES ===
1. READ the actual [E0] log content - do NOT copy from examples
2. IDENTIFY component and severity FROM THE LOGS 
3. CREATE a unique signature that describes THIS specific anomaly
4. QUANTIFY: count errors, specify line ranges (L1 to L10 = 10 lines inclusive, not 9)
5. CITE specific evidence_spans for each claim
6. RESPECT LINE RANGES: Each evidence block shows its valid range (e.g., "10 lines: E0-L1 to E0-L10"). NEVER reference a line number beyond the stated maximum.
7. SPAN FORMAT: Each evidence_span string MUST be either a single line "E0-L8" or a range with " to " separator "E0-L5 to E0-L10".  NEVER use bare evidence IDs ("E5"), "E0-L5-E0-L10", "E0-L5/E0-L10", or any other separator.  For contrast claims where normal evidence has NO errors, cite the full range: "E5-L1 to E5-L35".
8. NEVER use "STRUCTURAL" as an evidence_span value.  The word STRUCTURAL describes a category of anomaly, NOT a line reference.  Always cite actual line numbers like "E0-L15" or "E0-L20 to E0-L25".
9. evidence_spans MUST NOT be empty — every claim MUST cite at least one span in E<n>-L<line> format.
10. You MUST output exactly 3 claims: one "observation", one "pattern_match", one "contrast".  Do NOT omit any claim type.
11. Your claim text MUST include keywords that actually appear in the cited evidence lines (e.g., if E0-L8 says "Got exception while serving", your claim must mention "exception" or "serving").
12. NEVER use placeholder line numbers such as "Lx", "L?", "L??", or "Lnn".  Every span MUST have a concrete numeric line number (e.g., "E1-L3").  If you cannot determine the exact line, OMIT that span entirely rather than guessing.

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
2. IDENTIFY the component (KERNEL, APP, MMCS, LINKCARD for BGL; DATANODE, NAMENODE for HDFS)
3. CREATE a signature: COMPONENT__ERROR_TYPE (no severity in the name)
4. COUNT errors, note LINE NUMBERS (E0-L5, E0-L8, etc.)
5. COMPARE with E1-E4 (anomaly exemplars) and E5 (normal session) to support your analysis

Output ONLY a valid JSON object with no additional text."""


# Keep old constants for backward compatibility
SYSTEM_PROMPT = get_system_prompt("BGL")
EXPLANATION_PROMPT_TEMPLATE = get_explanation_prompt_template("BGL")


def format_evidence_block(hits: List[RetrievalHit], max_chars_per_evidence: int = 50000) -> str:
    """Format retrieved evidence for the prompt with type, label, and LINE NUMBERS."""
    if not hits:
        return "No evidence retrieved."
    
    output_lines = []
    for i, hit in enumerate(hits, 1):
        evidence_id = f"E{i}"
        
        # Get evidence type and label from metadata
        evidence_type = hit.metadata.get("evidence_type", "session") if hit.metadata else "session"
        label = hit.metadata.get("label", None) if hit.metadata else None
        # Label indicates whether this reference evidence is an anomaly
        # exemplar or a normal session.  This is NOT leakage — labels are
        # on E1-E5 (reference evidence), not on E0 (query session).
        # The LLM uses these labels to generate pattern_match claims
        # (comparing E0 against known anomalies) and contrast claims
        # (comparing E0 against normal sessions).
        label_str = "anomaly" if label == 1 else "normal" if label == 0 else "unknown"
        
        # Split text into lines first
        text_lines = hit.text.split("\n")
        total_evidence_lines = len(text_lines)
        
        # Pre-compute how many lines are visible within the char budget
        char_count = 0
        visible_lines = 0
        for line in text_lines:
            if char_count + len(line) > max_chars_per_evidence:
                break
            char_count += len(line) + 1
            visible_lines += 1
        
        # Header advertises only the visible line range to avoid
        # information asymmetry (LLM seeing range it cannot read)
        if visible_lines >= total_evidence_lines:
            range_str = f"{total_evidence_lines} lines: {evidence_id}-L1 to {evidence_id}-L{total_evidence_lines}"
        else:
            range_str = f"{visible_lines}/{total_evidence_lines} lines shown: {evidence_id}-L1 to {evidence_id}-L{visible_lines}"
        output_lines.append(f"[{evidence_id}] (type={evidence_type}, label={label_str}, score={hit.score:.2f}, {range_str})")
        
        # Add line numbers to each line of evidence
        char_count = 0
        for line_num, line in enumerate(text_lines, 1):
            if char_count + len(line) > max_chars_per_evidence:
                output_lines.append(f"{evidence_id}-L{line_num}: ... (truncated)")
                break
            output_lines.append(f"{evidence_id}-L{line_num}: {line}")
            char_count += len(line) + 1
        
        output_lines.append("")  # Empty line separator
    
    return "\n".join(output_lines)


def format_query_session_with_lines(
    session_lines: List[str],
    max_chars: int = 100000,
    tail_ratio: float = 0.3,
    structural_summary: Optional[str] = None,
) -> str:
    """Format the query session (E0) with line numbers.

    Dynamic strategy — adapts to the actual session length:
      * If the full session fits within *max_chars*, show ALL lines.
      * Otherwise, apply head+tail truncation using *tail_ratio* to
        decide how many tail lines to keep so late-appearing anomaly
        signals are never hidden.

    Args:
        session_lines: The raw log lines for this session.
        max_chars: Soft character budget for the formatted E0 block.
            Default 100 000 chars — effectively shows all content
            for BGL/HDFS sessions (GPT-5.1 has 400K token context).
            Head+tail truncation only activates for rare outliers.
        tail_ratio: Fraction of the displayable lines reserved for
            the tail section (0.0–0.5). Default 0.3 means 30 % of
            lines come from the end of the session.
        structural_summary: Optional structural annotation string
            (e.g. from HDFSNormalizer.structural_summary()).  When
            provided it is appended after the log lines so the LLM
            sees the same structural tags that appear in E1-E5.
    """
    total_lines = len(session_lines)
    header = f"({total_lines} lines total, valid span range: E0-L1 to E0-L{total_lines})"
    output: List[str] = [header]

    # Fast path: build full output and check whether it fits.
    full_output = [header]
    for line_num, line in enumerate(session_lines, 1):
        full_output.append(f"E0-L{line_num}: {line}")

    if len("\n".join(full_output)) <= max_chars:
        if structural_summary:
            full_output.append(structural_summary)
        return "\n".join(full_output)

    # --- Truncation needed — use head + tail ------------------
    # Determine how many lines we can afford.
    # Estimate avg chars per formatted line.
    avg_line_len = len("\n".join(full_output)) / total_lines
    affordable_lines = max(10, int(max_chars / avg_line_len))
    affordable_lines = min(affordable_lines, total_lines)

    tail_lines = max(3, int(affordable_lines * tail_ratio))
    head_lines = affordable_lines - tail_lines

    # Head section
    for line_num, line in enumerate(session_lines[:head_lines], 1):
        output.append(f"E0-L{line_num}: {line}")

    # Gap marker
    gap_start = head_lines + 1
    gap_end = total_lines - tail_lines
    gap = gap_end - gap_start + 1
    if gap > 0:
        output.append(f"... ({gap} lines omitted, E0-L{gap_start} to E0-L{gap_end})")

    # Tail section
    tail_start = total_lines - tail_lines
    for idx, line in enumerate(session_lines[tail_start:]):
        line_num = tail_start + idx + 1
        output.append(f"E0-L{line_num}: {line}")

    if structural_summary:
        output.append(structural_summary)
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
        max_log_chars: int = 100000,
        tail_ratio: float = 0.3,
        max_chars_per_evidence: int = 50000,
        max_evidence_items: int = 5,
        dataset: str = "BGL",
        normalizer=None,
    ):
        """
        Initialize prompt builder.
        
        Args:
            max_log_chars: Soft character budget for the E0 log block.
                Sessions that fit are shown in full; longer sessions
                are dynamically truncated with a head+tail strategy.
                Default 100K effectively shows all BGL/HDFS sessions
                in full (GPT-5.1 has 400K token context window).
            tail_ratio: Fraction of displayable lines reserved for
                the tail section when truncation is needed (0.0–0.5).
            max_chars_per_evidence: Max characters per evidence item
            max_evidence_items: Maximum number of evidence items to include
            dataset: Dataset type ("BGL" or "HDFS") for dataset-specific prompts
            normalizer: Optional LogNormalizer instance for structural
                summary injection into E0.  When provided, E0 will
                include the same STRUCTURAL tags that appear in E1-E5.
        """
        self.max_log_chars = max_log_chars
        self.tail_ratio = tail_ratio
        self.max_chars_per_evidence = max_chars_per_evidence
        self.max_evidence_items = max_evidence_items
        self.dataset = dataset.upper()
        self.normalizer = normalizer
        
        # Get dataset-specific prompts
        self._system_prompt = get_system_prompt(self.dataset)
        self._explanation_template = get_explanation_prompt_template(self.dataset)
    
    def build_prompt(
        self,
        session: Session,
        screener_output: ScreenerOutput,
        evidence_hits: List[RetrievalHit],
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
        # Build structural summary (if normalizer available)
        structural = None
        if self.normalizer is not None:
            structural = self.normalizer.structural_summary(session)

        # Normalize E0 lines so they use the same representation as E1-E5.
        # Only for HDFS: block IDs are true noise that normalization helps.
        # BGL: normalization is too aggressive (replaces timestamps/IPs with
        # <NUM> tokens), causing the 8B model to produce a simplified schema.
        if self.normalizer is not None and self.dataset.upper() == "HDFS":
            norm_result = self.normalizer.normalize_lines(session.lines)
            display_lines = norm_result.normalized_text.split("\n")
        else:
            display_lines = session.lines

        # Format log content WITH LINE NUMBERS (dynamic strategy)
        log_content = format_query_session_with_lines(
            display_lines,
            self.max_log_chars,
            self.tail_ratio,
            structural_summary=structural,
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
    
    # Verification (attached after verifier runs)
    verification: Optional[Dict] = None
    
    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        # Extract normalized signature (already canonicalized in-place by pipeline)
        sig_name = "UNKNOWN"
        if (self.explanation and self.explanation.signature
                and self.explanation.signature.name):
            sig_name = self.explanation.signature.name
        
        return {
            "session_id": self.session_id,
            "label": self.session.label,  # Ground truth (for analysis only)
            "normalized_signature": sig_name,
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
            "created_at": self.created_at,
            "verification": self.verification or {}
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
