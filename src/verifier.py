"""
Verifier for explanation faithfulness.

Provides rule-based verification to ensure explanations are grounded in evidence
and don't contain hallucinations.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
import re
from enum import Enum

from .prompt_builder import TraceExplanation, Claim, ExplanationResult
from .retriever import RetrievalHit


class VerificationStatus(Enum):
    """Status of a verification check."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


@dataclass
class VerificationIssue:
    """A single verification issue."""
    check_name: str
    status: VerificationStatus
    message: str
    details: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "check": self.check_name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details
        }


@dataclass
class VerificationResult:
    """Result of verifying an explanation."""
    session_id: str
    passed: bool
    issues: List[VerificationIssue]
    
    # Summary metrics
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    warning_checks: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "passed": self.passed,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "warning_checks": self.warning_checks,
            "issues": [i.to_dict() for i in self.issues]
        }
    
    @property
    def pass_rate(self) -> float:
        if self.total_checks == 0:
            return 0.0
        return self.passed_checks / self.total_checks


class Verifier:
    """
    Rule-based verifier for explanation faithfulness.
    
    Checks:
    1. All referenced evidence IDs exist
    2. Claims contain keywords found in referenced evidence
    3. JSON structure is valid
    4. Evidence coverage is sufficient
    5. Evidence spans are valid (NEW)
    6. Span keyword matches (NEW)
    7. Signature exists (NEW)
    8. Cited severity check — E0 lines actually contain errors (NEW)
    """
    
    # Keywords that indicate a genuine anomaly in log lines
    SEVERITY_KEYWORDS = [
        'WARN', 'ERROR', 'FATAL',
        'exception', 'Exception', 'IOException',
        'Could not read from stream',
        'Got exception while serving',
        'Redundant addStoredBlock',
        'BlockInfo not found',
        'does not belong to any file',
        'Receiving empty packet',
        'failed', 'FAILED',
    ]
    
    def __init__(
        self,
        min_evidence_coverage: float = 0.8,
        require_keyword_match: bool = True,
        min_keyword_match_ratio: float = 0.3,
        require_spans: bool = True,
        require_signature: bool = True,
        check_cited_severity: bool = True,
        dataset: str = "BGL",
    ):
        """
        Initialize verifier.
        
        Args:
            min_evidence_coverage: Minimum fraction of claims with evidence
            require_keyword_match: Whether to check keyword matches
            min_keyword_match_ratio: Minimum keyword match ratio per claim
            require_spans: Whether to require evidence_spans in claims
            require_signature: Whether to require a signature
            check_cited_severity: Whether to check that cited E0 lines
                contain genuine error/warning keywords (Phase 3a)
            dataset: Dataset name (severity check is most useful for HDFS)
        """
        self.min_evidence_coverage = min_evidence_coverage
        self.require_keyword_match = require_keyword_match
        self.min_keyword_match_ratio = min_keyword_match_ratio
        self.require_spans = require_spans
        self.require_signature = require_signature
        self.check_cited_severity = check_cited_severity
        self.dataset = dataset.upper()
    
    def verify(
        self,
        explanation: TraceExplanation,
        evidence_hits: List[RetrievalHit],
        evidence_id_mapping: Dict[str, str],
        query_session_text: Optional[str] = None
    ) -> VerificationResult:
        """
        Verify an explanation against its evidence.
        
        Args:
            explanation: The explanation to verify
            evidence_hits: The evidence that was provided
            evidence_id_mapping: Mapping from E1, E2 to original IDs
            query_session_text: The raw text of E0 (query session) for keyword matching
            
        Returns:
            VerificationResult with all check outcomes
        """
        issues = []
        
        # Check 1: JSON structure
        issues.append(self._check_structure(explanation))
        
        # Check 2: Evidence ID validity
        issues.append(self._check_evidence_ids(
            explanation, evidence_id_mapping
        ))
        
        # Check 3: Evidence coverage
        issues.append(self._check_evidence_coverage(explanation))
        
        # Check 4: Keyword matching (if enabled)
        if self.require_keyword_match:
            keyword_issues = self._check_keyword_matches(
                explanation, evidence_hits, evidence_id_mapping, query_session_text
            )
            issues.extend(keyword_issues)
        
        # Check 5: Empty claims
        issues.append(self._check_empty_claims(explanation))
        
        # Check 6: Evidence spans validity (NEW)
        if self.require_spans:
            e0_lines = len(query_session_text.split("\n")) if query_session_text else 0
            span_issues = self._check_evidence_spans(
                explanation, evidence_hits, evidence_id_mapping,
                query_session_lines=e0_lines,
            )
            issues.extend(span_issues)
        
        # Check 7: Signature existence (NEW)
        if self.require_signature:
            issues.append(self._check_signature(explanation))
        
        # Check 8: Span keyword matching (NEW - stronger than evidence_id matching)
        if self.require_spans and self.require_keyword_match:
            span_kw_issues = self._check_span_keyword_matches(
                explanation, evidence_hits, evidence_id_mapping, query_session_text
            )
            issues.extend(span_kw_issues)
        
        # Check 9: Cited severity — do cited E0 lines actually contain errors? (Phase 3a)
        if self.check_cited_severity and query_session_text:
            issues.append(self._check_cited_severity(
                explanation, query_session_text
            ))
        
        # Calculate summary
        total = len(issues)
        passed = sum(1 for i in issues if i.status == VerificationStatus.PASS)
        failed = sum(1 for i in issues if i.status == VerificationStatus.FAIL)
        warnings = sum(1 for i in issues if i.status == VerificationStatus.WARNING)
        
        return VerificationResult(
            session_id=explanation.raw_response[:50] if explanation.raw_response else "unknown",
            passed=(failed == 0),
            issues=issues,
            total_checks=total,
            passed_checks=passed,
            failed_checks=failed,
            warning_checks=warnings
        )
    
    def _check_structure(self, explanation: TraceExplanation) -> VerificationIssue:
        """Check that explanation has required fields."""
        missing = []
        
        if not explanation.prediction:
            missing.append("prediction")
        if not explanation.summary:
            missing.append("summary")
        if not explanation.claims:
            missing.append("claims")
        
        if missing:
            return VerificationIssue(
                check_name="structure",
                status=VerificationStatus.FAIL,
                message=f"Missing required fields: {missing}",
                details={"missing_fields": missing}
            )
        
        return VerificationIssue(
            check_name="structure",
            status=VerificationStatus.PASS,
            message="All required fields present"
        )
    
    def _check_evidence_ids(
        self,
        explanation: TraceExplanation,
        evidence_id_mapping: Dict[str, str]
    ) -> VerificationIssue:
        """Check that all referenced evidence IDs exist."""
        valid_ids = set(evidence_id_mapping.keys())
        referenced_ids = set(explanation.all_evidence_ids)
        
        invalid_ids = referenced_ids - valid_ids
        
        if invalid_ids:
            return VerificationIssue(
                check_name="evidence_ids",
                status=VerificationStatus.FAIL,
                message=f"Invalid evidence IDs referenced: {invalid_ids}",
                details={
                    "invalid_ids": list(invalid_ids),
                    "valid_ids": list(valid_ids)
                }
            )
        
        if not referenced_ids:
            return VerificationIssue(
                check_name="evidence_ids",
                status=VerificationStatus.WARNING,
                message="No evidence IDs referenced in any claim"
            )
        
        return VerificationIssue(
            check_name="evidence_ids",
            status=VerificationStatus.PASS,
            message=f"All {len(referenced_ids)} evidence IDs are valid"
        )
    
    def _check_evidence_coverage(
        self,
        explanation: TraceExplanation
    ) -> VerificationIssue:
        """Check that sufficient claims have evidence."""
        coverage = explanation.evidence_coverage
        
        if coverage < self.min_evidence_coverage:
            return VerificationIssue(
                check_name="evidence_coverage",
                status=VerificationStatus.FAIL,
                message=f"Evidence coverage {coverage:.0%} below minimum {self.min_evidence_coverage:.0%}",
                details={
                    "coverage": coverage,
                    "minimum": self.min_evidence_coverage,
                    "claims_with_evidence": sum(1 for c in explanation.claims if c.evidence_ids),
                    "total_claims": len(explanation.claims)
                }
            )
        
        return VerificationIssue(
            check_name="evidence_coverage",
            status=VerificationStatus.PASS,
            message=f"Evidence coverage {coverage:.0%} meets minimum"
        )
    
    def _check_keyword_matches(
        self,
        explanation: TraceExplanation,
        evidence_hits: List[RetrievalHit],
        evidence_id_mapping: Dict[str, str],
        query_session_text: Optional[str] = None
    ) -> List[VerificationIssue]:
        """Check that claims contain keywords from their referenced evidence."""
        issues = []
        
        # Build evidence text lookup (include E0 = query session)
        evidence_texts = {}
        if query_session_text:
            evidence_texts["E0"] = query_session_text.lower()
        for i, hit in enumerate(evidence_hits, 1):
            evidence_texts[f"E{i}"] = hit.text.lower()
        
        # Extract significant keywords from claim
        def extract_keywords(text: str) -> Set[str]:
            # Simple keyword extraction: words > 4 chars, not common
            stopwords = {
                'this', 'that', 'with', 'from', 'have', 'been', 'were',
                'they', 'their', 'which', 'there', 'about', 'would',
                'could', 'should', 'these', 'those', 'being', 'other'
            }
            words = re.findall(r'\b[a-z]{4,}\b', text.lower())
            return set(w for w in words if w not in stopwords)
        
        for i, claim in enumerate(explanation.claims):
            if not claim.evidence_ids:
                continue
            
            claim_keywords = extract_keywords(claim.claim)
            if not claim_keywords:
                continue
            
            # Check if any keyword appears in referenced evidence
            matches_found = 0
            for eid in claim.evidence_ids:
                if eid in evidence_texts:
                    for kw in claim_keywords:
                        if kw in evidence_texts[eid]:
                            matches_found += 1
                            break
            
            match_ratio = matches_found / len(claim.evidence_ids) if claim.evidence_ids else 0
            
            if match_ratio < self.min_keyword_match_ratio:
                issues.append(VerificationIssue(
                    check_name=f"keyword_match_claim_{i}",
                    status=VerificationStatus.WARNING,
                    message=f"Claim {i} has low keyword overlap with evidence",
                    details={
                        "claim": claim.claim[:100],
                        "evidence_ids": claim.evidence_ids,
                        "keywords_checked": list(claim_keywords)[:10],
                        "match_ratio": match_ratio
                    }
                ))
        
        if not issues:
            issues.append(VerificationIssue(
                check_name="keyword_match",
                status=VerificationStatus.PASS,
                message="Claims have sufficient keyword overlap with evidence"
            ))
        
        return issues
    
    def _check_empty_claims(
        self,
        explanation: TraceExplanation
    ) -> VerificationIssue:
        """Check for empty or very short claims."""
        empty_claims = []
        
        for i, claim in enumerate(explanation.claims):
            if len(claim.claim.strip()) < 10:
                empty_claims.append(i)
        
        if empty_claims:
            return VerificationIssue(
                check_name="empty_claims",
                status=VerificationStatus.WARNING,
                message=f"Found {len(empty_claims)} empty or very short claims",
                details={"claim_indices": empty_claims}
            )
        
        return VerificationIssue(
            check_name="empty_claims",
            status=VerificationStatus.PASS,
            message="All claims have sufficient content"
        )
    
    def _parse_span(self, span: str) -> Tuple[Optional[str], Optional[int]]:
        """Parse a span like 'E0-L12' into (evidence_id, line_number).
        
        Also handles range formats:
          - 'E0-L6 to E0-L10'       → returns first span (E0, 6)
          - 'E0-L6, E0-L8'          → returns first span (E0, 6)
          - 'E0-L6-L10'             → returns first line  (E0, 6)
          - 'E0-L7-E0-L12'          → returns first span (E0, 7)
          - 'E5-L3/E5-L4'           → returns first span (E5, 3)
          - 'E5'                    → whole-document ref  (E5, None)
        """
        if not span:
            return None, None

        # Coerce to string (LLM sometimes returns int in spans list)
        span = str(span)

        # Handle bare evidence ID like 'E5' (whole-document reference)
        bare_match = re.fullmatch(r'E(\d+)', span.strip())
        if bare_match:
            return f"E{bare_match.group(1)}", None

        if '-L' not in span:
            return None, None
        try:
            # Handle 'E0-L6 to E0-L10' range format
            clean = span.split(' to ')[0].strip()
            # Handle 'E0-L6, E0-L8' comma format
            clean = clean.split(',')[0].strip()
            # Handle 'E5-L3/E5-L4' slash format
            clean = clean.split('/')[0].strip()
            # Handle 'E0-L7-E0-L12' repeated-ID range format
            # Regex: extract first occurrence of E{n}-L{n}
            m = re.match(r'^(E\d+)-L(\d+)', clean)
            if m:
                return m.group(1), int(m.group(2))
            # Fallback: split on -L
            parts = clean.split('-L')
            evidence_id = parts[0]
            line_num = int(parts[1])
            return evidence_id, line_num
        except (ValueError, IndexError):
            return None, None
    
    def _build_evidence_lines(self, evidence_hits: List[RetrievalHit], query_session_text: str = None) -> Dict[str, List[str]]:
        """Build a lookup of evidence_id -> list of lines."""
        evidence_lines = {}
        
        # E0 is the query session (if provided)
        if query_session_text:
            evidence_lines["E0"] = query_session_text.split("\n")
        
        # E1, E2, ... are retrieved evidence
        for i, hit in enumerate(evidence_hits, 1):
            evidence_lines[f"E{i}"] = hit.text.split("\n")
        
        return evidence_lines
    
    def _check_evidence_spans(
        self,
        explanation: TraceExplanation,
        evidence_hits: List[RetrievalHit],
        evidence_id_mapping: Dict[str, str],
        query_session_lines: int = 0,
    ) -> List[VerificationIssue]:
        """Check that evidence_spans reference valid lines."""
        issues = []
        
        # Build line count lookup — use actual E0 line count if available
        if query_session_lines > 0:
            line_counts = {"E0": query_session_lines}
        else:
            line_counts = {"E0": 100}  # Fallback for backward compatibility
        for i, hit in enumerate(evidence_hits, 1):
            line_counts[f"E{i}"] = len(hit.text.split("\n"))
        
        valid_eids = set(evidence_id_mapping.keys())
        
        claims_without_spans = 0
        invalid_spans = []
        
        for i, claim in enumerate(explanation.claims):
            # Check if claim has evidence_spans
            if not claim.evidence_spans:
                claims_without_spans += 1
                continue
            
            for span in claim.evidence_spans:
                eid, line_num = self._parse_span(span)
                
                if eid is None:
                    # Only fail for truly unparseable spans (not range formats)
                    invalid_spans.append((i, span, "malformed"))
                    continue
                
                if eid not in valid_eids and eid != "E0":
                    invalid_spans.append((i, span, f"unknown evidence_id {eid}"))
                    continue
                
                # line_num is None for bare evidence ID refs (whole-document)
                if line_num is not None and eid in line_counts and line_num > line_counts[eid]:
                    invalid_spans.append((i, span, f"line {line_num} > max {line_counts[eid]}"))
        
        # Report issues
        if claims_without_spans > 0:
            issues.append(VerificationIssue(
                check_name="evidence_spans_coverage",
                status=VerificationStatus.WARNING,
                message=f"{claims_without_spans}/{len(explanation.claims)} claims lack evidence_spans",
                details={"claims_without_spans": claims_without_spans}
            ))
        
        if invalid_spans:
            issues.append(VerificationIssue(
                check_name="evidence_spans_validity",
                status=VerificationStatus.FAIL,
                message=f"Found {len(invalid_spans)} invalid spans",
                details={"invalid_spans": invalid_spans[:10]}  # Limit output
            ))
        else:
            issues.append(VerificationIssue(
                check_name="evidence_spans_validity",
                status=VerificationStatus.PASS,
                message="All evidence spans are valid"
            ))
        
        return issues
    
    def _check_signature(
        self,
        explanation: TraceExplanation
    ) -> VerificationIssue:
        """Check that explanation has a signature."""
        if not explanation.signature:
            return VerificationIssue(
                check_name="signature",
                status=VerificationStatus.WARNING,
                message="No signature provided"
            )
        
        if not explanation.signature.name or explanation.signature.name == "UNKNOWN":
            return VerificationIssue(
                check_name="signature",
                status=VerificationStatus.WARNING,
                message="Signature name is missing or UNKNOWN"
            )
        
        # Check signature format: should contain double underscore
        if "__" not in explanation.signature.name:
            return VerificationIssue(
                check_name="signature",
                status=VerificationStatus.WARNING,
                message=f"Signature '{explanation.signature.name}' doesn't follow COMPONENT__ERROR_TYPE format",
                details={"signature": explanation.signature.name}
            )
        
        return VerificationIssue(
            check_name="signature",
            status=VerificationStatus.PASS,
            message=f"Valid signature: {explanation.signature.name}"
        )
    
    def _check_span_keyword_matches(
        self,
        explanation: TraceExplanation,
        evidence_hits: List[RetrievalHit],
        evidence_id_mapping: Dict[str, str],
        query_session_text: Optional[str] = None
    ) -> List[VerificationIssue]:
        """Check that claim keywords appear in the specific referenced spans."""
        issues = []
        
        # Build evidence lines lookup (include E0 = query session)
        evidence_lines = {}
        if query_session_text:
            evidence_lines["E0"] = query_session_text.split("\n")
        for i, hit in enumerate(evidence_hits, 1):
            evidence_lines[f"E{i}"] = hit.text.split("\n")
        
        # Extract significant keywords
        def extract_keywords(text: str) -> Set[str]:
            stopwords = {
                'this', 'that', 'with', 'from', 'have', 'been', 'were',
                'they', 'their', 'which', 'there', 'about', 'would',
                'could', 'should', 'these', 'those', 'being', 'other',
                'contains', 'shows', 'matches', 'unlike', 'normal'
            }
            words = re.findall(r'\b[a-z]{4,}\b', text.lower())
            return set(w for w in words if w not in stopwords)
        
        claims_with_span_match = 0
        claims_checked = 0
        
        for i, claim in enumerate(explanation.claims):
            if not claim.evidence_spans:
                continue
            
            claims_checked += 1
            claim_keywords = extract_keywords(claim.claim)
            if not claim_keywords:
                continue
            
            # Check if any keyword appears in any referenced span
            found_match = False
            for span in claim.evidence_spans:
                eid, line_num = self._parse_span(span)
                if eid is None or eid not in evidence_lines:
                    continue
                
                lines = evidence_lines[eid]
                if line_num is None:
                    # Bare evidence ID: whole-document reference — check all lines
                    full_text = " ".join(lines).lower()
                    for kw in claim_keywords:
                        if kw in full_text:
                            found_match = True
                            break
                elif line_num <= len(lines):
                    span_text = lines[line_num - 1].lower()
                    for kw in claim_keywords:
                        if kw in span_text:
                            found_match = True
                            break
                if found_match:
                    break
            
            if found_match:
                claims_with_span_match += 1
        
        if claims_checked > 0:
            match_rate = claims_with_span_match / claims_checked
            if match_rate < 0.5:
                issues.append(VerificationIssue(
                    check_name="span_keyword_match",
                    status=VerificationStatus.WARNING,
                    message=f"Only {claims_with_span_match}/{claims_checked} claims have keywords in their spans",
                    details={"match_rate": match_rate}
                ))
            else:
                issues.append(VerificationIssue(
                    check_name="span_keyword_match",
                    status=VerificationStatus.PASS,
                    message=f"{claims_with_span_match}/{claims_checked} claims have keywords in their spans"
                ))
        
        return issues
    
    def _check_cited_severity(
        self,
        explanation: TraceExplanation,
        query_session_text: str,
    ) -> VerificationIssue:
        """Check that cited E0 lines contain genuine error/warning keywords.

        This catches the most common HDFS hallucination pattern: the LLM
        cites normal INFO lines (Receiving block, PacketResponder) as
        errors.  If EVERY cited E0 line is a plain INFO line with no
        severity keyword, this check FAILs.

        Returns:
            VerificationIssue with PASS/FAIL/WARNING status.
        """
        e0_lines = query_session_text.split("\n")

        # Collect all cited E0 line numbers from claims
        cited_line_nums: set[int] = set()
        for claim in explanation.claims:
            for span in (claim.evidence_spans or []):
                span = str(span)  # LLM sometimes returns int
                eid, line_num = self._parse_span(span)
                if eid == "E0" and line_num is not None:
                    cited_line_nums.add(line_num)
                # Handle ranges like "E0-L5 to E0-L10"
                if " to " in span:
                    parts = span.split(" to ")
                    if len(parts) == 2:
                        _, start = self._parse_span(parts[0].strip())
                        _, end = self._parse_span(parts[1].strip())
                        if start is not None and end is not None:
                            for n in range(start, end + 1):
                                cited_line_nums.add(n)

        if not cited_line_nums:
            return VerificationIssue(
                check_name="cited_severity",
                status=VerificationStatus.WARNING,
                message="No E0 lines cited — cannot check severity",
            )

        # Check which cited lines contain severity keywords
        lines_with_severity = 0
        lines_checked = 0
        severity_details: list[dict] = []

        for ln in sorted(cited_line_nums):
            if 1 <= ln <= len(e0_lines):
                lines_checked += 1
                line_text = e0_lines[ln - 1]
                has_kw = any(kw in line_text for kw in self.SEVERITY_KEYWORDS)
                if has_kw:
                    lines_with_severity += 1
                severity_details.append({
                    "line": ln,
                    "has_severity": has_kw,
                    "snippet": line_text[:80],
                })

        if lines_checked == 0:
            return VerificationIssue(
                check_name="cited_severity",
                status=VerificationStatus.WARNING,
                message="All cited E0 line numbers are out of range",
                details={"cited": sorted(cited_line_nums), "e0_total": len(e0_lines)},
            )

        severity_ratio = lines_with_severity / lines_checked

        if severity_ratio == 0.0:
            # ALL cited lines are normal INFO — classic hallucination
            return VerificationIssue(
                check_name="cited_severity",
                status=VerificationStatus.FAIL,
                message=(
                    f"0/{lines_checked} cited E0 lines contain error/warning keywords "
                    f"— likely hallucination"
                ),
                details={"lines": severity_details},
            )
        elif severity_ratio < 0.5:
            return VerificationIssue(
                check_name="cited_severity",
                status=VerificationStatus.WARNING,
                message=(
                    f"Only {lines_with_severity}/{lines_checked} cited E0 lines "
                    f"contain error/warning keywords"
                ),
                details={"lines": severity_details},
            )
        else:
            return VerificationIssue(
                check_name="cited_severity",
                status=VerificationStatus.PASS,
                message=(
                    f"{lines_with_severity}/{lines_checked} cited E0 lines "
                    f"contain error/warning keywords"
                ),
                details={"lines": severity_details},
            )

    def verify_batch(
        self,
        results: List[ExplanationResult]
    ) -> Tuple[List[VerificationResult], Dict]:
        """
        Verify a batch of explanations.
        
        Returns:
            Tuple of (verification_results, summary_stats)
        """
        verifications = []
        
        for result in results:
            v = self.verify(
                explanation=result.explanation,
                evidence_hits=result.evidence_hits,
                evidence_id_mapping=result.evidence_id_mapping
            )
            v.session_id = result.session_id
            verifications.append(v)
        
        # Calculate summary stats
        total = len(verifications)
        passed = sum(1 for v in verifications if v.passed)
        
        all_issues = [i for v in verifications for i in v.issues]
        issue_counts = {}
        for issue in all_issues:
            key = issue.check_name.split("_claim_")[0]  # Group claim-specific checks
            if key not in issue_counts:
                issue_counts[key] = {"pass": 0, "fail": 0, "warning": 0}
            issue_counts[key][issue.status.value] += 1
        
        summary = {
            "total_explanations": total,
            "passed_explanations": passed,
            "pass_rate": passed / total if total > 0 else 0,
            "issue_breakdown": issue_counts
        }
        
        return verifications, summary


# Quick test
if __name__ == "__main__":
    from .prompt_builder import TraceExplanation, Claim
    from .retriever import RetrievalHit
    
    # Create test data
    explanation = TraceExplanation(
        prediction="anomaly",
        summary="Memory errors detected in this session",
        claims=[
            Claim(
                claim="Multiple DDR memory errors occurred",
                evidence_ids=["E1", "E2"]
            ),
            Claim(
                claim="Error pattern suggests hardware failure",
                evidence_ids=["E1"]
            ),
            Claim(
                claim="This is suspicious",  # No evidence
                evidence_ids=[]
            )
        ]
    )
    
    evidence_hits = [
        RetrievalHit(
            evidence_id="E_BGL_00001",
            score=5.2,
            text="DDR memory error detected at address 0x1234. Memory controller reported failure.",
            rank=1
        ),
        RetrievalHit(
            evidence_id="E_BGL_00002",
            score=4.8,
            text="Hardware diagnostic shows DDR module failure pattern.",
            rank=2
        )
    ]
    
    evidence_id_mapping = {"E1": "E_BGL_00001", "E2": "E_BGL_00002"}
    
    # Verify
    verifier = Verifier()
    result = verifier.verify(explanation, evidence_hits, evidence_id_mapping)
    
    print("Verification Result:")
    print(f"  Passed: {result.passed}")
    print(f"  Pass rate: {result.pass_rate:.0%}")
    print(f"  Issues:")
    for issue in result.issues:
        status_symbol = "✓" if issue.status == VerificationStatus.PASS else "✗" if issue.status == VerificationStatus.FAIL else "⚠"
        print(f"    {status_symbol} {issue.check_name}: {issue.message}")
