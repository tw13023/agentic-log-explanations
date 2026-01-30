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
    """
    
    def __init__(
        self,
        min_evidence_coverage: float = 0.8,
        require_keyword_match: bool = True,
        min_keyword_match_ratio: float = 0.3
    ):
        """
        Initialize verifier.
        
        Args:
            min_evidence_coverage: Minimum fraction of claims with evidence
            require_keyword_match: Whether to check keyword matches
            min_keyword_match_ratio: Minimum keyword match ratio per claim
        """
        self.min_evidence_coverage = min_evidence_coverage
        self.require_keyword_match = require_keyword_match
        self.min_keyword_match_ratio = min_keyword_match_ratio
    
    def verify(
        self,
        explanation: TraceExplanation,
        evidence_hits: List[RetrievalHit],
        evidence_id_mapping: Dict[str, str]
    ) -> VerificationResult:
        """
        Verify an explanation against its evidence.
        
        Args:
            explanation: The explanation to verify
            evidence_hits: The evidence that was provided
            evidence_id_mapping: Mapping from E1, E2 to original IDs
            
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
                explanation, evidence_hits, evidence_id_mapping
            )
            issues.extend(keyword_issues)
        
        # Check 5: Empty claims
        issues.append(self._check_empty_claims(explanation))
        
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
        evidence_id_mapping: Dict[str, str]
    ) -> List[VerificationIssue]:
        """Check that claims contain keywords from their referenced evidence."""
        issues = []
        
        # Build evidence text lookup
        evidence_texts = {}
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
