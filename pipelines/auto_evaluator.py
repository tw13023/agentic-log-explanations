"""
Automated Human-Evaluation Proxy for Log Explanations.

Rule-based scoring that mirrors manual human evaluation criteria:
  C  (Correctness 1-5): Is the explanation factually correct?
  Co (Coherence 1-5):   Is the explanation logically structured?
  E  (Evidence 1-5):    Are evidence references accurate and sufficient?
  Y/N:                  Overall acceptability (avg >= 3.0 -> Y)

Usage:
    from pipelines.auto_evaluator import AutoEvaluator
    evaluator = AutoEvaluator()
    scores = evaluator.evaluate_pipeline(pipeline)
    evaluator.print_report(scores)
"""

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class ExplanationScore:
    """Score for a single explanation."""
    session_id: str
    correctness: float          # C: 1-5
    coherence: float            # Co: 1-5
    evidence_quality: float     # E: 1-5
    acceptable: bool            # Y/N
    deductions: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def avg(self) -> float:
        return (self.correctness + self.coherence + self.evidence_quality) / 3

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "C": round(self.correctness, 2),
            "Co": round(self.coherence, 2),
            "E": round(self.evidence_quality, 2),
            "avg": round(self.avg, 2),
            "Y": self.acceptable,
            "deductions": self.deductions,
        }


@dataclass
class EvaluationReport:
    """Aggregate evaluation report."""
    dataset: str
    total: int
    scores: List[ExplanationScore]

    # Averages
    avg_c: float = 0.0
    avg_co: float = 0.0
    avg_e: float = 0.0
    pct_y: float = 0.0

    def compute(self):
        if not self.scores:
            return
        self.avg_c = statistics.mean(s.correctness for s in self.scores)
        self.avg_co = statistics.mean(s.coherence for s in self.scores)
        self.avg_e = statistics.mean(s.evidence_quality for s in self.scores)
        self.pct_y = sum(1 for s in self.scores if s.acceptable) / len(self.scores) * 100

    def to_dict(self) -> Dict:
        return {
            "dataset": self.dataset,
            "total_evaluated": self.total,
            "avg_C": round(self.avg_c, 2),
            "avg_Co": round(self.avg_co, 2),
            "avg_E": round(self.avg_e, 2),
            "pct_Y": round(self.pct_y, 1),
            "scores": [s.to_dict() for s in self.scores],
        }


class AutoEvaluator:
    """
    Rule-based automatic evaluator that produces scores comparable
    to manual human evaluation.

    Scoring logic:

    CORRECTNESS (C) — starts at 5.0:
      -2.0  verification overall FAIL
      -0.5  each keyword_match WARNING (max -1.5)
      -1.0  signature missing or UNKNOWN
      -0.5  cited_severity WARNING
      -0.5  prediction != "anomaly" (for anomaly sessions)

    COHERENCE (Co) — starts at 5.0:
      +0   3 claim types present (obs + pat + con)
      -1.0 only 2 types present
      -2.0 only 1 type present
      -0.5 summary too short (<30 chars)
      -0.5 duplicate claims detected (>80% word overlap)
      -0.5 claims < 2

    EVIDENCE (E) — starts at 5.0:
      -X   based on evidence_coverage gap: -(1 - coverage) * 3
      -1.0 evidence_spans_validity FAIL
      -0.5 span_keyword_match WARNING
      -0.5 no E0 spans cited at all
      -0.5 each claim with no spans (max -1.5)

    Y/N: avg(C, Co, E) >= 3.0
    """

    def evaluate_pipeline(self, pipeline) -> EvaluationReport:
        """Evaluate all results in a completed pipeline.

        Args:
            pipeline: ExplainAllPipeline with .results and .verifications populated.

        Returns:
            EvaluationReport with per-explanation scores.
        """
        scores: List[ExplanationScore] = []

        # Build verification lookup
        verif_map: Dict[str, object] = {}
        for v in pipeline.verifications:
            verif_map[v.session_id] = v

        for result in pipeline.results:
            v = verif_map.get(result.session_id)
            score = self._score_one(result, v)
            scores.append(score)

        report = EvaluationReport(
            dataset=pipeline.config.dataset,
            total=len(scores),
            scores=scores,
        )
        report.compute()
        return report

    def evaluate_jsonl(self, jsonl_path: str, verifications=None, dataset: str = "UNKNOWN") -> EvaluationReport:
        """Evaluate from a saved JSONL file (without pipeline object).

        This is a simpler path that scores based on the explanation dict alone
        (no verification issues available unless passed separately).
        """
        scores: List[ExplanationScore] = []
        with open(jsonl_path, "r") as f:
            content = f.read()

        # Parse JSONL (may be pretty-printed)
        objects = self._parse_jsonl(content)

        verif_map = {}
        if verifications:
            for v in verifications:
                verif_map[v.session_id] = v

        for obj in objects:
            session_id = obj.get("session_id", "?")
            exp = obj.get("explanation", {})
            v = verif_map.get(session_id)
            # Fall back to verification embedded in JSONL if no external verification
            if v is None:
                v_dict = obj.get("verification", {})
                if v_dict and v_dict.get("issues"):
                    v = self._verif_from_dict(v_dict)
            score = self._score_from_dict(session_id, exp, v)
            scores.append(score)

        report = EvaluationReport(dataset=dataset, total=len(scores), scores=scores)
        report.compute()
        return report

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_one(self, result, verification) -> ExplanationScore:
        """Score a single ExplanationResult + VerificationResult."""
        exp = result.explanation
        session_id = result.session_id

        # Extract verification issues into a lookup
        issues = self._issues_lookup(verification)

        c_score, c_ded = self._score_correctness(exp, issues)
        co_score, co_ded = self._score_coherence(exp)
        e_score, e_ded = self._score_evidence(exp, issues)

        avg = (c_score + co_score + e_score) / 3

        return ExplanationScore(
            session_id=session_id,
            correctness=c_score,
            coherence=co_score,
            evidence_quality=e_score,
            acceptable=(avg >= 3.0),
            deductions={"C": c_ded, "Co": co_ded, "E": e_ded},
        )

    def _score_from_dict(self, session_id: str, exp_dict: Dict, verification=None) -> ExplanationScore:
        """Score from a raw explanation dict (JSONL path)."""
        issues = self._issues_lookup(verification)

        c_score, c_ded = self._score_correctness_dict(exp_dict, issues)
        co_score, co_ded = self._score_coherence_dict(exp_dict)
        e_score, e_ded = self._score_evidence_dict(exp_dict, issues)

        avg = (c_score + co_score + e_score) / 3

        return ExplanationScore(
            session_id=session_id,
            correctness=c_score,
            coherence=co_score,
            evidence_quality=e_score,
            acceptable=(avg >= 3.0),
            deductions={"C": c_ded, "Co": co_ded, "E": e_ded},
        )

    # ------------------------------------------------------------------
    # CORRECTNESS
    # ------------------------------------------------------------------

    def _score_correctness(self, exp, issues: Dict) -> Tuple[float, List[str]]:
        ded: List[str] = []
        score = 5.0

        # Verification overall FAIL
        if issues.get("_overall_fail"):
            score -= 2.0
            ded.append("-2.0 verification FAIL")

        # keyword_match WARNINGs
        kw_warns = sum(1 for k, v in issues.items() if "keyword_match" in k and v == "warning")
        if kw_warns:
            penalty = min(kw_warns * 0.5, 1.5)
            score -= penalty
            ded.append(f"-{penalty} keyword_match WARNINGs x{kw_warns}")

        # Signature missing
        sig = exp.signature
        if sig is None or sig.name in ("UNKNOWN", "", None):
            score -= 1.0
            ded.append("-1.0 signature missing/UNKNOWN")

        # cited_severity WARNING
        if issues.get("cited_severity") == "warning":
            score -= 0.5
            ded.append("-0.5 cited_severity WARNING")

        # Prediction check
        if exp.prediction != "anomaly":
            score -= 0.5
            ded.append("-0.5 prediction != anomaly")

        return max(1.0, min(5.0, score)), ded

    def _score_correctness_dict(self, exp: Dict, issues: Dict) -> Tuple[float, List[str]]:
        ded: List[str] = []
        score = 5.0

        if issues.get("_overall_fail"):
            score -= 2.0
            ded.append("-2.0 verification FAIL")

        kw_warns = sum(1 for k, v in issues.items() if "keyword_match" in k and v == "warning")
        if kw_warns:
            penalty = min(kw_warns * 0.5, 1.5)
            score -= penalty
            ded.append(f"-{penalty} keyword_match WARNINGs x{kw_warns}")

        sig = exp.get("signature", {})
        sig_name = sig.get("name", "UNKNOWN") if isinstance(sig, dict) else str(sig)
        if sig_name in ("UNKNOWN", "", "None", None):
            score -= 1.0
            ded.append("-1.0 signature missing/UNKNOWN")

        if issues.get("cited_severity") == "warning":
            score -= 0.5
            ded.append("-0.5 cited_severity WARNING")

        if exp.get("prediction", "") != "anomaly":
            score -= 0.5
            ded.append("-0.5 prediction != anomaly")

        return max(1.0, min(5.0, score)), ded

    # ------------------------------------------------------------------
    # COHERENCE
    # ------------------------------------------------------------------

    def _score_coherence(self, exp) -> Tuple[float, List[str]]:
        return self._coherence_logic(
            claims=exp.claims,
            summary=exp.summary,
            get_type=lambda c: c.type,
            get_text=lambda c: c.claim,
        )

    def _score_coherence_dict(self, exp: Dict) -> Tuple[float, List[str]]:
        claims = exp.get("claims", [])
        return self._coherence_logic(
            claims=claims,
            summary=exp.get("summary", ""),
            get_type=lambda c: c.get("type", ""),
            get_text=lambda c: c.get("claim", ""),
        )

    def _coherence_logic(self, claims, summary, get_type, get_text) -> Tuple[float, List[str]]:
        ded: List[str] = []
        score = 5.0

        # Claim type diversity
        # Accept LLM synonyms: comparison->con, structural->obs (structural observations)
        types_present = set()
        for c in claims:
            t = get_type(c)
            if t in ("observation", "obs", "structural"):
                types_present.add("obs")
            elif t in ("pattern_match", "pat"):
                types_present.add("pat")
            elif t in ("contrast", "con", "comparison"):
                types_present.add("con")

        if len(types_present) == 2:
            score -= 1.0
            ded.append(f"-1.0 only 2 claim types: {types_present}")
        elif len(types_present) <= 1:
            score -= 2.0
            ded.append(f"-2.0 only {len(types_present)} claim type(s)")

        # Summary length
        if len(summary or "") < 30:
            score -= 0.5
            ded.append(f"-0.5 summary too short ({len(summary or '')} chars)")

        # Duplicate claims (>80% word overlap)
        texts = [get_text(c) for c in claims]
        if self._has_duplicates(texts):
            score -= 0.5
            ded.append("-0.5 duplicate claims detected")

        # Too few claims
        if len(claims) < 2:
            score -= 0.5
            ded.append(f"-0.5 only {len(claims)} claim(s)")

        return max(1.0, min(5.0, score)), ded

    # ------------------------------------------------------------------
    # EVIDENCE
    # ------------------------------------------------------------------

    def _score_evidence(self, exp, issues: Dict) -> Tuple[float, List[str]]:
        return self._evidence_logic(
            claims=exp.claims,
            issues=issues,
            get_spans=lambda c: c.evidence_spans or [],
        )

    def _score_evidence_dict(self, exp: Dict, issues: Dict) -> Tuple[float, List[str]]:
        claims = exp.get("claims", [])
        return self._evidence_logic(
            claims=claims,
            issues=issues,
            get_spans=lambda c: c.get("evidence_spans", []),
        )

    def _evidence_logic(self, claims, issues: Dict, get_spans) -> Tuple[float, List[str]]:
        ded: List[str] = []
        score = 5.0

        # Evidence coverage gap (from verification)
        coverage = issues.get("_coverage_ratio")
        if coverage is not None and coverage < 1.0:
            gap_penalty = (1.0 - coverage) * 3.0
            score -= gap_penalty
            ded.append(f"-{gap_penalty:.1f} evidence coverage {coverage:.0%}")

        # evidence_spans_validity FAIL
        if issues.get("evidence_spans_validity") == "fail":
            score -= 1.0
            ded.append("-1.0 evidence_spans_validity FAIL")

        # span_keyword_match WARNING
        if issues.get("span_keyword_match") == "warning":
            score -= 0.5
            ded.append("-0.5 span_keyword_match WARNING")

        # Check if any E0 spans are cited at all
        has_e0 = False
        claims_no_spans = 0
        for c in claims:
            spans = get_spans(c)
            if not spans:
                claims_no_spans += 1
            for s in spans:
                if str(s).startswith("E0"):
                    has_e0 = True

        if not has_e0 and len(claims) > 0:
            score -= 0.5
            ded.append("-0.5 no E0 spans cited")

        if claims_no_spans > 0:
            penalty = min(claims_no_spans * 0.5, 1.5)
            score -= penalty
            ded.append(f"-{penalty} {claims_no_spans} claim(s) with no spans")

        return max(1.0, min(5.0, score)), ded

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _verif_from_dict(self, v_dict: Dict):
        """Reconstruct a lightweight verification object from a JSONL dict.

        Returns an object with .passed, .issues (each with .check_name,
        .status, .details) -- enough for _issues_lookup().
        """
        from types import SimpleNamespace

        issues = []
        for iss_d in v_dict.get("issues", []):
            ns = SimpleNamespace(
                check_name=iss_d.get("check", ""),
                status=SimpleNamespace(value=iss_d.get("status", "pass")),
                message=iss_d.get("message", ""),
                details=iss_d.get("details", {}),
            )
            issues.append(ns)

        return SimpleNamespace(
            session_id=v_dict.get("session_id", "?"),
            passed=v_dict.get("passed", True),
            issues=issues,
            total_checks=v_dict.get("total_checks", 0),
            passed_checks=v_dict.get("passed_checks", 0),
            failed_checks=v_dict.get("failed_checks", 0),
            warning_checks=v_dict.get("warning_checks", 0),
        )

    def _issues_lookup(self, verification) -> Dict:
        """Convert VerificationResult into a flat lookup dict."""
        issues: Dict = {}
        if verification is None:
            return issues

        issues["_overall_fail"] = not verification.passed

        for iss in verification.issues:
            issues[iss.check_name] = iss.status.value
            # Extract coverage ratio from message
            if iss.check_name == "evidence_coverage" and iss.details:
                ratio = iss.details.get("coverage_ratio")
                if ratio is not None:
                    issues["_coverage_ratio"] = ratio

        return issues

    def _has_duplicates(self, texts: List[str], threshold: float = 0.8) -> bool:
        """Check if any two claim texts have >threshold word overlap."""
        if len(texts) < 2:
            return False
        word_sets = [set(t.lower().split()) for t in texts if t]
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                if not word_sets[i] or not word_sets[j]:
                    continue
                overlap = len(word_sets[i] & word_sets[j])
                smaller = min(len(word_sets[i]), len(word_sets[j]))
                if smaller > 0 and overlap / smaller > threshold:
                    return True
        return False

    def _parse_jsonl(self, content: str) -> List[Dict]:
        """Parse JSONL that may be pretty-printed (multi-line JSON objects)."""
        objects = []
        buf = []
        depth = 0
        for line in content.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            depth += stripped.count("{") - stripped.count("}")
            buf.append(line)
            if depth == 0 and buf:
                try:
                    objects.append(json.loads("\n".join(buf)))
                except json.JSONDecodeError:
                    pass
                buf = []
        return objects

    # ------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------

    def print_report(self, report: EvaluationReport, show_all: bool = False):
        """Print formatted evaluation report."""
        print("\n" + "=" * 70)
        print(f"  AUTO-EVALUATION REPORT  [{report.dataset}]")
        print("=" * 70)
        print(f"  Evaluated: {report.total} explanations")
        print(f"  Avg C  (Correctness):  {report.avg_c:.2f} / 5.00")
        print(f"  Avg Co (Coherence):    {report.avg_co:.2f} / 5.00")
        print(f"  Avg E  (Evidence):     {report.avg_e:.2f} / 5.00")
        print(f"  %Y (Acceptable):       {report.pct_y:.1f}%")
        print("=" * 70)

        # Score distribution
        c_dist = self._distribution([s.correctness for s in report.scores])
        co_dist = self._distribution([s.coherence for s in report.scores])
        e_dist = self._distribution([s.evidence_quality for s in report.scores])

        print("\nScore Distribution:")
        print(f"  C:  {c_dist}")
        print(f"  Co: {co_dist}")
        print(f"  E:  {e_dist}")

        # Show low-scoring explanations
        low = [s for s in report.scores if not s.acceptable]
        if low:
            print(f"\n[WARN] {len(low)} explanations scored below threshold (avg < 3.0):")
            for s in low:
                print(f"  {s.session_id}: C={s.correctness:.1f} Co={s.coherence:.1f} E={s.evidence_quality:.1f} avg={s.avg:.2f}")
                for dim, deds in s.deductions.items():
                    for d in deds:
                        print(f"    [{dim}] {d}")

        if show_all:
            print("\nAll scores:")
            for s in report.scores:
                tag = "Y" if s.acceptable else "N"
                print(f"  [{tag}] {s.session_id}: C={s.correctness:.1f} Co={s.coherence:.1f} E={s.evidence_quality:.1f} avg={s.avg:.2f}")

    def _distribution(self, values: List[float]) -> str:
        if not values:
            return "no data"
        bins = {"5.0": 0, "4.0-4.9": 0, "3.0-3.9": 0, "2.0-2.9": 0, "1.0-1.9": 0}
        for v in values:
            if v >= 5.0:
                bins["5.0"] += 1
            elif v >= 4.0:
                bins["4.0-4.9"] += 1
            elif v >= 3.0:
                bins["3.0-3.9"] += 1
            elif v >= 2.0:
                bins["2.0-2.9"] += 1
            else:
                bins["1.0-1.9"] += 1
        parts = [f"{k}:{v}" for k, v in bins.items() if v > 0]
        return " | ".join(parts)

    def save_report(self, report: EvaluationReport, path: str):
        """Save evaluation report to JSON."""
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"[OK] Evaluation saved to: {path}")


if __name__ == "__main__":
    print("AutoEvaluator ready. Use evaluate_pipeline() or evaluate_jsonl().")
