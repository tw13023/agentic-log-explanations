"""
Explain-All Pipeline (Baseline)

This is the baseline pipeline that explains ALL predicted anomalies.
Supports two gating modes (see src/gating.py):
  Mode a (explain-all): every predicted anomaly is explained (default).
  Mode b (top-K):       only the K most uncertain anomalies are explained.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from tqdm import tqdm
import numpy as np

# Import all components
from src.data_loader import Session, BGLDataLoader, HDFSDataLoader, get_data_loader
from src.normalizer import get_normalizer
from src.screener import Screener, ScreenerOutput
from src.evidence_store import EvidenceStore, EvidenceDoc, build_evidence_store
from src.retriever import Retriever, RetrievalHit
from src.prompt_builder import (PromptBuilder, TraceExplanation, Claim, Signature,
                                ExplanationResult)
from src.llm_client import LLMClient, LLMResponse
from src.verifier import Verifier, VerificationResult
from src.gating import GatingMode, GatingConfig, gate


@dataclass
class PipelineConfig:
    """Configuration for the Explain-All pipeline."""
    # Dataset
    dataset: str = "BGL"
    log_file: str = "./logs/BGL.log"
    model_path: str = "./best_model/best_model_20250724_072857.pth"
    
    # RAG settings
    top_k: int = 5
    top_k_anomaly: int = 4  # Number of anomaly evidence to retrieve
    top_k_normal: int = 1   # Number of normal evidence for contrast claims
    use_mixed_retrieval: bool = True  # Enable mixed retrieval (anomaly + normal)
    retriever_method: str = "bm25"
    
    # LLM settings
    llm_provider: str = "ollama"
    llm_model: str = "llama3.1:8b"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    llm_timeout: int = 120
    
    # Patterns (data-driven signature cards)
    patterns_dir: str = "./patterns"  # directory with {dataset}_patterns.json
    
    # Output
    output_dir: str = "./results/explanations"
    save_evidence_store: bool = True
    
    # Gating
    gating_mode: str = "explain_all"  # "explain_all" or "top_k"
    gating_budget: float = 1.0         # fraction of anomalies to explain
    
    # Limits (for testing)
    max_sessions: Optional[int] = None  # None = process all
    
    def to_dict(self) -> Dict:
        return {
            "dataset": self.dataset,
            "log_file": self.log_file,
            "model_path": self.model_path,
            "top_k": self.top_k,
            "top_k_anomaly": self.top_k_anomaly,
            "top_k_normal": self.top_k_normal,
            "use_mixed_retrieval": self.use_mixed_retrieval,
            "retriever_method": self.retriever_method,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
            "patterns_dir": self.patterns_dir,
            "max_sessions": self.max_sessions,
            "gating_mode": self.gating_mode,
            "gating_budget": self.gating_budget,
        }


@dataclass
class PipelineMetrics:
    """Metrics collected during pipeline execution."""
    # Counts
    total_sessions: int = 0
    anomaly_sessions: int = 0
    explained_sessions: int = 0
    successful_explanations: int = 0
    failed_explanations: int = 0
    
    # Verification
    verification_passed: int = 0
    verification_failed: int = 0
    verification_warnings: int = 0
    
    # Signatures (NEW)
    signature_counts: Dict[str, int] = field(default_factory=dict)
    
    # Tokens & Latency
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    latencies: List[float] = field(default_factory=list)
    
    # Timing
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    @property
    def avg_tokens_per_explanation(self) -> float:
        if self.successful_explanations == 0:
            return 0.0
        return self.total_tokens / self.successful_explanations
    
    @property
    def avg_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        return np.mean(self.latencies)
    
    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        return np.percentile(self.latencies, 95)
    
    @property
    def total_time_seconds(self) -> float:
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time
    
    @property
    def trigger_rate(self) -> float:
        """Fraction of sessions that triggered explanation."""
        if self.total_sessions == 0:
            return 0.0
        return self.anomaly_sessions / self.total_sessions
    
    @property
    def success_rate(self) -> float:
        """Fraction of explanations that succeeded."""
        if self.explained_sessions == 0:
            return 0.0
        return self.successful_explanations / self.explained_sessions
    
    @property
    def verification_pass_rate(self) -> float:
        """Fraction of explanations that passed verification."""
        total_verified = self.verification_passed + self.verification_failed
        if total_verified == 0:
            return 0.0
        return self.verification_passed / total_verified
    
    def to_dict(self) -> Dict:
        return {
            "counts": {
                "total_sessions": self.total_sessions,
                "anomaly_sessions": self.anomaly_sessions,
                "explained_sessions": self.explained_sessions,
                "successful_explanations": self.successful_explanations,
                "failed_explanations": self.failed_explanations
            },
            "verification": {
                "passed": self.verification_passed,
                "failed": self.verification_failed,
                "warnings": self.verification_warnings,
                "pass_rate": self.verification_pass_rate
            },
            "signatures": {
                "unique_count": len(self.signature_counts),
                "distribution": self.signature_counts
            },
            "tokens": {
                "total": self.total_tokens,
                "avg_per_explanation": self.avg_tokens_per_explanation
            },
            "latency": {
                "avg_ms": self.avg_latency_ms,
                "p95_ms": self.p95_latency_ms,
                "total_llm_ms": self.total_latency_ms
            },
            "rates": {
                "trigger_rate": self.trigger_rate,
                "success_rate": self.success_rate
            },
            "total_time_seconds": self.total_time_seconds
        }


class ExplainAllPipeline:
    """
    Baseline Explain-All Pipeline.
    
    Flow:
    1. Load data and split
    2. Run Screener on test set
    3. Build Evidence Store from train set
    4. For each predicted anomaly:
       a. Retrieve top-k evidence
       b. Build prompt
       c. Call LLM for explanation
       d. Parse and verify explanation
    5. Save results and metrics
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.metrics = PipelineMetrics()
        
        # Components (initialized in setup)
        self.data_loader = None
        self.screener = None
        self.evidence_store = None
        self.retriever = None
        self.prompt_builder = None
        self.llm_client = None
        self.verifier = None
        
        # Results
        self.results: List[ExplanationResult] = []
        self.verifications: List[VerificationResult] = []
    
    def setup(self) -> "ExplainAllPipeline":
        """Initialize all pipeline components."""
        print("="*60)
        print("EXPLAIN-ALL PIPELINE SETUP")
        print("="*60)
        
        # 1. Data Loader
        print(f"\n[1/6] Loading {self.config.dataset} data...")
        if self.config.dataset.upper() == "BGL":
            self.data_loader = BGLDataLoader(log_file=self.config.log_file)
        else:
            # For HDFS, need label file too
            self.data_loader = HDFSDataLoader(
                log_file=self.config.log_file,
                label_file="./logs/anomaly_label_HDFS.csv"
            )
        self.data_loader.load()
        self.data_loader.print_stats()
        
        # 2. Screener
        print(f"\n[2/6] Loading Screener model...")
        self.screener = Screener.from_pretrained(
            dataset=self.config.dataset,
            model_path=self.config.model_path
        )
        
        # 3. Evidence Store
        print(f"\n[3/6] Building Evidence Store...")
        evidence_path = Path(self.config.output_dir) / f"evidence_store_{self.config.dataset}.json"
        if evidence_path.exists():
            print(f"  Loading from {evidence_path}")
            self.evidence_store = EvidenceStore(self.config.dataset)
            self.evidence_store.load(str(evidence_path))
        else:
            self.evidence_store = build_evidence_store(
                self.data_loader,
                self.config.dataset,
                save_path=str(evidence_path) if self.config.save_evidence_store else None
            )
        
        # 4. Normalizer (needed by PromptBuilder and Retriever)
        self.normalizer = get_normalizer(self.config.dataset)
        
        # 5. Signature cards — add BEFORE building BM25 index so cards are searchable
        self._load_signature_cards()
        
        # 6. Retriever — build index AFTER signature cards are in the evidence store
        print(f"\n[4/7] Building Retriever...")
        self.retriever = Retriever(
            self.evidence_store,
            method=self.config.retriever_method
        )
        self.retriever.build_index()
        
        # 7. Prompt Builder (with normalizer for structural summary injection)
        print(f"\n[5/7] Initializing Prompt Builder...")
        self.prompt_builder = PromptBuilder(
            dataset=self.config.dataset,
            normalizer=self.normalizer,
        )
        
        # 8. LLM Client
        print(f"\n[6/7] Initializing LLM Client...")
        self.llm_client = LLMClient(
            provider=self.config.llm_provider,
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
            timeout=self.config.llm_timeout
        )
        
        if not self.llm_client.is_available():
            print(f"  [WARN] LLM ({self.config.llm_model}) is not available!")
            print(f"    Start Ollama with: ollama serve")
            print(f"    Pull model with: ollama pull {self.config.llm_model}")
        else:
            print(f"  [OK] LLM ({self.config.llm_model}) is available")
        
        # 9. Verifier (keyword match at 0.15 — catch pure-hallucination)
        self.verifier = Verifier(
            min_keyword_match_ratio=0.15,
            dataset=self.config.dataset,
        )
        
        print("\n" + "="*60)
        print("SETUP COMPLETE")
        print("="*60)
        
        return self
    
    def _load_signature_cards(self) -> None:
        """Load data-driven patterns from JSON and add as signature cards.
        
        IMPORTANT: Must be called BEFORE retriever.build_index() so that
        signature cards are included in the BM25 index and can be retrieved.
        """
        dataset = self.config.dataset.lower()
        patterns_file = Path(self.config.patterns_dir) / f"{dataset}_patterns.json"
        
        if not patterns_file.exists():
            print(f"\n[SIG] No patterns file found at {patterns_file} -- skipping signature cards")
            return
        
        print(f"\n[SIG] Loading signature cards from {patterns_file}...")
        with open(patterns_file, 'r') as f:
            patterns = json.load(f)
        for pattern_id, info in patterns.items():
            # BGL uses 'fingerprint', HDFS uses 'merge_key'
            pattern_key = info.get('merge_key', info.get('fingerprint', 'N/A'))
            sig_text = f"""ERROR SIGNATURE: {info['name']}
Description: {info['description']}

Key Indicators: {', '.join(info['keywords'])}
Frequency: {info['frequency']} occurrences in training data

Pattern Characteristics:
  - Fingerprint: {pattern_key}
"""
            doc = EvidenceDoc(
                evidence_id=f"E_SIG_{pattern_id}",
                session_id=pattern_id,
                text=sig_text,
                evidence_type="signature",
                metadata={
                    "label": 1,
                    "dataset": self.config.dataset,
                    "signature_name": info['name'],
                    "frequency": info['frequency'],
                    "keywords": info['keywords'],
                }
            )
            self.evidence_store.documents.append(doc)
            self.evidence_store._id_to_doc[doc.evidence_id] = doc
        
        print(f"  Added {len(patterns)} signature cards")
        print(f"  Evidence store now has {len(self.evidence_store.documents):,} documents")
    
    def run(self) -> "ExplainAllPipeline":
        """Run the full pipeline."""
        self.metrics.start_time = time.time()
        
        print("\n" + "="*60)
        print("RUNNING EXPLAIN-ALL PIPELINE")
        print("="*60)
        
        # Get test sessions
        test_sessions = self.data_loader.get_test()
        if self.config.max_sessions:
            test_sessions = test_sessions[:self.config.max_sessions]
        
        self.metrics.total_sessions = len(test_sessions)
        print(f"\nProcessing {len(test_sessions)} test sessions...")
        
        # Step 1: Screen all sessions
        print("\n[Step 1] Running Screener...")
        screener_outputs = self.screener.screen_sessions(test_sessions)
        
        # Get anomalies (apply gating)
        gating_cfg = GatingConfig(
            mode=GatingMode(self.config.gating_mode),
            budget=self.config.gating_budget,
        )
        anomaly_pairs = gate(test_sessions, screener_outputs, gating_cfg)
        all_anomalies = sum(1 for o in screener_outputs if o.is_anomaly)
        self.metrics.anomaly_sessions = all_anomalies
        if gating_cfg.mode == GatingMode.TOP_K and gating_cfg.budget < 1.0:
            print(f"  Found {all_anomalies} predicted anomalies, "
                  f"gated to {len(anomaly_pairs)} (B={gating_cfg.budget:.0%})")
        else:
            print(f"  Found {len(anomaly_pairs)} predicted anomalies ({len(anomaly_pairs)/len(test_sessions):.1%})")
        
        # Step 2: Explain each anomaly
        print(f"\n[Step 2] Explaining anomalies...")
        
        for session, screener_output in tqdm(anomaly_pairs, desc="Explaining"):
            self.metrics.explained_sessions += 1
            
            try:
                result = self._explain_session(session, screener_output)
                self.results.append(result)
                self.metrics.successful_explanations += 1
                self.metrics.total_tokens += result.total_tokens
                self.metrics.total_latency_ms += result.latency_ms
                self.metrics.latencies.append(result.latency_ms)
                
                # Track signature (NEW)
                if result.explanation.signature:
                    sig_name = result.explanation.signature.name
                    self.metrics.signature_counts[sig_name] = self.metrics.signature_counts.get(sig_name, 0) + 1
                
            except Exception as e:
                self.metrics.failed_explanations += 1
                print(f"\n  ✗ Failed to explain {session.session_id}: {e}")
        
        # Step 3: Verify explanations (with E0 text for keyword matching)
        print(f"\n[Step 3] Verifying explanations...")
        self._verify_all_explanations()
        
        self.metrics.end_time = time.time()
        
        # Print summary
        self._print_summary()
        
        return self
    
    def _verify_all_explanations(self) -> None:
        """Verify all explanations with E0 text for accurate keyword matching."""
        for result in self.results:
            query_session_text = "\n".join(result.session.lines)
            v = self.verifier.verify(
                explanation=result.explanation,
                evidence_hits=result.evidence_hits,
                evidence_id_mapping=result.evidence_id_mapping,
                query_session_text=query_session_text
            )
            v.session_id = result.session_id
            self.verifications.append(v)
            
            if v.passed:
                self.metrics.verification_passed += 1
            else:
                self.metrics.verification_failed += 1
            self.metrics.verification_warnings += v.warning_checks
    
    def _explain_session(
        self,
        session: Session,
        screener_output: ScreenerOutput
    ) -> ExplanationResult:
        """Generate explanation for a single session."""
        # Retrieve evidence (mixed or standard)
        if self.config.use_mixed_retrieval:
            # Mixed retrieval: anomaly exemplars + normal for contrast claims
            evidence_hits = self.retriever.retrieve_for_session_mixed(
                session,
                top_k_anomaly=self.config.top_k_anomaly,
                top_k_normal=self.config.top_k_normal
            )
        else:
            # Standard retrieval: top-k any label
            evidence_hits = self.retriever.retrieve_for_session(
                session,
                top_k=self.config.top_k
            )
        
        # Build prompt
        system_prompt, user_prompt = self.prompt_builder.build_prompt(
            session, screener_output, evidence_hits
        )
        evidence_id_mapping = self.prompt_builder.build_evidence_id_mapping(session, evidence_hits)
        
        # Call LLM
        try:
            parsed_json, llm_response = self.llm_client.generate_json(
                prompt=user_prompt,
                system_prompt=system_prompt
            )
            explanation = TraceExplanation.from_dict(parsed_json)
            explanation.raw_response = llm_response.content
            
            # Normalize signature name (strip severity, canonical error types)
            if explanation.signature and explanation.signature.name:
                explanation.signature.name = self.normalizer.normalize_signature(
                    explanation.signature.name
                )
            
        except json.JSONDecodeError as e:
            # Try to salvage partial response
            explanation = TraceExplanation(
                prediction="anomaly",
                summary=f"JSON parse error: {e}",
                claims=[],
                insufficient_evidence=True,
                raw_response=llm_response.content if 'llm_response' in dir() else ""
            )
            llm_response = LLMResponse(
                content="",
                model=self.config.llm_model,
                latency_ms=0
            )
        
        return ExplanationResult(
            session_id=session.session_id,
            session=session,
            screener_output=screener_output,
            evidence_hits=evidence_hits,
            explanation=explanation,
            evidence_id_mapping=evidence_id_mapping,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            total_tokens=llm_response.total_tokens,
            latency_ms=llm_response.latency_ms
        )
    
    def _print_summary(self) -> None:
        """Print pipeline execution summary."""
        print("\n" + "="*60)
        print("PIPELINE SUMMARY")
        print("="*60)
        
        m = self.metrics
        
        print(f"\nSessions:")
        print(f"  Total: {m.total_sessions:,}")
        print(f"  Anomalies (trigger): {m.anomaly_sessions:,} ({m.trigger_rate:.1%})")
        
        print(f"\nExplanations:")
        print(f"  Attempted: {m.explained_sessions:,}")
        print(f"  Successful: {m.successful_explanations:,} ({m.success_rate:.1%})")
        print(f"  Failed: {m.failed_explanations:,}")
        
        print(f"\nVerification:")
        print(f"  Passed: {m.verification_passed:,}")
        print(f"  Failed: {m.verification_failed:,}")
        print(f"  Warnings: {m.verification_warnings:,}")
        print(f"  Pass rate: {m.verification_pass_rate:.1%}")
        
        # Signature distribution (NEW)
        if m.signature_counts:
            print(f"\nSignatures ({len(m.signature_counts)} unique):")
            sorted_sigs = sorted(m.signature_counts.items(), key=lambda x: -x[1])
            for sig_name, count in sorted_sigs[:10]:  # Top 10
                print(f"  {sig_name}: {count:,}")
            if len(sorted_sigs) > 10:
                print(f"  ... and {len(sorted_sigs) - 10} more")
        
        print(f"\nTokens:")
        print(f"  Total: {m.total_tokens:,}")
        print(f"  Avg/explanation: {m.avg_tokens_per_explanation:.0f}")
        
        print(f"\nLatency:")
        print(f"  Total LLM time: {m.total_latency_ms/1000:.1f}s")
        print(f"  Avg/explanation: {m.avg_latency_ms:.0f}ms")
        print(f"  P95: {m.p95_latency_ms:.0f}ms")
        print(f"  Total pipeline time: {m.total_time_seconds:.1f}s")
    
    def save_results(self, output_path: Optional[str] = None) -> str:
        """Save results to JSONL file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(self.config.output_dir) / f"explanations_{self.config.dataset}_{timestamp}.jsonl"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for result in self.results:
                f.write(result.to_json() + "\n")
        
        print(f"\n💾 Results saved to: {output_path}")
        
        # Also save metrics
        metrics_path = output_path.with_suffix(".metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({
                "config": self.config.to_dict(),
                "metrics": self.metrics.to_dict()
            }, f, indent=2)
        
        print(f"📈 Metrics saved to: {metrics_path}")
        
        return str(output_path)


def run_explain_all_pipeline(
    dataset: str = "BGL",
    max_sessions: Optional[int] = None,
    llm_model: str = "llama3.1:8b"
) -> ExplainAllPipeline:
    """
    Convenience function to run the Explain-All pipeline.
    
    Args:
        dataset: "BGL" or "HDFS"
        max_sessions: Limit number of test sessions (for testing)
        llm_model: Ollama model to use
        
    Returns:
        Completed pipeline object
    """
    # Set paths based on dataset
    if dataset.upper() == "BGL":
        config = PipelineConfig(
            dataset="BGL",
            log_file="./logs/BGL.log",
            model_path="./best_model/best_model_20250724_072857.pth",
            llm_model=llm_model,
            max_sessions=max_sessions,
            output_dir="./results",
        )
    else:
        config = PipelineConfig(
            dataset="HDFS",
            log_file="./logs/HDFS.log",
            model_path="./best_model_HDFS/best_model_HDFS20250804_201746.pth",
            llm_model=llm_model,
            max_sessions=max_sessions,
            output_dir="./results_HDFS",
        )
    
    pipeline = ExplainAllPipeline(config)
    pipeline.setup()
    pipeline.run()
    pipeline.save_results()
    
    return pipeline


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Explain-All Pipeline")
    parser.add_argument("--dataset", type=str, default="BGL", choices=["BGL", "HDFS"])
    parser.add_argument("--max-sessions", type=int, default=None, help="Limit test sessions")
    parser.add_argument("--llm-model", type=str, default="llama3.1:8b", help="Ollama model")
    
    args = parser.parse_args()
    
    run_explain_all_pipeline(
        dataset=args.dataset,
        max_sessions=args.max_sessions,
        llm_model=args.llm_model
    )
