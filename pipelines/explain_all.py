"""
Explain-All Pipeline (Baseline)

This is the baseline pipeline that explains ALL predicted anomalies.
No gating/filtering - every anomaly detected by the Screener gets an explanation.
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
from src.evidence_store import EvidenceStore, build_evidence_store
from src.retriever import Retriever, RetrievalHit
from src.prompt_builder import PromptBuilder, TraceExplanation, ExplanationResult
from src.llm_client import LLMClient, LLMResponse
from src.verifier import Verifier, VerificationResult


@dataclass
class PipelineConfig:
    """Configuration for the Explain-All pipeline."""
    # Dataset
    dataset: str = "BGL"
    log_file: str = "./logs/BGL.log"
    model_path: str = "./best_model/best_model_20250724_072857.pth"
    
    # RAG settings
    top_k: int = 5
    retriever_method: str = "bm25"
    
    # LLM settings
    llm_provider: str = "ollama"
    llm_model: str = "llama3.2"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    llm_timeout: int = 120
    
    # Output
    output_dir: str = "./results/explanations"
    save_evidence_store: bool = True
    
    # Limits (for testing)
    max_sessions: Optional[int] = None  # None = process all
    
    def to_dict(self) -> Dict:
        return {
            "dataset": self.dataset,
            "log_file": self.log_file,
            "model_path": self.model_path,
            "top_k": self.top_k,
            "retriever_method": self.retriever_method,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
            "max_sessions": self.max_sessions
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
                "pass_rate": self.verification_pass_rate
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
        
        # 4. Retriever
        print(f"\n[4/6] Building Retriever...")
        self.retriever = Retriever(
            self.evidence_store,
            method=self.config.retriever_method
        )
        self.retriever.build_index()
        
        # 5. Prompt Builder
        print(f"\n[5/6] Initializing Prompt Builder...")
        self.prompt_builder = PromptBuilder()
        
        # 6. LLM Client
        print(f"\n[6/6] Initializing LLM Client...")
        self.llm_client = LLMClient(
            provider=self.config.llm_provider,
            model=self.config.llm_model,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
            timeout=self.config.llm_timeout
        )
        
        if not self.llm_client.is_available():
            print(f"  ⚠ WARNING: LLM ({self.config.llm_model}) is not available!")
            print(f"    Start Ollama with: ollama serve")
            print(f"    Pull model with: ollama pull {self.config.llm_model}")
        else:
            print(f"  ✓ LLM ({self.config.llm_model}) is available")
        
        # 7. Verifier
        self.verifier = Verifier()
        
        print("\n" + "="*60)
        print("SETUP COMPLETE")
        print("="*60)
        
        return self
    
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
        
        # Get anomalies
        anomaly_pairs = [
            (session, output)
            for session, output in zip(test_sessions, screener_outputs)
            if output.is_anomaly
        ]
        self.metrics.anomaly_sessions = len(anomaly_pairs)
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
                
            except Exception as e:
                self.metrics.failed_explanations += 1
                print(f"\n  ✗ Failed to explain {session.session_id}: {e}")
        
        # Step 3: Verify explanations
        print(f"\n[Step 3] Verifying explanations...")
        self.verifications, verify_summary = self.verifier.verify_batch(self.results)
        
        for v in self.verifications:
            if v.passed:
                self.metrics.verification_passed += 1
            else:
                self.metrics.verification_failed += 1
        
        self.metrics.end_time = time.time()
        
        # Print summary
        self._print_summary()
        
        return self
    
    def _explain_session(
        self,
        session: Session,
        screener_output: ScreenerOutput
    ) -> ExplanationResult:
        """Generate explanation for a single session."""
        # Retrieve evidence
        evidence_hits = self.retriever.retrieve_for_session(
            session,
            top_k=self.config.top_k
        )
        
        # Build prompt
        system_prompt, user_prompt = self.prompt_builder.build_prompt(
            session, screener_output, evidence_hits
        )
        evidence_id_mapping = self.prompt_builder.build_evidence_id_mapping(evidence_hits)
        
        # Call LLM
        try:
            parsed_json, llm_response = self.llm_client.generate_json(
                prompt=user_prompt,
                system_prompt=system_prompt
            )
            explanation = TraceExplanation.from_dict(parsed_json)
            explanation.raw_response = llm_response.content
            
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
        
        print(f"\n📊 Sessions:")
        print(f"  Total: {m.total_sessions:,}")
        print(f"  Anomalies (trigger): {m.anomaly_sessions:,} ({m.trigger_rate:.1%})")
        
        print(f"\n📝 Explanations:")
        print(f"  Attempted: {m.explained_sessions:,}")
        print(f"  Successful: {m.successful_explanations:,} ({m.success_rate:.1%})")
        print(f"  Failed: {m.failed_explanations:,}")
        
        print(f"\n✓ Verification:")
        print(f"  Passed: {m.verification_passed:,}")
        print(f"  Failed: {m.verification_failed:,}")
        print(f"  Pass rate: {m.verification_pass_rate:.1%}")
        
        print(f"\n💰 Cost (tokens):")
        print(f"  Total: {m.total_tokens:,}")
        print(f"  Avg/explanation: {m.avg_tokens_per_explanation:.0f}")
        
        print(f"\n⏱ Latency:")
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
    llm_model: str = "llama3.2"
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
            max_sessions=max_sessions
        )
    else:
        config = PipelineConfig(
            dataset="HDFS",
            log_file="./logs/HDFS.log",
            model_path="./best_model_HDFS/best_model_HDFS20250804_201746.pth",
            llm_model=llm_model,
            max_sessions=max_sessions
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
    parser.add_argument("--llm-model", type=str, default="llama3.2", help="Ollama model")
    
    args = parser.parse_args()
    
    run_explain_all_pipeline(
        dataset=args.dataset,
        max_sessions=args.max_sessions,
        llm_model=args.llm_model
    )
