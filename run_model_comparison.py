"""
Multi-Model Edge Case Comparison

Runs the 24 HDFS verification-FAIL edge cases through multiple
open-source LLMs (via Ollama) with the prompt fix applied,
then compares verification pass rates and auto-eval scores.

Usage:
    # Run all models sequentially:
    python run_model_comparison.py

    # Run a single model:
    python run_model_comparison.py --model qwen2.5:14b

    # Just compare existing results:
    python run_model_comparison.py --compare-only
"""

import json
import sys
import argparse
import statistics
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from pipelines.explain_all import ExplainAllPipeline, PipelineConfig
from pipelines.auto_evaluator import AutoEvaluator


# ---- Models to compare ----
MODELS = [
    "llama3.1:8b",      # 4.9 GB -- current baseline
    "gemma2:9b",         # 5.4 GB -- Google, strong reasoning
    "qwen2.5:14b",       # 8.7 GB -- Alibaba, largest that fits 12GB
    "mistral:7b",        # 4.1 GB -- Mistral AI, efficient
    "phi3:14b",          # 7.9 GB -- Microsoft, strong instruction-following
]

# ---- 24 HDFS edge cases (verification FAIL from 500-anomaly run) ----
EDGE_CASE_IDS = [
    # Cat A: Empty claims (LLM returned no claims)
    "HDFS_blk_-3661881463166428296",
    "HDFS_blk_-4014392925001710803",
    "HDFS_blk_1340637939534925227",
    "HDFS_blk_-6958758037020501321",
    "HDFS_blk_-9139390120172868222",
    "HDFS_blk_-2981635085495510991",
    "HDFS_blk_-3646517331515187756",
    # Cat B: All evidence_spans empty
    "HDFS_blk_-2565130857684426419",
    "HDFS_blk_-8178334713387015260",
    "HDFS_blk_9053361956697310554",
    # Cat C: Spans with 'STRUCTURAL' format
    "HDFS_blk_7963243402010652895",
    "HDFS_blk_-6108988464398509549",
    "HDFS_blk_5703197246022264715",
    "HDFS_blk_9216586288937763843",
    "HDFS_blk_8095512464329197839",
    "HDFS_blk_-3371701595281520594",
    "HDFS_blk_8332836013426864517",
    "HDFS_blk_8239323489440610674",
    # Cat D: Proper spans but keyword mismatch
    "HDFS_blk_8065120248309205717",
    "HDFS_blk_3338061113250311986",
    "HDFS_blk_3387839675473728056",
    "HDFS_blk_-4184262965734141875",
    "HDFS_blk_6757687595653348398",
    "HDFS_blk_4690101219411104590",
]

# Category labels
CAT_LABELS = {}
for sid in EDGE_CASE_IDS[:7]:
    CAT_LABELS[sid] = "A-empty_claims"
for sid in EDGE_CASE_IDS[7:10]:
    CAT_LABELS[sid] = "B-empty_spans"
for sid in EDGE_CASE_IDS[10:18]:
    CAT_LABELS[sid] = "C-STRUCTURAL"
for sid in EDGE_CASE_IDS[18:]:
    CAT_LABELS[sid] = "D-keyword"

CAT_ORDER = ["A-empty_claims", "B-empty_spans", "C-STRUCTURAL", "D-keyword"]


def model_tag(model_name: str) -> str:
    """Convert model name to safe file tag, e.g. 'qwen2.5:14b' -> 'qwen2.5_14b'"""
    return model_name.replace(":", "_").replace(".", "p")


def ensure_model_available(model_name: str) -> bool:
    """Check if model is pulled; return True if available."""
    try:
        result = subprocess.run(
            ["ollama", "show", model_name],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print(f"  [OK] {model_name} is available")
            return True
        # Try pulling
        print(f"  [PULL] Pulling {model_name}...")
        pull = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=False, timeout=1800  # 30 min timeout
        )
        return pull.returncode == 0
    except Exception as e:
        print(f"  [ERR] {model_name}: {e}")
        return False


def run_single_model(model_name: str, reuse_pipeline: Optional[ExplainAllPipeline] = None):
    """Run edge cases with a specific model. Reuses setup if pipeline provided."""
    tag = model_tag(model_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results_HDFS")

    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"{'='*60}")

    # Configure pipeline
    config = PipelineConfig(
        dataset="HDFS",
        log_file="./logs/HDFS.log",
        model_path="./best_model_HDFS/best_model_HDFS20250804_201746.pth",
        output_dir="./results_HDFS",
        llm_model=model_name,
        session_ids=EDGE_CASE_IDS,
    )

    pipeline = ExplainAllPipeline(config)

    # Reuse heavy components (data, screener model, evidence store, retriever)
    # if a previous pipeline is provided
    if reuse_pipeline is not None:
        print("  Reusing data loader, screener, evidence store, retriever...")
        pipeline.data_loader = reuse_pipeline.data_loader
        pipeline.screener = reuse_pipeline.screener
        pipeline.evidence_store = reuse_pipeline.evidence_store
        pipeline.retriever = reuse_pipeline.retriever
        pipeline.prompt_builder = reuse_pipeline.prompt_builder
        pipeline.verifier = reuse_pipeline.verifier
        pipeline.normalizer = reuse_pipeline.normalizer

        # Only reinitialize the LLM client for the new model
        from src.llm_client import LLMClient
        pipeline.llm_client = LLMClient(
            provider="ollama",
            model=model_name,
        )
        # Check availability
        if not pipeline.llm_client.is_available():
            print(f"  [WARN] LLM ({model_name}) is not available!")
            print(f"    Pull model with: ollama pull {model_name}")
            return None, None
        print(f"  [OK] LLM ({model_name}) is available")

        # Reset results and metrics
        pipeline.results = []
        pipeline.verifications = []
        from pipelines.explain_all import PipelineMetrics
        pipeline.metrics = PipelineMetrics()
    else:
        pipeline.setup()

    # Run pipeline
    pipeline.run()

    # Save results
    out_file = out_dir / f"model_cmp_{tag}_{timestamp}.jsonl"
    pipeline.save_results(str(out_file))

    # Evaluate
    evaluator = AutoEvaluator()
    report = evaluator.evaluate_pipeline(pipeline)

    # Print per-model summary
    print(f"\n--- {model_name} Auto-Eval ---")
    evaluator.print_report(report)

    # Category breakdown
    cat_stats = {}
    for cat_name in CAT_ORDER:
        cat_scores = [sc for sc in report.scores if CAT_LABELS.get(sc.session_id) == cat_name]
        if cat_scores:
            verif_fail = sum(1 for sc in cat_scores
                            if any("verification FAIL" in d
                                   for dim_deds in (sc.deductions.values() if isinstance(sc.deductions, dict) else [])
                                   for d in (dim_deds if isinstance(dim_deds, list) else [])))
            cat_stats[cat_name] = {
                "n": len(cat_scores),
                "C": statistics.mean(sc.correctness for sc in cat_scores),
                "Co": statistics.mean(sc.coherence for sc in cat_scores),
                "E": statistics.mean(sc.evidence_quality for sc in cat_scores),
                "verif_fail": verif_fail,
            }

    print(f"\n--- Category breakdown ---")
    for cat_name in CAT_ORDER:
        s = cat_stats.get(cat_name)
        if s:
            print(f"  {cat_name} (n={s['n']}): C={s['C']:.2f} Co={s['Co']:.2f} E={s['E']:.2f} verif_FAIL={s['verif_fail']}")

    # Collect pipeline-level stats
    verif_pass = sum(1 for v in pipeline.verifications if v.passed)
    verif_total = len(pipeline.verifications)
    avg_latency = statistics.mean(r.latency_ms for r in pipeline.results if hasattr(r, 'latency_ms') and r.latency_ms) if pipeline.results else 0
    total_tokens = sum(r.total_tokens for r in pipeline.results if hasattr(r, 'total_tokens') and r.total_tokens)

    model_result = {
        "model": model_name,
        "tag": tag,
        "timestamp": timestamp,
        "avg_C": report.avg_c,
        "avg_Co": report.avg_co,
        "avg_E": report.avg_e,
        "pct_Y": report.pct_y,
        "verif_pass": verif_pass,
        "verif_total": verif_total,
        "verif_rate": verif_pass / verif_total * 100 if verif_total > 0 else 0,
        "avg_latency_ms": avg_latency,
        "total_tokens": total_tokens,
        "cat_stats": cat_stats,
        "results_file": str(out_file),
    }

    # Save eval (full model result with scores)
    eval_file = out_dir / f"model_cmp_{tag}_{timestamp}_eval.json"
    eval_data = {**model_result, "scores": [s.to_dict() for s in report.scores]}
    with open(eval_file, "w") as f:
        json.dump(eval_data, f, indent=2)
    model_result["eval_file"] = str(eval_file)

    print(f"\nResults: {out_file}")
    print(f"Eval: {eval_file}")

    return pipeline, model_result


def print_comparison(all_results: List[Dict]):
    """Print a comparison table across all models."""
    print(f"\n{'='*80}")
    print(f"{'MODEL COMPARISON':^80}")
    print(f"{'='*80}")

    # Sort by composite score: 0.4*C + 0.2*Co + 0.2*E + 0.2*verif_rate/20
    for r in all_results:
        r["composite"] = (
            0.40 * r["avg_C"]
            + 0.20 * r["avg_Co"]
            + 0.20 * r["avg_E"]
            + 0.20 * (r["verif_rate"] / 20.0)  # scale 0-100 to 0-5
        )

    ranked = sorted(all_results, key=lambda x: x["composite"], reverse=True)

    # Header
    print(f"\n{'Rank':<5} {'Model':<18} {'C':>5} {'Co':>5} {'E':>5} {'%Y':>6} {'Verif':>8} {'Latency':>10} {'Tokens':>8} {'Score':>6}")
    print("-" * 80)

    for i, r in enumerate(ranked, 1):
        vr = f"{r['verif_pass']}/{r['verif_total']}"
        lat = f"{r['avg_latency_ms']:.0f}ms"
        print(f"  {i:<3} {r['model']:<18} {r['avg_C']:>5.2f} {r['avg_Co']:>5.2f} {r['avg_E']:>5.2f} {r['pct_Y']:>5.1f}% {vr:>8} {lat:>10} {r['total_tokens']:>8} {r['composite']:>6.2f}")

    # Per-category comparison
    print(f"\n{'--- Per-Category Correctness ---':^80}")
    print(f"{'Model':<18}", end="")
    for cat in CAT_ORDER:
        print(f" {cat:>16}", end="")
    print()
    print("-" * 82)

    for r in ranked:
        print(f"{r['model']:<18}", end="")
        for cat in CAT_ORDER:
            cs = r["cat_stats"].get(cat, {})
            c_val = cs.get("C", 0)
            vf = cs.get("verif_fail", 0)
            print(f" {c_val:>5.2f}(F={vf})", end="    ")
        print()

    # Best model
    best = ranked[0]
    print(f"\n{'='*80}")
    print(f"RECOMMENDED: {best['model']}")
    print(f"  Composite Score: {best['composite']:.2f}")
    print(f"  Correctness: {best['avg_C']:.2f}")
    print(f"  Coherence: {best['avg_Co']:.2f}")
    print(f"  Evidence: {best['avg_E']:.2f}")
    print(f"  Verification: {best['verif_pass']}/{best['verif_total']} ({best['verif_rate']:.1f}%)")
    print(f"  Avg Latency: {best['avg_latency_ms']:.0f}ms")
    print(f"{'='*80}")

    return ranked


def load_existing_results() -> List[Dict]:
    """Load previously saved model comparison results."""
    results_dir = Path("results_HDFS")
    all_results = []
    for eval_file in sorted(results_dir.glob("model_cmp_*_eval.json")):
        with open(eval_file) as f:
            data = json.load(f)
        # Extract model name from filename
        # model_cmp_qwen2p5_14b_20260228_...
        parts = eval_file.stem.replace("model_cmp_", "").rsplit("_", 2)
        tag = parts[0] if parts else "?"
        all_results.append({
            "model": tag,
            "tag": tag,
            "avg_C": data.get("avg_C", 0),
            "avg_Co": data.get("avg_Co", 0),
            "avg_E": data.get("avg_E", 0),
            "pct_Y": data.get("pct_Y", 0),
            "verif_pass": data.get("verif_pass", 0),
            "verif_total": data.get("verif_total", 24),
            "verif_rate": data.get("verif_pass", 0) / max(data.get("verif_total", 24), 1) * 100,
            "avg_latency_ms": data.get("avg_latency_ms", 0),
            "total_tokens": data.get("total_tokens", 0),
            "cat_stats": data.get("cat_stats", {}),
            "eval_file": str(eval_file),
        })
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Multi-model edge case comparison")
    parser.add_argument("--model", type=str, help="Run a single model instead of all")
    parser.add_argument("--compare-only", action="store_true", help="Just compare existing results")
    parser.add_argument("--models", nargs="+", help="Specify which models to run")
    args = parser.parse_args()

    if args.compare_only:
        results = load_existing_results()
        if results:
            print_comparison(results)
        else:
            print("No results found. Run models first.")
        return

    models_to_run = [args.model] if args.model else (args.models if args.models else MODELS)

    print(f"{'='*60}")
    print(f"MULTI-MODEL EDGE CASE COMPARISON")
    print(f"{'='*60}")
    print(f"Models: {', '.join(models_to_run)}")
    print(f"Edge cases: {len(EDGE_CASE_IDS)}")
    print(f"Prompt fix: APPLIED")
    print()

    # Check model availability first
    available = []
    for m in models_to_run:
        if ensure_model_available(m):
            available.append(m)
        else:
            print(f"  [SKIP] {m} -- not available, skipping")

    if not available:
        print("No models available. Pull models with: ollama pull <model>")
        sys.exit(1)

    print(f"\nRunning {len(available)} models: {', '.join(available)}")

    # Run models sequentially, reusing pipeline setup
    all_results = []
    shared_pipeline = None

    for i, model_name in enumerate(available):
        print(f"\n[{i+1}/{len(available)}] Running {model_name}...")
        pipeline, result = run_single_model(model_name, reuse_pipeline=shared_pipeline)
        if result:
            all_results.append(result)
        if pipeline and shared_pipeline is None:
            shared_pipeline = pipeline  # Reuse for subsequent models

    # Save combined results
    if all_results:
        combo_file = Path("results_HDFS") / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(combo_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nCombined results: {combo_file}")

        # Print comparison
        ranked = print_comparison(all_results)

        # Save ranking
        rank_file = Path("results_HDFS") / f"model_ranking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(rank_file, "w") as f:
            json.dump(ranked, f, indent=2, default=str)
        print(f"Ranking: {rank_file}")


if __name__ == "__main__":
    main()
