"""
Edge Case A/B Test for HDFS Prompt Fix

Runs the 24 verification-FAIL sessions from the 500-anomaly eval
through the pipeline and evaluates them with the auto-evaluator.

Usage:
    # Run with current prompt (after fix is applied):
    python run_edge_case_ab_test.py

    # To get baseline, stash prompt changes first:
    git stash
    python run_edge_case_ab_test.py --tag baseline
    git stash pop
"""

import json
import sys
import argparse
from datetime import datetime
from pathlib import Path

from pipelines.explain_all import ExplainAllPipeline, PipelineConfig
from pipelines.auto_evaluator import AutoEvaluator


# 24 HDFS sessions that had verification FAIL in the 500-anomaly run
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


def run_edge_case_test(tag: str = "prompt_fix"):
    """Run only the edge case sessions and evaluate."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"{'='*60}")
    print(f"EDGE CASE A/B TEST - {tag}")
    print(f"{'='*60}")
    print(f"Sessions: {len(EDGE_CASE_IDS)}")
    print(f"Tag: {tag}")

    # Configure pipeline with session_ids filter
    config = PipelineConfig(
        dataset="HDFS",
        log_file="./logs/HDFS.log",
        model_path="./best_model_HDFS/best_model_HDFS20250804_201746.pth",
        output_dir="./results_HDFS",
        session_ids=EDGE_CASE_IDS,
    )

    pipeline = ExplainAllPipeline(config)
    pipeline.setup()
    pipeline.run()

    # Save results
    out_dir = Path("results_HDFS")
    out_file = out_dir / f"edge_case_{tag}_{timestamp}.jsonl"
    pipeline.save_results(str(out_file))

    # Evaluate
    evaluator = AutoEvaluator()
    report = evaluator.evaluate_pipeline(pipeline)

    # Save eval
    eval_file = out_dir / f"edge_case_{tag}_{timestamp}_eval.json"
    with open(eval_file, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    # Print detailed results
    print(f"\n{'='*60}")
    print(f"EDGE CASE RESULTS - {tag}")
    print(f"{'='*60}")

    evaluator.print_report(report)

    # Per-session breakdown
    print(f"\n--- Per-session scores ---")
    cat_labels = {}
    for sid in EDGE_CASE_IDS[:7]:
        cat_labels[sid] = "A-empty_claims"
    for sid in EDGE_CASE_IDS[7:10]:
        cat_labels[sid] = "B-empty_spans"
    for sid in EDGE_CASE_IDS[10:18]:
        cat_labels[sid] = "C-STRUCTURAL"
    for sid in EDGE_CASE_IDS[18:]:
        cat_labels[sid] = "D-keyword"

    for sc in report.scores:
        cat = cat_labels.get(sc.session_id, "?")
        c_deds = "; ".join(sc.deductions.get("C", []) if isinstance(sc.deductions, dict) else [])
        print(f"  [{cat}] {sc.session_id}: C={sc.correctness:.1f} Co={sc.coherence:.1f} E={sc.evidence_quality:.1f} Y={'Y' if sc.acceptable else 'N'}")
        if c_deds:
            print(f"    C deds: {c_deds}")

    # Category summary
    print(f"\n--- Category summary ---")
    for cat_name in ["A-empty_claims", "B-empty_spans", "C-STRUCTURAL", "D-keyword"]:
        cat_scores = [sc for sc in report.scores if cat_labels.get(sc.session_id) == cat_name]
        if cat_scores:
            import statistics
            avg_c = statistics.mean(sc.correctness for sc in cat_scores)
            avg_co = statistics.mean(sc.coherence for sc in cat_scores)
            avg_e = statistics.mean(sc.evidence_quality for sc in cat_scores)
            n_pass = sum(1 for sc in cat_scores if sc.acceptable)
            verif_fail = sum(1 for sc in cat_scores
                            if any("verification FAIL" in d
                                   for dim_deds in (sc.deductions.values() if isinstance(sc.deductions, dict) else [])
                                   for d in (dim_deds if isinstance(dim_deds, list) else [])))
            print(f"  {cat_name} (n={len(cat_scores)}): C={avg_c:.2f} Co={avg_co:.2f} E={avg_e:.2f} Y={n_pass}/{len(cat_scores)} verif_FAIL={verif_fail}")

    print(f"\nResults: {out_file}")
    print(f"Eval: {eval_file}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="prompt_fix", help="Tag for this run (e.g., baseline, prompt_fix)")
    args = parser.parse_args()
    run_edge_case_test(tag=args.tag)
