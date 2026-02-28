"""
500 Anomaly Sample Evaluation - HDFS & BGL
Run both datasets with enough sessions to capture ~500 anomalies each,
then auto-evaluate and compare with previous baselines.

HDFS: 2.93% anomaly rate -> need ~17000 sessions
BGL:  8.22% anomaly rate -> need ~6100 sessions
"""

import time
import json
from datetime import datetime
from pipelines.explain_all import run_explain_all_pipeline
from pipelines.auto_evaluator import AutoEvaluator

BASELINES = {
    "v1 (pre-fix)": {
        "BGL":  {"C": 4.24, "Co": 4.60, "E": 4.74, "Y%": 96.0, "n": 20},
        "HDFS": {"C": 3.06, "Co": 3.04, "E": 3.00, "Y%": 64.0, "n": 11},
    },
    "v2 (200 sessions)": {
        "BGL":  {"C": 4.97, "Co": 5.00, "E": 5.00, "Y%": 100.0, "n": 20},
        "HDFS": {"C": 4.14, "Co": 4.91, "E": 4.82, "Y%": 100.0, "n": 11},
    },
}


def run_dataset(dataset: str, max_sessions: int):
    """Run pipeline + auto-eval for one dataset."""
    print("\n" + "=" * 70)
    print(f"  {dataset} - 500 ANOMALY EVALUATION (max_sessions={max_sessions})")
    print("=" * 70)

    t0 = time.time()
    pipeline = run_explain_all_pipeline(dataset=dataset, max_sessions=max_sessions)
    elapsed_pipeline = time.time() - t0

    n_anomalies = sum(1 for r in pipeline.results if True)  # all results are anomalies
    print(f"\n[PIPELINE] {dataset} done in {elapsed_pipeline/60:.1f} min, {n_anomalies} anomalies explained")

    # Auto-evaluate
    evaluator = AutoEvaluator()
    report = evaluator.evaluate_pipeline(pipeline)
    evaluator.print_report(report, show_all=False)

    # Save evaluation report
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = "results" if dataset == "BGL" else "results_HDFS"
    eval_path = f"{out_dir}/eval_{dataset}_500anom_{ts}.json"
    evaluator.save_report(report, eval_path)
    print(f"[SAVED] {eval_path}")

    return {
        "dataset": dataset,
        "n_anomalies": n_anomalies,
        "elapsed_min": round(elapsed_pipeline / 60, 1),
        "C": round(report.avg_c, 2),
        "Co": round(report.avg_co, 2),
        "E": round(report.avg_e, 2),
        "Y%": round(report.pct_y, 1),
    }


def print_comparison(results):
    """Print comparison table with baselines."""
    print("\n" + "=" * 70)
    print("  COMPARISON TABLE")
    print("=" * 70)

    header = f"{'Version':<25} {'Dataset':<6} {'n':>5} {'C':>6} {'Co':>6} {'E':>6} {'Y%':>6}"
    print(header)
    print("-" * len(header))

    for version, datasets in BASELINES.items():
        for ds, metrics in datasets.items():
            print(f"{version:<25} {ds:<6} {metrics['n']:>5} {metrics['C']:>6.2f} {metrics['Co']:>6.2f} {metrics['E']:>6.2f} {metrics['Y%']:>6.1f}")

    print("-" * len(header))
    for r in results:
        label = f"v3 (500 anomalies)"
        print(f"{label:<25} {r['dataset']:<6} {r['n_anomalies']:>5} {r['C']:>6.2f} {r['Co']:>6.2f} {r['E']:>6.2f} {r['Y%']:>6.1f}")

    print()


if __name__ == "__main__":
    total_start = time.time()

    results = []

    # HDFS first (slower due to more sessions needed)
    r_hdfs = run_dataset("HDFS", max_sessions=17000)
    results.append(r_hdfs)

    # BGL
    r_bgl = run_dataset("BGL", max_sessions=6100)
    results.append(r_bgl)

    total_elapsed = time.time() - total_start

    print_comparison(results)

    print(f"Total elapsed: {total_elapsed/60:.1f} min ({total_elapsed/3600:.1f} hours)")

    # Save combined results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined = {
        "timestamp": ts,
        "total_elapsed_min": round(total_elapsed / 60, 1),
        "results": results,
        "baselines": BASELINES,
    }
    with open(f"results/eval_500anom_combined_{ts}.json", "w") as f:
        json.dump(combined, f, indent=2)
    print(f"[SAVED] results/eval_500anom_combined_{ts}.json")
