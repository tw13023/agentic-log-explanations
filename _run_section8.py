"""Standalone runner for NB11 Section 8: Semantic Faithfulness (LLM-as-Judge)."""
import json
import os
import re as _re
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import BGLDataLoader, HDFSDataLoader
from src.llm_client import get_client

PROJECT_ROOT = Path(__file__).parent
RESULTS_DIR = PROJECT_ROOT / 'results'
RESULTS_HDFS_DIR = PROJECT_ROOT / 'results_HDFS'

BGL_JSONL  = RESULTS_DIR / 'explanations_BGL_20260313_002116.jsonl'
HDFS_JSONL = RESULTS_HDFS_DIR / 'explanations_HDFS_20260311_230513.jsonl'


# ── helpers ────────────────────────────────────────────────────────────────────

def load_explanations(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_span_range(span_str: str):
    m = _re.match(r'(E\w+)-L(\d+)(?:\s+to\s+E\w+-L(\d+))?', span_str.strip())
    if not m:
        return None
    eid = m.group(1)
    start = int(m.group(2))
    end = int(m.group(3)) if m.group(3) else start
    return eid, start, end


def get_evidence_text_for_claim(claim: dict, eid_mapping: dict, sid_to_lines: dict):
    cited_ids = claim.get('evidence_ids', [])
    spans = claim.get('evidence_spans', [])
    for eid in cited_ids:
        if eid not in eid_mapping:
            return None
    if not spans:
        return None
    lines_collected = []
    for span_str in spans:
        parsed = parse_span_range(span_str)
        if parsed is None:
            continue
        eid, start, end = parsed
        if eid not in eid_mapping:
            return None
        raw_sid = eid_mapping[eid]
        session_id = raw_sid[2:] if raw_sid.startswith('E_') else raw_sid
        log_lines = sid_to_lines.get(session_id, [])
        for i in range(start, end + 1):
            if 1 <= i <= len(log_lines):
                lines_collected.append(log_lines[i - 1])
    return '\n'.join(lines_collected) if lines_collected else None


JUDGE_MODEL = 'gpt-4.1'
JUDGE_SYSTEM = "You are an expert log anomaly analyst evaluating explanation accuracy."


def judge_claim(claim_text: str, evidence_text: str, judge_llm) -> str:
    user_msg = (
        f"EVIDENCE (cited log lines):\n{evidence_text}\n\n"
        f"CLAIM:\n{claim_text}\n\n"
        "Is this claim factually supported by the cited evidence log lines?\n"
        "- Accept if error type, pattern, and key details are correct.\n"
        "- Reject if the claim states facts that contradict or are absent from the evidence.\n"
        "Answer with exactly YES or NO."
    )
    response = judge_llm.chat(
        messages=[
            {'role': 'system', 'content': JUDGE_SYSTEM},
            {'role': 'user',   'content': user_msg},
        ]
    )
    answer = response.content.strip().upper()
    return 'YES' if answer.startswith('YES') else 'NO'


def compute_semantic_faithfulness(records, sid_to_lines, cache, cache_prefix, judge_llm):
    per_session = []
    calls_made = 0
    calls_skipped = 0
    structural_fails = 0

    for rec in tqdm(records, desc=f'Judge [{cache_prefix}]'):
        session_id = rec['session_id']
        expl = rec.get('explanation', {})
        claims = expl.get('claims', [])
        eid_mapping = rec.get('evidence_id_mapping', {})

        if not claims:
            continue

        yes_count = 0
        total = 0

        for claim_idx, claim in enumerate(claims):
            total += 1
            cache_key = f"{cache_prefix}|{session_id}|{claim_idx}"

            if cache_key in cache:
                if cache[cache_key] == 'YES':
                    yes_count += 1
                calls_skipped += 1
                continue

            evidence_text = get_evidence_text_for_claim(claim, eid_mapping, sid_to_lines)

            if evidence_text is None:
                cache[cache_key] = 'NO'
                structural_fails += 1
                continue

            result = judge_claim(claim.get('claim', ''), evidence_text, judge_llm)
            cache[cache_key] = result

            if result == 'YES':
                yes_count += 1
            calls_made += 1
            time.sleep(0.05)

        faithfulness = yes_count / total if total > 0 else 0.0
        per_session.append({
            'session_id': session_id,
            'semantic_faithfulness': faithfulness,
            'supported_claims': yes_count,
            'total_claims': total,
            'signature': rec.get('normalized_signature', 'unknown'),
        })

    print(f"  API calls: {calls_made:,} new, {calls_skipped:,} cached, "
          f"{structural_fails:,} structural failures")
    return per_session


def compute_structural_faithfulness(records):
    total_supported = 0
    total_claims = 0
    for rec in records:
        expl = rec.get('explanation', {})
        claims = expl.get('claims', [])
        eid_mapping = rec.get('evidence_id_mapping', {})
        valid_ids = set(eid_mapping.keys())
        for claim in claims:
            total_claims += 1
            cited = claim.get('evidence_ids', [])
            spans = claim.get('evidence_spans', [])
            if any(eid in valid_ids for eid in cited) and spans:
                total_supported += 1
    return total_supported / total_claims if total_claims > 0 else 0.0


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("[1/7] Loading BGL data loader (~475k sessions, takes ~2 min)...")
    bgl_loader = BGLDataLoader(
        log_file=str(PROJECT_ROOT / 'logs' / 'BGL.log'),
        windows_size=10, step_size=10, train_ratio=0.7,
    )
    bgl_loader.load()
    bgl_sid_to_lines = {s.session_id: s.lines for s in bgl_loader.get_sessions()}
    print(f"  BGL sessions indexed: {len(bgl_sid_to_lines):,}")

    print("[2/7] Loading HDFS data loader...")
    hdfs_loader = HDFSDataLoader(
        log_file=str(PROJECT_ROOT / 'logs' / 'HDFS.log'),
        label_file=str(PROJECT_ROOT / 'logs' / 'anomaly_label_HDFS.csv'),
        train_ratio=0.7,
    )
    hdfs_loader.load()
    hdfs_sid_to_lines = {s.session_id: s.lines for s in hdfs_loader.get_sessions()}
    print(f"  HDFS sessions indexed: {len(hdfs_sid_to_lines):,}")

    print("[3/7] Loading explanation files...")
    bgl_records  = load_explanations(BGL_JSONL)
    hdfs_records = load_explanations(HDFS_JSONL)
    print(f"  BGL full records: {len(bgl_records):,}")
    print(f"  HDFS full records: {len(hdfs_records):,}")

    print("[4/7] Loading ablation no-RAG files...")
    bgl_no_rag_ablation  = load_explanations(RESULTS_DIR / 'ablation_no_rag_BGL.jsonl')
    hdfs_no_rag_ablation = load_explanations(RESULTS_HDFS_DIR / 'ablation_no_rag_HDFS.jsonl')
    ablation_ids      = {r['session_id'] for r in bgl_no_rag_ablation}
    hdfs_ablation_ids = {r['session_id'] for r in hdfs_no_rag_ablation}
    bgl_rag_on_ablation  = [r for r in bgl_records  if r['session_id'] in ablation_ids]
    hdfs_rag_on_ablation = [r for r in hdfs_records if r['session_id'] in hdfs_ablation_ids]
    print(f"  BGL  ablation: {len(bgl_no_rag_ablation)} no-RAG, {len(bgl_rag_on_ablation)} RAG-on")
    print(f"  HDFS ablation: {len(hdfs_no_rag_ablation)} no-RAG, {len(hdfs_rag_on_ablation)} RAG-on")

    judge_llm = get_client(provider='openai', model=JUDGE_MODEL, temperature=0.0)

    # ── BGL ──────────────────────────────────────────────────────────────────
    BGL_CACHE_PATH = RESULTS_DIR / 'semantic_faith_judge_BGL.json'
    if BGL_CACHE_PATH.exists():
        with open(BGL_CACHE_PATH) as f:
            bgl_cache = json.load(f)
        print(f"[5/7] BGL cache loaded: {len(bgl_cache):,} entries")
    else:
        bgl_cache = {}
        print("[5/7] BGL cache: starting fresh")

    print("  --- BGL RAG-on ---")
    bgl_rag_on_sem = compute_semantic_faithfulness(
        bgl_rag_on_ablation, bgl_sid_to_lines, bgl_cache, 'BGL_rag_on', judge_llm)
    with open(BGL_CACHE_PATH, 'w') as f:
        json.dump(bgl_cache, f)

    print("  --- BGL No-RAG ---")
    bgl_no_rag_sem = compute_semantic_faithfulness(
        bgl_no_rag_ablation, bgl_sid_to_lines, bgl_cache, 'BGL_no_rag', judge_llm)
    with open(BGL_CACHE_PATH, 'w') as f:
        json.dump(bgl_cache, f)
    print(f"  BGL cache saved: {len(bgl_cache):,} entries")

    # ── HDFS ─────────────────────────────────────────────────────────────────
    HDFS_CACHE_PATH = RESULTS_HDFS_DIR / 'semantic_faith_judge_HDFS.json'
    if HDFS_CACHE_PATH.exists():
        with open(HDFS_CACHE_PATH) as f:
            hdfs_cache = json.load(f)
        print(f"[6/7] HDFS cache loaded: {len(hdfs_cache):,} entries")
    else:
        hdfs_cache = {}
        print("[6/7] HDFS cache: starting fresh")

    print("  --- HDFS RAG-on ---")
    hdfs_rag_on_sem = compute_semantic_faithfulness(
        hdfs_rag_on_ablation, hdfs_sid_to_lines, hdfs_cache, 'HDFS_rag_on', judge_llm)
    with open(HDFS_CACHE_PATH, 'w') as f:
        json.dump(hdfs_cache, f)

    print("  --- HDFS No-RAG ---")
    hdfs_no_rag_sem = compute_semantic_faithfulness(
        hdfs_no_rag_ablation, hdfs_sid_to_lines, hdfs_cache, 'HDFS_no_rag', judge_llm)
    with open(HDFS_CACHE_PATH, 'w') as f:
        json.dump(hdfs_cache, f)
    print(f"  HDFS cache saved: {len(hdfs_cache):,} entries")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("[7/7] Computing summary...")
    bgl_ron_df  = pd.DataFrame(bgl_rag_on_sem)
    bgl_nrag_df = pd.DataFrame(bgl_no_rag_sem)
    hdfs_ron_df  = pd.DataFrame(hdfs_rag_on_sem)
    hdfs_nrag_df = pd.DataFrame(hdfs_no_rag_sem)

    bgl_ron_mean  = bgl_ron_df['semantic_faithfulness'].mean()
    bgl_ron_std   = bgl_ron_df['semantic_faithfulness'].std()
    bgl_nrag_mean = bgl_nrag_df['semantic_faithfulness'].mean()
    bgl_nrag_std  = bgl_nrag_df['semantic_faithfulness'].std()
    bgl_delta     = bgl_ron_mean - bgl_nrag_mean

    hdfs_ron_mean  = hdfs_ron_df['semantic_faithfulness'].mean()
    hdfs_ron_std   = hdfs_ron_df['semantic_faithfulness'].std()
    hdfs_nrag_mean = hdfs_nrag_df['semantic_faithfulness'].mean()
    hdfs_nrag_std  = hdfs_nrag_df['semantic_faithfulness'].std()
    hdfs_delta     = hdfs_ron_mean - hdfs_nrag_mean

    bgl_struct_ron  = compute_structural_faithfulness(bgl_rag_on_ablation)
    bgl_struct_nrag = compute_structural_faithfulness(bgl_no_rag_ablation)
    hdfs_struct_ron  = compute_structural_faithfulness(hdfs_rag_on_ablation)
    hdfs_struct_nrag = compute_structural_faithfulness(hdfs_no_rag_ablation)

    print()
    print("=" * 72)
    print("Table: Semantic Faithfulness (LLM-as-Judge) vs Structural Faithfulness")
    print("=" * 72)
    print(f"{'Metric':<38} {'BGL':>10} {'HDFS':>10}")
    print("-" * 60)
    print(f"{'Structural Faith. (RAG-on)':<38} {bgl_struct_ron:>10.4f} {hdfs_struct_ron:>10.4f}")
    print(f"{'Structural Faith. (No-RAG)':<38} {bgl_struct_nrag:>10.4f} {hdfs_struct_nrag:>10.4f}")
    print("-" * 60)
    print(f"{'Semantic Faith. (RAG-on)':<38} {bgl_ron_mean:>10.4f} {hdfs_ron_mean:>10.4f}")
    print(f"{'Semantic Faith. (No-RAG)':<38} {bgl_nrag_mean:>10.4f} {hdfs_nrag_mean:>10.4f}")
    print(f"{'Semantic Delta (RAG-on minus No-RAG)':<38} {bgl_delta:>+10.4f} {hdfs_delta:>+10.4f}")
    print("=" * 72)

    results = {
        'judge_model': JUDGE_MODEL,
        'BGL': {
            'rag_on':  {'mean': float(bgl_ron_mean),  'std': float(bgl_ron_std),  'n': len(bgl_rag_on_sem)},
            'no_rag':  {'mean': float(bgl_nrag_mean), 'std': float(bgl_nrag_std), 'n': len(bgl_no_rag_sem)},
            'delta':   float(bgl_delta),
            'structural': {'rag_on': float(bgl_struct_ron), 'no_rag': float(bgl_struct_nrag)},
        },
        'HDFS': {
            'rag_on':  {'mean': float(hdfs_ron_mean),  'std': float(hdfs_ron_std),  'n': len(hdfs_rag_on_sem)},
            'no_rag':  {'mean': float(hdfs_nrag_mean), 'std': float(hdfs_nrag_std), 'n': len(hdfs_no_rag_sem)},
            'delta':   float(hdfs_delta),
            'structural': {'rag_on': float(hdfs_struct_ron), 'no_rag': float(hdfs_struct_nrag)},
        },
    }
    out_path = RESULTS_DIR / 'semantic_faithfulness_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Saved results to {out_path}")


if __name__ == '__main__':
    main()
