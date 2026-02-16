"""Load the actual log lines for the two verification-failure sessions."""
import json
import sys
sys.path.insert(0, '.')
from src.data_loader import BGLDataLoader

loader = BGLDataLoader('configs/config.yaml')
loader.load()

targets = {"BGL_04647320", "BGL_04647460"}

for session in loader.test_sessions:
    if session.session_id in targets:
        print(f"{'='*80}")
        print(f"SESSION: {session.session_id}  (label={session.label})")
        print(f"Lines: {len(session.lines)}")
        print(f"{'='*80}")
        for i, line in enumerate(session.lines):
            print(f"  L{i+1}: {line.strip()[:120]}")
        print()
