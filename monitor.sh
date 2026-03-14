#!/bin/bash
# Quick status check for HDFS full run
RESULTS_DIR="results_HDFS"
LATEST=$(ls -t "$RESULTS_DIR"/explanations_HDFS_*.jsonl 2>/dev/null | head -1)
if [ -z "$LATEST" ]; then
    echo "[WARN] No results file found"
    exit 1
fi
TOTAL=2527
DONE=$(wc -l < "$LATEST")
ERRORS=$(grep -c '"error"' "$LATEST" 2>/dev/null || echo 0)
VFAIL=$(grep -c '"verification_passed": false' "$LATEST" 2>/dev/null || echo 0)
PCT=$(echo "scale=1; $DONE * 100 / $TOTAL" | bc)
echo "File:    $LATEST"
echo "Done:    $DONE / $TOTAL  ($PCT%)"
echo "Errors:  $ERRORS"
echo "V-fail:  $VFAIL"
