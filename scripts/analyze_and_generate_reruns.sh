#!/bin/bash
# Complete analysis and rerun generation workflow
# Usage: ./analyze_and_generate_reruns.sh <timestamp>
# Example: ./analyze_and_generate_reruns.sh 20260113_002155

set -e  # Exit on error

if [ $# -ne 1 ]; then
    echo "Usage: $0 <timestamp>"
    echo "Example: $0 20260113_002155"
    exit 1
fi

TIMESTAMP=$1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================================================"
echo "EXPERIMENTAL RUN ANALYSIS AND RERUN GENERATION"
echo "Timestamp: ${TIMESTAMP}"
echo "========================================================================"
echo ""

# Step 1: Check final epochs
echo "Step 1/3: Analyzing experimental results..."
echo "------------------------------------------------------------------------"
bash "${SCRIPT_DIR}/check_final_epochs.sh" "${TIMESTAMP}"
echo ""

# Step 2: Detect missing experiments
echo ""
echo "Step 2/3: Detecting missing experiments..."
echo "------------------------------------------------------------------------"
python "${SCRIPT_DIR}/detect_missing_experiments.py" "${TIMESTAMP}"
echo ""

# Step 3: Generate rerun scripts
echo ""
echo "Step 3/3: Generating rerun scripts for incomplete experiments..."
echo "------------------------------------------------------------------------"
python "${SCRIPT_DIR}/generate_rerun_jobs.py" "${TIMESTAMP}"
echo ""

# Summary
echo ""
echo "========================================================================"
echo "ANALYSIS COMPLETE"
echo "========================================================================"
echo ""
echo "Generated files:"
echo "  - results/${TIMESTAMP}/final_epochs_summary_${TIMESTAMP}.txt"
echo "  - scripts/missing_experiments_${TIMESTAMP}.txt"
echo "  - scripts/rerun_summary_${TIMESTAMP}.txt"
echo "  - scripts/rerun_*_${TIMESTAMP}.sh"
echo ""
echo "Next steps:"
echo "  1. Review the summaries:"
echo "     cat results/${TIMESTAMP}/final_epochs_summary_${TIMESTAMP}.txt"
echo "     cat scripts/missing_experiments_${TIMESTAMP}.txt"
echo "     cat scripts/rerun_summary_${TIMESTAMP}.txt"
echo ""
echo "  2. Submit rerun jobs for incomplete experiments:"
echo "     bash scripts/rerun_all_${TIMESTAMP}.sh"
echo ""
echo "  3. Handle missing experiments separately (if any):"
echo "     - Review scripts/missing_experiments_${TIMESTAMP}.txt"
echo "     - Manually submit jobs or re-run full suite"
echo ""
echo "========================================================================"
