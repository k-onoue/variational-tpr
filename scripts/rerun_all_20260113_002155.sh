#!/bin/bash

# Master script to rerun all incomplete experiments from 20260113_002155
# Generated on: 2026-01-21 19:48:10

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=================================================="
echo "Rerunning all incomplete experiments from 20260113_002155"
echo "=================================================="

echo "Launching XuSparseTPR jobs..."
bash "$SCRIPT_DIR/rerun_xusparsetpr_20260113_002155.sh"
sleep 1  # Small delay between model submissions

echo "=================================================="
echo "All 1 jobs have been submitted"
echo "=================================================="