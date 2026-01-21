#!/bin/bash

# Rerun script for incomplete XuSparseTPR experiments
# Analyzed timestamp: 20260113_002155
# Original timestamp: 20260113_002155
# Generated on: 2026-01-21 19:48:10
# Consolidate mode: Yes (overwrites in original dir)

# --- Configuration ---
MODEL_NAME="XuSparseTPR"
PYTHON_SCRIPT="experiments/sparse_v6.py"
ORIGINAL_TIMESTAMP="20260113_002155"

# Create output directories
OUTPUT_DIR="results/20260113_002155"
mkdir -p "${OUTPUT_DIR}/snapshots"
mkdir -p "${OUTPUT_DIR}/raw"
mkdir -p "${OUTPUT_DIR}/logs"

# Save lineage information
echo "20260113_002155" > "${OUTPUT_DIR}/ORIGINAL_TIMESTAMP"
echo "20260113_002155" >> "${OUTPUT_DIR}/RERUN_FROM"
date >> "${OUTPUT_DIR}/RERUN_FROM"

# Save snapshots
cp "${PYTHON_SCRIPT}" "${OUTPUT_DIR}/snapshots/"
cp "$0" "${OUTPUT_DIR}/snapshots/"

echo "Rerunning incomplete XuSparseTPR experiments..."
echo "Original experiment: 20260113_002155"
echo "Results will be saved in ${OUTPUT_DIR}"
echo "=================================================="

# Submit individual jobs for each incomplete run

# Dataset: Taxi

echo "Submitting job 1: XuSparseTPR, Taxi, split 2"
sbatch << 'EOF'
#!/bin/bash -l
#SBATCH --job-name=XuSparseTPR_Taxi_s2
#SBATCH --partition=gpu_short
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=${OUTPUT_DIR}/logs/%x_%j.out
#SBATCH --error=${OUTPUT_DIR}/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

echo "Job started on $(hostname) at $(date)"
echo "Running: Model=XuSparseTPR, Dataset=Taxi, Split=2"

python experiments/sparse_v6.py \
    --model "XuSparseTPR" \
    --dataset "Taxi" \
    --split 2 \
    --output_dir "results/20260113_002155/raw"

echo "Job finished at $(date)"
EOF


echo "Submitted 1 jobs for XuSparseTPR"
