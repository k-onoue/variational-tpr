#!/bin/bash

# --- Configuration ---
# 1. Models, datasets, and splits to check
MODEL_NAMES=("XuSparseTPR") # Add other models if needed, e.g., ("SparseGPR" "SparseTPR")
DATASETS=(
    'Protein' 
    'Protein_Outliers' 
)
NUM_SPLITS=10

# 2. Path and file settings
PYTHON_SCRIPT="experiments/sparse_v5.py" # Ensure this filename is correct
OUTPUT_DIR="results/20250920_055600"   # IMPORTANT: Set this to your actual results path
EXPECTED_LINES=1001                      # Expected line count for a complete run (Header + 1000 epochs)


# --- Step 1: Automatically Detect Missing or Incomplete Jobs ---
RETRY_JOBS=() # Initialize an empty array to hold jobs that need rerunning
echo "Scanning for incomplete or missing jobs in ${OUTPUT_DIR}/raw ..."
echo "=================================================="

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for DATASET_NAME in "${DATASETS[@]}"; do
        for SPLIT_INDEX in $(seq 0 $((NUM_SPLITS - 1))); do
            FILE_PATH="${OUTPUT_DIR}/raw/${MODEL_NAME}_${DATASET_NAME}_split${SPLIT_INDEX}.csv"

            if [ -f "${FILE_PATH}" ]; then
                # File exists, check its line count
                LINE_COUNT=$(wc -l < "${FILE_PATH}")
                if [ ${LINE_COUNT} -lt ${EXPECTED_LINES} ]; then
                    echo " -> INCOMPLETE: ${FILE_PATH} (has ${LINE_COUNT} lines, expected ${EXPECTED_LINES})"
                    RETRY_JOBS+=("${MODEL_NAME},${DATASET_NAME},${SPLIT_INDEX}")
                fi
            else
                # File does not exist, mark for retry
                echo " -> MISSING: ${FILE_PATH}"
                RETRY_JOBS+=("${MODEL_NAME},${DATASET_NAME},${SPLIT_INDEX}")
            fi
        done
    done
done

echo "=================================================="
# --- Step 2: Submit Slurm Jobs for the Detected List ---
NUM_RETRY_JOBS=${#RETRY_JOBS[@]}
if [ $NUM_RETRY_JOBS -eq 0 ]; then
    echo "All jobs are complete. Nothing to do. Exiting."
    exit 0
fi

TOTAL_JOBS_TO_RUN=$((NUM_RETRY_JOBS - 1)) # Job array indices are 0-based
echo "Found ${NUM_RETRY_JOBS} jobs to retry. Submitting to Slurm..."

sbatch << EOF
#!/bin/bash -l
#SBATCH --job-name=RetrySparseJobs    # A general name for the retry job
#SBATCH --partition=gpu_short         # Use the GPU short partition
#SBATCH --array=0-${TOTAL_JOBS_TO_RUN}  # Job array for the specific jobs to retry
#SBATCH --time=4:00:00                # Max runtime
#SBATCH --gres=gpu:1                  # Request one GPU
#SBATCH --output=${OUTPUT_DIR}/logs/retry_%A_%a.out  # Log files prefixed with "retry"
#SBATCH --error=${OUTPUT_DIR}/logs/retry_%A_%a.err   #
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=4             # Number of CPUs per task
#SBATCH --mem=16G                     # Memory per node

echo "Slurm Job ID: \${SLURM_JOB_ID}, Array Task ID: \${SLURM_ARRAY_TASK_ID}"
echo "Job started on \$(hostname) at \$(date)"

# --- Determine Model, Dataset, and Split from the Retry List ---
RETRY_JOBS_ARRAY=(${RETRY_JOBS[@]})
# Get the job info string (e.g., "XuSparseTPR,Bike,6") using the task ID as an index
JOB_INFO=\${RETRY_JOBS_ARRAY[\$SLURM_ARRAY_TASK_ID]}

# Parse the string to get Model, Dataset, and Split
MODEL_NAME=\$(echo "\$JOB_INFO" | cut -d',' -f1)
DATASET_NAME=\$(echo "\$JOB_INFO" | cut -d',' -f2)
SPLIT_INDEX=\$(echo "\$JOB_INFO" | cut -d',' -f3)

echo "Retrying: Model=\${MODEL_NAME}, Dataset=\${DATASET_NAME}, Split=\${SPLIT_INDEX}"

# --- Execute the Python Script ---
python ${PYTHON_SCRIPT} \\
    --model "\${MODEL_NAME}" \\
    --dataset "\${DATASET_NAME}" \\
    --split "\${SPLIT_INDEX}" \\
    --output_dir "${OUTPUT_DIR}/raw"

echo "Job finished at \$(date)"

EOF

# Check the exit code of the sbatch command
if [ $? -eq 0 ]; then
    echo " -> Successfully submitted retry job array."
else
    echo " -> ERROR: Failed to submit retry job array."
fi
echo "--------------------------------------------------"