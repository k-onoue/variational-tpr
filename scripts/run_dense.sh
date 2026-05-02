#!/bin/bash

# --- Configuration ---
# List of models to run
# MODEL_NAMES=("GPR" "TPR" "XuTPR" "TangTPR")
MODEL_NAMES=("TPR")

# Path to the Python script to be executed
PYTHON_SCRIPT="experiments/dense.py"

# List of datasets (should match the config in the Python script)
DATASETS=(
    'Boston' 'Diabetes' 'ELE' 'MPG'
    'Machine_CPU' 'Neal' 'Neal_XOutlier' 'Yacht'
    'Boston_Outliers' 'Diabetes_Outliers' 'ELE_Outliers' 'MPG_Outliers'
    'Machine_CPU_Outliers' 'Neal_Outliers' 'Neal_YOutlier' 'Yacht_Outliers'
)
NUM_SPLITS=10
NUM_DATASETS=${#DATASETS[@]}
TOTAL_JOBS=$((NUM_DATASETS * NUM_SPLITS - 1)) # Job array indices are 0-based, so subtract 1

# --- Prepare Execution Directory ---
# Generate a single timestamp for the entire run
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
OUTPUT_DIR="results/${TIMESTAMP}"

# Create the necessary directories
mkdir -p "${OUTPUT_DIR}/snapshots"
mkdir -p "${OUTPUT_DIR}/raw"
mkdir -p "${OUTPUT_DIR}/logs"

# Save snapshots of the scripts
cp "${PYTHON_SCRIPT}" "${OUTPUT_DIR}/snapshots/"
cp "$0" "${OUTPUT_DIR}/snapshots/" # Copy this script itself

echo "Starting experiments. Results will be saved in ${OUTPUT_DIR}"
echo "=================================================="

# --- Submit Slurm Jobs for Each Model ---
for MODEL_NAME in "${MODEL_NAMES[@]}"
do
    echo "Submitting Slurm job for model: ${MODEL_NAME}..."

    # --- Submit Slurm job using a here document ---
    # The structure here is CRITICAL:
    # 1. #!/bin/bash -l MUST be the first line.
    # 2. #SBATCH directives MUST come immediately after, with no blank lines in between.
    sbatch << EOF
#!/bin/bash -l
#SBATCH --job-name=${MODEL_NAME}_dense    # Job name (unique for each model)
#SBATCH --partition=cluster_short       # Partition (queue) name
#SBATCH --array=0-${TOTAL_JOBS}         # Job array indices
#SBATCH --time=4:00:00                  # Set maximum runtime to 4 hours
#SBATCH --output=${OUTPUT_DIR}/logs/%x_%A_%a.out  # Standard output log file
#SBATCH --error=${OUTPUT_DIR}/logs/%x_%A_%a.err   # Standard error log file
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --cpus-per-task=4               # Number of CPUs per task
#SBATCH --mem=8G                       # Memory per node

echo "Slurm Job ID: \${SLURM_JOB_ID}, Array Task ID: \${SLURM_ARRAY_TASK_ID}"
echo "Job started on \$(hostname) at \$(date)"

# --- Determine Dataset and Split from Task ID ---
DATASETS_ARRAY=(${DATASETS[@]})
NUM_SPLITS_PER_DATASET=${NUM_SPLITS}

DATASET_INDEX=\$((SLURM_ARRAY_TASK_ID / NUM_SPLITS_PER_DATASET))
SPLIT_INDEX=\$((SLURM_ARRAY_TASK_ID % NUM_SPLITS_PER_DATASET))
DATASET_NAME=\${DATASETS_ARRAY[\$DATASET_INDEX]}

echo "Running: Model=${MODEL_NAME}, Dataset=\${DATASET_NAME}, Split=\${SPLIT_INDEX}"

# --- Execute the Python Script ---
# Assuming this script is run from the project root
python ${PYTHON_SCRIPT} \\
    --model "${MODEL_NAME}" \\
    --dataset "\${DATASET_NAME}" \\
    --split "\${SPLIT_INDEX}" \\
    --output_dir "${OUTPUT_DIR}/raw"

echo "Job finished at \$(date)"

EOF

    # Check the exit code of the sbatch command
    if [ $? -eq 0 ]; then
        echo " -> Successfully submitted job for ${MODEL_NAME}."
    else
        echo " -> ERROR: Failed to submit job for ${MODEL_NAME}."
    fi
    echo "--------------------------------------------------"
done

echo "All model jobs have been submitted."