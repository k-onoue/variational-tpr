#!/bin/bash
#SBATCH --job-name=tprt_eval_split # Job name
#SBATCH --partition=cluster_short  # 使用するキューに合わせて変更
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2          # 必要なCPUコア数
#SBATCH --mem=4G                   # 必要なメモリ
#SBATCH --time=00:30:00            # 1スプリットあたりの最大実行時間

# ログファイルのパス
#SBATCH --output=logs/tprt_eval_split_%A_%a.out
#SBATCH --error=logs/tprt_eval_split_%A_%a.err

### --- 引数のチェック --- ###
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Error: Missing arguments."
  echo "Usage: sbatch --array=<...> run_evaluation.sh <MODEL_NAME> <SAVE_DIR>"
  exit 1
fi

MODEL_NAME=$1
SAVE_DIR=$2

### --- 環境の準備 --- ###
# source /path/to/your/venv/bin/activate
# export PYTHONPATH=$PYTHONPATH:/path/to/your/project

### --- データセットとスプリット数の定義 --- ###
DATASETS=(
    "Bike" "Concrete" "Diabetes" "ELE" "MPG" "Machine_CPU" "Neal" "Neal_XOutlier"
)
NUM_SPLITS=10

NUM_DATASETS=${#DATASETS[@]}
TOTAL_TASKS=$((NUM_DATASETS * NUM_SPLITS))
TASK_ID=$SLURM_ARRAY_TASK_ID

if [ "$TASK_ID" -ge "$TOTAL_TASKS" ]; then
    echo "Task ID $TASK_ID is out of bounds. Total tasks: $TOTAL_TASKS. Exiting."
    exit 0
fi

# --- 1次元のタスクIDからデータセットとスプリットのインデックスを計算 ---
# データセットのインデックス = floor(タスクID / スプリット数)
DATASET_INDEX=$((TASK_ID / NUM_SPLITS))
# スプリットのインデックス = タスクID % スプリット数
SPLIT_ID=$((TASK_ID % NUM_SPLITS))

DATASET_NAME=${DATASETS[$DATASET_INDEX]}

echo "===== Slurm Task Start ====="
echo "Job ID: $SLURM_JOB_ID, Task ID: $TASK_ID"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_NAME (Index: $DATASET_INDEX)"
echo "Split: $SPLIT_ID"
echo "Save Directory: $SAVE_DIR"
echo "=========================="

### --- Pythonスクリプトの実行 --- ###
python non_sparse_evaluation.py \
    --model_name    "$MODEL_NAME" \
    --dataset_name  "$DATASET_NAME" \
    --split_id      "$SPLIT_ID" \
    --save_dir      "$SAVE_DIR"

echo "===== Slurm Task End ====="