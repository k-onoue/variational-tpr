#!/bin/bash
#SBATCH --job-name=svtp_eval        # Job name
#SBATCH --partition=cluster_short   # 使用するキューに合わせて変更
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16           # CPUコア数
#SBATCH --mem=8G                    # メモリ
#SBATCH --time=04:00:00             # エポック数が増えたので時間を延長

# ログファイルのパス
#SBATCH --output=logs/svtp_eval_%A_%a.out
#SBATCH --error=logs/svtp_eval_%A_%a.err

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

# --- データセットとスプリット数の定義 (新しい構成) --- ###
DATASETS=(
    "Boston" "Concrete" "Concrete_Outliers" "Energy" "Yacht"
)
NUM_SPLITS=5

NUM_DATASETS=${#DATASETS[@]}
TASK_ID=$SLURM_ARRAY_TASK_ID

# --- 1次元のタスクIDからデータセットとスプリットのインデックスを計算 ---
DATASET_INDEX=$((TASK_ID / NUM_SPLITS))
SPLIT_ID=$((TASK_ID % NUM_SPLITS))

# 範囲外のタスクIDをチェック
if [ "$DATASET_INDEX" -ge "$NUM_DATASETS" ]; then
    echo "Task ID $TASK_ID results in an invalid dataset index. Exiting."
    exit 0
fi

DATASET_NAME=${DATASETS[$DATASET_INDEX]}

echo "===== Slurm Task Start ====="
echo "Job ID: $SLURM_JOB_ID, Task ID: $TASK_ID"
echo "Model: $MODEL_NAME"
echo "Dataset: $DATASET_NAME (Index: $DATASET_INDEX)"
echo "Split: $SPLIT_ID"
echo "Save Directory: $SAVE_DIR"
echo "=========================="

### --- Pythonスクリプトの実行 --- ###
python experiments/sparse_evaluation.py \
    --model_name    "$MODEL_NAME" \
    --dataset_name  "$DATASET_NAME" \
    --split_id      "$SPLIT_ID" \
    --save_dir      "$SAVE_DIR"

echo "===== Slurm Task End ====="