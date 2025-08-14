#!/bin/bash
#SBATCH --job-name=gpflow_eval      # ジョブ名
#SBATCH --partition=gpu_short       # GPUが利用できるパーティションを指定
#SBATCH --gres=gpu:1                # GPUを1つリクエスト
#SBATCH --nodes=1                   # 1ノードを使用
#SBATCH --ntasks=1                  # 1タスクを実行
#SBATCH --cpus-per-task=8           # 1タスクあたり8CPUコアをリクエスト
#SBATCH --mem=16G                   # メモリを16GBリクエスト
#SBATCH --time=04:00:00             # 最大実行時間

# ログファイルのパス
#SBATCH --output=logs/gpflow_eval_%A_%a.out
#SBATCH --error=logs/gpflow_eval_%A_%a.err

### --- 引数のチェック --- ###
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Error: Missing arguments."
  echo "Usage: sbatch --array=<...> run_evaluation_gpflow.sh <MODEL_NAME> <SAVE_DIR>"
  exit 1
fi

MODEL_NAME=$1
SAVE_DIR=$2

### --- 環境の準備 --- ###
# ==============================================================================
# ★★★★★ FIX v5: Conda環境のパスを $HOME を使って正しく指定 ★★★★★
# これが根本原因でした。/work/keisuke-o/ を $HOME に変更します。
# ==============================================================================
CONDA_BASE_PATH="$HOME/anaconda3"
CONDA_ENV_NAME="tprt"
CONDA_ACTIVATE_SCRIPT="${CONDA_BASE_PATH}/envs/${CONDA_ENV_NAME}/bin/activate"

# Condaの初期化と環境の有効化
if [ -f "${CONDA_BASE_PATH}/bin/conda" ]; then
    # Condaの初期化スクリプトを実行
    source "${CONDA_BASE_PATH}/etc/profile.d/conda.sh"
    # 目的の環境を有効化
    conda activate "${CONDA_ENV_NAME}"
    echo "Conda environment '${CONDA_ENV_NAME}' activated successfully."
    echo "CONDA_PREFIX is now: $CONDA_PREFIX"
else
    echo "Error: Conda base path not found at ${CONDA_BASE_PATH}"
    exit 1
fi

# Conda環境内のCUDAツールキットのパスをXLA_FLAGSで明示的に指定
if [ -n "$CONDA_PREFIX" ]; then
  export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CONDA_PREFIX"
  echo "Set XLA_FLAGS to: $XLA_FLAGS"
else
  echo "Error: CONDA_PREFIX is not set after attempting to activate conda environment."
  exit 1
fi
# ==============================================================================

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

### --- データセットとスプリット数の定義 --- ###
DATASETS=(
    "Elevators" "Kin8nm" "Kin8nm_Outliers" "Protein"
)
NUM_SPLITS=5
NUM_DATASETS=${#DATASETS[@]}
TASK_ID=$SLURM_ARRAY_TASK_ID

DATASET_INDEX=$((TASK_ID / NUM_SPLITS))
SPLIT_ID=$((TASK_ID % NUM_SPLITS))

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
# XLA_FLAGSはexport済みなので、ここではコマンドの前に置く必要はありません
python experiments/sparse_evaluation_gp.py \
    --model_name    "$MODEL_NAME" \
    --dataset_name  "$DATASET_NAME" \
    --split_id      "$SPLIT_ID" \
    --base_path     "datasets/dataset_xu_2024/" \
    --save_dir      "$SAVE_DIR"

echo "===== Slurm Task End ====="
