#!/bin/bash

# --- この実験全体で共有するタイムスタンプと保存先を定義 ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_SAVE_DIR="results/tprt_full_parallel_${TIMESTAMP}"

# --- 評価するモデルのリスト ---
MODELS_TO_EVALUATE=("TPRT-VEM" "TPRT-LA")

# --- データセットとスプリットの総数を定義 ---
# run_evaluation.sh内のリストと一致させる
NUM_DATASETS=8
NUM_SPLITS=10
TOTAL_TASKS=$((NUM_DATASETS * NUM_SPLITS))
ARRAY_MAX_INDEX=$((TOTAL_TASKS - 1)) # --arrayは0から始まる

# --- 必要なディレクトリを作成 ---
mkdir -p "$BASE_SAVE_DIR"
mkdir -p slurm_logs

# --- スクリプトのスナップショットを保存 ---
cp "$0" "$BASE_SAVE_DIR/launcher_snapshot.sh"
cp run_evaluation.sh "$BASE_SAVE_DIR/run_evaluation_snapshot.sh"
cp evaluation.py "$BASE_SAVE_DIR/evaluation_snapshot.py"

echo "Submitting evaluation jobs."
echo "Total tasks per model: $TOTAL_TASKS"
echo "Results will be saved in: $BASE_SAVE_DIR"
echo "-------------------------------------"

for model in "${MODELS_TO_EVALUATE[@]}"; do
    MODEL_SAVE_DIR="${BASE_SAVE_DIR}/${model}"
    mkdir -p "$MODEL_SAVE_DIR"

    echo "Submitting job array for model: $model"
    
    # sbatchでジョブ配列を投入 (例: 8データセット x 10スプリット = 80タスク)
    sbatch --array=0-$ARRAY_MAX_INDEX run_evaluation.sh "$model" "$MODEL_SAVE_DIR"
done

echo "All jobs have been submitted."
echo "Monitor progress with: squeue -u $USER"
echo "After all jobs complete, run 'python aggregate_results.py ${BASE_SAVE_DIR}' to get the summary."