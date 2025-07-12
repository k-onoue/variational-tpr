#!/bin/bash

# --- この実験全体で共有するタイムスタンプと保存先を定義 ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_SAVE_DIR="results/sparse_${TIMESTAMP}"

# --- 評価するモデルのリスト ---
# MODELS_TO_EVALUATE=("TPRT-SCAVI" "SVTP-UB" "SVTP-MC")
MODELS_TO_EVALUATE=("TPRT-SCAVI")

# --- データセットとスプリットの総数を定義 ---
NUM_DATASETS=9
NUM_SPLITS=5
TOTAL_TASKS=$((NUM_DATASETS * NUM_SPLITS))
ARRAY_MAX_INDEX=$((TOTAL_TASKS - 1))

# --- 必要なディレクトリを作成 ---
mkdir -p "$BASE_SAVE_DIR"
mkdir -p logs

# --- スクリプトのスナップショットを保存 ---
cp "$0" "$BASE_SAVE_DIR/launcher_snapshot.sh"
cp sparse_run_evaluation.sh "$BASE_SAVE_DIR/run_evaluation_snapshot.sh"
cp sparse_evaluation.py "$BASE_SAVE_DIR/evaluation_snapshot.py"

echo "Submitting sparse model evaluation jobs based on Xu et al. (2023)."
echo "Total tasks per model: $TOTAL_TASKS (from 0 to $ARRAY_MAX_INDEX)"
echo "Results will be saved in: $BASE_SAVE_DIR"
echo "-------------------------------------"

for model in "${MODELS_TO_EVALUATE[@]}"; do
    MODEL_SAVE_DIR="${BASE_SAVE_DIR}/${model}"
    mkdir -p "$MODEL_SAVE_DIR"

    echo "Submitting job array for model: $model"
    
    sbatch --array=0-$ARRAY_MAX_INDEX sparse_run_evaluation.sh "$model" "$MODEL_SAVE_DIR"
done

echo "All jobs have been submitted."
echo "Monitor progress with: squeue -u $USER"
echo "After all jobs complete, run 'python aggregate_results.py ${BASE_SAVE_DIR}' to get the summary."