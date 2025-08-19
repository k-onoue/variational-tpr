#!/bin/bash

# --- この実験全体で共有するタイムスタンプと保存先を定義 ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_SAVE_DIR="results/gpflow_sparse_${TIMESTAMP}"

# --- 評価するモデルのリスト (GPflowモデルに変更) ---
MODELS_TO_EVALUATE=("GPyTorchSVGP")

# --- データセットとスプリットの総数を定義 ---
# 使用するデータセットのリスト
DATASETS=("Elevators" "Kin8nm" "Kin8nm_Outliers" "Protein")
DATASETS=("Protein")
NUM_DATASETS=${#DATASETS[@]}
NUM_SPLITS=1
TOTAL_TASKS=$((NUM_DATASETS * NUM_SPLITS))
ARRAY_MAX_INDEX=$((TOTAL_TASKS - 1))

# --- 必要なディレクトリを作成 ---
mkdir -p "$BASE_SAVE_DIR"
mkdir -p logs

# --- スクリプトのスナップショットを保存 (再現性の確保) ---
# このランチャースクリプト自体
cp "$0" "$BASE_SAVE_DIR/launcher_snapshot.sh"
# Slurmジョブスクリプト
cp "run_evaluation_gpflow.sh" "$BASE_SAVE_DIR/run_evaluation_snapshot.sh"
# Python評価スクリプト
cp "experiments/sparse_evaluation_gp.py" "$BASE_SAVE_DIR/evaluation_snapshot.py"

echo "Submitting GPflow SVGP model evaluation jobs."
echo "Total tasks per model: $TOTAL_TASKS (from 0 to $ARRAY_MAX_INDEX)"
echo "Results will be saved in: $BASE_SAVE_DIR"
echo "-------------------------------------"

for model in "${MODELS_TO_EVALUATE[@]}"; do
    MODEL_SAVE_DIR="${BASE_SAVE_DIR}/${model}"
    mkdir -p "$MODEL_SAVE_DIR"

    echo "Submitting job array for model: $model"
    
    # Slurmジョブスクリプトをsbatchで投入
    # 引数としてモデル名と保存先ディレクトリを渡す
    sbatch --array=0-$ARRAY_MAX_INDEX run_evaluation_gpflow.sh "$model" "$MODEL_SAVE_DIR"
done

echo "All jobs have been submitted."
echo "Monitor progress with: squeue -u $USER"
echo "After all jobs complete, run a result aggregation script to get the summary."
