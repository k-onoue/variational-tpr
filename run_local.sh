#!/bin/bash
set -e # エラーが発生したらスクリプトを停止する

# --- ここで実験パラメータを指定 ---
MODEL_TO_EVALUATE="TPRT-SCAVI"
DATASET_TO_EVALUATE="Protein"
SPLIT_ID_TO_EVALUATE=1
# ---------------------------------

# 1. タイムスタンプ付きの保存先ディレクトリを作成
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MODEL_SAVE_DIR="results/sparse_${TIMESTAMP}/${MODEL_TO_EVALUATE}"
mkdir -p "$MODEL_SAVE_DIR"

echo "========================================"
echo "Starting Local Evaluation"
echo "Model:    ${MODEL_TO_EVALUATE}"
echo "Dataset:  ${DATASET_TO_EVALUATE}"
echo "Split ID: ${SPLIT_ID_TO_EVALUATE}"
echo "Results will be saved in: ${MODEL_SAVE_DIR}"
echo "========================================"

# 2. Pythonスクリプトを実行
python experiments/sparse_evaluation_gpu.py \
    --model_name    "$MODEL_TO_EVALUATE" \
    --dataset_name  "$DATASET_TO_EVALUATE" \
    --split_id      "$SPLIT_ID_TO_EVALUATE" \
    --save_dir      "$MODEL_SAVE_DIR"

echo "✅ Evaluation finished successfully."