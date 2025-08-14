#!/bin/bash -l
#================================================================
# Slurm BATCH SCRIPT
#================================================================
#SBATCH --job-name=gpu-gpflow      # ジョブの名前
#SBATCH -p gpu_short               # 投入するパーティション(キュー)名 (資料P.7)
#SBATCH --gres=gpu:1               # 使用するGPUの数 (資料P.7)
#SBATCH -c 4                       # 使用するCPUコア数
#SBATCH --time=00:30:00            # 実行時間 (HH:MM:SS)。gpu_shortの最大は4時間 (資料P.7)
#SBATCH -o slurm-%j.out            # 標準出力ファイル
#SBATCH -e slurm-%j.err            # 標準エラーファイル

# --- 環境設定 ---
# モジュールの読み込み (資料P.9-10)


# Pythonスクリプトの実行
# 事前にpipでライブラリをインストールしておく必要があります
python test_slurm.py

echo "Job Finished at $(date)"