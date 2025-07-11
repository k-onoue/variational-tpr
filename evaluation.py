# evaluation.py (Slurm対応版)

import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from collections import defaultdict
import warnings
import time
import logging
from pathlib import Path
import argparse # argparseを追加

# --- (tprtのインポートとグローバル設定は変更なし) ---
from tprt import TPRTFullBatch, TPRTFullBatch_Tang
torch.set_default_dtype(torch.float64)
warnings.filterwarnings('ignore')

# === 実験パラメータの一元管理 ===
EVALUATION_CONFIG = {
    'TPRT-VEM': {
        'model_class': TPRTFullBatch,
        'init_params': { 'nu_f': 2.0+1e-6, 'nu_e': 2.0+1e-6, 'kernel_lengthscale': 1.0, 'kernel_variance': 1.0, 'likelihood_sigma': 1.0 },
        'fit_params': { 'max_iter_global': 100, 'cavi_max_iter': 20, 'lr': 0.01 }
    },
    'TPRT-LA': {
        'model_class': TPRTFullBatch_Tang,
        'init_params': { 'nu_f': 2.0+1e-6, 'nu_e': 2.0+1e-6, 'kernel_lengthscale': 1.0, 'kernel_variance': 1.0, 'likelihood_sigma': 1.0 },
        'fit_params': { 'max_iter_global': 100, 'mode_finding_iter': 20, 'cg_restarts': 10 } # CG版のパラメータ
    }
}

def load_data(dataset_path: Path, dtype=torch.float64):
    """指定されたパスから学習データとテストデータを読み込む"""
    train_features_path = dataset_path / 'train_features.csv'
    train_target_path = dataset_path / 'train_target.csv'
    test_features_path = dataset_path / 'test_features.csv'
    test_target_path = dataset_path / 'test_target.csv'

    train_features = pd.read_csv(train_features_path, header=None).values
    train_target = pd.read_csv(train_target_path, header=None).values
    test_features = pd.read_csv(test_features_path, header=None).values
    test_target = pd.read_csv(test_target_path, header=None).values

    # Torchテンソルに変換
    X_train = torch.tensor(train_features, dtype=dtype)
    y_train = torch.tensor(train_target, dtype=dtype)
    X_test = torch.tensor(test_features, dtype=dtype)
    y_test = torch.tensor(test_target, dtype=dtype)
    
    return X_train, y_train, X_test, y_test

def run_single_split_evaluation(model_name, dataset_name, split_id, datasets_base_path: Path, save_dir: Path):
    """指定された単一のモデル、データセット、スプリットで評価を行う"""
    logging.info(f"===== Evaluating Model: {model_name} on Dataset: {dataset_name}, Split: {split_id} =====")
    
    config = EVALUATION_CONFIG[model_name]
    model_class = config['model_class']
    init_params = config['init_params']
    fit_params = config['fit_params']

    split_path = datasets_base_path / dataset_name / f'split_{split_id}'
    if not split_path.exists():
        logging.error(f"Split path not found: {split_path}. Exiting.")
        return

    X_train, y_train, X_test, y_test = load_data(split_path)
    
    model = model_class(X_train, y_train, **init_params)
    
    start_time = time.time()
    logging.info(f"Training...")
    model.fit(**fit_params)
    end_time = time.time()
    logging.info(f"Training finished.")

    with torch.no_grad():
        pred_mean, _, _ = model.predict(X_test)
        
    rmse = np.sqrt(mean_squared_error(y_test.numpy(), pred_mean.numpy()))
    elapsed_time = end_time - start_time
    
    result_data = {
        'model': [model_name],
        'dataset': [dataset_name],
        'split_id': [split_id],
        'rmse': [rmse],
        'time_s': [elapsed_time]
    }
    df = pd.DataFrame(result_data)
    
    # 結果を個別のCSVファイルに保存
    result_file = save_dir / f"result_{model_name}_{dataset_name}_split_{split_id}.csv"
    df.to_csv(result_file, index=False)
    logging.info(f"Result saved to {result_file}")
    logging.info(f"  => RMSE: {rmse:.4f} | Time: {elapsed_time:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Run a single split evaluation for TPRT.")
    parser.add_argument('--model_name', type=str, required=True, choices=EVALUATION_CONFIG.keys())
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--split_id', type=int, required=True)
    parser.add_argument('--base_path', type=str, default='datasets/dataset_tang_2017/')
    parser.add_argument('--save_dir', type=str, required=True)
    
    args = parser.parse_args()
    
    # ログ設定
    log_dir = Path(args.save_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{args.model_name}_{args.dataset_name}_split_{args.split_id}.log"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

    datasets_base_path = Path(args.base_path)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    run_single_split_evaluation(
        args.model_name, args.dataset_name, args.split_id, datasets_base_path, save_dir
    )

if __name__ == "__main__":
    main()
