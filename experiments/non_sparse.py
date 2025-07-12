# evaluation.py (Fullモデル用、エラーハンドリング機能付き)

import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
import warnings
import time
import logging
from pathlib import Path
import argparse
import traceback # エラー詳細表示のためにインポート

# --- モデルのインポート ---
from tprt import TPRTFullBatch, TPRTFullBatch_Tang

# --- グローバル設定 ---
torch.set_default_dtype(torch.float64)
warnings.filterwarnings('ignore')

# === 実験パラメータの一元管理 ===
EVALUATION_CONFIG = {
    'TPRT-VEM': {
        'model_class': TPRTFullBatch,
        'init_params': { 'nu_f': 2.0+1e-6, 'nu_e': 2.0+1e-6, 'kernel_lengthscale': 1.0, 'kernel_variance': 1.0, 'likelihood_sigma': 1.0 },
        'fit_params': { 'max_iter_global': 200, 'cavi_max_iter': 20, 'lr': 0.1 }
    },
    'TPRT-LA': {
        'model_class': TPRTFullBatch_Tang,
        'init_params': { 'nu_f': 2.0+1e-6, 'nu_e': 2.0+1e-6, 'kernel_lengthscale': 1.0, 'kernel_variance': 1.0, 'likelihood_sigma': 1.0 },
        # fit_paramsはtprt_tang.pyの実装に合わせる
        # (CG版を想定し、lrは不要、cg_restartsを追加)
        'fit_params': { 'max_iter_global': 100, 'mode_finding_iter': 20, 'cg_restarts': 10 } 
    }
}
# 注意: TPRT-LAのfit_paramsは、お使いのtprt_tang.pyの実装に合わせてください。
# Adam版をお使いの場合は 'lr' を、CG版をお使いの場合は 'cg_restarts' を指定します。

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

    X_train = torch.tensor(train_features, dtype=dtype)
    y_train = torch.tensor(train_target, dtype=dtype)
    X_test = torch.tensor(test_features, dtype=dtype)
    y_test = torch.tensor(test_target, dtype=dtype)
    
    return X_train, y_train, X_test, y_test

def run_single_split_evaluation(model_name, dataset_name, split_id, datasets_base_path: Path, save_dir: Path):
    """指定された単一のモデル、データセット、スプリットで評価を行う"""
    try: # === メインの処理を try ブロックで囲む ===
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
        logging.info(f"Training with params: {fit_params}")
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
        
        result_file = save_dir / f"result_{model_name}_{dataset_name}_split_{split_id}.csv"
        df.to_csv(result_file, index=False)
        logging.info(f"Result saved to {result_file}")
        logging.info(f"  => SUCCESS: RMSE: {rmse:.4f} | Time: {elapsed_time:.2f}s")

    except Exception as e:
        # === 予期せぬエラーが発生した場合の処理 ===
        logging.error(f"An unexpected error occurred during evaluation of {model_name} on {dataset_name}, Split {split_id}.")
        logging.error(traceback.format_exc())
        
        # 失敗したことを示す結果ファイルを作成
        error_data = {
            'model': [model_name],
            'dataset': [dataset_name],
            'split_id': [split_id],
            'rmse': [np.nan],
            'time_s': [np.nan]
        }
        df = pd.DataFrame(error_data)
        
        error_file = save_dir / f"ERROR_result_{model_name}_{dataset_name}_split_{split_id}.csv"
        df.to_csv(error_file, index=False)
        logging.info(f"Error summary saved to {error_file}")


def main():
    parser = argparse.ArgumentParser(description="Run a single split evaluation for Full TPRT models.")
    parser.add_argument('--model_name', type=str, required=True, choices=EVALUATION_CONFIG.keys())
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--split_id', type=int, required=True)
    parser.add_argument('--base_path', type=str, default='datasets/dataset_tang_2017/', help='Base path to the datasets directory.')
    parser.add_argument('--save_dir', type=str, required=True)
    
    args = parser.parse_args()
    
    log_dir = Path(args.save_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{args.model_name}_{args.dataset_name}_split_{args.split_id}.log"
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

    run_single_split_evaluation(
        args.model_name, args.dataset_name, args.split_id, Path(args.base_path), Path(args.save_dir)
    )

if __name__ == "__main__":
    main()