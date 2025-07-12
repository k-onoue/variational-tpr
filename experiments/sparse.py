# evaluation_sparse.py (Xu et al. 2023 対応版)

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

from tprt import SparseTPRTMiniBatch, SparseTPRTMiniBatch_Xu

torch.set_default_dtype(torch.float64)
warnings.filterwarnings('ignore')

# === 実験パラメータの一元管理 (Xu et al. 2023 に準拠) ===
# 論文のSVTPに対応するモデルは SparseTPRTMiniBatch_Xu
SPARSE_EVALUATION_CONFIG = {
    'TPRT-SCAVI': {
        'model_class': SparseTPRTMiniBatch,
        'init_params': { 'nu_f': 2.0+1e-6, 'nu_e': 2.0+1e-6, 'kernel_lengthscale': 1.0, 'kernel_variance': 1.0, 'likelihood_sigma': 1.0 },
        'fit_params': { 'epochs': 5000, 'batch_size': 1024, 'cavi_max_iter': 20, 'lr': 0.01 }
    },
    'SVTP-UB': {
        'model_class': SparseTPRTMiniBatch_Xu,
        'init_params': { 'nu_f': 2.0+1e-6, 'nu_e': 2.0+1e-6, 'kernel_lengthscale': 1.0, 'kernel_variance': 1.0, 'likelihood_sigma': 0.1 },
        'fit_params': { 'epochs': 5000, 'batch_size': 1024, 'lr': 0.01, 'kl_method': 'UB' }
    },
    'SVTP-MC': {
        'model_class': SparseTPRTMiniBatch_Xu,
        'init_params': { 'nu_f': 2.0+1e-6, 'nu_e': 2.0+1e-6, 'kernel_lengthscale': 1.0, 'kernel_variance': 1.0, 'likelihood_sigma': 0.1 },
        'fit_params': { 'epochs': 5000, 'batch_size': 1024, 'lr': 0.01, 'kl_method': 'MC', 'num_samples_kl': 10 }
    }
}

# (load_data, run_single_split_evaluation, main 関数は前の回答と同じでOK)
# ... (前の回答の `evaluation_sparse.py` の残りの部分をここに貼り付け) ...
# ...
def load_data(dataset_path: Path, dtype=torch.float64):
    """指定されたパスから学習データとテストデータを読み込む"""
    train_features = pd.read_csv(dataset_path / 'train_features.csv', header=None).values
    train_target = pd.read_csv(dataset_path / 'train_target.csv', header=None).values
    test_features = pd.read_csv(dataset_path / 'test_features.csv', header=None).values
    test_target = pd.read_csv(dataset_path / 'test_target.csv', header=None).values

    return (torch.tensor(train_features, dtype=dtype),
            torch.tensor(train_target, dtype=dtype),
            torch.tensor(test_features, dtype=dtype),
            torch.tensor(test_target, dtype=dtype))

# def run_single_split_evaluation(model_name, dataset_name, split_id, datasets_base_path: Path, save_dir: Path):
#     """指定された単一のモデル、データセット、スプリットで評価を行う"""
#     logging.info(f"===== Evaluating Model: {model_name} on Dataset: {dataset_name}, Split: {split_id} =====")
    
#     config = SPARSE_EVALUATION_CONFIG[model_name]
#     model_class = config['model_class']
#     init_params = config['init_params'].copy()
#     fit_params = config['fit_params'].copy()

#     split_path = datasets_base_path / dataset_name / f'split_{split_id}'
#     if not split_path.exists():
#         logging.error(f"Split path not found: {split_path}. Exiting.")
#         return

#     X_train, y_train, X_test, y_test = load_data(split_path)
    
#     # --- 1. 実際のバッチサイズを決定 ---
#     # デフォルトのバッチサイズを取得
#     default_batch_size = fit_params.get('batch_size', 1024) # configにない場合は1024をデフォルトに
    
#     # データセットサイズがデフォルトのバッチサイズより小さい場合は、データセットサイズに調整
#     if default_batch_size > len(X_train):
#         actual_batch_size = len(X_train)
#         logging.warning(f"Default batch size ({default_batch_size}) is larger than dataset size ({len(X_train)}). Adjusting batch size to {actual_batch_size}.")
#     else:
#         actual_batch_size = default_batch_size
    
#     # fit_paramsに実際のバッチサイズをセット
#     fit_params['batch_size'] = actual_batch_size
    
#     # --- 2. 決定されたバッチサイズに基づいて M を計算 ---
#     M = actual_batch_size // 4
#     logging.info(f"Setting M = actual_batch_size / 4 = {actual_batch_size} / 4 = {M}")
    
#     # Mが0にならないように最低値を保証
#     if M < 2: 
#         M = 2
#         logging.warning(f"Calculated M is too small. Setting M to minimum value of 2.")

#     # 計算したMを初期化パラメータに追加
#     init_params['M'] = M

#     # モデルの初期化
#     model = model_class(X_train, y_train, **init_params)
    
#     start_time = time.time()
#     logging.info(f"Training with M={M} and fit_params: {fit_params}")
#     model.fit(**fit_params)
#     end_time = time.time()
#     logging.info(f"Training finished.")

#     # 計算したMを初期化パラメータに追加
#     init_params['M'] = M
#     # =================================================

#     # バッチサイズがデータセットサイズより大きい場合、データセットサイズに調整
#     if 'batch_size' in fit_params and fit_params['batch_size'] > len(X_train):
#         logging.warning(f"Batch size ({fit_params['batch_size']}) is larger than dataset size ({len(X_train)}). Adjusting to {len(X_train)}.")
#         fit_params['batch_size'] = len(X_train)
    
#     # モデルの初期化
#     model = model_class(X_train, y_train, **init_params)
    
#     start_time = time.time()
#     logging.info(f"Training with M={M} and fit_params: {fit_params}")
#     model.fit(**fit_params)
#     end_time = time.time()
#     logging.info(f"Training finished.")

#     with torch.no_grad():
#         # SVTPは予測にもサンプリングを使うので、時間がかかる可能性がある
#         pred_start_time = time.time()
#         pred_mean, _, _ = model.predict(X_test)
#         pred_end_time = time.time()
#         logging.info(f"Prediction took {pred_end_time - pred_start_time:.2f}s.")
        
#     rmse = np.sqrt(mean_squared_error(y_test.numpy(), pred_mean.numpy()))
#     elapsed_time = end_time - start_time
    
#     result_data = {
#         'model': [model_name],
#         'dataset': [dataset_name],
#         'split_id': [split_id],
#         'rmse': [rmse],
#         'time_s': [elapsed_time]
#     }
#     df = pd.DataFrame(result_data)
    
#     result_file = save_dir / f"result_{model_name}_{dataset_name}_split_{split_id}.csv"
#     df.to_csv(result_file, index=False)
#     logging.info(f"Result saved to {result_file}")
#     logging.info(f"  => RMSE: {rmse:.4f} | Time: {elapsed_time:.2f}s")

def run_single_split_evaluation(model_name, dataset_name, split_id, datasets_base_path: Path, save_dir: Path):
    """指定された単一のモデル、データセット、スプリットで評価を行う"""
    try: # === メインの処理を try ブロックで囲む ===
        logging.info(f"===== Evaluating Model: {model_name} on Dataset: {dataset_name}, Split: {split_id} =====")
        
        config = SPARSE_EVALUATION_CONFIG[model_name]
        model_class = config['model_class']
        init_params = config['init_params'].copy()
        fit_params = config['fit_params'].copy()

        split_path = datasets_base_path / dataset_name / f'split_{split_id}'
        if not split_path.exists():
            logging.error(f"Split path not found: {split_path}. Exiting.")
            return

        X_train, y_train, X_test, y_test = load_data(split_path)
        
        # --- 1. 実際のバッチサイズを決定 ---
        default_batch_size = fit_params.get('batch_size', 1024)
        if default_batch_size > len(X_train):
            actual_batch_size = len(X_train)
            logging.warning(f"Default batch size ({default_batch_size}) is larger than dataset size ({len(X_train)}). Adjusting to {actual_batch_size}.")
        else:
            actual_batch_size = default_batch_size
        fit_params['batch_size'] = actual_batch_size
        
        # --- 2. 決定されたバッチサイズに基づいて M を計算 ---
        M = actual_batch_size // 4
        logging.info(f"Setting M = actual_batch_size / 4 = {actual_batch_size} / 4 = {M}")
        if M < 2: 
            M = 2
            logging.warning(f"Calculated M is too small. Setting M to minimum value of 2.")
        init_params['M'] = M

        # モデルの初期化
        model = model_class(X_train, y_train, **init_params)
        
        start_time = time.time()
        logging.info(f"Training with M={M} and fit_params: {fit_params}")
        model.fit(**fit_params)
        end_time = time.time()
        logging.info(f"Training finished.")

        with torch.no_grad():
            pred_start_time = time.time()
            pred_mean, _, _ = model.predict(X_test)
            pred_end_time = time.time()
            logging.info(f"Prediction took {pred_end_time - pred_start_time:.2f}s.")
            
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
        # traceback をインポートして、詳細なエラー情報をログに出力
        import traceback
        logging.error(traceback.format_exc())
        
        # 失敗したことを示す結果ファイルを作成
        error_data = {
            'model': [model_name],
            'dataset': [dataset_name],
            'split_id': [split_id],
            'rmse': [np.nan], # エラーなのでRMSEはNaN
            'time_s': [np.nan]
        }
        df = pd.DataFrame(error_data)
        
        # ファイル名に "ERROR" を含める
        error_file = save_dir / f"ERROR_result_{model_name}_{dataset_name}_split_{split_id}.csv"
        df.to_csv(error_file, index=False)
        logging.info(f"Error summary saved to {error_file}")


def main():
    parser = argparse.ArgumentParser(description="Run a single split evaluation for Sparse TPRT models.")
    parser.add_argument('--model_name', type=str, required=True, choices=SPARSE_EVALUATION_CONFIG.keys())
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--split_id', type=int, required=True)
    parser.add_argument('--base_path', type=str, default='datasets/dataset_xu_2024/', help='Base path to the datasets directory.')
    parser.add_argument('--save_dir', type=str, required=True)
    
    args = parser.parse_args()
    
    log_dir = Path(args.save_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{args.model_name}_{args.dataset_name}_split_{args.split_id}.log"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

    run_single_split_evaluation(
        args.model_name, args.dataset_name, args.split_id, Path(args.base_path), Path(args.save_dir)
    )

if __name__ == "__main__":
    main()