# evaluation.py
# tprt パッケージと同じディレクトリに配置して実行してください。

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

# --- 作成した tprt パッケージからモデルをインポート ---
from tprt import TPRTFullBatch, TPRTFullBatch_Tang

# --- グローバル設定 ---
torch.set_default_dtype(torch.float64)
warnings.filterwarnings('ignore')

# === 1. ロギング設定の追加 ===
def setup_logging(log_file='evaluation_results.log'):
    """コンソールとファイルへのロギングを設定する"""
    # 古いログファイルを削除
    if os.path.exists(log_file):
        os.remove(log_file)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler() # コンソールにも出力
        ]
    )

# === 2. 実験パラメータの一元管理 ===
EVALUATION_CONFIG = {
    'TPRT-VEM': {
        'model_class': TPRTFullBatch,
        'init_params': {
            'nu_f': 2+1e-6,
            'nu_e': 2+1e-6,
            'kernel_lengthscale': 1.0,
            'kernel_variance': 1.0,
            'likelihood_sigma': 1.0, 
        }, # デフォルトの初期化パラメータを使用
        'fit_params': {
            'max_iter_global': 100,
            'cavi_max_iter':20,
            'lr': 0.01
        }
    },
    'TPRT-LA': {
        'model_class': TPRTFullBatch_Tang,
        'init_params': { # 論文設定に合わせて初期値を指定
            'nu_f': 2+1e-6,
            'nu_e': 2+1e-6,
            'kernel_lengthscale': 1.0,
            'kernel_variance': 1.0,
            'likelihood_sigma': 1.0 # 論文設定 (σ^2=1.0) に合わせる
        },
        'fit_params': {
            'max_iter_global': 100,
            'mode_finding_iter': 20,
            'lr': 0.01
        }
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

def evaluate_model(model_name, config, datasets_base_path: Path):
    """指定されたモデルを全てのデータセットで評価する"""
    logging.info(f"\n===== Evaluating Model: {model_name} =====")
    
    datasets = sorted([d.name for d in datasets_base_path.iterdir() if d.is_dir()])
    results = defaultdict(list)
    
    model_class = config['model_class']
    init_params = config['init_params']
    fit_params = config['fit_params']
    
    for dataset in datasets:
        logging.info(f"--- Dataset: {dataset} ---")
        dataset_rmses = []
        dataset_times = []
        
        for i in range(10): # 10スプリットで評価
            split_path = datasets_base_path / dataset / f'split_{i}'
            
            try:
                X_train, y_train, X_test, y_test = load_data(split_path)
            except FileNotFoundError:
                logging.warning(f"  Split {i} for dataset {dataset} not found. Skipping.")
                continue

            # モデルの初期化 (設定ファイルからパラメータを渡す)
            model = model_class(X_train, y_train, **init_params)
            
            start_time = time.time()
            
            # モデルの学習 (設定ファイルからパラメータを渡す)
            # モデル内部のprint文はパフォーマンスのためコメントアウト推奨
            logging.info(f"  Training on split {i}...")
            model.fit(**fit_params)
            logging.info(f"  Training finished for split {i}.")

            # 予測
            with torch.no_grad():
                pred_mean, _, _ = model.predict(X_test)

            end_time = time.time()
            
            # 評価
            rmse = np.sqrt(mean_squared_error(y_test.numpy(), pred_mean.numpy()))
            dataset_rmses.append(rmse)
            dataset_times.append(end_time - start_time)
            
        if dataset_rmses:
            avg_rmse = np.mean(dataset_rmses)
            std_rmse = np.std(dataset_rmses)
            avg_time = np.mean(dataset_times)
            results[dataset] = {'avg_rmse': avg_rmse, 'std_rmse': std_rmse, 'avg_time_s': avg_time}
            logging.info(f"  => Avg RMSE: {avg_rmse:.4f} ± {std_rmse:.4f} | Avg Time: {avg_time:.2f}s")
            
    return results

def main():
    """メインの実行関数"""
    setup_logging() # ロギングを開始

    base_path_str = 'datasets/dataset_tang_2017/'
    base_path = Path(base_path_str)
    
    if not base_path.exists():
        logging.warning(f"Dataset directory not found at {base_path_str}")
        # Colabや他の環境用の代替パス
        base_path_str_alt = '../datasets/dataset_tang_2017/'
        base_path = Path(base_path_str_alt)
        if not base_path.exists():
            logging.error("Dataset directory not found. Please check the path.")
            return

    all_results = {}
    for name, config in EVALUATION_CONFIG.items():
        all_results[name] = evaluate_model(name, config, base_path)

    # 結果を整形してDataFrameで表示
    summary_data = []
    # 最初のモデルの結果からデータセット一覧を取得
    datasets = sorted(all_results[list(EVALUATION_CONFIG.keys())[0]].keys())

    for dataset in datasets:
        row = {'Dataset': dataset}
        for model_name in EVALUATION_CONFIG.keys():
            res = all_results[model_name].get(dataset)
            if res:
                row[f'{model_name}_RMSE'] = f"{res['avg_rmse']:.4f} ± {res['std_rmse']:.4f}"
                row[f'{model_name}_Time(s)'] = f"{res['avg_time_s']:.2f}"
        summary_data.append(row)

    results_df = pd.DataFrame(summary_data)
    logging.info("\n\n--- 📊 Final Performance Summary 📊 ---")
    
    # DataFrameをログに出力
    summary_string = results_df.to_string(index=False)
    logging.info(f"\n{summary_string}")

    return results_df


if __name__ == "__main__":
    results_df = main()