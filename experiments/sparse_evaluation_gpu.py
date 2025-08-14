# evaluation_sparse.py (Xu et al. 2023 対応版)

import pandas as pd
import torch
import warnings
import time
import logging
from pathlib import Path
import argparse

from tprt import SparseTPRTMiniBatch, SparseTPRTMiniBatch_Xu

torch.set_default_dtype(torch.float64)
warnings.filterwarnings('ignore')

torch.manual_seed(42)

# === 実験パラメータの一元管理 (Xu et al. 2023 に準拠) ===
# 論文のSVTPに対応するモデルは SparseTPRTMiniBatch_Xu
SPARSE_EVALUATION_CONFIG = {
    'TPRT-SCAVI': {
        'model_class': SparseTPRTMiniBatch,
        'init_params': { 'nu_f': 2.0, 'nu_e': 2.0, 'kernel_lengthscale': 1.0, 'kernel_variance': 1.0, 'likelihood_sigma': 1.0 },
        'fit_params': { 'epochs': 5000, 'batch_size': 1024, 'lr': 0.01, 'eval_interval': 10 }
    },
    'SVTP-UB': {
        'model_class': SparseTPRTMiniBatch_Xu,
        'init_params': { 'nu_f': 2.0, 'nu_e': 2.0, 'kernel_lengthscale': 1.0, 'kernel_variance': 1.0, 'likelihood_sigma': 1.0 },
        'fit_params': { 'epochs': 5000, 'batch_size': 1024, 'lr': 0.01, 'kl_method': 'UB', 'eval_interval': 10 }
    },
    'SVTP-MC': {
        'model_class': SparseTPRTMiniBatch_Xu,
        'init_params': { 'nu_f': 2.0, 'nu_e': 2.0, 'kernel_lengthscale': 1.0, 'kernel_variance': 1.0, 'likelihood_sigma': 1.0 },
        'fit_params': { 'epochs': 5000, 'batch_size': 1024, 'lr': 0.01, 'kl_method': 'MC', 'num_samples_kl': 10, 'eval_interval': 10 }
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

def run_single_split_evaluation(model_name, dataset_name, split_id, datasets_base_path: Path, save_dir: Path):
    """
    指定された単一のモデル、データセット、スプリットで評価を行い、
    途中経過をCSVファイルに追記保存する。(スパースモデル版)
    """
    try:
        logging.info(f"===== Evaluating Model: {model_name} on Dataset: {dataset_name}, Split: {split_id} =====")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        
        config = SPARSE_EVALUATION_CONFIG[model_name]
        model_class = config['model_class']
        init_params = config['init_params'].copy()
        fit_params = config['fit_params'].copy()
        # ★ 変更点: eval_intervalを取得
        eval_interval = fit_params.pop('eval_interval', 100) 

        # ★ 変更点: 結果保存用のパスを定義
        result_file = save_dir / f"result_{model_name}_{dataset_name}_split_{split_id}.csv"

        split_path = datasets_base_path / dataset_name / f'split_{split_id}'
        if not split_path.exists():
            logging.error(f"Split path not found: {split_path}. Exiting.")
            return

        X_train, y_train, X_test, y_test = load_data(split_path)
        
        # バッチサイズとMを決定 (このロジックはスパースモデル特有のため維持)
        default_batch_size = fit_params.get('batch_size', 1024)
        if default_batch_size > len(X_train):
            actual_batch_size = len(X_train)
            logging.warning(f"Default batch size ({default_batch_size}) is larger than dataset size ({len(X_train)}). Adjusting to {actual_batch_size}.")
        else:
            actual_batch_size = default_batch_size
        fit_params['batch_size'] = actual_batch_size
        
        M = actual_batch_size // 4
        logging.info(f"Setting M = actual_batch_size / 4 = {actual_batch_size} / 4 = {M}")
        if M < 2: 
            M = 2
            logging.warning(f"Calculated M is too small. Setting M to minimum value of 2.")
        init_params['M'] = M

        # モデルを初期化
        model = model_class(X_train, y_train, **init_params, device=device)
        # model.to(device) # モデル内部でdevice指定しているのでこれは不要

        start_time = time.time()
        logging.info(f"Training with M={M} and params: {fit_params}. Results will be saved to {result_file}")
        
        # === ★ 変更点: evaluate_model を呼び出し、結果パスを渡す ===
        model.evaluate_model(
            **fit_params,
            X_test=X_test,
            y_test=y_test,
            eval_interval=eval_interval,
            result_path=result_file
        )
        # ========================================================
        
        end_time = time.time()
        total_time = end_time - start_time
        logging.info(f"Evaluation process finished in {total_time:.2f}s.")

        # --- ★ 変更点: 完了後、保存されたファイルから最終結果を報告 ---
        if result_file.exists() and result_file.stat().st_size > 0:
            final_results_df = pd.read_csv(result_file)
            if not final_results_df.empty:
                last_row = final_results_df.iloc[-1]
                logging.info(f"  => SUCCESS: Final RMSE at epoch {last_row['epoch']}: {last_row['rmse']:.4f} | Total Time: {total_time:.2f}s")
            else:
                logging.warning("Result file is empty.")
        else:
            logging.error("Result file was not created or is empty.")
        # -----------------------------------------------------------

    except Exception as e:
        import traceback
        logging.error(f"An unexpected error occurred in evaluation script for {model_name} on {dataset_name}, Split {split_id}.")
        logging.error(traceback.format_exc())
        logging.error(f"Error details: {e}")
        # タイムアウトの場合、途中までの結果が残るので、特別なエラーファイルは不要


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