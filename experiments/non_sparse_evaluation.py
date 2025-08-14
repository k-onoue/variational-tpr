# evaluation.py (Slurm対応版)

import pandas as pd
import torch
import warnings
import time
import logging
from pathlib import Path
import argparse # argparseを追加

# --- (tprtのインポートとグローバル設定は変更なし) ---
from tprt import TPRTFullBatch, TPRTFullBatch_Tang
torch.set_default_dtype(torch.float64)
warnings.filterwarnings('ignore')

torch.manual_seed(42)  

# === 実験パラメータの一元管理 ===
EVALUATION_CONFIG = {
    'TPRT-VEM': {
        'model_class': TPRTFullBatch,
        'init_params': { 'nu_f': 2.0, 'nu_e': 2.0, 'kernel_lengthscale': 1.0, 'kernel_variance': 1.0, 'likelihood_sigma': 1.0 },
        'fit_params': { 'max_iter_global': 1000, 'cavi_max_iter': 20, 'lr': 0.01, 'eval_interval': 10 }
    },
    'TPRT-LA': {
        'model_class': TPRTFullBatch_Tang,
        'init_params': { 'nu_f': 2.0, 'nu_e': 2.0, 'kernel_lengthscale': 1.0, 'kernel_variance': 1.0, 'likelihood_sigma': 1.0 },
        'fit_params': { 'max_iter_global': 1000, 'mode_finding_iter': 20, 'lr': 0.01, 'eval_interval': 10 } 
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
    """
    指定された単一のモデル、データセット、スプリットで評価を行い、
    途中経過をCSVファイルに追記保存する。
    """
    try:
        logging.info(f"===== Evaluating Model: {model_name} on Dataset: {dataset_name}, Split: {split_id} =====")
        
        config = EVALUATION_CONFIG[model_name]
        model_class = config['model_class']
        init_params = config['init_params']
        fit_params = config['fit_params'].copy()
        eval_interval = fit_params.pop('eval_interval', 100)

        # --- ★ 変更点: 結果保存用のパスを定義 ---
        result_file = save_dir / f"result_{model_name}_{dataset_name}_split_{split_id}.csv"
        # --------------------------------------

        split_path = datasets_base_path / dataset_name / f'split_{split_id}'
        if not split_path.exists():
            logging.error(f"Split path not found: {split_path}. Exiting.")
            return

        X_train, y_train, X_test, y_test = load_data(split_path)
        
        model = model_class(X_train, y_train, **init_params)
        
        start_time = time.time()
        logging.info(f"Training with params: {fit_params}. Results will be saved to {result_file}")
        
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
        if result_file.exists():
            # 正常に終了した場合、最終的な結果をログに出力
            final_results_df = pd.read_csv(result_file)
            if not final_results_df.empty:
                last_row = final_results_df.iloc[-1]
                logging.info(f"  => SUCCESS: Final RMSE at iter {last_row['iteration']}: {last_row['rmse']:.4f} | Total Time: {total_time:.2f}s")
            else:
                logging.warning("Result file is empty.")
        else:
            logging.error("Result file was not created.")
        # -----------------------------------------------------------

    except Exception as e:
        import traceback
        # (エラーハンドリング部分は変更なし、ただしエラーファイルは作らない)
        logging.error(f"An unexpected error occurred in evaluation script for {model_name} on {dataset_name}, Split {split_id}.")
        logging.error(traceback.format_exc())
        logging.error(f"Error details: {e}")
        # タイムアウトの場合、途中までの結果が残るので、特別なエラーファイルは不要
        # ただし、エラーが発生したことはログで明確にする


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
