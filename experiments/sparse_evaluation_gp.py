import pandas as pd
import numpy as np
# ==============================================================================
# ★★★ GPyTorch (PyTorch) ベースのライブラリをインポート ★★★
# ==============================================================================
import torch
import gpytorch

from sklearn.cluster import KMeans
import warnings
import time
import logging
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

# GPyTorch/PyTorchのデフォルトのデータ型を設定
torch.set_default_dtype(torch.float64)
# 不要な警告を非表示にする
warnings.filterwarnings('ignore')
# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# =============================================================================
# ★★★ GPyTorch SVGPモデルの内部定義 ★★★
# =============================================================================
class _InternalSparseGPModel(gpytorch.models.ApproximateGP):
    """
    GPyTorchのApproximateGPを継承した実際のモデルクラス。
    GPyTorchSVGPWrapper内で使用される。
    """
    def __init__(self, inducing_points: torch.Tensor, kernel):
        # q(u)の分布を定義
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        # スパースGPの近似戦略を定義
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        
        # 平均関数とカーネル関数を定義
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x: torch.Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# =============================================================================
# ★★★ GPyTorch SVGP Wrapper Class ★★★
# =============================================================================
class GPyTorchSVGPWrapper:
    """
    GPyTorchのApproximateGPモデルをラップし、学習、評価、予測のための便利なメソッドを提供するクラス。
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, M: int, kernel=None):
        """
        コンストラクタ

        Args:
            X (np.ndarray): 入力学習データ (N, D)
            y (np.ndarray): 出力学習データ (N, 1)
            M (int): 誘導点の数
            kernel (gpytorch.kernels.Kernel, optional): GPyTorchカーネル。指定されない場合はRBFカーネルが使用される。
        """
        # --- GPUデバイスの設定 ---
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- データとパラメータの準備 (PyTorch Tensorに変換) ---
        self.X_full = torch.from_numpy(X).to(self.device)
        self.y_full = torch.from_numpy(y).squeeze(-1).to(self.device) # GPyTorchでは(N,)を期待
        self.N, self.D = self.X_full.shape
        self.M = M

        # --- GPyTorchモデルの構築 ---
        if kernel is None:
            # ★★★ RBFカーネルを使用 ★★★
            # ARD (Automatic Relevance Determination) を有効にする
            kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=self.D)
            )
        
        # 誘導点の初期位置をK-Meansで決定
        kmeans = KMeans(n_clusters=self.M, random_state=0, n_init='auto').fit(X)
        inducing_points = torch.from_numpy(kmeans.cluster_centers_).to(self.device)

        # 内部モデルと尤度をインスタンス化し、GPUへ送る
        self.model = _InternalSparseGPModel(inducing_points, kernel).to(self.device)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        
        self.optimizer = None
        # ELBO損失関数を定義
        self.mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=self.N)

    def predict(self, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """新しいデータ点に対する予測を行う。"""
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            X_test_tensor = torch.from_numpy(X_test).to(self.device)
            # 尤度を通して観測ノイズを含んだ予測分布を取得
            observed_pred = self.likelihood(self.model(X_test_tensor))
            mean = observed_pred.mean
            var = observed_pred.variance
        # 結果をnumpy配列としてCPUに戻す
        return mean.cpu().numpy()[:, np.newaxis], var.cpu().numpy()[:, np.newaxis]

    def evaluate_model(self, epochs: int = 100, batch_size: int = 256, lr: float = 0.01,
                       X_test: np.ndarray = None, y_test: np.ndarray = None, 
                       eval_interval: int = 10, result_path: Path = None):
        """
        モデルを学習させながら、定期的にテストデータで性能を評価し、結果をファイルに保存する。
        """
        # --- 最適化とデータローダーの準備 ---
        self.optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
            {'params': self.likelihood.parameters()},
        ], lr=lr)
        
        train_dataset = torch.utils.data.TensorDataset(self.X_full, self.y_full)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        can_evaluate = X_test is not None and y_test is not None and result_path is not None
        if can_evaluate:
            if not result_path.parent.exists():
                result_path.parent.mkdir(parents=True, exist_ok=True)
            with open(result_path, 'w') as f:
                f.write("epoch,rmse,elbo\n")

        logging.info(f"Starting training and evaluation for {epochs} epochs...")
        elbo_val = 0.0

        # --- 学習ループ ---
        self.model.train()
        self.likelihood.train()
        
        for epoch in range(1, epochs + 1):
            for x_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                output = self.model(x_batch)  # 順伝播
                loss = -self.mll(output, y_batch) # 損失(負のELBO)を計算
                loss.backward()  # 勾配を計算
                self.optimizer.step() # パラメータを更新
                elbo_val = -loss.item()

            if can_evaluate and (epoch % eval_interval == 0):
                pred_mean, _ = self.predict(X_test)
                rmse = np.sqrt(np.mean((y_test.flatten() - pred_mean.flatten())**2))
                logging.info(f"Epoch {epoch:4d}/{epochs}, Test RMSE: {rmse:.4f}, Final Batch ELBO: {elbo_val:.4f}")
                with open(result_path, 'a') as f:
                    f.write(f"{epoch},{rmse},{elbo_val}\n")
        
        if can_evaluate and (epochs > 0 and epochs % eval_interval != 0):
            pred_mean, _ = self.predict(X_test)
            rmse = np.sqrt(np.mean((y_test.flatten() - pred_mean.flatten())**2))
            with open(result_path, 'a') as f:
                f.write(f"{epochs},{rmse},{elbo_val}\n")
            logging.info(f"Final evaluation - Epoch {epochs:4d}, Test RMSE: {rmse:.4f}")

        logging.info("Optimization finished.")

# =============================================================================
# ★★★ Evaluation Script (GPyTorch版) ★★★
# =============================================================================
SPARSE_EVALUATION_CONFIG = {
    'GPyTorchSVGP': { # モデル名を変更
        'model_class': GPyTorchSVGPWrapper, # Wrapperクラスを差し替え
        'init_params': {},
        'fit_params': {
            'epochs': 5000,
            'batch_size': 1024,
            'lr': 0.01,
            'eval_interval': 10
        }
    },
}

def load_data(dataset_path: Path):
    """指定されたパスから学習データとテストデータをNumpy配列として読み込む (変更なし)"""
    train_features = pd.read_csv(dataset_path / 'train_features.csv', header=None).values
    train_target = pd.read_csv(dataset_path / 'train_target.csv', header=None).values
    test_features = pd.read_csv(dataset_path / 'test_features.csv', header=None).values
    test_target = pd.read_csv(dataset_path / 'test_target.csv', header=None).values
    return train_features, train_target, test_features, test_target

def run_single_split_evaluation(model_name, dataset_name, split_id, datasets_base_path: Path, save_dir: Path):
    """
    指定された単一のモデル、データセット、スプリットで評価を行い、
    途中経過をCSVファイルに追記保存する。
    """
    try:
        logging.info(f"===== Evaluating Model: {model_name} on Dataset: {dataset_name}, Split: {split_id} =====")
        
        # --- GPUチェックをPyTorch方式に変更 ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == 'cuda':
            logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logging.info("Using CPU")
        
        config = SPARSE_EVALUATION_CONFIG[model_name]
        model_class = config['model_class']
        init_params = config['init_params'].copy()
        fit_params = config['fit_params'].copy()
        eval_interval = fit_params.pop('eval_interval', 100) 

        result_file = save_dir / f"result_{model_name}_{dataset_name}_split_{split_id}.csv"

        split_path = datasets_base_path / dataset_name / f'split_{split_id}'
        if not split_path.exists():
            logging.error(f"Split path not found: {split_path}. Exiting.")
            return

        X_train, y_train, X_test, y_test = load_data(split_path)
        
        default_batch_size = fit_params.get('batch_size', 1024)
        if default_batch_size > len(X_train):
            actual_batch_size = len(X_train)
            logging.warning(f"Default batch size ({default_batch_size}) is larger than dataset size ({len(X_train)}). Adjusting to {actual_batch_size}.")
        else:
            actual_batch_size = default_batch_size
        fit_params['batch_size'] = actual_batch_size
        
        M = max(2, actual_batch_size // 4)
        logging.info(f"Setting M (inducing points) = {M}")
        init_params['M'] = M

        model = model_class(X_train, y_train, **init_params)

        start_time = time.time()
        logging.info(f"Training with M={M} and params: {fit_params}. Results will be saved to {result_file}")
        
        model.evaluate_model(
            **fit_params,
            X_test=X_test,
            y_test=y_test,
            eval_interval=eval_interval,
            result_path=result_file
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        logging.info(f"Evaluation process finished in {total_time:.2f}s.")

        if result_file.exists() and result_file.stat().st_size > 0:
            final_results_df = pd.read_csv(result_file)
            if not final_results_df.empty:
                last_row = final_results_df.iloc[-1]
                logging.info(f"  => SUCCESS: Final RMSE at epoch {last_row['epoch']}: {last_row['rmse']:.4f} | Total Time: {total_time:.2f}s")
            else:
                logging.warning("Result file is empty.")
        else:
            logging.error("Result file was not created or is empty.")

    except Exception as e:
        import traceback
        logging.error(f"An unexpected error occurred in evaluation script for {model_name} on {dataset_name}, Split {split_id}.")
        logging.error(traceback.format_exc())
        logging.error(f"Error details: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run a single split evaluation for Sparse GP models using GPyTorch.")
    # --- モデル名の選択肢をGPyTorch版に変更 ---
    parser.add_argument('--model_name', type=str, required=True, choices=SPARSE_EVALUATION_CONFIG.keys())
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--split_id', type=int, required=True)
    parser.add_argument('--base_path', type=str, default='datasets/dataset_xu_2024/', help='Base path to the datasets directory.')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save results and logs.')
    
    args = parser.parse_args()
    
    save_path = Path(args.save_dir)
    log_dir = save_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"{args.model_name}_{args.dataset_name}_split_{args.split_id}.log"
    
    # ログハンドラの設定 (変更なし)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

    run_single_split_evaluation(
        args.model_name, args.dataset_name, args.split_id, Path(args.base_path), save_path
    )

if __name__ == "__main__":
    main()