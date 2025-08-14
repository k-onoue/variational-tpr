import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Matplotlibのスタイル設定
plt.style.use("ggplot")

# GPUが利用可能かチェックし、デバイスを設定 (GPyTorch/PyTorchでの作法)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 大規模なダミーデータの生成
# ------------------------------------
# N: データ点数
N = 20000
# X: 入力データ (-10から10までの一様分布)
# (ここは変更なし)
X_np = np.random.uniform(-10.0, 10.0, (N, 1))
# f(x) = sin(x) + 0.2 * cos(3x)
Y_np = np.sin(X_np) + 0.2 * np.cos(3 * X_np) + np.random.randn(N, 1) * 0.1

# データをTensorFlowのテンソルからPyTorchのテンソルに変換し、GPUへ送る
# .float()はgpflow.default_float()に相当
X = torch.from_numpy(X_np).float().to(device)
Y = torch.from_numpy(Y_np).float().squeeze(-1).to(device) # GPyTorchではYは(N,)の1次元テンソルを期待する

# 2. データパイプラインの構築 (`torch.utils.data.DataLoader`)
# ------------------------------------
# ミニバッチサイズ
batch_size = 256

# tf.data.DatasetからPyTorchのTensorDatasetとDataLoaderに変更
train_dataset = torch.utils.data.TensorDataset(X, Y)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# 3. モデルの定義 (`gpytorch.models.ApproximateGP`)
# ------------------------------------
# M: 誘導点の数
M = 50

# 誘導点の初期位置をK-Meansで決定 (このロジックは流用)
kmeans = KMeans(n_clusters=M, random_state=0, n_init='auto').fit(X_np)
inducing_points = torch.from_numpy(kmeans.cluster_centers_).float().to(device)

# GPyTorchでは、モデルをクラスとして定義するのが一般的
class SparseGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        # VariationalDistribution: q(u)の分布を定義
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        # VariationalStrategy: スパースGPの近似戦略を定義
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(SparseGPModel, self).__init__(variational_strategy)
        
        # 平均関数とカーネル関数を定義
        self.mean_module = gpytorch.means.ConstantMean()
        # Matern52カーネルは nu=2.5 のMaternKernelに相当
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # ガウス過程の事後分布を返す
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# モデルと尤度(likelihood)をインスタンス化
model = SparseGPModel(inducing_points=inducing_points).to(device)
likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)


# 4. 最適化の準備
# ------------------------------------
# オプティマイザの選択 (PyTorchのAdamを使用)
# モデルのパラメータと尤度のパラメータの両方を渡す
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.01)

# 損失関数を定義 (GPyTorchのELBOを使用)
mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=Y.size(0))

# 学習の進捗を記録するためのリスト
log_elbo = []

# 5. 最適化ループの実行
# ------------------------------------
# トレーニングステップ数からエポック数に変換
epochs = 2
training_steps_per_epoch = N // batch_size
training_steps = epochs * training_steps_per_epoch

print(f"Starting training for {epochs} epochs ({training_steps} steps)...")

# モデルを学習モードに設定
model.train()
likelihood.train()

# 最適化ループ
# tf.functionの代わりに、PyTorchの標準的な学習ループを記述
global_step = 0
for i in range(epochs):
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(x_batch) # 順伝播
        loss = -mll(output, y_batch) # 損失(負のELBO)を計算
        loss.backward() # 勾配を計算
        optimizer.step() # パラメータを更新
        
        if global_step % 100 == 0:
            elbo = -loss.item()
            log_elbo.append(elbo)
            print(f"Epoch {i+1}/{epochs} - Step {global_step:5d}: ELBO = {elbo:.4f}")
        
        global_step += 1

print("Optimization finished.")

# 6. 結果の可視化
# ------------------------------------
# モデルを評価モードに設定
model.eval()
likelihood.eval()

# テスト用の入力データを作成
xx_np = np.linspace(-12, 12, 200).reshape(-1, 1)
xx = torch.from_numpy(xx_np).float().to(device)

# 予測平均と予測分散を計算 (勾配計算は不要)
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(xx))
    mean = observed_pred.mean.cpu().numpy()
    # 95%信頼区間を取得
    lower, upper = observed_pred.confidence_region()
    lower = lower.cpu().numpy()
    upper = upper.cpu().numpy()

# プロット
plt.figure(figsize=(12, 6))

# 元のデータ点をプロット (多すぎるのでサンプリングして表示)
plt.plot(X_np[::20], Y_np[::20], "kx", mew=2, alpha=0.5, label="Training Data (Sampled)")

# 予測平均をプロット
plt.plot(xx_np, mean, "C0", lw=2, label="Predictive Mean")

# 95%信頼区間をプロット
plt.fill_between(
    xx_np[:, 0], lower, upper, color="C0", alpha=0.2, label="95% Confidence Interval"
)

# 誘導点をプロット
inducing_points = model.variational_strategy.inducing_points.detach().cpu().numpy()
plt.plot(inducing_points, np.zeros_like(inducing_points) - 2.5, "C4^", ms=8, label="Inducing Points")

plt.title("Sparse Gaussian Process Regression with GPyTorch")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(loc="upper left")
plt.ylim(-3, 3)
plt.savefig("sparse_gp_regression_gpytorch.png")

# ELBOの学習曲線
plt.figure(figsize=(8, 4))
plt.plot(log_elbo)
plt.title("ELBO (Evidence Lower Bound) over training steps")
plt.xlabel("Iteration (x100)")
plt.ylabel("ELBO")
plt.savefig("elbo_training_curve_gpytorch.png")