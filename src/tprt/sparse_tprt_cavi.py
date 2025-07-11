from .kernels import rbf_kernel
import torch
import torch.nn as nn 
import torch.optim as optim
import math


class SparseTPRTFullBatch:
    """
    Implementation of Variational EM for Student-t Process Regression.
    - E-Step: CAVI updates for variational parameters.
    - M-Step: Gradient-based optimization of model hyperparameters and inducing points.
    """
    def __init__(self, X, y, M, nu_f=2.1, nu_e=2.1, kernel_lengthscale=1.0, kernel_variance=1.0, likelihood_sigma=1.0):
        """
        Args:
            X (torch.Tensor): Training inputs, shape (N, D).
            y (torch.Tensor): Training outputs, shape (N, 1) or (N,).
            M (int): Number of inducing points.
            ... (hyperparameters)
        """
        self.X = X
        # ★★★ 修正点1: yの形状を(N, 1)に統一 ★★★
        self.y = y.view(-1, 1)
        self.N, self.D = X.shape
        self.M = M

        # --- Initialize Inducing Points using Sobol sequence ---
        Z_initial = self._initialize_inducing_points()
        self.Z = nn.Parameter(Z_initial)

        # --- Initialize Hyperparameters (M-step) ---
        self.log_nu_f = nn.Parameter(torch.log(torch.tensor(nu_f, dtype=X.dtype)))
        self.log_nu_epsilon = nn.Parameter(torch.log(torch.tensor(nu_e, dtype=X.dtype)))
        self.log_sigma_sq = nn.Parameter(torch.log(torch.tensor(likelihood_sigma**2, dtype=X.dtype)))
        self.log_kernel_lengthscale = nn.Parameter(torch.log(torch.tensor(kernel_lengthscale, dtype=X.dtype)))
        self.log_kernel_variance = nn.Parameter(torch.log(torch.tensor(kernel_variance, dtype=X.dtype)))

        # --- Initialize Variational Parameters (E-step) ---
        self.m_u = torch.zeros(self.M, 1, dtype=X.dtype, device=X.device)
        self.S_u = torch.eye(self.M, dtype=X.dtype, device=X.device)
        self.alpha_r = torch.tensor(1.0, dtype=X.dtype, device=X.device)
        self.beta_r = torch.tensor(1.0, dtype=X.dtype, device=X.device)
        # ★★★ 修正点2: alpha/beta_lambdaの形状を(N, 1)に統一 ★★★
        self.alpha_lambda = torch.ones(self.N, 1, dtype=X.dtype, device=X.device)
        self.beta_lambda = torch.ones(self.N, 1, dtype=X.dtype, device=X.device)

    def _initialize_inducing_points(self):
        """
        Initializes inducing points using a Sobol sequence scaled to the data's bounds.
        """
        min_bounds = self.X.min(dim=0).values
        max_bounds = self.X.max(dim=0).values
        sobol_engine = torch.quasirandom.SobolEngine(dimension=self.D, scramble=True, seed=0)
        sobol_points_unit = sobol_engine.draw(self.M).to(self.X.dtype)
        return min_bounds + sobol_points_unit * (max_bounds - min_bounds)
    
    def _get_hyperparams(self):
        """Returns the exponentiated (positive) hyperparameters."""
        return {
            "nu_f": torch.exp(self.log_nu_f),
            "nu_epsilon": torch.exp(self.log_nu_epsilon),
            "sigma_sq": torch.exp(self.log_sigma_sq),
            "lengthscale": torch.exp(self.log_kernel_lengthscale),
            "variance": torch.exp(self.log_kernel_variance)
        }
        
    def _update_q_lambda(self, params, L_ZZ, K_XZ, K_ZX, k_ii):
        KZZ_inv_m_u = torch.cholesky_solve(self.m_u, L_ZZ)
        expected_f_mean = K_XZ @ KZZ_inv_m_u # Shape: (N, 1)

        if self.alpha_r > 1: expected_r_inv = self.beta_r / (self.alpha_r - 1.0)
        else: expected_r_inv = self.beta_r 

        KXZ_KZZ_inv = torch.cholesky_solve(K_ZX, L_ZZ).T
        
        var_f_term1 = expected_r_inv * (k_ii - (KXZ_KZZ_inv * K_XZ).sum(dim=1))
        var_f_term2 = (KXZ_KZZ_inv @ self.S_u * KXZ_KZZ_inv).sum(dim=1)
        # ★★★ 修正点3: var_fの形状を(N, 1)に調整 ★★★
        var_f = (var_f_term1 + var_f_term2).unsqueeze(1)

        # y と expected_f_mean が (N, 1) なので .squeeze() は不要
        expected_sq_error = (self.y - expected_f_mean).pow(2) + var_f
        self.alpha_lambda = params['nu_epsilon'] / 2.0 + 0.5
        self.beta_lambda = params['nu_epsilon'] / 2.0 + (0.5 / params['sigma_sq']) * expected_sq_error

    def _update_q_r(self, params, L_ZZ):
        trace_term = torch.trace(torch.cholesky_solve(self.S_u, L_ZZ))
        KZZ_inv_m_u = torch.cholesky_solve(self.m_u, L_ZZ)
        mean_term = self.m_u.T @ KZZ_inv_m_u
        expected_u_quadratic_form = trace_term + mean_term
        self.alpha_r = params['nu_f'] / 2.0 + self.M / 2.0
        self.beta_r = params['nu_f'] / 2.0 + 0.5 * expected_u_quadratic_form.squeeze()

    def _update_q_u(self, params, K_ZZ, K_XZ, K_ZX):
        expected_r = self.alpha_r / self.beta_r
        # alpha/beta_lambdaは(N, 1)なのでsqueeze()が必要
        expected_lambda = self.alpha_lambda.squeeze() / self.beta_lambda.squeeze()
        c = expected_lambda / params['sigma_sq']

        B = (K_ZX * c) @ K_XZ 
        precision_inner = expected_r * K_ZZ + B
        L_precision_inner = torch.linalg.cholesky(precision_inner)
        
        tmp_S = torch.cholesky_solve(K_ZZ, L_precision_inner)
        self.S_u = K_ZZ @ tmp_S

        # yは(N, 1)なのでsqueeze()が必要
        y_term = K_ZX @ (self.y.squeeze() * c)
        m_u_unscaled = torch.cholesky_solve(y_term.unsqueeze(1), L_precision_inner)
        self.m_u = K_ZZ @ m_u_unscaled

    def _cavi_step(self, params, K_ZZ, L_ZZ, K_XZ, K_ZX, k_ii):
        self._update_q_lambda(params, L_ZZ, K_XZ, K_ZX, k_ii)
        self._update_q_r(params, L_ZZ)
        self._update_q_u(params, K_ZZ, K_XZ, K_ZX)

    def _e_step(self, cavi_max_iter=20, cavi_tol=1e-6):
        with torch.no_grad(): 
            params = self._get_hyperparams()
            K_ZZ = rbf_kernel(self.Z, self.Z, params['lengthscale'], params['variance']) + torch.eye(self.M, device=self.Z.device) * 1e-6
            L_ZZ = torch.linalg.cholesky(K_ZZ)
            K_XZ = rbf_kernel(self.X, self.Z, params['lengthscale'], params['variance'])
            K_ZX = K_XZ.T
            k_ii = params['variance'].expand(self.X.shape[0])

            for _ in range(cavi_max_iter):
                m_u_prev = self.m_u.clone()
                self._cavi_step(params, K_ZZ, L_ZZ, K_XZ, K_ZX, k_ii)
                m_u_rel_change = torch.norm(self.m_u - m_u_prev) / torch.norm(m_u_prev)
                if m_u_rel_change < cavi_tol:
                    break

    def _m_step(self, optimizer):
        """Performs the gradient update for the hyperparameters (M-Step)."""
        optimizer.zero_grad()
        elbo = self._calculate_elbo()
        loss = -elbo
        loss.backward()
        optimizer.step()
        return elbo.item()

    def fit(self, max_iter_global=100, cavi_max_iter=10, cavi_tol=1e-5, lr=0.01):
        """Runs the full Variational EM algorithm."""
        parameters_to_optimize = [
            self.log_nu_f, self.log_nu_epsilon, self.log_sigma_sq,
            self.log_kernel_lengthscale, self.log_kernel_variance, self.Z
        ]
        optimizer = optim.Adam(parameters_to_optimize, lr=lr)
        
        elbo_history = []
        print("Starting Variational EM optimization...")
        for i in range(max_iter_global):
            self._e_step(cavi_max_iter=cavi_max_iter, cavi_tol=cavi_tol)
            elbo = self._m_step(optimizer)
            elbo_history.append(elbo)
            
            if (i + 1) % 50 == 0:
                print(f"EM Iteration {i+1}/{max_iter_global}, ELBO: {elbo:.4f}")
        
        print("\nOptimization finished.")
        return elbo_history

    def _calculate_elbo(self):
        params = self._get_hyperparams()
        K_ZZ = rbf_kernel(self.Z, self.Z, params['lengthscale'], params['variance']) + torch.eye(self.M, device=self.Z.device) * 1e-6
        L_ZZ = torch.linalg.cholesky(K_ZZ)
        K_XZ = rbf_kernel(self.X, self.Z, params['lengthscale'], params['variance'])
        K_ZX = K_XZ.T
        k_ii = params['variance'].expand(self.X.shape[0])
        
        L_S = torch.linalg.cholesky(self.S_u)

        KZZ_inv_m_u = torch.cholesky_solve(self.m_u, L_ZZ)
        KXZ_KZZ_inv = torch.cholesky_solve(K_ZX, L_ZZ).T

        # --- 1. Expected Log-Likelihood ---
        expected_log_lambda = torch.digamma(self.alpha_lambda) - torch.log(self.beta_lambda)
        expected_lambda = self.alpha_lambda / self.beta_lambda
        
        expected_f_mean = K_XZ @ KZZ_inv_m_u # Shape: (N, 1)
        
        if self.alpha_r > 1: expected_r_inv = self.beta_r / (self.alpha_r - 1.0)
        else: expected_r_inv = self.beta_r
        
        var_f_term1 = expected_r_inv * (k_ii - (KXZ_KZZ_inv * K_XZ).sum(dim=1))
        var_f_term2 = (KXZ_KZZ_inv @ self.S_u * KXZ_KZZ_inv).sum(dim=1)
        # ★★★ 修正点4: var_fの形状を(N, 1)に調整 ★★★
        var_f = (var_f_term1 + var_f_term2).unsqueeze(1)
        
        # yとexpected_f_meanは(N, 1)なのでsqueeze不要
        expected_sq_error = (self.y - expected_f_mean).pow(2) + var_f
        
        log_lik = 0.5 * torch.sum(expected_log_lambda - math.log(2 * math.pi) - torch.log(params['sigma_sq']) - \
                                  (expected_lambda / params['sigma_sq']) * expected_sq_error)

        # --- 2. KL Divergences ---
        p_alpha_r, p_beta_r = params['nu_f'] / 2.0, params['nu_f'] / 2.0
        kl_r = (self.alpha_r - p_alpha_r) * torch.digamma(self.alpha_r) - torch.lgamma(self.alpha_r) + torch.lgamma(p_alpha_r) + \
               p_alpha_r * (torch.log(self.beta_r) - torch.log(p_beta_r)) + self.alpha_r * (p_beta_r - self.beta_r) / self.beta_r
        
        expected_log_r = torch.digamma(self.alpha_r) - torch.log(self.beta_r)
        expected_r = self.alpha_r / self.beta_r
        
        logdet_S_u = 2 * torch.sum(torch.log(torch.diag(L_S)))
        logdet_K_ZZ = 2 * torch.sum(torch.log(torch.diag(L_ZZ)))
        
        trace_KZZinv_Su = torch.trace(torch.cholesky_solve(self.S_u, L_ZZ))
        m_T_KZZinv_m = self.m_u.T @ KZZ_inv_m_u 
        expected_u_quadratic = trace_KZZinv_Su + m_T_KZZinv_m
        
        kl_u = 0.5 * (
            -logdet_S_u - self.M * expected_log_r + logdet_K_ZZ + \
            expected_r * expected_u_quadratic - self.M
        ).squeeze()
        
        p_alpha_lambda, p_beta_lambda = params['nu_epsilon'] / 2.0, params['nu_epsilon'] / 2.0
        kl_lambda = torch.sum((self.alpha_lambda - p_alpha_lambda) * torch.digamma(self.alpha_lambda) - \
                    torch.lgamma(self.alpha_lambda) + torch.lgamma(p_alpha_lambda) + \
                    p_alpha_lambda * (torch.log(self.beta_lambda) - torch.log(p_beta_lambda)) + \
                    self.alpha_lambda * (p_beta_lambda - self.beta_lambda) / self.beta_lambda)

        return log_lik - kl_u - kl_r - kl_lambda

    def predict(self, X_test):
        """
        Makes predictions for new data X_test.
        """
        with torch.no_grad():
            params = self._get_hyperparams()
            
            K_ZZ = rbf_kernel(self.Z, self.Z, params['lengthscale'], params['variance']) + torch.eye(self.M, device=self.Z.device) * 1e-6
            L_ZZ = torch.linalg.cholesky(K_ZZ)
            K_star_Z = rbf_kernel(X_test, self.Z, params['lengthscale'], params['variance'])
            k_star_star = rbf_kernel(X_test, X_test, params['lengthscale'], params['variance']).diag()

            KZZ_inv_m_u = torch.cholesky_solve(self.m_u, L_ZZ)
            pred_mean = K_star_Z @ KZZ_inv_m_u

            K_star_Z_K_ZZ_inv = torch.cholesky_solve(K_star_Z.T, L_ZZ, upper=False).T
            
            gp_var = k_star_star - (K_star_Z_K_ZZ_inv * K_star_Z).sum(dim=1) + \
                     (K_star_Z_K_ZZ_inv @ self.S_u * K_star_Z_K_ZZ_inv).sum(dim=1)
            
            pred_nu = 2 * self.alpha_r
            pred_scale_sq = (gp_var * (self.beta_r / self.alpha_r)).unsqueeze(1)
            
            return pred_mean, pred_scale_sq, pred_nu
        




# if __name__ == '__main__':

#     import matplotlib.pyplot as plt
#     from scipy.stats import t
#     import math # 不要ですが、他のコードとの互換性のため残すこともあります

#     # --- グローバル設定 ---
#     torch.set_default_dtype(torch.float64)
#     torch.manual_seed(42)

#     # --- 1. 1次元データの生成 ---
#     N = 100
#     X_train = torch.linspace(-5, 5, N).unsqueeze(1)
#     y_true = torch.sin(X_train) * 2
    
#     # ノイズと外れ値の追加
#     noise = torch.randn(N, 1) * 0.1
#     t_dist_sample = torch.distributions.StudentT(df=2)
#     outlier_noise = t_dist_sample.sample((N, 1)) * 0.5
#     y_train = y_true + noise
#     outlier_indices = torch.randperm(N)[:10]
#     y_train[outlier_indices] += outlier_noise[outlier_indices] * 3
#     # 手動でさらに強い外れ値を追加
#     y_train[30] = -4.0
#     y_train[70] = 5.0

#     # --- 2. モデルのセットアップ ---
    
#     # 誘導点の数を指定
#     M = 20
    
#     # モデルのインスタンス化 (Z_initial の代わりに M を渡す)
#     model = SparseTPRTFullBatch(
#         X=X_train,
#         y=y_train,
#         M=M, # <- ここが変更点
#         nu_f=5.0,
#         nu_e=5.0,
#         kernel_lengthscale=0.5,
#         kernel_variance=2.0,
#         likelihood_sigma=0.5
#     )

#     # 最適化前の初期誘導点を保存しておく
#     initial_Z = model.Z.clone().detach()

#     # --- 3. モデルの学習とELBOの記録 ---
#     elbo_history = model.fit(max_iter_global=300, cavi_max_iter=5, lr=0.01)

#     # --- 4. 誘導点の位置を表示 ---
#     print("\n--- Initial Z ---")
#     print(initial_Z.squeeze().numpy())
#     print("\n--- Optimized Z ---")
#     print(model.Z.detach().squeeze().numpy())
#     print("-------------------")

#     # --- 5. 予測の実行 ---
#     X_test = torch.linspace(-6, 6, 200).unsqueeze(1)
#     pred_mean, pred_var, pred_nu = model.predict(X_test)

#     # --- 6. 回帰結果の可視化 ---
#     plt.figure(figsize=(12, 8))

#     # 予測分布（Student-t）の95%信頼区間を計算
#     pred_scale = torch.sqrt(pred_var.clamp(min=1e-9))
#     df = pred_nu.item()
#     # scipy.stats.tを使ってパーセント点関数(ppf)を計算
#     lower_quantile = t.ppf(0.025, df=df)
#     upper_quantile = t.ppf(0.975, df=df)
#     lower = pred_mean + lower_quantile * pred_scale
#     upper = pred_mean + upper_quantile * pred_scale

#     plt.fill_between(X_test.squeeze(), lower.squeeze(), upper.squeeze(), color='orange', alpha=0.3, label='95% Predictive Interval (Student-t)')
#     plt.plot(X_test, pred_mean, 'r-', lw=2, label='Predictive Mean')
#     plt.plot(X_train, y_train, 'kx', mew=2, label='Training Data (with outliers)')
#     plt.plot(model.Z.detach(), torch.full_like(model.Z.detach(), -5.5), 'r|', ms=20, mew=2, label='Optimized Inducing Points')

#     plt.title('Sparse TP Regression (with Z and Hyperparameter Optimization)', fontsize=16)
#     plt.legend(loc='upper left')
#     plt.grid(True)
#     plt.xlim(-6, 6)
#     plt.ylim(-6, 6)
    
#     # --- 7. ELBOの履歴を可視化 ---
#     plt.figure(figsize=(12, 6))
#     plt.plot(range(1, len(elbo_history) + 1), elbo_history, marker='.', linestyle='-')
#     plt.title('Sparse TP ELBO Convergence', fontsize=16)
#     plt.xlabel('EM Iteration', fontsize=12)
#     plt.ylabel('ELBO', fontsize=12)
#     plt.grid(True)
#     plt.tight_layout()

#     plt.show()