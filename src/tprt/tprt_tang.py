from .priors import GammaPrior, LogNormalPrior
from .kernels import rbf_kernel
import torch
import torch.nn as nn
import tqdm


torch.set_default_dtype(torch.float64)

class TPRTFullBatch_Tang(nn.Module):
    # +++ 2. __init__ を修正 +++
    def __init__(self, X, y, kernel=rbf_kernel, nu_f=2.1, nu_e=2.1,
                 kernel_lengthscale=1.0, kernel_variance=1.0, likelihood_sigma=0.1):
        super().__init__()

        self.register_buffer('X', X)
        self.register_buffer('y', y.squeeze())
        self.N, self.D = X.shape
        
        # カーネル関数をインスタンス変数として保持
        self.kernel = kernel

        self.log_kernel_lengthscale = nn.Parameter(torch.log(torch.tensor(kernel_lengthscale)))
        self.log_kernel_variance = nn.Parameter(torch.log(torch.tensor(kernel_variance)))
        self.log_likelihood_sigma = nn.Parameter(torch.log(torch.tensor(likelihood_sigma)))
        self.log_nu_f = nn.Parameter(torch.log(torch.tensor(nu_f)))
        self.log_nu_e = nn.Parameter(torch.log(torch.tensor(nu_e)))

        self.register_buffer('f_hat', torch.zeros(self.N))

        self.lengthscale_prior = GammaPrior(3.0, 6.0)
        self.variance_prior = GammaPrior(2.0, 0.15)
        self.sigma_sq_prior = GammaPrior(1.1, 0.05) 
        self.nu_prior = LogNormalPrior(loc=1.0, scale=1.0)

    def _get_hyperparams(self):
        lengthscale = torch.exp(self.log_kernel_lengthscale)
        variance = torch.exp(self.log_kernel_variance)
        sigma = torch.exp(self.log_likelihood_sigma)
        nu_f = torch.exp(self.log_nu_f)
        nu_e = torch.exp(self.log_nu_e)
        return lengthscale, variance, sigma, nu_f, nu_e

    def _calculate_ln_Q(self, f, K_inv, hyperparams):
        _, _, sigma, nu_f, nu_e = hyperparams
        log_lik_term = ((nu_e + 1) / 2) * torch.log(1 + (1 / nu_e) * ((self.y - f) / sigma).pow(2))
        fT_K_inv_f = f @ K_inv @ f
        log_prior_term = ((nu_f + self.N) / 2) * torch.log(1 + (1 / nu_f) * fT_K_inv_f)
        return torch.sum(log_lik_term) + log_prior_term

    def _find_f_hat(self, K_inv_detached, hyperparams_detached, max_iter=10, tol=1e-5):
        f = nn.Parameter(self.f_hat.clone())
        optimizer = torch.optim.LBFGS([f], lr=0.5, max_iter=20, line_search_fn="strong_wolfe")
        def closure():
            optimizer.zero_grad()
            loss = self._calculate_ln_Q(f, K_inv_detached, hyperparams_detached)
            loss.backward()
            return loss
        for _ in range(max_iter):
            f_old = f.clone().detach()
            optimizer.step(closure)
            if torch.norm(f.detach() - f_old) < tol: break
        return f.detach()

    def _calculate_hessian_A_inv(self, f_hat, K_inv, hyperparams):
        _, _, sigma, nu_f, nu_e = hyperparams
        err = self.y - f_hat
        err2 = err.pow(2)
        sigma2 = sigma.pow(2)
        W_diag = (nu_e + 1) * (nu_e * sigma2 - err2) / (nu_e * sigma2 + err2).pow(2)
        W = torch.diag(W_diag)
        fT_K_inv_f = f_hat @ K_inv @ f_hat
        hess_prior_den = (nu_f + fT_K_inv_f).pow(2)
        K_inv_f = K_inv @ f_hat
        fT_K_inv = K_inv_f.T
        hess_prior_num = (nu_f + self.N) * (K_inv * (nu_f + fT_K_inv_f) - 2 * torch.outer(K_inv_f, fT_K_inv))
        hess_prior = hess_prior_num / hess_prior_den
        A_inv = W + hess_prior
        jitter = torch.eye(self.N, device=self.X.device) * 1e-6
        return A_inv + jitter

    def _calculate_neg_log_marginal_likelihood(self):
        hyperparams = self._get_hyperparams()
        f_hat = self.f_hat
        lengthscale, variance, sigma, nu_f, nu_e = hyperparams
        
        # +++ 3. self.kernel をハイパラ付きで呼び出し +++
        K = self.kernel(self.X, self.X, lengthscale, variance) + torch.eye(self.N) * 1e-6
        try: L = torch.linalg.cholesky(K)
        except torch.linalg.LinAlgError: return torch.tensor(float('inf'))
        K_inv = torch.cholesky_inverse(L)

        ln_Q_at_f_hat = self._calculate_ln_Q(f_hat, K_inv, hyperparams)
        A_inv = self._calculate_hessian_A_inv(f_hat, K_inv, hyperparams)

        log_det_K = 2 * torch.sum(torch.log(torch.diag(L)))
        sign, log_det_A_inv = torch.linalg.slogdet(A_inv)
        if sign.item() <= 0: return torch.tensor(float('inf'))

        log_det_B = log_det_K + log_det_A_inv

        c_nu_f_term = torch.lgamma((nu_f + self.N)/2) - torch.lgamma(nu_f/2) - (self.N/2)*torch.log(nu_f)
        c_nu_e_term = self.N * (torch.lgamma((nu_e + 1)/2) - torch.lgamma(nu_e/2) - 0.5*torch.log(nu_e))
        c_sigma_term = -self.N * torch.log(sigma)

        neg_log_marginal_lik = ln_Q_at_f_hat - 0.5 * log_det_B - (c_nu_f_term + c_nu_e_term + c_sigma_term)
        return neg_log_marginal_lik

    def _calculate_neg_log_prior_plob(self):
        lengthscale, variance, sigma, nu_f, nu_e = self._get_hyperparams()
        log_prior_lengthscale = self.lengthscale_prior.log_prob(lengthscale)
        log_prior_variance = self.variance_prior.log_prob(variance)
        log_prior_sigma = self.sigma_sq_prior.log_prob(sigma.pow(2))
        log_prior_nu_f = self.nu_prior.log_prob(nu_f)
        log_prior_nu_e = self.nu_prior.log_prob(nu_e)
        neg_log_prior = -(log_prior_lengthscale.sum() + log_prior_variance + log_prior_sigma + log_prior_nu_f + log_prior_nu_e)
        return neg_log_prior
    
    def _e_step(self, mode_finding_iter=10):
        with torch.no_grad():
            hyperparams_detached = self._get_hyperparams()
            lengthscale_d, variance_d, _, _, _ = hyperparams_detached
            K_detached = self.kernel(self.X, self.X, lengthscale_d, variance_d) + torch.eye(self.N) * 1e-6
            L_detached = torch.linalg.cholesky(K_detached)
            K_inv_detached = torch.cholesky_inverse(L_detached)
        
        f_hat = self._find_f_hat(K_inv_detached, hyperparams_detached, max_iter=mode_finding_iter)
        self.f_hat.copy_(f_hat)

    def _m_step(self, optimizer):
        optimizer.zero_grad()
        nll = self._calculate_neg_log_marginal_likelihood()
        neg_log_prior = self._calculate_neg_log_prior_plob()
        loss = nll + neg_log_prior
        if torch.isinf(loss) or torch.isnan(loss):
            print(f"Warning: Loss is {loss.item()} during M-step, skipping update.")
            return loss.item()
        loss.backward()
        optimizer.step()
        return -nll.item()

    def fit(self, max_iter_global=100, mode_finding_iter=10, lr=0.01):
        print("Starting EM-like optimization...")
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        pbar = tqdm.trange(max_iter_global)
        for _ in pbar:
            self._e_step(mode_finding_iter=mode_finding_iter)
            ll = self._m_step(optimizer)
            pbar.set_description(f"Approx. Log Likelihood: {ll:.2f}")

    def predict(self, X_test):
        with torch.no_grad():
            hyperparams = self._get_hyperparams()
            lengthscale, variance, _, _, _ = hyperparams
            f_hat = self.f_hat
            
            K = self.kernel(self.X, self.X, lengthscale, variance) + torch.eye(self.N) * 1e-6
            L = torch.linalg.cholesky(K)
            K_inv = torch.cholesky_inverse(L)
            K_star_x = self.kernel(X_test, self.X, lengthscale, variance)
            K_star_star_diag = self.kernel(X_test, X_test, lengthscale, variance).diag()

            pred_mean = K_star_x @ K_inv @ f_hat
            A_inv = self._calculate_hessian_A_inv(f_hat, K_inv, hyperparams)
            try:
                L_A_inv = torch.linalg.cholesky(A_inv)
                A = torch.cholesky_inverse(L_A_inv)
            except torch.linalg.LinAlgError:
                print("Warning: Hessian not positive definite during prediction. Using pseudo-inverse.")
                A = torch.linalg.pinv(A_inv)
            tmp = torch.cholesky_solve(K_star_x.T, L).T
            posterior_uncertainty = (tmp @ A @ tmp.T).diag()
            pred_var = K_star_star_diag - (tmp @ K_star_x.T).diag() + posterior_uncertainty
            pred_var = pred_var.clamp(min=1e-9)
            return pred_mean.unsqueeze(1), pred_var.unsqueeze(1)
        
    def evaluate_model(self, max_iter_global=100, mode_finding_iter=10, lr=0.01,
                       X_test=None, y_test=None, eval_interval=10,
                       result_path=None):
        """
        Fits the model using EM-like optimization and periodically evaluates/saves performance on test data.
        This method is resilient to timeouts by appending results to a file.

        Args:
            max_iter_global (int): Total number of EM iterations.
            mode_finding_iter (int): Max iterations for the inner mode-finding loop.
            lr (float): Learning rate for the Adam optimizer.
            X_test (torch.Tensor, optional): Test inputs for evaluation.
            y_test (torch.Tensor, optional): Test targets for evaluation.
            eval_interval (int): The interval (in iterations) at which to evaluate.
            result_path (pathlib.Path, optional): Path to save the intermediate results CSV.
                                                  If provided, results are appended.
        """
        # --- Optimizer Setup ---
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # --- Evaluation Setup ---
        can_evaluate = X_test is not None and y_test is not None and result_path is not None
        if can_evaluate:
            # Create parent directory if it doesn't exist
            result_path.parent.mkdir(parents=True, exist_ok=True)
            # Write header if the file is new
            if not result_path.exists():
                with open(result_path, 'w') as f:
                    f.write("iteration,rmse,ll\n")
        # ------------------------

        print("Starting EM-like optimization with evaluation...")
        pbar = tqdm.trange(max_iter_global)
        for i in pbar:
            # 1. Training Step
            self._e_step(mode_finding_iter=mode_finding_iter)
            ll = self._m_step(optimizer)
            pbar.set_description(f"Approx. Log Likelihood: {ll:.2f}")

            # 2. Periodic Evaluation and Saving
            if can_evaluate and (i + 1) % eval_interval == 0:
                with torch.no_grad():
                    pred_mean, _ = self.predict(X_test)
                    rmse = torch.sqrt(torch.mean((y_test.squeeze() - pred_mean.squeeze())**2)).item()
                
                # Append the result to the file
                with open(result_path, 'a') as f:
                    f.write(f"{i + 1},{rmse},{ll}\n")
        
        # 3. Final Evaluation (if the last iteration was not an evaluation point)
        if can_evaluate and max_iter_global % eval_interval != 0:
             with torch.no_grad():
                pred_mean, _ = self.predict(X_test)
                rmse = torch.sqrt(torch.mean((y_test.squeeze() - pred_mean.squeeze())**2)).item()
             with open(result_path, 'a') as f:
                f.write(f"{max_iter_global},{rmse},{ll}\n")



# if __name__ == '__main__':
#     N = 60
#     X_train = torch.linspace(-5, 5, N).unsqueeze(1)
#     y_true = torch.sin(X_train) * 2
    
#     torch.manual_seed(42)
#     noise = torch.randn(N, 1) * 0.2
#     y_train = y_true + noise
    
#     outlier_indices = [5, 15, 30, 45, 55]
#     y_train[outlier_indices] = torch.tensor([-3.0, 4.0, -4.5, 5.0, -2.5]).unsqueeze(1)

#     # モデルのインスタンス化時にカーネル関数を渡す
#     model = TPRT_Laplace(X=X_train, y=y_train, kernel=rbf_kernel, nu_f=2.1, nu_e=2.1,
#                          kernel_lengthscale=1.0, kernel_variance=1.0, likelihood_sigma=0.5)
    
#     model.fit(max_iter=100, mode_finding_iter=10, lr=0.01)

#     X_test = torch.linspace(-6, 6, 200).unsqueeze(1)
#     pred_mean, pred_var = model.predict(X_test)

#     plt.figure(figsize=(12, 8))
#     pred_std = torch.sqrt(pred_var)
#     lower_ci = pred_mean - 1.96 * pred_std
#     upper_ci = pred_mean + 1.96 * pred_std

#     plt.fill_between(X_test.squeeze(), lower_ci.squeeze(), upper_ci.squeeze(), color='orange', alpha=0.3, label='95% Confidence Interval')
#     plt.plot(X_test, pred_mean, 'r-', lw=2, label='Predictive Mean')
#     plt.plot(X_train, y_train, 'kx', mew=2, label='Training Data (with outliers)')
#     plt.plot(X_test, torch.sin(X_test) * 2, 'b--', label='True Function')
#     plt.title('TPRT with Laplace Approximation (EM-like training)', fontsize=16)
#     plt.legend(loc='upper left')
#     plt.grid(True)
#     plt.xlim(-6, 6)
#     plt.ylim(-6, 6)
    
#     l, v, s, n1, n2 = model._get_hyperparams()
#     print("\n--- Learned Hyperparameters ---")
#     print(f"Kernel Lengthscale: {l.item():.3f}")
#     print(f"Kernel Variance: {v.item():.3f}")
#     print(f"Likelihood Sigma: {s.item():.3f}")
#     print(f"Prior DoF (nu_f): {n1.item():.3f}")
#     print(f"Likelihood DoF (nu_e): {n2.item():.3f}")
    
#     plt.show()