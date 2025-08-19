from .kernels import rbf_kernel
from .priors import GammaPrior, LogNormalPrior
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import StudentT, Gamma
from torch.utils.data import DataLoader, TensorDataset
import math
import copy
import matplotlib.pyplot as plt
# import tqdm # tqdmを削除
import pandas as pd
import logging
from pathlib import Path
from sklearn.cluster import KMeans


# Set default tensor type
torch.set_default_dtype(torch.float64)


def logpdf_st(x, mu, K, nu):
    """
    Calculates the log-pdf of a multivariate Student-t distribution.
    (This is part of the original model's logic and is preserved.)
    """
    d = mu.shape[0]
    K_stable = K + 1e-6 * torch.eye(d, device=K.device)
    L = torch.linalg.cholesky(K_stable)
    log_det_K = 2 * torch.sum(torch.log(torch.diag(L)))

    diff = x - mu.unsqueeze(-1)
    mahalanobis_dist = torch.sum(torch.square(torch.linalg.solve_triangular(L, diff, upper=False)), dim=0)

    term1 = torch.lgamma((nu + d) / 2) - torch.lgamma(nu / 2)
    term2 = -0.5 * log_det_K - (d / 2) * math.log((nu - 2) * math.pi)
    term3 = -((nu + d) / 2) * torch.log(1 + mahalanobis_dist / (nu - 2))

    return term3.squeeze() + term1 + term2

class SparseTPRTMiniBatch_Xu:
    """
    Implementation of Sparse Variational Student-t Process (SVTP) from Xu et al. (2021).
    This version is aligned with the structure of the other SparseTPRTMiniBatch classes
    and allows switching between UB and MC methods for the KL term.
    THE INTERNAL CALCULATION LOGIC IS PRESERVED FROM THE ORIGINAL WORKING SCRIPT.
    """
    def __init__(self, X, y, M, nu_f=3.0, nu_e=3.0, kernel_lengthscale=1.0, kernel_variance=1.0, likelihood_sigma=0.1, device=None):
        if device is None:
            self.device = X.device if isinstance(X, torch.Tensor) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.X_full = X.to(self.device)
        self.y_full = y.view(-1, 1).to(self.device)
        
        self.N, self.D = self.X_full.shape
        self.M = M

        # --- Trainable Parameters ---
        Z_initial = self._initialize_inducing_points()
        self.Z = nn.Parameter(Z_initial)
        
        self.log_kernel_lengthscale = nn.Parameter(torch.log(torch.tensor(kernel_lengthscale, dtype=self.X_full.dtype, device=self.device)))
        self.log_kernel_variance = nn.Parameter(torch.log(torch.tensor(kernel_variance, dtype=self.X_full.dtype, device=self.device)))
        self.log_likelihood_sigma = nn.Parameter(torch.log(torch.tensor(likelihood_sigma, dtype=self.X_full.dtype, device=self.device)))
        self.log_nu_f_minus_2 = nn.Parameter(torch.log(torch.tensor(nu_f - 2.0, dtype=self.X_full.dtype, device=self.device)))
        self.log_nu_e_minus_2 = nn.Parameter(torch.log(torch.tensor(nu_e - 2.0, dtype=self.X_full.dtype, device=self.device)))

        # --- Initialize Priors for Hyperparameters ---
        self.lengthscale_prior = GammaPrior(3.0, 6.0)
        self.variance_prior = GammaPrior(2.0, 0.15)
        self.sigma_sq_prior = GammaPrior(1.1, 0.05)
        self.nu_prior = LogNormalPrior(loc=1.0, scale=1.0)

        # --- Initialize Variational Parameters ---
        self.mu_q = nn.Parameter(torch.zeros(self.M, dtype=self.X_full.dtype, device=self.device))
        self.S_chol_q = nn.Parameter(torch.eye(self.M, dtype=self.X_full.dtype, device=self.device))

    def _initialize_inducing_points(self):
        min_bounds = self.X_full.min(dim=0).values
        max_bounds = self.X_full.max(dim=0).values
        sobol_engine = torch.quasirandom.SobolEngine(dimension=self.D, scramble=True, seed=0)
        sobol_points_unit = sobol_engine.draw(self.M).to(dtype=self.X_full.dtype, device=self.device)
        return min_bounds + sobol_points_unit * (max_bounds - min_bounds)

    def _get_hyperparams(self):
        """Helper to get positive parameters from their transformed storage."""
        nu_f = torch.exp(self.log_nu_f_minus_2) + 2.0
        nu_q = nu_f + self.N
        nu_e = torch.exp(self.log_nu_e_minus_2) + 2.0
        
        return {
            "nu_f": nu_f, "nu_q": nu_q, "nu_e": nu_e,
            "likelihood_sigma": torch.exp(self.log_likelihood_sigma),
            "lengthscale": torch.exp(self.log_kernel_lengthscale),
            "variance": torch.exp(self.log_kernel_variance),
            "sigma_sq": torch.exp(self.log_likelihood_sigma)**2
        }

    # === CORE LOGIC METHODS (PRESERVED FROM ORIGINAL) ===
    def _sample_q_u(self, nu_q, num_samples=1):
        r_inv_dist = Gamma(nu_q / 2, 0.5)
        r_inv = r_inv_dist.sample((num_samples,))
        r = 1.0 / r_inv
        eps = torch.randn(self.M, num_samples, device=self.device)
        u_samples = self.mu_q.unsqueeze(1) + self.S_chol_q @ (eps * torch.sqrt(r).T)
        return u_samples

    def _kl_divergence(self, K_mm, S_q, nu_prior, nu_q, method='UB', num_samples_kl=10):
        """Calculates KL(q(u) || p(u)) using either Upper Bound (UB) or Monte Carlo (MC)."""
        if method == 'MC':
            u_samples = self._sample_q_u(nu_q, num_samples=num_samples_kl)
            log_q_u = logpdf_st(u_samples, self.mu_q, S_q, nu_q)
            log_p_u = logpdf_st(u_samples, torch.zeros(self.M, device=self.device), K_mm, nu_prior)
            return torch.mean(log_q_u - log_p_u)
        
        elif method == 'UB':
            K_mm_inv = torch.inverse(K_mm)
            l1 = torch.digamma((nu_q + self.M) / 2) - torch.digamma(nu_q / 2)
            tr_term = torch.trace(K_mm_inv @ S_q)
            mean_term = self.mu_q.T @ K_mm_inv @ self.mu_q
            l2_star = torch.log(1 + (tr_term + mean_term) / (nu_prior - 2))
            sign_S, logdet_S = torch.linalg.slogdet(S_q)
            if sign_S.item() <= 0: return float('inf')
            sign_K, logdet_K = torch.linalg.slogdet(K_mm)
            if sign_K.item() <= 0: return float('inf')
            log_nu_diff = torch.log(nu_prior - 2) - torch.log(nu_q - 2)
            C = 0.5 * (logdet_S - logdet_K + self.M * log_nu_diff)
            kl_approx = C - ((nu_q + self.M) / 2) * l1 + ((nu_prior + self.M) / 2) * l2_star
            return kl_approx
        else:
            raise ValueError("kl_method must be 'UB' or 'MC'")

    def _expected_log_likelihood(self, X_batch, y_batch, K_mm_inv, K_nm, nu_q, nu_lik, sigma_n, num_samples=1):
        u_samples = self._sample_q_u(nu_q, num_samples)
        f_est_samples = K_nm @ K_mm_inv @ u_samples
        dist = StudentT(df=nu_lik)
        log_p_y_given_f = dist.log_prob((y_batch.unsqueeze(1) - f_est_samples) / sigma_n) - torch.log(sigma_n)
        return torch.mean(torch.sum(log_p_y_given_f, dim=0))

    def _calculate_elbo(self, X_batch, y_batch, kl_method, num_samples_elbo, num_samples_kl):
        params = self._get_hyperparams()
        S_q = self.S_chol_q @ self.S_chol_q.T + 1e-6 * torch.eye(self.M, device=self.device)
        K_mm = rbf_kernel(self.Z, self.Z, params['lengthscale'], params['variance']) + 1e-6 * torch.eye(self.M, device=self.device)
        K_nm = rbf_kernel(X_batch, self.Z, params['lengthscale'], params['variance'])
        K_mm_inv = torch.inverse(K_mm)

        kl = self._kl_divergence(K_mm, S_q, params['nu_f'], params['nu_q'], method=kl_method, num_samples_kl=num_samples_kl)
        exp_log_lik = self._expected_log_likelihood(X_batch, y_batch.squeeze(), K_mm_inv, K_nm, params['nu_q'], params['nu_e'], params['likelihood_sigma'], num_samples_elbo)
        
        scale = self.N / X_batch.shape[0]

        # Calculate log prior for hyperparameters
        log_prior = 0.0
        log_prior += self.lengthscale_prior.log_prob(params['lengthscale'])
        log_prior += self.variance_prior.log_prob(params['variance'])
        log_prior += self.sigma_sq_prior.log_prob(params['sigma_sq'])
        log_prior += self.nu_prior.log_prob(params['nu_f'])
        log_prior += self.nu_prior.log_prob(params['nu_e'])

        return scale * exp_log_lik - kl + log_prior

    def get_trainable_parameters(self):
        return [
            self.Z, self.log_kernel_lengthscale, self.log_kernel_variance,
            self.log_likelihood_sigma, self.log_nu_f_minus_2, self.log_nu_e_minus_2,
            self.mu_q, self.S_chol_q
        ]

    def fit(self, epochs=100, batch_size=64, lr=0.01, kl_method='UB', num_samples_elbo=1, num_samples_kl=10):
        """Runs the full optimization algorithm using mini-batches."""
        optimizer = optim.Adam(self.get_trainable_parameters(), lr=lr)
        
        dataset = TensorDataset(self.X_full, self.y_full)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        elbo_history = []
        print(f"🚀 Starting training with method: {kl_method}")
        for epoch in range(epochs):
            final_elbo = 0.0
            for i, (X_batch, y_batch) in enumerate(dataloader):
                optimizer.zero_grad()
                elbo = self._calculate_elbo(X_batch, y_batch, kl_method, num_samples_elbo, num_samples_kl)
                loss = -elbo
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    loss.backward()
                    optimizer.step()
                
                final_elbo = elbo.item()
                elbo_history.append(final_elbo)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Final Batch ELBO: {final_elbo:.2f}")
        
        print(f"✓ Training complete for method: {kl_method}")
        return elbo_history

    def predict(self, X_test, num_samples=500):
        """Makes predictions for new data X_test."""
        # ===== 修正箇所 (ここから) =====
        X_test = X_test.to(self.device)
        # ===== 修正箇所 (ここまで) =====
        with torch.no_grad():
            params = self._get_hyperparams()
            
            K_mm = rbf_kernel(self.Z, self.Z, params['lengthscale'], params['variance']) + 1e-6 * torch.eye(self.M, device=self.device)
            K_star_m = rbf_kernel(X_test, self.Z, params['lengthscale'], params['variance'])
            K_star_star_diag = torch.diag(rbf_kernel(X_test, X_test, params['lengthscale'], params['variance']))
            K_mm_inv = torch.inverse(K_mm)
            
            u_samples = self._sample_q_u(params['nu_q'], num_samples=num_samples)
            f_star_mean_samples = K_star_m @ K_mm_inv @ u_samples
            
            beta = torch.sum((u_samples.T @ K_mm_inv) * u_samples.T, dim=1)
            scale_factor = (params['nu_f'] + beta - 2) / (params['nu_f'] + self.M - 2)
            
            K_star_m_K_inv = K_star_m @ K_mm_inv
            var_f_cond_u_diag = K_star_star_diag - torch.sum(K_star_m_K_inv * K_star_m, dim=1)
            f_star_var_samples = var_f_cond_u_diag.clamp(min=1e-6).unsqueeze(1) * scale_factor.unsqueeze(0)
            
            mu_pred = torch.mean(f_star_mean_samples, dim=1)
            var_f_total = torch.mean(f_star_var_samples, dim=1) + torch.var(f_star_mean_samples, dim=1)
            var_likelihood = (params['likelihood_sigma']**2 * params['nu_e']) / (params['nu_e'] - 2)
            var_pred = var_f_total + var_likelihood
            
            pred_nu = torch.tensor(float('inf'), device=self.device)
            return mu_pred.unsqueeze(1), var_pred.unsqueeze(1), pred_nu
            
    def evaluate_model(self, epochs=100, batch_size=64, lr=0.01, kl_method='UB', 
                       num_samples_elbo=1, num_samples_kl=10,
                       X_test=None, y_test=None, eval_interval=10, result_path=None):
        """
        Trains the model while periodically evaluating on test data and saving results.
        """
        optimizer = optim.Adam(self.get_trainable_parameters(), lr=lr)
        dataset = TensorDataset(self.X_full, self.y_full)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        can_evaluate = X_test is not None and y_test is not None and result_path is not None
        if can_evaluate:
            result_path = Path(result_path)
            if not result_path.exists():
                result_path.parent.mkdir(parents=True, exist_ok=True)
                with open(result_path, 'w') as f:
                    f.write("epoch,rmse,elbo\n")

        print(f"🚀 Starting training with evaluation (method: {kl_method})...")
        final_elbo = 0.0

        for epoch in range(epochs):
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                elbo = self._calculate_elbo(X_batch, y_batch, kl_method, num_samples_elbo, num_samples_kl)
                loss = -elbo
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    loss.backward()
                    optimizer.step()
                
                final_elbo = elbo.item()
            
            # Simplified progress printing
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, ELBO: {final_elbo:.2f}")

            if can_evaluate and (epoch + 1) % eval_interval == 0:
                with torch.no_grad():
                    pred_mean, _, _ = self.predict(X_test)
                    # Ensure tensors are on the same device for comparison
                    y_test_device = y_test.to(pred_mean.device)
                    rmse = torch.sqrt(torch.mean((y_test_device.view(-1) - pred_mean.view(-1))**2)).item()
                
                log_msg = f"Epoch {epoch+1}/{epochs}, ELBO: {final_elbo:.4f}, Test RMSE: {rmse:.4f}"
                logging.info(log_msg)
                print(log_msg) # Also print to console
                
                with open(result_path, 'a') as f:
                    f.write(f"{epoch+1},{rmse},{final_elbo}\n")

        # Perform final evaluation if the last epoch was not an eval interval
        if can_evaluate and epochs > 0 and epochs % eval_interval != 0:
             with torch.no_grad():
                pred_mean, _, _ = self.predict(X_test)
                y_test_device = y_test.to(pred_mean.device)
                rmse = torch.sqrt(torch.mean((y_test_device.view(-1) - pred_mean.view(-1))**2)).item()
                print(f"Final Test RMSE: {rmse:.4f}")
             with open(result_path, 'a') as f:
                f.write(f"{epochs},{rmse},{final_elbo}\n")
        
        print(f"✓ Training and evaluation complete for method: {kl_method}")