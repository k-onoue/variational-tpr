from .kernels import rbf_kernel, matern52_kernel
from .priors import GammaPrior, LogNormalPrior

import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import DataLoader, TensorDataset
import logging
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans



class SparseTPRTMiniBatch(nn.Module):
    """
    This version implements a batch update for global variational parameters
    at the end of each epoch, as requested.
    """
    def __init__(self, X, y, M, nu_f=2.1, nu_e=2.1,
                 kernel_lengthscale=None, kernel_variance=1.0,
                 likelihood_sigma=1.0, device=None):
        super().__init__()

        if device is None:
            self.device = X.device if isinstance(X, torch.Tensor) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.register_buffer('X_full', X.to(self.device))
        self.register_buffer('y_full', y.view(-1, 1).to(self.device))

        if self.X_full.ndim == 1: self.X_full = self.X_full.unsqueeze(1)
        if self.y_full.ndim == 1: self.y_full = self.y_full.unsqueeze(1)

        self.N, self.D = self.X_full.shape
        self.M = M
        dtype = self.X_full.dtype

        if kernel_lengthscale is None:
            kernel_lengthscale = torch.ones(self.D, dtype=dtype)
        else:
            kernel_lengthscale = torch.as_tensor(kernel_lengthscale, dtype=dtype)
        if kernel_lengthscale.ndim == 0:
            kernel_lengthscale = kernel_lengthscale.repeat(self.D)
        if kernel_lengthscale.shape[0] != self.D:
            raise ValueError(f"lengthscale must be a scalar or a vector of length D={self.D}")

        self.log_kernel_lengthscale = nn.Parameter(torch.log(kernel_lengthscale))
        self.log_kernel_variance = nn.Parameter(torch.log(torch.tensor(kernel_variance, dtype=dtype)))
        self.log_nu_f = nn.Parameter(torch.log(torch.tensor(nu_f, dtype=dtype)))
        self.log_nu_epsilon = nn.Parameter(torch.log(torch.tensor(nu_e, dtype=dtype)))
        self.log_sigma_sq = nn.Parameter(torch.log(torch.tensor(likelihood_sigma**2, dtype=dtype)))
        self.Z = nn.Parameter(self._initialize_inducing_points())

        self.lengthscale_prior = GammaPrior(3.0, 6.0)
        self.variance_prior = GammaPrior(2.0, 0.15)
        self.sigma_sq_prior = GammaPrior(1.1, 0.05)
        self.nu_prior = LogNormalPrior(loc=1.0, scale=1.0)

        self.register_buffer('m_u', torch.zeros(self.M, 1, dtype=dtype))
        self.register_buffer('S_u', torch.eye(self.M, dtype=dtype))
        self.register_buffer('alpha_r', torch.tensor(1.0, dtype=dtype))
        self.register_buffer('beta_r', torch.tensor(1.0, dtype=dtype))

        self.to(self.device)

        self.kernel = rbf_kernel

    def _initialize_inducing_points(self):
        if self.N >= self.M:
            X_np = self.X_full.cpu().numpy()
            kmeans = KMeans(n_clusters=self.M, random_state=0, n_init='auto').fit(X_np)
            Z_init = torch.from_numpy(kmeans.cluster_centers_)
        else:
            indices = np.random.choice(self.N, self.M, replace=True)
            Z_init = self.X_full[indices].clone()
        return Z_init.to(dtype=self.X_full.dtype, device=self.device)

    def _get_hyperparams(self):
        return {
            "nu_f": torch.exp(self.log_nu_f), "nu_epsilon": torch.exp(self.log_nu_epsilon),
            "sigma_sq": torch.exp(self.log_sigma_sq), "lengthscale": torch.exp(self.log_kernel_lengthscale),
            "variance": torch.exp(self.log_kernel_variance)
        }

    def _compute_common_terms(self, X_batch, params):
        K_ZZ = self.kernel(self.Z, self.Z, params['lengthscale'], params['variance']) + torch.eye(self.M, device=self.Z.device) * 1e-6
        L_ZZ = torch.linalg.cholesky(K_ZZ)
        K_XZ_batch = self.kernel(X_batch, self.Z, params['lengthscale'], params['variance'])
        KXZ_KZZ_inv = torch.linalg.solve(K_ZZ, K_XZ_batch.T).T
        return K_ZZ, L_ZZ, K_XZ_batch, KXZ_KZZ_inv

    def _e_step(self, X_batch, y_batch, common_terms):
        with torch.no_grad():
            params = self._get_hyperparams()
            K_ZZ, L_ZZ, K_XZ_batch, KXZ_KZZ_inv = common_terms

            k_ii_batch = params['variance'].expand(X_batch.shape[0])

            expected_r_inv = self.beta_r / (self.alpha_r - 1.0) if self.alpha_r > 1 else self.beta_r

            KZZ_inv_m_u = torch.cholesky_solve(self.m_u, L_ZZ)
            expected_f_mean = K_XZ_batch @ KZZ_inv_m_u

            var_f_term1 = expected_r_inv * (k_ii_batch - (KXZ_KZZ_inv * K_XZ_batch).sum(dim=1))
            var_f_term2 = (KXZ_KZZ_inv @ self.S_u @ KXZ_KZZ_inv.T).diag()
            var_f = (var_f_term1 + var_f_term2).unsqueeze(1)

            expected_sq_error = (y_batch - expected_f_mean).pow(2) + var_f
            alpha_lambda_local = params['nu_epsilon'] / 2.0 + 0.5
            beta_lambda_local = params['nu_epsilon'] / 2.0 + (0.5 / params['sigma_sq']) * expected_sq_error

            expected_r = self.alpha_r / self.beta_r
            expected_lambda = alpha_lambda_local.squeeze() / beta_lambda_local.squeeze()
            c = expected_lambda / params['sigma_sq']

            B = (K_XZ_batch.T * c) @ K_XZ_batch
            precision_inner = expected_r * K_ZZ + B
            L_precision_inner = torch.linalg.cholesky(precision_inner + torch.eye(self.M, device=K_ZZ.device) * 1e-6)

            S_u_local = K_ZZ @ torch.cholesky_solve(K_ZZ, L_precision_inner)
            y_term = K_XZ_batch.T @ (y_batch.squeeze() * c)
            m_u_local = K_ZZ @ torch.cholesky_solve(y_term.unsqueeze(1), L_precision_inner)

            trace_term = torch.trace(torch.cholesky_solve(S_u_local, L_ZZ))
            KZZ_inv_m_u_local = torch.cholesky_solve(m_u_local, L_ZZ)
            mean_term = m_u_local.T @ KZZ_inv_m_u_local
            expected_u_quadratic = trace_term + mean_term

            alpha_r_local = params['nu_f'] / 2.0 + self.M / 2.0
            beta_r_local_val = params['nu_f'] / 2.0 + 0.5 * expected_u_quadratic.squeeze()

            # Convert local variational parameters to natural parameters to be returned
            S_local_inv = torch.linalg.inv(S_u_local + torch.eye(self.M, dtype=S_u_local.dtype, device=S_u_local.device) * 1e-6)
            eta_u1_local = S_local_inv @ m_u_local
            eta_u2_local = -0.5 * S_local_inv
            eta_r1_local = alpha_r_local - 1.0
            eta_r2_local = -beta_r_local_val

            # Return local natural parameters and local lambda parameters
            return eta_u1_local, eta_u2_local, eta_r1_local, eta_r2_local, alpha_lambda_local, beta_lambda_local

    def _m_step_and_elbo(self, optimizer, X_batch, y_batch, alpha_lambda_batch, beta_lambda_batch, common_terms):
        optimizer.zero_grad()
        elbo = self._calculate_elbo(X_batch, y_batch, alpha_lambda_batch, beta_lambda_batch, common_terms)
        loss = -elbo
        loss.backward()
        # Add gradient clipping for stability
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
        optimizer.step()
        return elbo.item()

    def _calculate_elbo(self, X_batch, y_batch, alpha_lambda_batch, beta_lambda_batch, common_terms):
        B = X_batch.shape[0]
        params = self._get_hyperparams()

        K_ZZ, L_ZZ, K_XZ_batch, KXZ_KZZ_inv = common_terms
        k_ii_batch = params['variance'].expand(B)

        expected_log_lambda = torch.digamma(alpha_lambda_batch) - torch.log(beta_lambda_batch)
        expected_lambda = alpha_lambda_batch / beta_lambda_batch

        KZZ_inv_m_u = torch.cholesky_solve(self.m_u, L_ZZ)
        expected_f_mean = K_XZ_batch @ KZZ_inv_m_u

        expected_r_inv = self.beta_r / (self.alpha_r - 1.0) if self.alpha_r > 1 else self.beta_r

        var_f_term1 = expected_r_inv * (k_ii_batch - (KXZ_KZZ_inv * K_XZ_batch).sum(dim=1))
        var_f_term2 = (KXZ_KZZ_inv @ self.S_u @ KXZ_KZZ_inv.T).diag()
        var_f = (var_f_term1 + var_f_term2).unsqueeze(1)

        expected_sq_error = (y_batch - expected_f_mean).pow(2) + var_f
        log_lik_batch = 0.5 * torch.sum(expected_log_lambda - math.log(2 * math.pi) - torch.log(params['sigma_sq']) - \
                                  (expected_lambda / params['sigma_sq']) * expected_sq_error)
        log_lik = log_lik_batch * (self.N / B)

        p_alpha_r, p_beta_r = params['nu_f'] / 2.0, params['nu_f'] / 2.0
        kl_r = (self.alpha_r - p_alpha_r) * torch.digamma(self.alpha_r) - torch.lgamma(self.alpha_r) + torch.lgamma(p_alpha_r) + \
               p_alpha_r * (torch.log(self.beta_r) - torch.log(p_beta_r)) + self.alpha_r * (p_beta_r - self.beta_r) / self.beta_r

        expected_log_r = torch.digamma(self.alpha_r) - torch.log(self.beta_r)
        expected_r = self.alpha_r / self.beta_r

        L_S = torch.linalg.cholesky(self.S_u + torch.eye(self.M, dtype=self.S_u.dtype, device=self.S_u.device) * 1e-6)
        logdet_S_u = 2 * torch.sum(torch.log(torch.diag(L_S)))
        logdet_K_ZZ = 2 * torch.sum(torch.log(torch.diag(L_ZZ)))

        trace_KZZinv_Su = torch.trace(torch.cholesky_solve(self.S_u, L_ZZ))
        m_T_KZZinv_m = self.m_u.T @ KZZ_inv_m_u
        kl_u = 0.5 * (-logdet_S_u - self.M * expected_log_r + logdet_K_ZZ + expected_r * (trace_KZZinv_Su + m_T_KZZinv_m) - self.M).squeeze()

        p_alpha_lambda, p_beta_lambda = params['nu_epsilon'] / 2.0, params['nu_epsilon'] / 2.0
        kl_lambda_batch = torch.sum((alpha_lambda_batch - p_alpha_lambda) * torch.digamma(alpha_lambda_batch) - \
                    torch.lgamma(alpha_lambda_batch) + torch.lgamma(p_alpha_lambda) + \
                    p_alpha_lambda * (torch.log(beta_lambda_batch) - torch.log(p_beta_lambda)) + \
                    alpha_lambda_batch * (p_beta_lambda - beta_lambda_batch) / beta_lambda_batch)
        kl_lambda = kl_lambda_batch * (self.N / B)

        log_prior = self.lengthscale_prior.log_prob(params['lengthscale']).sum() + \
                    self.variance_prior.log_prob(params['variance']) + \
                    self.sigma_sq_prior.log_prob(params['sigma_sq']) + \
                    self.nu_prior.log_prob(params['nu_f']) + \
                    self.nu_prior.log_prob(params['nu_epsilon'])

        # log_prior = 0
        return log_lik - kl_u - kl_r - kl_lambda + log_prior

    def fit(self, epochs=100, batch_size=64, lr=0.01, 
            base_lr=1.0, kappa_e=0.7, tau_e=1.0, tau_b=1.0,
            X_test=None, y_test=None, eval_interval=10, result_path=None):
        
        # parameters_to_optimize = [
        #     self.log_nu_f, self.log_nu_epsilon, self.log_sigma_sq,
        #     self.log_kernel_lengthscale, self.log_kernel_variance, self.Z
        # ]
        parameters_to_optimize = [
            self.log_sigma_sq,
            self.log_kernel_lengthscale, self.log_kernel_variance, self.Z
        ]
        optimizer = optim.Adam(parameters_to_optimize, lr=lr)
        dataset = TensorDataset(self.X_full, self.y_full)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        elbo_history = []
        
        can_evaluate = X_test is not None and y_test is not None
        if can_evaluate and result_path:
            result_path = Path(result_path)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            with open(result_path, 'w') as f: f.write("epoch,rmse,elbo\n")

        print(f"Starting Variational EM optimization for {epochs} epochs...")

        for epoch in range(epochs):
            elbo_val = 0.0 
            
            # --- 追加部分：エポック間減衰項の計算 ---
            epoch_num = epoch + 1
            gamma_e = (epoch_num + tau_e)**(-kappa_e)

            for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
                
                params = self._get_hyperparams()
                common_terms = self._compute_common_terms(X_batch, params)
                
                eta_u1_local, eta_u2_local, eta_r1_local, eta_r2_local, alpha_lambda, beta_lambda = self._e_step(X_batch, y_batch, common_terms)

                # --- 変更部分：階層的学習率rhoの計算 ---
                scale = self.N / len(X_batch)
                batch_num = batch_idx + 1
                lambda_b = (batch_num + tau_b)**(-1.0)
                rho = base_lr * gamma_e * lambda_b

                S_inv = torch.linalg.inv(self.S_u + torch.eye(self.M, dtype=self.S_u.dtype, device=self.S_u.device) * 1e-6)
                eta_u1_global = S_inv @ self.m_u
                eta_u2_global = -0.5 * S_inv
                eta_r1_global = self.alpha_r - 1.0
                eta_r2_global = -self.beta_r

                eta_u1_updated = (1 - rho) * eta_u1_global + rho * eta_u1_local * scale
                eta_u2_updated = (1 - rho) * eta_u2_global + rho * eta_u2_local * scale
                eta_r1_updated = (1 - rho) * eta_r1_global + rho * eta_r1_local * scale
                eta_r2_updated = (1 - rho) * eta_r2_global + rho * eta_r2_local * scale
    
                S_u_updated = torch.linalg.inv(-2.0 * eta_u2_updated + torch.eye(self.M, dtype=self.S_u.dtype, device=self.S_u.device) * 1e-6)
                m_u_updated = S_u_updated @ eta_u1_updated
                alpha_r_updated = eta_r1_updated + 1.0
                beta_r_updated = -eta_r2_updated

                alpha_r_updated = torch.clamp(alpha_r_updated, min=1e-6)
                beta_r_updated = torch.clamp(beta_r_updated, min=1e-6)

                self.m_u.data = m_u_updated
                self.S_u.data = S_u_updated
                self.alpha_r.data = alpha_r_updated
                self.beta_r.data = beta_r_updated

                elbo_val = self._m_step_and_elbo(optimizer, X_batch, y_batch, alpha_lambda, beta_lambda, common_terms)
                elbo_history.append(elbo_val)
            
            if (epoch + 1) % 1 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Final Batch ELBO: {elbo_val:.4f}, rho: {rho:.4f}")
                print(f"          Hyperparameters: nu_f={torch.exp(self.log_nu_f).item():.4f}, nu_epsilon={torch.exp(self.log_nu_epsilon).item():.4f}")

        print("\nOptimization finished.")
        return elbo_history

    def predict(self, X_test):
        X_test_dev = X_test.to(self.device)
        with torch.no_grad():
            params = self._get_hyperparams()
            k_star_star = self.kernel(X_test_dev, X_test_dev, params['lengthscale'], params['variance']).diag()

            _, L_ZZ, K_star_Z, K_star_Z_K_ZZ_inv = self._compute_common_terms(X_test_dev, params)
            
            KZZ_inv_m_u = torch.cholesky_solve(self.m_u, L_ZZ)
            pred_mean = K_star_Z @ KZZ_inv_m_u
            
            gp_var = k_star_star - (K_star_Z_K_ZZ_inv * K_star_Z).sum(dim=1) + \
                     (K_star_Z_K_ZZ_inv @ self.S_u @ K_star_Z_K_ZZ_inv.T).diag()
            
            pred_nu = 2 * self.alpha_r
            pred_scale_sq = (gp_var * (self.beta_r / self.alpha_r)).unsqueeze(1)
            
            return pred_mean, pred_scale_sq, pred_nu

    def evaluate_model(self, epochs=100, batch_size=64, lr=0.01,
                       base_lr=1.0, kappa_e=0.7, tau_e=1.0, tau_b=1.0,
                       X_test=None, y_test=None, eval_interval=10,
                       result_path=None):
        
        parameters_to_optimize = [
            self.log_nu_f, self.log_nu_epsilon, self.log_sigma_sq,
            self.log_kernel_lengthscale, self.log_kernel_variance, self.Z
        ]
        optimizer = optim.Adam(parameters_to_optimize, lr=lr)
        dataset = TensorDataset(self.X_full, self.y_full)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        can_evaluate = X_test is not None and y_test is not None and result_path is not None
        if can_evaluate:
            result_path = Path(result_path)
            if not result_path.exists():
                result_path.parent.mkdir(parents=True, exist_ok=True)
                with open(result_path, 'w') as f:
                    f.write("epoch,rmse,elbo\n")

        print(f"Starting Variational EM optimization for {epochs} epochs with evaluation...")

        elbo_history = []
        
        num_batches_per_epoch = math.ceil(self.N / batch_size)

        for epoch in range(epochs):
            elbo_val = 0.0 
            
            # --- 追加部分：エポック間減衰項の計算 ---
            epoch_num = epoch + 1
            gamma_e = (epoch_num + tau_e)**(-kappa_e)

            for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
                
                params = self._get_hyperparams()
                common_terms = self._compute_common_terms(X_batch, params)
                
                eta_u1_local, eta_u2_local, eta_r1_local, eta_r2_local, alpha_lambda, beta_lambda = self._e_step(X_batch, y_batch, common_terms)
                
                # --- 変更部分：階層的学習率rhoの計算 ---
                scale = self.N / len(X_batch)
                batch_num = batch_idx + 1
                lambda_b = (batch_num + tau_b)**(-1.0)
                rho = base_lr * gamma_e * lambda_b

                S_inv = torch.linalg.inv(self.S_u + torch.eye(self.M, dtype=self.S_u.dtype, device=self.S_u.device) * 1e-6)
                eta_u1_global = S_inv @ self.m_u
                eta_u2_global = -0.5 * S_inv
                eta_r1_global = self.alpha_r - 1.0
                eta_r2_global = -self.beta_r

                eta_u1_updated = (1 - rho) * eta_u1_global + rho * eta_u1_local * scale
                eta_u2_updated = (1 - rho) * eta_u2_global + rho * eta_u2_local * scale
                eta_r1_updated = (1 - rho) * eta_r1_global + rho * eta_r1_local * scale
                eta_r2_updated = (1 - rho) * eta_r2_global + rho * eta_r2_local * scale
    
                S_u_updated = torch.linalg.inv(-2.0 * eta_u2_updated + torch.eye(self.M, dtype=self.S_u.dtype, device=self.S_u.device) * 1e-6)
                m_u_updated = S_u_updated @ eta_u1_updated
                alpha_r_updated = eta_r1_updated + 1.0
                beta_r_updated = -eta_r2_updated

                alpha_r_updated = torch.clamp(alpha_r_updated, min=1e-6)
                beta_r_updated = torch.clamp(beta_r_updated, min=1e-6)

                self.m_u.data = m_u_updated
                self.S_u.data = S_u_updated
                self.alpha_r.data = alpha_r_updated
                self.beta_r.data = beta_r_updated

                elbo_val = self._m_step_and_elbo(optimizer, X_batch, y_batch, alpha_lambda, beta_lambda, common_terms)
                elbo_history.append(elbo_val)
            

            if can_evaluate and (epoch + 1) % eval_interval == 0:
                with torch.no_grad():
                    pred_mean, _, _ = self.predict(X_test)
                    rmse = torch.sqrt(torch.mean((y_test.to(self.device).view(-1) - pred_mean.view(-1))**2)).item()
                
                log_msg = f"Epoch {epoch+1}/{epochs}, ELBO: {elbo_val:.4f}, Test RMSE: {rmse:.4f}"
                logging.info(log_msg)
                print(log_msg)
                
                with open(result_path, 'a') as f:
                    f.write(f"{epoch+1},{rmse},{elbo_val}\n")
            elif (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Final Batch ELBO: {elbo_val:.4f}, rho: {rho}")

        if can_evaluate and epochs > 0 and epochs % eval_interval != 0:
             with torch.no_grad():
                pred_mean, _, _ = self.predict(X_test)
                rmse = torch.sqrt(torch.mean((y_test.to(self.device).view(-1) - pred_mean.view(-1))**2)).item()
                print(f"rmse: {rmse:.4f}")
             with open(result_path, 'a') as f:
                f.write(f"{epochs},{rmse},{elbo_val}\n")
        
        print("\nOptimization finished.")












# from .kernels import rbf_kernel, matern52_kernel
# from .priors import GammaPrior, LogNormalPrior

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import math
# from torch.utils.data import DataLoader, TensorDataset
# import logging
# from pathlib import Path
# import numpy as np
# from sklearn.cluster import KMeans



# class SparseTPRTMiniBatch(nn.Module):
#     """
#     This version implements a batch update for global variational parameters
#     at the end of each epoch, as requested.
#     """
#     def __init__(self, X, y, M, nu_f=2.1, nu_e=2.1,
#                  kernel_lengthscale=None, kernel_variance=1.0,
#                  likelihood_sigma=1.0, device=None):
#         super().__init__()

#         if device is None:
#             self.device = X.device if isinstance(X, torch.Tensor) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         else:
#             self.device = torch.device(device)

#         self.register_buffer('X_full', X.to(self.device))
#         self.register_buffer('y_full', y.view(-1, 1).to(self.device))

#         if self.X_full.ndim == 1: self.X_full = self.X_full.unsqueeze(1)
#         if self.y_full.ndim == 1: self.y_full = self.y_full.unsqueeze(1)

#         self.N, self.D = self.X_full.shape
#         self.M = M
#         dtype = self.X_full.dtype

#         if kernel_lengthscale is None:
#             kernel_lengthscale = torch.ones(self.D, dtype=dtype)
#         else:
#             kernel_lengthscale = torch.as_tensor(kernel_lengthscale, dtype=dtype)
#         if kernel_lengthscale.ndim == 0:
#             kernel_lengthscale = kernel_lengthscale.repeat(self.D)
#         if kernel_lengthscale.shape[0] != self.D:
#             raise ValueError(f"lengthscale must be a scalar or a vector of length D={self.D}")

#         self.log_kernel_lengthscale = nn.Parameter(torch.log(kernel_lengthscale))
#         self.log_kernel_variance = nn.Parameter(torch.log(torch.tensor(kernel_variance, dtype=dtype)))
#         self.log_nu_f = nn.Parameter(torch.log(torch.tensor(nu_f, dtype=dtype)))
#         self.log_nu_epsilon = nn.Parameter(torch.log(torch.tensor(nu_e, dtype=dtype)))
#         self.log_sigma_sq = nn.Parameter(torch.log(torch.tensor(likelihood_sigma**2, dtype=dtype)))
#         self.Z = nn.Parameter(self._initialize_inducing_points())

#         self.lengthscale_prior = GammaPrior(3.0, 6.0)
#         self.variance_prior = GammaPrior(2.0, 0.15)
#         self.sigma_sq_prior = GammaPrior(1.1, 0.05)
#         self.nu_prior = LogNormalPrior(loc=1.0, scale=1.0)

#         self.register_buffer('m_u', torch.zeros(self.M, 1, dtype=dtype))
#         self.register_buffer('S_u', torch.eye(self.M, dtype=dtype))
#         self.register_buffer('alpha_r', torch.tensor(1.0, dtype=dtype))
#         self.register_buffer('beta_r', torch.tensor(1.0, dtype=dtype))

#         self.to(self.device)

#         self.kernel = rbf_kernel

#     def _initialize_inducing_points(self):
#         if self.N >= self.M:
#             X_np = self.X_full.cpu().numpy()
#             kmeans = KMeans(n_clusters=self.M, random_state=0, n_init='auto').fit(X_np)
#             Z_init = torch.from_numpy(kmeans.cluster_centers_)
#         else:
#             indices = np.random.choice(self.N, self.M, replace=True)
#             Z_init = self.X_full[indices].clone()
#         return Z_init.to(dtype=self.X_full.dtype, device=self.device)

#     def _get_hyperparams(self):
#         return {
#             "nu_f": torch.exp(self.log_nu_f), "nu_epsilon": torch.exp(self.log_nu_epsilon),
#             "sigma_sq": torch.exp(self.log_sigma_sq), "lengthscale": torch.exp(self.log_kernel_lengthscale),
#             "variance": torch.exp(self.log_kernel_variance)
#         }

#     def _compute_common_terms(self, X_batch, params):
#         K_ZZ = self.kernel(self.Z, self.Z, params['lengthscale'], params['variance']) + torch.eye(self.M, device=self.Z.device) * 1e-6
#         L_ZZ = torch.linalg.cholesky(K_ZZ)
#         K_XZ_batch = self.kernel(X_batch, self.Z, params['lengthscale'], params['variance'])
#         KXZ_KZZ_inv = torch.linalg.solve(K_ZZ, K_XZ_batch.T).T
#         return K_ZZ, L_ZZ, K_XZ_batch, KXZ_KZZ_inv

#     def _e_step(self, X_batch, y_batch, common_terms):
#         with torch.no_grad():
#             params = self._get_hyperparams()
#             K_ZZ, L_ZZ, K_XZ_batch, KXZ_KZZ_inv = common_terms

#             k_ii_batch = params['variance'].expand(X_batch.shape[0])

#             expected_r_inv = self.beta_r / (self.alpha_r - 1.0) if self.alpha_r > 1 else self.beta_r

#             KZZ_inv_m_u = torch.cholesky_solve(self.m_u, L_ZZ)
#             expected_f_mean = K_XZ_batch @ KZZ_inv_m_u

#             var_f_term1 = expected_r_inv * (k_ii_batch - (KXZ_KZZ_inv * K_XZ_batch).sum(dim=1))
#             var_f_term2 = (KXZ_KZZ_inv @ self.S_u @ KXZ_KZZ_inv.T).diag()
#             var_f = (var_f_term1 + var_f_term2).unsqueeze(1)

#             expected_sq_error = (y_batch - expected_f_mean).pow(2) + var_f
#             alpha_lambda_local = params['nu_epsilon'] / 2.0 + 0.5
#             beta_lambda_local = params['nu_epsilon'] / 2.0 + (0.5 / params['sigma_sq']) * expected_sq_error

#             expected_r = self.alpha_r / self.beta_r
#             expected_lambda = alpha_lambda_local.squeeze() / beta_lambda_local.squeeze()
#             c = expected_lambda / params['sigma_sq']

#             B = (K_XZ_batch.T * c) @ K_XZ_batch
#             precision_inner = expected_r * K_ZZ + B
#             L_precision_inner = torch.linalg.cholesky(precision_inner + torch.eye(self.M, device=K_ZZ.device) * 1e-6)

#             S_u_local = K_ZZ @ torch.cholesky_solve(K_ZZ, L_precision_inner)
#             y_term = K_XZ_batch.T @ (y_batch.squeeze() * c)
#             m_u_local = K_ZZ @ torch.cholesky_solve(y_term.unsqueeze(1), L_precision_inner)

#             trace_term = torch.trace(torch.cholesky_solve(S_u_local, L_ZZ))
#             KZZ_inv_m_u_local = torch.cholesky_solve(m_u_local, L_ZZ)
#             mean_term = m_u_local.T @ KZZ_inv_m_u_local
#             expected_u_quadratic = trace_term + mean_term

#             alpha_r_local = params['nu_f'] / 2.0 + self.M / 2.0
#             beta_r_local_val = params['nu_f'] / 2.0 + 0.5 * expected_u_quadratic.squeeze()

#             # Convert local variational parameters to natural parameters to be returned
#             S_local_inv = torch.linalg.inv(S_u_local + torch.eye(self.M, dtype=S_u_local.dtype, device=S_u_local.device) * 1e-6)
#             eta_u1_local = S_local_inv @ m_u_local
#             eta_u2_local = -0.5 * S_local_inv
#             eta_r1_local = alpha_r_local - 1.0
#             eta_r2_local = -beta_r_local_val

#             # Return local natural parameters and local lambda parameters
#             return eta_u1_local, eta_u2_local, eta_r1_local, eta_r2_local, alpha_lambda_local, beta_lambda_local

#     def _m_step_and_elbo(self, optimizer, X_batch, y_batch, alpha_lambda_batch, beta_lambda_batch, common_terms):
#         optimizer.zero_grad()
#         elbo = self._calculate_elbo(X_batch, y_batch, alpha_lambda_batch, beta_lambda_batch, common_terms)
#         loss = -elbo
#         loss.backward()
#         # Add gradient clipping for stability
#         nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.0)
#         optimizer.step()
#         return elbo.item()

#     def _calculate_elbo(self, X_batch, y_batch, alpha_lambda_batch, beta_lambda_batch, common_terms):
#         B = X_batch.shape[0]
#         params = self._get_hyperparams()

#         K_ZZ, L_ZZ, K_XZ_batch, KXZ_KZZ_inv = common_terms
#         k_ii_batch = params['variance'].expand(B)

#         expected_log_lambda = torch.digamma(alpha_lambda_batch) - torch.log(beta_lambda_batch)
#         expected_lambda = alpha_lambda_batch / beta_lambda_batch

#         KZZ_inv_m_u = torch.cholesky_solve(self.m_u, L_ZZ)
#         expected_f_mean = K_XZ_batch @ KZZ_inv_m_u

#         expected_r_inv = self.beta_r / (self.alpha_r - 1.0) if self.alpha_r > 1 else self.beta_r

#         var_f_term1 = expected_r_inv * (k_ii_batch - (KXZ_KZZ_inv * K_XZ_batch).sum(dim=1))
#         var_f_term2 = (KXZ_KZZ_inv @ self.S_u @ KXZ_KZZ_inv.T).diag()
#         var_f = (var_f_term1 + var_f_term2).unsqueeze(1)

#         expected_sq_error = (y_batch - expected_f_mean).pow(2) + var_f
#         log_lik_batch = 0.5 * torch.sum(expected_log_lambda - math.log(2 * math.pi) - torch.log(params['sigma_sq']) - \
#                                   (expected_lambda / params['sigma_sq']) * expected_sq_error)
#         log_lik = log_lik_batch * (self.N / B)

#         p_alpha_r, p_beta_r = params['nu_f'] / 2.0, params['nu_f'] / 2.0
#         kl_r = (self.alpha_r - p_alpha_r) * torch.digamma(self.alpha_r) - torch.lgamma(self.alpha_r) + torch.lgamma(p_alpha_r) + \
#                p_alpha_r * (torch.log(self.beta_r) - torch.log(p_beta_r)) + self.alpha_r * (p_beta_r - self.beta_r) / self.beta_r

#         expected_log_r = torch.digamma(self.alpha_r) - torch.log(self.beta_r)
#         expected_r = self.alpha_r / self.beta_r

#         L_S = torch.linalg.cholesky(self.S_u + torch.eye(self.M, dtype=self.S_u.dtype, device=self.S_u.device) * 1e-6)
#         logdet_S_u = 2 * torch.sum(torch.log(torch.diag(L_S)))
#         logdet_K_ZZ = 2 * torch.sum(torch.log(torch.diag(L_ZZ)))

#         trace_KZZinv_Su = torch.trace(torch.cholesky_solve(self.S_u, L_ZZ))
#         m_T_KZZinv_m = self.m_u.T @ KZZ_inv_m_u
#         kl_u = 0.5 * (-logdet_S_u - self.M * expected_log_r + logdet_K_ZZ + expected_r * (trace_KZZinv_Su + m_T_KZZinv_m) - self.M).squeeze()

#         p_alpha_lambda, p_beta_lambda = params['nu_epsilon'] / 2.0, params['nu_epsilon'] / 2.0
#         kl_lambda_batch = torch.sum((alpha_lambda_batch - p_alpha_lambda) * torch.digamma(alpha_lambda_batch) - \
#                     torch.lgamma(alpha_lambda_batch) + torch.lgamma(p_alpha_lambda) + \
#                     p_alpha_lambda * (torch.log(beta_lambda_batch) - torch.log(p_beta_lambda)) + \
#                     alpha_lambda_batch * (p_beta_lambda - beta_lambda_batch) / beta_lambda_batch)
#         kl_lambda = kl_lambda_batch * (self.N / B)

#         log_prior = self.lengthscale_prior.log_prob(params['lengthscale']).sum() + \
#                     self.variance_prior.log_prob(params['variance']) + \
#                     self.sigma_sq_prior.log_prob(params['sigma_sq']) + \
#                     self.nu_prior.log_prob(params['nu_f']) + \
#                     self.nu_prior.log_prob(params['nu_epsilon'])

#         # log_prior = 0
#         return log_lik - kl_u - kl_r - kl_lambda + log_prior

#     def fit(self, epochs=100, batch_size=64, lr=0.01, 
#             base_lr=1.0, kappa_e=0.7, tau_e=1.0, tau_b=1.0,
#             X_test=None, y_test=None, eval_interval=10, result_path=None):
        
#         parameters_to_optimize = [
#             self.log_nu_f, self.log_nu_epsilon, self.log_sigma_sq,
#             self.log_kernel_lengthscale, self.log_kernel_variance, self.Z
#         ]
#         optimizer = optim.Adam(parameters_to_optimize, lr=lr)
#         dataset = TensorDataset(self.X_full, self.y_full)
#         dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
#         elbo_history = []
        
#         can_evaluate = X_test is not None and y_test is not None
#         if can_evaluate and result_path:
#             result_path = Path(result_path)
#             result_path.parent.mkdir(parents=True, exist_ok=True)
#             with open(result_path, 'w') as f: f.write("epoch,rmse,elbo\n")

#         print(f"Starting Variational EM optimization for {epochs} epochs...")
        
#         num_batches_per_epoch = math.ceil(self.N / batch_size)

#         for epoch in range(epochs):
#             elbo_val = 0.0 
            
#             epoch_num = epoch + 1
#             gamma_e = (epoch_num + tau_e)**(-kappa_e)

#             for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
                
#                 # --- 1. E-Step (現在のパラメータに基づき、初期のローカルパラメータを計算) ---
#                 params = self._get_hyperparams()
#                 common_terms = self._compute_common_terms(X_batch, params)
#                 K_ZZ, L_ZZ, K_XZ_batch, _ = common_terms
                
#                 # _e_stepはuとr両方のローカルパラメータを一度に計算する
#                 # ここではまず、両方の更新の方向性を得るために呼び出す
#                 initial_eta_u1, initial_eta_u2, eta_r1_local, eta_r2_local, alpha_lambda, beta_lambda = self._e_step(X_batch, y_batch, common_terms)

#                 # 学習率の計算
#                 scale = self.N / len(X_batch)
#                 batch_num = batch_idx + 1
#                 lambda_b = (batch_num + tau_b)**(-1.0)
#                 rho = base_lr * gamma_e * lambda_b

#                 rho=0.02

#                 # --- 2. グローバルな q(r) を先に更新 ---
#                 eta_r1_global = self.alpha_r - 1.0
#                 eta_r2_global = -self.beta_r
                
#                 eta_r1_updated = (1 - rho) * eta_r1_global + rho * eta_r1_local * scale
#                 eta_r2_updated = (1 - rho) * eta_r2_global + rho * eta_r2_local * scale
    
#                 alpha_r_updated = torch.clamp(eta_r1_updated + 1.0, min=1e-6)
#                 beta_r_updated = torch.clamp(-eta_r2_updated, min=1e-6)

#                 # モデルの状態を新しいrで更新
#                 self.alpha_r.data = alpha_r_updated
#                 self.beta_r.data = beta_r_updated

#                 # --- 3. 新しい q(r) を使って、q(u) のローカルパラメータを再計算 ---
#                 # B行列はrに依存しないため、E-stepの結果を再利用できる
#                 expected_lambda = alpha_lambda.squeeze() / beta_lambda.squeeze()
#                 c = expected_lambda / params['sigma_sq']
#                 B = (K_XZ_batch.T * c) @ K_XZ_batch
                
#                 # 更新されたrの期待値を計算
#                 expected_r_new = self.alpha_r / self.beta_r 
                
#                 # この新しい期待値を使ってprecision_innerを再計算
#                 precision_inner_new = expected_r_new * K_ZZ + B
#                 L_precision_inner_new = torch.linalg.cholesky(precision_inner_new + torch.eye(self.M, device=K_ZZ.device) * 1e-6)

#                 # precision_innerからeta_u_localまでの計算をやり直す
#                 S_u_local_new = K_ZZ @ torch.cholesky_solve(K_ZZ, L_precision_inner_new)
#                 y_term = K_XZ_batch.T @ (y_batch.squeeze() * c)
#                 m_u_local_new = K_ZZ @ torch.cholesky_solve(y_term.unsqueeze(1), L_precision_inner_new)
                
#                 S_local_inv_new = torch.linalg.inv(S_u_local_new + torch.eye(self.M, dtype=S_u_local_new.dtype, device=S_u_local_new.device) * 1e-6)
#                 eta_u1_local_new = S_local_inv_new @ m_u_local_new
#                 eta_u2_local_new = -0.5 * S_local_inv_new
                
#                 # --- 4. グローバルな q(u) を更新 ---
#                 S_inv_old = torch.linalg.inv(self.S_u + torch.eye(self.M, dtype=self.S_u.dtype, device=self.S_u.device) * 1e-6)
#                 eta_u1_global_old = S_inv_old @ self.m_u
#                 eta_u2_global_old = -0.5 * S_inv_old

#                 eta_u1_updated = (1 - rho) * eta_u1_global_old + rho * eta_u1_local_new * scale
#                 eta_u2_updated = (1 - rho) * eta_u2_global_old + rho * eta_u2_local_new * scale

#                 S_u_updated = torch.linalg.inv(-2.0 * eta_u2_updated + torch.eye(self.M, dtype=self.S_u.dtype, device=self.S_u.device) * 1e-6)
#                 m_u_updated = S_u_updated @ eta_u1_updated

#                 # モデルの状態を新しいuで更新
#                 self.m_u.data = m_u_updated
#                 self.S_u.data = S_u_updated
                
#                 # --- 5. M-Step (ハイパーパラメータの更新) ---
#                 # すべての変分パラメータが更新された状態でELBOを計算し、M-Stepを実行
#                 elbo_val = self._m_step_and_elbo(optimizer, X_batch, y_batch, alpha_lambda, beta_lambda, common_terms)
#                 elbo_history.append(elbo_val)
            
#             if (epoch + 1) % 1 == 0:
#                 print(f"Epoch {epoch+1}/{epochs}, Final Batch ELBO: {elbo_val:.4f}, rho: {rho:.4f}")

#         print("\nOptimization finished.")
#         return elbo_history

#     def predict(self, X_test):
#         X_test_dev = X_test.to(self.device)
#         with torch.no_grad():
#             params = self._get_hyperparams()
#             k_star_star = self.kernel(X_test_dev, X_test_dev, params['lengthscale'], params['variance']).diag()

#             _, L_ZZ, K_star_Z, K_star_Z_K_ZZ_inv = self._compute_common_terms(X_test_dev, params)
            
#             KZZ_inv_m_u = torch.cholesky_solve(self.m_u, L_ZZ)
#             pred_mean = K_star_Z @ KZZ_inv_m_u
            
#             gp_var = k_star_star - (K_star_Z_K_ZZ_inv * K_star_Z).sum(dim=1) + \
#                      (K_star_Z_K_ZZ_inv @ self.S_u @ K_star_Z_K_ZZ_inv.T).diag()
            
#             pred_nu = 2 * self.alpha_r
#             pred_scale_sq = (gp_var * (self.beta_r / self.alpha_r)).unsqueeze(1)
            
#             return pred_mean, pred_scale_sq, pred_nu

#     def evaluate_model(self, epochs=100, batch_size=64, lr=0.01,
#                        base_lr=1.0, kappa_e=0.7, tau_e=1.0, tau_b=1.0,
#                        X_test=None, y_test=None, eval_interval=10,
#                        result_path=None):
        
#         parameters_to_optimize = [
#             self.log_nu_f, self.log_nu_epsilon, self.log_sigma_sq,
#             self.log_kernel_lengthscale, self.log_kernel_variance, self.Z
#         ]
#         optimizer = optim.Adam(parameters_to_optimize, lr=lr)
#         dataset = TensorDataset(self.X_full, self.y_full)
#         dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#         can_evaluate = X_test is not None and y_test is not None and result_path is not None
#         if can_evaluate:
#             result_path = Path(result_path)
#             if not result_path.exists():
#                 result_path.parent.mkdir(parents=True, exist_ok=True)
#                 with open(result_path, 'w') as f:
#                     f.write("epoch,rmse,elbo\n")

#         print(f"Starting Variational EM optimization for {epochs} epochs with evaluation...")

#         elbo_history = []
        
#         num_batches_per_epoch = math.ceil(self.N / batch_size)

#         for epoch in range(epochs):
#             elbo_val = 0.0 
            
#             epoch_num = epoch + 1
#             gamma_e = (epoch_num + tau_e)**(-kappa_e)

#             for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
                
#                 # --- 1. E-Step (現在のパラメータに基づき、初期のローカルパラメータを計算) ---
#                 params = self._get_hyperparams()
#                 common_terms = self._compute_common_terms(X_batch, params)
#                 K_ZZ, L_ZZ, K_XZ_batch, _ = common_terms
                
#                 # _e_stepはuとr両方のローカルパラメータを一度に計算する
#                 # ここではまず、両方の更新の方向性を得るために呼び出す
#                 initial_eta_u1, initial_eta_u2, eta_r1_local, eta_r2_local, alpha_lambda, beta_lambda = self._e_step(X_batch, y_batch, common_terms)

#                 # 学習率の計算
#                 scale = self.N / len(X_batch)
#                 batch_num = batch_idx + 1
#                 lambda_b = (batch_num + tau_b)**(-1.0)
#                 rho = base_lr * gamma_e * lambda_b

#                 # --- 2. グローバルな q(r) を先に更新 ---
#                 eta_r1_global = self.alpha_r - 1.0
#                 eta_r2_global = -self.beta_r
                
#                 eta_r1_updated = (1 - rho) * eta_r1_global + rho * eta_r1_local * scale
#                 eta_r2_updated = (1 - rho) * eta_r2_global + rho * eta_r2_local * scale
    
#                 alpha_r_updated = torch.clamp(eta_r1_updated + 1.0, min=1e-6)
#                 beta_r_updated = torch.clamp(-eta_r2_updated, min=1e-6)

#                 # モデルの状態を新しいrで更新
#                 self.alpha_r.data = alpha_r_updated
#                 self.beta_r.data = beta_r_updated

#                 # --- 3. 新しい q(r) を使って、q(u) のローカルパラメータを再計算 ---
#                 # B行列はrに依存しないため、E-stepの結果を再利用できる
#                 expected_lambda = alpha_lambda.squeeze() / beta_lambda.squeeze()
#                 c = expected_lambda / params['sigma_sq']
#                 B = (K_XZ_batch.T * c) @ K_XZ_batch
                
#                 # 更新されたrの期待値を計算
#                 expected_r_new = self.alpha_r / self.beta_r 
                
#                 # この新しい期待値を使ってprecision_innerを再計算
#                 precision_inner_new = expected_r_new * K_ZZ + B
#                 L_precision_inner_new = torch.linalg.cholesky(precision_inner_new + torch.eye(self.M, device=K_ZZ.device) * 1e-6)

#                 # precision_innerからeta_u_localまでの計算をやり直す
#                 S_u_local_new = K_ZZ @ torch.cholesky_solve(K_ZZ, L_precision_inner_new)
#                 y_term = K_XZ_batch.T @ (y_batch.squeeze() * c)
#                 m_u_local_new = K_ZZ @ torch.cholesky_solve(y_term.unsqueeze(1), L_precision_inner_new)
                
#                 S_local_inv_new = torch.linalg.inv(S_u_local_new + torch.eye(self.M, dtype=S_u_local_new.dtype, device=S_u_local_new.device) * 1e-6)
#                 eta_u1_local_new = S_local_inv_new @ m_u_local_new
#                 eta_u2_local_new = -0.5 * S_local_inv_new
                
#                 # --- 4. グローバルな q(u) を更新 ---
#                 S_inv_old = torch.linalg.inv(self.S_u + torch.eye(self.M, dtype=self.S_u.dtype, device=self.S_u.device) * 1e-6)
#                 eta_u1_global_old = S_inv_old @ self.m_u
#                 eta_u2_global_old = -0.5 * S_inv_old

#                 eta_u1_updated = (1 - rho) * eta_u1_global_old + rho * eta_u1_local_new * scale
#                 eta_u2_updated = (1 - rho) * eta_u2_global_old + rho * eta_u2_local_new * scale

#                 S_u_updated = torch.linalg.inv(-2.0 * eta_u2_updated + torch.eye(self.M, dtype=self.S_u.dtype, device=self.S_u.device) * 1e-6)
#                 m_u_updated = S_u_updated @ eta_u1_updated

#                 # モデルの状態を新しいuで更新
#                 self.m_u.data = m_u_updated
#                 self.S_u.data = S_u_updated
                
#                 # --- 5. M-Step (ハイパーパラメータの更新) ---
#                 # すべての変分パラメータが更新された状態でELBOを計算し、M-Stepを実行
#                 elbo_val = self._m_step_and_elbo(optimizer, X_batch, y_batch, alpha_lambda, beta_lambda, common_terms)
#                 elbo_history.append(elbo_val)
            

#             if can_evaluate and (epoch + 1) % eval_interval == 0:
#                 with torch.no_grad():
#                     pred_mean, _, _ = self.predict(X_test)
#                     rmse = torch.sqrt(torch.mean((y_test.to(self.device).view(-1) - pred_mean.view(-1))**2)).item()
                
#                 log_msg = f"Epoch {epoch+1}/{epochs}, ELBO: {elbo_val:.4f}, Test RMSE: {rmse:.4f}"
#                 logging.info(log_msg)
#                 print(log_msg)
                
#                 with open(result_path, 'a') as f:
#                     f.write(f"{epoch+1},{rmse},{elbo_val}\n")
#             elif (epoch + 1) % 10 == 0:
#                 print(f"Epoch {epoch+1}/{epochs}, Final Batch ELBO: {elbo_val:.4f}, rho: {rho}")

#         if can_evaluate and epochs > 0 and epochs % eval_interval != 0:
#              with torch.no_grad():
#                 pred_mean, _, _ = self.predict(X_test)
#                 rmse = torch.sqrt(torch.mean((y_test.to(self.device).view(-1) - pred_mean.view(-1))**2)).item()
#                 print(f"rmse: {rmse:.4f}")
#              with open(result_path, 'a') as f:
#                 f.write(f"{epochs},{rmse},{elbo_val}\n")
        
#         print("\nOptimization finished.")