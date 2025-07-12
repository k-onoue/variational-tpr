from .kernels import rbf_kernel
import torch
import torch.nn as nn
import math
import logging


# class TPRTFullBatch:
#     """
#     Implementation of the full (non-sparse) Student-t Process regression model.
#     This version is aligned with the structure of the SparseTPRTFullBatch class.
#     """
#     def __init__(self, X, y, nu_f=2.1, nu_e=2.1,
#                  kernel_lengthscale=1.0, kernel_variance=1.0, likelihood_sigma=1.0):
        
#         self.X = X
#         self.y = y.view(-1, 1) # Ensure y is always of shape (N, 1)
#         self.N, self.D = X.shape

#         # --- Hyperparameters as nn.Parameter for optimization ---
#         self.log_kernel_lengthscale = nn.Parameter(torch.log(torch.tensor(kernel_lengthscale, dtype=X.dtype)))
#         self.log_kernel_variance = nn.Parameter(torch.log(torch.tensor(kernel_variance, dtype=X.dtype)))
#         self.log_likelihood_sigma_sq = nn.Parameter(torch.log(torch.tensor(likelihood_sigma**2, dtype=X.dtype)))
#         self.log_nu_f = nn.Parameter(torch.log(torch.tensor(nu_f, dtype=X.dtype)))
#         self.log_nu_e = nn.Parameter(torch.log(torch.tensor(nu_e, dtype=X.dtype)))

#         # --- Variational Parameters (updated via CAVI) ---
#         self.m_f = torch.zeros(self.N, 1, dtype=X.dtype, device=X.device)
#         self.L_f = torch.eye(self.N, dtype=X.dtype, device=X.device)
#         self.alpha_r = torch.tensor(1.0, dtype=X.dtype, device=X.device)
#         self.beta_r = torch.tensor(1.0, dtype=X.dtype, device=X.device)
#         self.alpha_lambda = torch.ones(self.N, 1, dtype=X.dtype, device=X.device)
#         self.beta_lambda = torch.ones(self.N, 1, dtype=X.dtype, device=X.device)

#     def _get_hyperparams(self):
#         """Helper to get positive hyperparameters from their log-transformed storage."""
#         return {
#             "lengthscale": torch.exp(self.log_kernel_lengthscale),
#             "variance": torch.exp(self.log_kernel_variance),
#             "sigma_sq": torch.exp(self.log_likelihood_sigma_sq),
#             "nu_f": torch.exp(self.log_nu_f),
#             "nu_e": torch.exp(self.log_nu_e)
#         }

#     def _update_q_lambda(self, params):
#         """CAVI update for q(lambda_i). Receives params as an argument."""
#         # params = self._get_hyperparams() <- 削除
#         S_f = self.L_f @ self.L_f.T
#         var_f = S_f.diag().unsqueeze(1)
#         expected_f_mean = self.m_f

#         self.alpha_lambda = params['nu_e'] / 2.0 + 0.5
#         expected_sq_error = (self.y - expected_f_mean).pow(2) + var_f
#         self.beta_lambda = params['nu_e'] / 2.0 + (1.0 / (2.0 * params['sigma_sq'])) * expected_sq_error

#     def _update_q_r(self, params, Lxx):
#         """CAVI update for q(r). Receives params and Lxx as arguments."""
#         # params = self._get_hyperparams() <- 削除
#         S_f = self.L_f @ self.L_f.T
        
#         trace_term = torch.trace(torch.cholesky_solve(S_f, Lxx))
#         mean_term = self.m_f.T @ torch.cholesky_solve(self.m_f, Lxx)
#         expected_f_quadratic_form = trace_term + mean_term

#         self.alpha_r = params['nu_f'] / 2.0 + self.N / 2.0
#         self.beta_r = params['nu_f'] / 2.0 + 0.5 * expected_f_quadratic_form.squeeze()

#     def _update_q_f(self, params, Lxx):
#         """CAVI update for q(f). Receives params and Lxx as arguments."""
#         # params = self._get_hyperparams() <- 削除
#         expected_r = self.alpha_r / self.beta_r
#         expected_lambda = self.alpha_lambda / self.beta_lambda

#         Kxx_inv = torch.cholesky_inverse(Lxx)
#         S_f_inv = expected_r * Kxx_inv + torch.diag(expected_lambda.squeeze() / params['sigma_sq'])
        
#         L_f_inv = torch.linalg.cholesky(S_f_inv)
#         self.L_f = torch.triangular_solve(torch.eye(self.N, device=self.X.device), L_f_inv, upper=False)[0]

#         temp_vec = (expected_lambda / params['sigma_sq']) * self.y
#         self.m_f = torch.cholesky_solve(temp_vec, L_f_inv)

#     def _cavi_step(self, params, Lxx):
#         """
#         Wrapper for a single CAVI update cycle.
#         This follows the structure of the sparse model template.
#         """
#         self._update_q_lambda(params)
#         self._update_q_r(params, Lxx)
#         self._update_q_f(params, Lxx)
        
#     def _e_step(self, cavi_max_iter=10, cavi_tol=1e-5):
#         """
#         Performs the E-Step by running CAVI updates.
#         This method now computes its own kernel matrix based on the current hyperparameters.
#         """
#         with torch.no_grad():
#             # 1. Get hyperparameters at the start of the E-step
#             params = self._get_hyperparams()
            
#             # 2. Compute common components for the CAVI loop
#             Kxx = rbf_kernel(self.X, self.X, params['lengthscale'], params['variance'])
#             Kxx += torch.eye(self.N, device=self.X.device) * 1e-6
#             Lxx = torch.linalg.cholesky(Kxx)

#             # 3. Run the CAVI update loop
#             for _ in range(cavi_max_iter):
#                 m_f_prev = self.m_f.clone()
#                 self._cavi_step(params, Lxx)
#                 m_f_rel_change = torch.norm(self.m_f - m_f_prev) / (torch.norm(m_f_prev) + 1e-8)
#                 if m_f_rel_change < cavi_tol:
#                     # print(f"CAVI converged in {_+1} iterations with relative change {m_f_rel_change:.6f}.")
#                     break

#     def _m_step(self, optimizer):
#         """Performs the M-Step by updating hyperparameters."""
#         optimizer.zero_grad()
#         elbo = self._calculate_elbo()
#         loss = -elbo
#         loss.backward()
#         optimizer.step()
#         return elbo.item()

#     def fit(self, max_iter_global=100, cavi_max_iter=10, cavi_tol=1e-5, lr=0.01):
#         """Fits the model using the Variational EM algorithm."""
#         optimizer = torch.optim.Adam([
#             self.log_kernel_lengthscale, self.log_kernel_variance,
#             self.log_likelihood_sigma_sq, self.log_nu_f, self.log_nu_e
#         ], lr=lr)

#         elbo_history = []
#         print("Starting Variational EM optimization...")
#         for i in range(max_iter_global):
#             self._e_step(cavi_max_iter=cavi_max_iter, cavi_tol=cavi_tol)
#             elbo = self._m_step(optimizer)
#             elbo_history.append(elbo)
            
#             if (i + 1) % 50 == 0:
#                 print(f"EM Iteration {i+1}/{max_iter_global}, ELBO: {elbo:.4f}")

#         print("\nOptimization finished.")
#         return elbo_history

#     def _calculate_elbo(self):
#         """
#         Calculates the Evidence Lower Bound (ELBO).
#         This version strictly maintains the original calculation logic.
#         """
#         params = self._get_hyperparams()
#         Kxx = rbf_kernel(self.X, self.X, params['lengthscale'], params['variance'])
#         Kxx += torch.eye(self.N, device=self.X.device) * 1e-6
#         Lxx = torch.linalg.cholesky(Kxx)
#         S_f = self.L_f @ self.L_f.T

#         # --- 1. Expected Log Likelihood (E[log p(y|f, lambda)]) ---
#         E_q_f = self.m_f
#         Var_q_f = S_f.diag().unsqueeze(1)
#         expected_sq_error = (self.y - E_q_f).pow(2) + Var_q_f

#         E_lambda = self.alpha_lambda / self.beta_lambda
#         E_log_lambda = torch.digamma(self.alpha_lambda) - torch.log(self.beta_lambda)

#         e_log_lik = -0.5 * self.N * math.log(2 * math.pi) - 0.5 * self.N * torch.log(params['sigma_sq']) + \
#                     0.5 * torch.sum(E_log_lambda) - \
#                     0.5 / params['sigma_sq'] * torch.sum(E_lambda * expected_sq_error)

#         # --- 2. KL Divergence for f and r (as in original code) ---
#         # Note: This part calculates -(E[log p(f,r)] - E[log q(f,r)])
#         E_r = self.alpha_r / self.beta_r
#         E_log_r = torch.digamma(self.alpha_r) - torch.log(self.beta_r)

#         # E[log q(f)]
#         # Original code simplifies this term, so we follow it.
#         # It's related to the entropy of a Gaussian.
#         log_q_f = -torch.sum(torch.log(torch.diag(self.L_f)))
        
#         # E[log q(r)]
#         log_q_r = self.alpha_r * torch.log(self.beta_r) - torch.lgamma(self.alpha_r) + \
#                   (self.alpha_r - 1) * E_log_r - self.beta_r * E_r

#         # E[log p(f|r)]
#         trace_term = torch.trace(torch.cholesky_solve(S_f, Lxx))
#         quad_form_term = self.m_f.T @ torch.cholesky_solve(self.m_f, Lxx)
#         E_quad_form_f = trace_term + quad_form_term
#         log_det_Kxx = 2 * torch.sum(torch.log(torch.diag(Lxx)))
#         E_log_p_f_r = -0.5 * log_det_Kxx + 0.5 * self.N * E_log_r - 0.5 * E_r * E_quad_form_f

#         # E[log p(r)]
#         p_alpha_r, p_beta_r = params['nu_f'] / 2.0, params['nu_f'] / 2.0
#         E_log_p_r = p_alpha_r * torch.log(p_beta_r) - torch.lgamma(p_alpha_r) + \
#                     (p_alpha_r - 1) * E_log_r - p_beta_r * E_r

#         # The constant -0.5 * N * log(2*pi) from log q(f) and E[log p(f|r)] cancel out.
#         # Original grouping: log_q_f + log_q_r - E_log_p_f_r - E_log_p_r
#         kl_f_r_grouped = (log_q_f + log_q_r) - (E_log_p_f_r + E_log_p_r)

#         # --- 3. KL Divergence for lambda_i (as in original code) ---
#         # Note: This part calculates - sum(E[log p(lambda_i)] - E[log q(lambda_i)])
#         p_alpha_lambda, p_beta_lambda = params['nu_e'] / 2.0, params['nu_e'] / 2.0
        
#         # Replicating the exact structure from the original snippet
#         kl_lambda = torch.lgamma(self.alpha_lambda) - self.alpha_lambda * torch.log(self.beta_lambda) - \
#                     (torch.lgamma(p_alpha_lambda) - p_alpha_lambda * torch.log(p_beta_lambda)) - \
#                     (self.alpha_lambda - p_alpha_lambda) * E_log_lambda + \
#                     (self.beta_lambda - p_beta_lambda) * E_lambda

#         kl_lambda_sum = torch.sum(kl_lambda)

#         # Combine terms exactly as in the original formulation
#         elbo = e_log_lik - kl_f_r_grouped - kl_lambda_sum
        
#         return elbo


#     def predict(self, X_test):
#         """Make predictions at new test points X_test."""
#         with torch.no_grad():
#             params = self._get_hyperparams()
#             K_star_x = rbf_kernel(X_test, self.X, params['lengthscale'], params['variance'])
#             K_star_star_diag = rbf_kernel(X_test, X_test, params['lengthscale'], params['variance']).diag()
#             Kxx = rbf_kernel(self.X, self.X, params['lengthscale'], params['variance'])
#             Kxx += torch.eye(self.N, device=self.X.device) * 1e-6
#             Lxx = torch.linalg.cholesky(Kxx)

#             Kxx_inv_mf = torch.cholesky_solve(self.m_f, Lxx)
#             pred_mean = K_star_x @ Kxx_inv_mf

#             S_f = self.L_f @ self.L_f.T
#             Kxx_inv_k_x_star = torch.cholesky_solve(K_star_x.T, Lxx)
            
#             var_from_q_f = (K_star_x @ torch.cholesky_solve(S_f @ Kxx_inv_k_x_star, Lxx)).diag()
            
#             E_inv_r = self.beta_r / (self.alpha_r - 1.0) if self.alpha_r > 1 else self.beta_r
#             var_from_prior = E_inv_r * (K_star_star_diag - (K_star_x * Kxx_inv_k_x_star.T).sum(dim=1))

#             pred_var = var_from_prior + var_from_q_f
#             pred_nu = 2 * self.alpha_r

#             return pred_mean, pred_var.unsqueeze(1), pred_nu
        

class TPRTFullBatch:
    """
    Implementation of the full (non-sparse) Student-t Process regression model.
    This version is aligned with the structure of the SparseTPRTFullBatch class.
    """
    def __init__(self, X, y, nu_f=2.1, nu_e=2.1,
                 kernel_lengthscale=1.0, kernel_variance=1.0, likelihood_sigma=1.0):
        
        self.X = X
        self.y = y.view(-1, 1) # Ensure y is always of shape (N, 1)
        self.N, self.D = X.shape

        # --- Hyperparameters as nn.Parameter for optimization ---
        self.log_kernel_lengthscale = nn.Parameter(torch.log(torch.tensor(kernel_lengthscale, dtype=X.dtype)))
        self.log_kernel_variance = nn.Parameter(torch.log(torch.tensor(kernel_variance, dtype=X.dtype)))
        self.log_likelihood_sigma_sq = nn.Parameter(torch.log(torch.tensor(likelihood_sigma**2, dtype=X.dtype)))
        self.log_nu_f = nn.Parameter(torch.log(torch.tensor(nu_f, dtype=X.dtype)))
        self.log_nu_e = nn.Parameter(torch.log(torch.tensor(nu_e, dtype=X.dtype)))

        # --- Variational Parameters (updated via CAVI) ---
        self.m_f = torch.zeros(self.N, 1, dtype=X.dtype, device=X.device)
        self.L_f = torch.eye(self.N, dtype=X.dtype, device=X.device)
        self.alpha_r = torch.tensor(1.0, dtype=X.dtype, device=X.device)
        self.beta_r = torch.tensor(1.0, dtype=X.dtype, device=X.device)
        self.alpha_lambda = torch.ones(self.N, 1, dtype=X.dtype, device=X.device)
        self.beta_lambda = torch.ones(self.N, 1, dtype=X.dtype, device=X.device)

    def _get_hyperparams(self):
        """Helper to get positive hyperparameters from their log-transformed storage."""
        return {
            "lengthscale": torch.exp(self.log_kernel_lengthscale),
            "variance": torch.exp(self.log_kernel_variance),
            "sigma_sq": torch.exp(self.log_likelihood_sigma_sq),
            "nu_f": torch.exp(self.log_nu_f),
            "nu_e": torch.exp(self.log_nu_e)
        }

    def _update_q_lambda(self, params):
        """CAVI update for q(lambda_i). Receives params as an argument."""
        # params = self._get_hyperparams() <- 削除
        S_f = self.L_f @ self.L_f.T
        var_f = S_f.diag().unsqueeze(1)
        expected_f_mean = self.m_f

        self.alpha_lambda = params['nu_e'] / 2.0 + 0.5
        expected_sq_error = (self.y - expected_f_mean).pow(2) + var_f
        self.beta_lambda = params['nu_e'] / 2.0 + (1.0 / (2.0 * params['sigma_sq'])) * expected_sq_error

    def _update_q_r(self, params, Lxx):
        """CAVI update for q(r). Receives params and Lxx as arguments."""
        # params = self._get_hyperparams() <- 削除
        S_f = self.L_f @ self.L_f.T
        
        trace_term = torch.trace(torch.cholesky_solve(S_f, Lxx))
        mean_term = self.m_f.T @ torch.cholesky_solve(self.m_f, Lxx)
        expected_f_quadratic_form = trace_term + mean_term

        self.alpha_r = params['nu_f'] / 2.0 + self.N / 2.0
        self.beta_r = params['nu_f'] / 2.0 + 0.5 * expected_f_quadratic_form.squeeze()

    def _update_q_f(self, params, Lxx):
        """CAVI update for q(f). Receives params and Lxx as arguments."""
        # params = self._get_hyperparams() <- 削除
        expected_r = self.alpha_r / self.beta_r
        expected_lambda = self.alpha_lambda / self.beta_lambda

        Kxx_inv = torch.cholesky_inverse(Lxx)
        S_f_inv = expected_r * Kxx_inv + torch.diag(expected_lambda.squeeze() / params['sigma_sq'])
        
        L_f_inv = torch.linalg.cholesky(S_f_inv)
        self.L_f = torch.triangular_solve(torch.eye(self.N, device=self.X.device), L_f_inv, upper=False)[0]

        temp_vec = (expected_lambda / params['sigma_sq']) * self.y
        self.m_f = torch.cholesky_solve(temp_vec, L_f_inv)

    def _cavi_step(self, params, Lxx):
        """
        Wrapper for a single CAVI update cycle.
        This follows the structure of the sparse model template.
        """
        self._update_q_lambda(params)
        self._update_q_r(params, Lxx)
        self._update_q_f(params, Lxx)
        
    def _e_step(self, cavi_max_iter=10, cavi_tol=1e-5):
        """
        Performs the E-Step by running CAVI updates.
        This method now computes its own kernel matrix based on the current hyperparameters.
        """
        with torch.no_grad():
            # 1. Get hyperparameters at the start of the E-step
            params = self._get_hyperparams()
            
            # 2. Compute common components for the CAVI loop
            Kxx = rbf_kernel(self.X, self.X, params['lengthscale'], params['variance'])
            Kxx += torch.eye(self.N, device=self.X.device) * 1e-6
            Lxx = torch.linalg.cholesky(Kxx)

            # 3. Run the CAVI update loop
            for _ in range(cavi_max_iter):
                m_f_prev = self.m_f.clone()
                self._cavi_step(params, Lxx)
                m_f_rel_change = torch.norm(self.m_f - m_f_prev) / (torch.norm(m_f_prev) + 1e-8)
                if m_f_rel_change < cavi_tol:
                    # print(f"CAVI converged in {_+1} iterations with relative change {m_f_rel_change:.6f}.")
                    break

    def _m_step(self, optimizer):
        """Performs the M-Step by updating hyperparameters."""
        optimizer.zero_grad()
        elbo = self._calculate_elbo()
        loss = -elbo
        loss.backward()
        optimizer.step()
        return elbo.item()

    def fit(self, max_iter_global=100, cavi_max_iter=10, cavi_tol=1e-5, lr=0.01):
        """Fits the model using the Variational EM algorithm."""
        optimizer = torch.optim.Adam([
            self.log_kernel_lengthscale, self.log_kernel_variance,
            self.log_likelihood_sigma_sq, self.log_nu_f, self.log_nu_e
        ], lr=lr)

        elbo_history = []
        print("Starting Variational EM optimization...")
        for i in range(max_iter_global):
            self._e_step(cavi_max_iter=cavi_max_iter, cavi_tol=cavi_tol)
            elbo = self._m_step(optimizer)
            elbo_history.append(elbo)
            
            if (i + 1) % 50 == 0:
                print(f"EM Iteration {i+1}/{max_iter_global}, ELBO: {elbo:.4f}")
                params = self._get_hyperparams()
                nu_f, nu_e = params['nu_f'].item(), params['nu_e'].item()
                lengthscale, variance = params['lengthscale'].item(), params['variance'].item()
                sigma = torch.sqrt(params['sigma_sq']).item()
                print(f"Current hyperparams: nu_f={nu_f:.2f}, nu_e={nu_e:.2f}, "
                      f"lengthscale={lengthscale:.3f}, variance={variance:.3f}, sigma={sigma:.3f}")

        print("\nOptimization finished.")
        return elbo_history

    def _calculate_elbo(self):
        """
        Calculates the Evidence Lower Bound (ELBO).
        This version strictly maintains the original calculation logic.
        """
        params = self._get_hyperparams()
        Kxx = rbf_kernel(self.X, self.X, params['lengthscale'], params['variance'])
        Kxx += torch.eye(self.N, device=self.X.device) * 1e-6
        Lxx = torch.linalg.cholesky(Kxx)
        S_f = self.L_f @ self.L_f.T

        # --- 1. Expected Log Likelihood (E[log p(y|f, lambda)]) ---
        E_q_f = self.m_f
        Var_q_f = S_f.diag().unsqueeze(1)
        expected_sq_error = (self.y - E_q_f).pow(2) + Var_q_f

        E_lambda = self.alpha_lambda / self.beta_lambda
        E_log_lambda = torch.digamma(self.alpha_lambda) - torch.log(self.beta_lambda)

        e_log_lik = -0.5 * self.N * math.log(2 * math.pi) - 0.5 * self.N * torch.log(params['sigma_sq']) + \
                    0.5 * torch.sum(E_log_lambda) - \
                    0.5 / params['sigma_sq'] * torch.sum(E_lambda * expected_sq_error)

        # --- 2. KL Divergence for f and r (as in original code) ---
        # Note: This part calculates -(E[log p(f,r)] - E[log q(f,r)])
        E_r = self.alpha_r / self.beta_r
        E_log_r = torch.digamma(self.alpha_r) - torch.log(self.beta_r)

        # E[log q(f)]
        # Original code simplifies this term, so we follow it.
        # It's related to the entropy of a Gaussian.
        log_q_f = -torch.sum(torch.log(torch.diag(self.L_f)))
        
        # E[log q(r)]
        log_q_r = self.alpha_r * torch.log(self.beta_r) - torch.lgamma(self.alpha_r) + \
                  (self.alpha_r - 1) * E_log_r - self.beta_r * E_r

        # E[log p(f|r)]
        trace_term = torch.trace(torch.cholesky_solve(S_f, Lxx))
        quad_form_term = self.m_f.T @ torch.cholesky_solve(self.m_f, Lxx)
        E_quad_form_f = trace_term + quad_form_term
        log_det_Kxx = 2 * torch.sum(torch.log(torch.diag(Lxx)))
        E_log_p_f_r = -0.5 * log_det_Kxx + 0.5 * self.N * E_log_r - 0.5 * E_r * E_quad_form_f

        # E[log p(r)]
        p_alpha_r, p_beta_r = params['nu_f'] / 2.0, params['nu_f'] / 2.0
        E_log_p_r = p_alpha_r * torch.log(p_beta_r) - torch.lgamma(p_alpha_r) + \
                    (p_alpha_r - 1) * E_log_r - p_beta_r * E_r

        # The constant -0.5 * N * log(2*pi) from log q(f) and E[log p(f|r)] cancel out.
        # Original grouping: log_q_f + log_q_r - E_log_p_f_r - E_log_p_r
        kl_f_r_grouped = (log_q_f + log_q_r) - (E_log_p_f_r + E_log_p_r)

        # --- 3. KL Divergence for lambda_i (as in original code) ---
        # Note: This part calculates - sum(E[log p(lambda_i)] - E[log q(lambda_i)])
        p_alpha_lambda, p_beta_lambda = params['nu_e'] / 2.0, params['nu_e'] / 2.0
        
        # Replicating the exact structure from the original snippet
        kl_lambda = torch.lgamma(self.alpha_lambda) - self.alpha_lambda * torch.log(self.beta_lambda) - \
                    (torch.lgamma(p_alpha_lambda) - p_alpha_lambda * torch.log(p_beta_lambda)) - \
                    (self.alpha_lambda - p_alpha_lambda) * E_log_lambda + \
                    (self.beta_lambda - p_beta_lambda) * E_lambda

        kl_lambda_sum = torch.sum(kl_lambda)

        # Combine terms exactly as in the original formulation
        elbo = e_log_lik - kl_f_r_grouped - kl_lambda_sum
        
        return elbo


    def predict(self, X_test):
        """Make predictions at new test points X_test."""
        with torch.no_grad():
            params = self._get_hyperparams()
            K_star_x = rbf_kernel(X_test, self.X, params['lengthscale'], params['variance'])
            K_star_star_diag = rbf_kernel(X_test, X_test, params['lengthscale'], params['variance']).diag()
            Kxx = rbf_kernel(self.X, self.X, params['lengthscale'], params['variance'])
            Kxx += torch.eye(self.N, device=self.X.device) * 1e-6
            Lxx = torch.linalg.cholesky(Kxx)

            Kxx_inv_mf = torch.cholesky_solve(self.m_f, Lxx)
            pred_mean = K_star_x @ Kxx_inv_mf

            S_f = self.L_f @ self.L_f.T
            Kxx_inv_k_x_star = torch.cholesky_solve(K_star_x.T, Lxx)
            
            var_from_q_f = (K_star_x @ torch.cholesky_solve(S_f @ Kxx_inv_k_x_star, Lxx)).diag()
            
            E_inv_r = self.beta_r / (self.alpha_r - 1.0) if self.alpha_r > 1 else self.beta_r
            var_from_prior = E_inv_r * (K_star_star_diag - (K_star_x * Kxx_inv_k_x_star.T).sum(dim=1))

            pred_var = var_from_prior + var_from_q_f
            pred_nu = 2 * self.alpha_r

            return pred_mean, pred_var.unsqueeze(1), pred_nu

    def evaluate_model(self, max_iter_global=100, cavi_max_iter=10, cavi_tol=1e-5, lr=0.01,
                       X_test=None, y_test=None, eval_interval=100,
                       result_path=None):
        """
        Fits the model and periodically evaluates/saves performance on test data.
        This method is resilient to timeouts by appending results to a file.

        Args:
            max_iter_global (int): Total number of EM iterations.
            cavi_max_iter (int): Max iterations for the inner CAVI loop.
            cavi_tol (float): Convergence tolerance for CAVI.
            lr (float): Learning rate for the optimizer.
            X_test (torch.Tensor, optional): Test inputs for evaluation.
            y_test (torch.Tensor, optional): Test targets for evaluation.
            eval_interval (int): The interval (in iterations) at which to evaluate.
            result_path (pathlib.Path, optional): Path to save the intermediate results CSV.
                                                  If provided, results are appended.
        """
        # --- オプティマイザのセットアップ ---
        # パラメータ名を元の `log_nu_f` と `log_nu_e` に修正
        optimizer = torch.optim.Adam([
            self.log_kernel_lengthscale, self.log_kernel_variance,
            self.log_likelihood_sigma_sq, self.log_nu_f, self.log_nu_e
        ], lr=lr)

        # --- タイムアウト対応のためのセットアップ ---
        can_evaluate = X_test is not None and y_test is not None and result_path is not None
        if can_evaluate:
            # ファイルが存在しない場合はヘッダーを書き込む
            if not result_path.exists():
                # ヘッダーを書き込む前にディレクトリが存在することを確認
                result_path.parent.mkdir(parents=True, exist_ok=True)
                with open(result_path, 'w') as f:
                    f.write("iteration,rmse\n")
        # -----------------------------------------

        for i in range(max_iter_global):
            # 1. 学習ステップ
            self._e_step(cavi_max_iter=cavi_max_iter, cavi_tol=cavi_tol)
            elbo_for_log = self._m_step(optimizer) # _m_stepはELBOを返す
            
            # 2. 定期的な評価と保存
            if can_evaluate and (i + 1) % eval_interval == 0:
                with torch.no_grad():
                    pred_mean, _, _ = self.predict(X_test)
                    rmse = torch.sqrt(torch.mean((y_test.view(-1) - pred_mean.view(-1))**2)).item()
                
                logging.info(f"EM Iteration {i+1}/{max_iter_global}, ELBO: {elbo_for_log:.4f}, Test RMSE: {rmse:.4f}")
                
                # 結果をファイルに追記
                with open(result_path, 'a') as f:
                    f.write(f"{i+1},{rmse}\n")
        
        # 3. 最終状態の評価と保存 (ループが正常に完了した場合)
        if can_evaluate:
             with torch.no_grad():
                pred_mean, _, _ = self.predict(X_test)
                rmse = torch.sqrt(torch.mean((y_test.view(-1) - pred_mean.view(-1))**2)).item()
             
             # 最後のイテレーションが評価間隔の倍数でない場合のみ、最終結果を追記
             if max_iter_global % eval_interval != 0:
                 with open(result_path, 'a') as f:
                    f.write(f"{max_iter_global},{rmse}\n")
        



# if __name__ == '__main__':

#     import matplotlib.pyplot as plt
#     from scipy.stats import t
#     torch.set_default_dtype(torch.float64)

#     # 1. Generate data
#     N = 60
#     X_train = torch.linspace(-5, 5, N).unsqueeze(1)
#     y_true = torch.sin(X_train) * 2
#     noise = torch.randn(N, 1) * 0.5
#     t_dist_sample = torch.distributions.StudentT(df=2)
#     outlier_noise = t_dist_sample.sample((N, 1)) * 0.3
#     y_train = y_true + noise
    
#     # Add some significant outliers
#     outlier_indices = torch.randperm(N)[:8]
#     y_train[outlier_indices] += outlier_noise[outlier_indices] * 3
#     y_train[15] = -4.0
#     y_train[45] = 5.0

#     # 2. Setup the non-sparse model
#     model = TPRTFullBatch(
#         X=X_train,
#         y=y_train,
#         nu_f=3.0,
#         nu_e=3.0,
#         kernel_lengthscale=1.0,
#         kernel_variance=1.0,
#         likelihood_sigma=1.0
#     )

#     # 3. Fit the model
#     elbo_history = model.fit(max_iter_global=100, cavi_max_iter=15, lr=0.05)

#     # 4. Make predictions
#     X_test = torch.linspace(-6, 6, 200).unsqueeze(1)
#     pred_mean, pred_var, pred_nu = model.predict(X_test)

#     # 5. Visualize the results
#     plt.figure(figsize=(12, 8))

#     pred_scale = torch.sqrt(pred_var.clamp(min=1e-9))
#     df = pred_nu.item()
#     lower_quantile = t.ppf(0.025, df=df)
#     upper_quantile = t.ppf(0.975, df=df)
    
#     lower = pred_mean + lower_quantile * pred_scale
#     upper = pred_mean + upper_quantile * pred_scale

#     plt.fill_between(X_test.squeeze(), lower.squeeze(), upper.squeeze(), color='orange', alpha=0.3, label='95% Predictive Interval (Student-t)')
#     plt.plot(X_test, pred_mean, 'r-', lw=2, label='Predictive Mean')
#     plt.plot(X_train, y_train, 'kx', mew=2, label='Training Data (with outliers)')

#     plt.title('Full TP Regression (Standardized Format)', fontsize=16)
#     plt.legend(loc='upper left')
#     plt.grid(True)
#     plt.xlim(-6, 6)
#     plt.ylim(-6, 6)
    
#     plt.figure(figsize=(12, 6))
#     plt.plot(elbo_history)
#     plt.title("ELBO Convergence")
#     plt.xlabel("EM Iteration")
#     plt.ylabel("ELBO")
#     plt.grid(True)
    
#     plt.show()