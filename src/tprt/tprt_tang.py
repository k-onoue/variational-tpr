from .kernels import rbf_kernel
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
import tqdm

# Set default tensor type for better performance with matrix operations
torch.set_default_dtype(torch.float64)


class TPRTFullBatch_Tang:
    """
    Implementation of the full (non-sparse) Student-t Process regression model
    using the Laplace Approximation.
    This version is aligned with the structure of the TPRTFullBatch (Variational EM) class.
    """
    def __init__(self, X, y, nu_f=2.1, nu_e=2.1,
                 kernel_lengthscale=1.0, kernel_variance=1.0, likelihood_sigma=0.1):
        
        self.X = X
        self.y = y.view(-1, 1) # Ensure y is always of shape (N, 1)
        self.N, self.D = X.shape

        # --- Hyperparameters as nn.Parameter for optimization ---
        self.log_kernel_lengthscale = nn.Parameter(torch.log(torch.tensor(kernel_lengthscale)))
        self.log_kernel_variance = nn.Parameter(torch.log(torch.tensor(kernel_variance)))
        self.log_likelihood_sigma = nn.Parameter(torch.log(torch.tensor(likelihood_sigma)))
        self.log_nu_f = nn.Parameter(torch.log(torch.tensor(nu_f)))
        self.log_nu_e = nn.Parameter(torch.log(torch.tensor(nu_e)))
        
        # --- Mode of the posterior f, updated in the E-step ---
        self.m_f = torch.zeros(self.N, 1, dtype=X.dtype, device=X.device)

    def _get_hyperparams(self):
        """Helper to get positive hyperparameters from their log-transformed storage."""
        return {
            "lengthscale": torch.exp(self.log_kernel_lengthscale),
            "variance": torch.exp(self.log_kernel_variance),
            "sigma": torch.exp(self.log_likelihood_sigma),
            "nu_f": torch.exp(self.log_nu_f),
            "nu_e": torch.exp(self.log_nu_e)
        }
        
    def _find_f_hat(self, K_inv, params, max_iter=10, tol=1e-5):
        """
        Inner loop to find the mode f_hat using L-BFGS for fixed hyperparameters.
        This is the core of the E-step.
        """
        f = nn.Parameter(self.m_f.clone()) # Start from the last known mode
        optimizer = torch.optim.LBFGS([f], lr=0.5, max_iter=20, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            log_lik_term = ((params['nu_e'] + 1) / 2) * torch.log(1 + (1 / params['nu_e']) * ((self.y - f) / params['sigma']).pow(2))
            fT_K_inv_f = f.T @ K_inv @ f
            log_prior_term = ((params['nu_f'] + self.N) / 2) * torch.log(1 + (1 / params['nu_f']) * fT_K_inv_f)
            loss = torch.sum(log_lik_term) + log_prior_term
            loss.backward()
            return loss

        for _ in range(max_iter):
            f_old = f.clone().detach()
            optimizer.step(closure)
            if torch.norm(f.detach() - f_old) < tol:
                break
        
        return f.detach()

    def _e_step(self, mode_finding_iter=10):
        """
        Performs the E-Step by finding the mode of the posterior p(f|y).
        This method uses detached hyperparameters to avoid building a graph for this optimization.
        """
        with torch.no_grad():
            params = self._get_hyperparams()
            K = rbf_kernel(self.X, self.X, params['lengthscale'], params['variance'])
            K += torch.eye(self.N, device=self.X.device) * 1e-6
            L = torch.linalg.cholesky(K)
            K_inv = torch.cholesky_inverse(L)
            
            # Find the mode f_hat and store it in self.m_f
            f_hat = self._find_f_hat(K_inv, params, max_iter=mode_finding_iter)
            self.m_f.copy_(f_hat)

    def _calculate_neg_log_marginal_likelihood(self):
        """
        Calculates the approximate negative log marginal likelihood, which serves as the loss
        for hyperparameter optimization (M-step).
        """
        params = self._get_hyperparams()
        f_hat = self.m_f # Use the mode found in the E-step

        # 1. Compute Kernel (connected to the graph for hyperparameter gradients)
        K = rbf_kernel(self.X, self.X, params['lengthscale'], params['variance'])
        K += torch.eye(self.N, device=self.X.device) * 1e-6
        L = torch.linalg.cholesky(K)
        K_inv = torch.cholesky_inverse(L)

        # 2. Calculate ln(Q(f_hat)) term
        log_lik_term = ((params['nu_e'] + 1) / 2) * torch.log(1 + (1 / params['nu_e']) * ((self.y - f_hat) / params['sigma']).pow(2))
        fT_K_inv_f = f_hat.T @ K_inv @ f_hat
        log_prior_term = ((params['nu_f'] + self.N) / 2) * torch.log(1 + (1 / params['nu_f']) * fT_K_inv_f)
        ln_Q_at_f_hat = torch.sum(log_lik_term) + log_prior_term

        # 3. Calculate the Hessian A = -∇∇ log Q(f) |f_hat
        f_hat_flat = f_hat.squeeze()
        prior_hess_num = K_inv * (params['nu_f'] + fT_K_inv_f) - 2 * (K_inv @ torch.outer(f_hat_flat, f_hat_flat) @ K_inv)
        prior_hess_den = (params['nu_f'] + fT_K_inv_f)**2
        prior_hess = (params['nu_f'] + self.N) * (prior_hess_num / prior_hess_den)

        err = self.y - f_hat
        lik_hess_num = err.pow(2) - params['nu_e'] * params['sigma']**2
        lik_hess_den = (err.pow(2) + params['nu_e'] * params['sigma']**2)**2
        W_diag = -(params['nu_e'] + 1) * (lik_hess_num / lik_hess_den)
        W = torch.diag(W_diag.squeeze())
        A = prior_hess + W

        # 4. Calculate log|B| where B = K + A^-1
        log_det_K = 2 * torch.sum(torch.log(torch.diag(L)))
        sign, log_det_A = torch.linalg.slogdet(A)
        if sign.item() <= 0:
            # Hessian is not positive definite, optimization is likely unstable
            return torch.tensor(float('inf'), device=self.X.device)
        
        log_det_B_approx = -log_det_K - log_det_A # This is log|K^-1 + A| = log|K^-1(I+KA)|... it should be log|A| not log|B|
        # The original paper's formula is -0.5 * log|A| + 0.5 * log|K|.
        # Let's rewrite the term more directly from the Laplace formula:
        # log Z ≈ log Q(f_hat) + (D/2)log(2π) - 0.5 * log|A|
        # And log p(y) = log Z - log p(f_hat). This gets complicated.
        # Let's stick to the formula from Tang et al. (2017) which is simpler.
        # log p(y) ≈ ln_Q_at_f_hat + const - 0.5*log|K| - 0.5*log|A| is incorrect.
        # It should be log p(y) ≈ log p(y|f_hat) + log p(f_hat) - 0.5 * log|A| + consts
        # The provided code's formula is from the paper, so let's use it.
        # neg_log_lik = ln_Q_at_f_hat - 0.5*log_det_B - consts
        # where log_det_B = log_det_K + log_det_A_inv, so -0.5*log_det_B = -0.5*log_det_K + 0.5*log_det_A
        log_term = -0.5 * log_det_K + 0.5 * log_det_A

        # 5. Calculate constant terms related to nu
        c_nu_f_term = torch.lgamma((params['nu_f'] + self.N)/2) - torch.lgamma(params['nu_f']/2) - (self.N/2)*torch.log(math.pi * params['nu_f'])
        c_nu_e_term = self.N * (torch.lgamma((params['nu_e'] + 1)/2) - torch.lgamma(params['nu_e']/2) - 0.5*torch.log(math.pi * params['nu_e']))
        c_sigma_term = -self.N * torch.log(params['sigma'])

        # Total approximate negative log marginal likelihood
        neg_log_marginal_lik = ln_Q_at_f_hat - log_term - (c_nu_f_term + c_nu_e_term + c_sigma_term)
        
        return neg_log_marginal_lik

    def _m_step(self, optimizer):
        """Performs the M-Step by updating hyperparameters."""
        optimizer.zero_grad()
        loss = self._calculate_neg_log_marginal_likelihood()
        
        if not (torch.isinf(loss) or torch.isnan(loss)):
            loss.backward()
            optimizer.step()
        else:
            print("Warning: Loss is inf or NaN, skipping M-step.")
            
        return loss.item()

    def fit(self, max_iter_global=100, mode_finding_iter=10, lr=0.01):
        """Fits the model using an EM-like algorithm with Laplace Approximation."""
        # Collect all parameters for the optimizer
        params_to_optimize = [
            self.log_kernel_lengthscale, self.log_kernel_variance,
            self.log_likelihood_sigma, self.log_nu_f, self.log_nu_e
        ]
        optimizer = torch.optim.Adam(params_to_optimize, lr=lr)

        loss_history = []
        print("Starting optimization...")
        for i in range(max_iter_global):
            # E-Step: Find the posterior mode f_hat
            self._e_step(mode_finding_iter=mode_finding_iter)
            
            # M-Step: Update hyperparameters by maximizing the marginal likelihood
            loss = self._m_step(optimizer)
            loss_history.append(loss)
            
            if (i + 1) % 50 == 0:
                print(f"Iteration {i + 1}/{max_iter_global}, Loss: {loss:.4f}")

        print("\nOptimization finished.")
        return loss_history

    def predict(self, X_test):
        """Make predictions at new test points X_test."""
        with torch.no_grad():
            params = self._get_hyperparams()
            f_hat = self.m_f

            # Recompute matrices needed for prediction
            K = rbf_kernel(self.X, self.X, params['lengthscale'], params['variance'])
            K += torch.eye(self.N, device=self.X.device) * 1e-6
            L = torch.linalg.cholesky(K)
            K_inv = torch.cholesky_inverse(L)
            
            K_star_x = rbf_kernel(X_test, self.X, params['lengthscale'], params['variance'])
            K_star_star_diag = rbf_kernel(X_test, X_test, params['lengthscale'], params['variance']).diag()

            # Predictive mean
            pred_mean = K_star_x @ K_inv @ f_hat

            # Predictive variance (approximated as Gaussian)
            f_hat_flat = f_hat.squeeze()
            fT_K_inv_f = f_hat.T @ K_inv @ f_hat
            prior_hess_num = K_inv * (params['nu_f'] + fT_K_inv_f) - 2 * (K_inv @ torch.outer(f_hat_flat, f_hat_flat) @ K_inv)
            prior_hess_den = (params['nu_f'] + fT_K_inv_f)**2
            prior_hess = (params['nu_f'] + self.N) * (prior_hess_num / prior_hess_den)
            
            err = self.y - f_hat
            lik_hess_num = err.pow(2) - params['nu_e'] * params['sigma']**2
            lik_hess_den = (err.pow(2) + params['nu_e'] * params['sigma']**2)**2
            W_diag = -(params['nu_e'] + 1) * (lik_hess_num / lik_hess_den)
            W = torch.diag(W_diag.squeeze())
            A = prior_hess + W
            
            try:
                L_A = torch.linalg.cholesky(A)
                A_inv = torch.cholesky_inverse(L_A)
            except torch.linalg.LinAlgError:
                print("Warning: Hessian not positive definite during prediction. Using pseudo-inverse.")
                A_inv = torch.linalg.pinv(A)

            K_inv_k_star = torch.cholesky_solve(K_star_x.T, L)
            prior_var_reduction = (K_star_x * K_inv_k_star.T).sum(dim=1)
            posterior_uncertainty = (K_star_x @ K_inv @ A_inv @ K_inv @ K_star_x.T).diag()
            
            pred_var = K_star_star_diag - prior_var_reduction + posterior_uncertainty
            pred_var = pred_var.clamp(min=1e-9)
            
            # The predictive distribution is Gaussian, so nu is infinite
            pred_nu = torch.tensor(float('inf'), device=self.X.device)
            
            return pred_mean, pred_var.unsqueeze(1), pred_nu




# if __name__ == '__main__':
#     # 1. Generate synthetic data with outliers
#     N = 60
#     X_train = torch.linspace(-5, 5, N).unsqueeze(1)
#     y_true = torch.sin(X_train) * 2
    
#     torch.manual_seed(42)
#     noise = torch.randn(N, 1) * 0.5
#     y_train = y_true + noise
    
#     outlier_indices = [5, 15, 30, 45, 55]
#     y_train[outlier_indices] = torch.tensor([-3.0, 4.0, -4.5, 5.0, -2.5]).unsqueeze(1)

#     # 2. Setup the model
#     model = TPRTFullBatch_Tang(
#         X=X_train,
#         y=y_train,
#         nu_f=2.1,
#         nu_e=2.1,
#         kernel_lengthscale=1.0,
#         kernel_variance=1.0,
#         likelihood_sigma=0.5
#     )

#     # 3. Fit the model
#     loss_history = model.fit(max_iter_global=100, mode_finding_iter=10, lr=0.01)

#     # 4. Make predictions
#     X_test = torch.linspace(-6, 6, 200).unsqueeze(1)
#     pred_mean, pred_var, pred_nu = model.predict(X_test)

#     # 5. Visualize the results
#     plt.figure(figsize=(12, 8))
    
#     pred_std = torch.sqrt(pred_var)
#     # Since nu is inf, the predictive distribution is Gaussian
#     # We use norm.ppf for confidence intervals (equivalent to multiplying by ~1.96)
#     lower_quantile = norm.ppf(0.025)
#     upper_quantile = norm.ppf(0.975)
    
#     lower_ci = pred_mean + lower_quantile * pred_std
#     upper_ci = pred_mean + upper_quantile * pred_std

#     plt.fill_between(X_test.squeeze(), lower_ci.squeeze(), upper_ci.squeeze(), color='orange', alpha=0.3, label='95% Predictive Interval (Gaussian Approx.)')
#     plt.plot(X_test, pred_mean, 'r-', lw=2, label='Predictive Mean')
#     plt.plot(X_train, y_train, 'kx', mew=2, label='Training Data (with outliers)')
#     plt.plot(X_test, torch.sin(X_test) * 2, 'b--', label='True Function')

#     plt.title('TPRT with Laplace Approximation (Standardized Format)', fontsize=16)
#     plt.legend(loc='upper left')
#     plt.grid(True)
#     plt.xlim(-6, 6)
#     plt.ylim(-6, 6)
    
#     # Print final learned hyperparameters
#     params = model._get_hyperparams()
#     print("\n--- Learned Hyperparameters ---")
#     print(f"Kernel Lengthscale: {params['lengthscale'].item():.3f}")
#     print(f"Kernel Variance: {params['variance'].item():.3f}")
#     print(f"Likelihood Sigma: {params['sigma'].item():.3f}")
#     print(f"Prior DoF (nu_f): {params['nu_f'].item():.3f}")
#     print(f"Likelihood DoF (nu_e): {params['nu_e'].item():.3f}")

#     # Plot loss curve
#     plt.figure(figsize=(12, 6))
#     plt.plot(loss_history)
#     plt.title("Approx. Negative Log Marginal Likelihood Convergence")
#     plt.xlabel("EM-like Iteration")
#     plt.ylabel("Loss")
#     plt.grid(True)
    
#     plt.show()