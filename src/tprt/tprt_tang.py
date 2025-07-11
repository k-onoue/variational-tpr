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
    using the Laplace Approximation, based on Tang et al. (2017).

    This extended version allows selecting the optimizer (LBFGS or Adam) for both
    the inner loop (mode-finding) and the outer loop (hyperparameter tuning).
    """
    def __init__(self, X, y, nu_f=2.1, nu_e=2.1,
                 kernel_lengthscale=1.0, kernel_variance=1.0, likelihood_sigma=1.0,
                 mode_optimizer='lbfgs', hyper_optimizer='adam'):
        """
        Initializes the model.

        Args:
            X (torch.Tensor): Training inputs, shape (N, D).
            y (torch.Tensor): Training outputs, shape (N,).
            nu_f (float): Initial prior degrees of freedom.
            nu_e (float): Initial likelihood degrees of freedom.
            kernel_lengthscale (float): Initial kernel lengthscale.
            kernel_variance (float): Initial kernel variance.
            likelihood_sigma (float): Initial likelihood noise standard deviation.
            mode_optimizer (str): Optimizer for the inner loop ('lbfgs' or 'adam').
            hyper_optimizer (str): Optimizer for the outer loop ('lbfgs' or 'adam').
        """
        self.X = X
        self.y = y.view(-1, 1) # Ensure y is always of shape (N, 1)
        self.N, self.D = X.shape

        # --- Hyperparameters as nn.Parameter for optimization ---
        self.log_kernel_lengthscale = nn.Parameter(torch.log(torch.tensor(kernel_lengthscale)))
        self.log_kernel_variance = nn.Parameter(torch.log(torch.tensor(kernel_variance)))
        self.log_likelihood_sigma = nn.Parameter(torch.log(torch.tensor(likelihood_sigma)))
        self.log_nu_f = nn.Parameter(torch.log(torch.tensor(nu_f)))
        self.log_nu_e = nn.Parameter(torch.log(torch.tensor(nu_e)))
        
        # --- Optimizer Choices ---
        self.mode_optimizer_name = mode_optimizer.lower()
        self.hyper_optimizer_name = hyper_optimizer.lower()
        
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
        Inner loop to find the mode f_hat using the chosen optimizer.
        This is the core of the E-step.
        """
        f = nn.Parameter(self.m_f.clone()) # Start from the last known mode

        def calculate_loss():
            """Calculates the negative log posterior of f."""
            log_lik_term = ((params['nu_e'] + 1) / 2) * torch.log(1 + (1 / params['nu_e']) * ((self.y - f) / params['sigma']).pow(2))
            fT_K_inv_f = f.T @ K_inv @ f
            log_prior_term = ((params['nu_f'] + self.N) / 2) * torch.log(1 + (1 / params['nu_f']) * fT_K_inv_f)
            return torch.sum(log_lik_term) + log_prior_term

        if self.mode_optimizer_name == 'lbfgs':
            optimizer = torch.optim.LBFGS([f], lr=0.5, max_iter=20, line_search_fn="strong_wolfe")
            def closure():
                optimizer.zero_grad()
                loss = calculate_loss()
                loss.backward()
                return loss
            
            for _ in range(max_iter):
                f_old = f.clone().detach()
                optimizer.step(closure)
                if torch.norm(f.detach() - f_old) < tol:
                    break
        
        elif self.mode_optimizer_name == 'adam':
            optimizer = torch.optim.Adam([f], lr=0.1)
            # Adam often requires more iterations than L-BFGS for this type of problem
            for _ in range(max_iter * 10):
                f_old = f.clone().detach()
                optimizer.zero_grad()
                loss = calculate_loss()
                loss.backward()
                optimizer.step()
                if torch.norm(f.detach() - f_old) < tol:
                    break
        else:
            raise ValueError(f"Unknown mode optimizer: '{self.mode_optimizer_name}'. Choose 'lbfgs' or 'adam'.")
        
        return f.detach()

    def _e_step(self, mode_finding_iter=10):
        """
        Performs the E-Step by finding the mode of the posterior p(f|y).
        This method uses detached hyperparameters to avoid building a graph for this optimization.
        """
        # Create a detached copy of the parameters for the E-step.
        # This prevents the inner optimization from affecting the main hyperparameter gradients.
        detached_params = {k: v.detach() for k, v in self._get_hyperparams().items()}

        # Compute K and K_inv using the detached parameters
        K = rbf_kernel(self.X, self.X, detached_params['lengthscale'], detached_params['variance'])
        K += torch.eye(self.N, device=self.X.device) * 1e-6
        L = torch.linalg.cholesky(K)
        K_inv = torch.cholesky_inverse(L)
        
        # Find the mode f_hat. This optimization builds its own local graph for `f`
        # and treats the detached params and K_inv as constants.
        f_hat = self._find_f_hat(K_inv, detached_params, max_iter=mode_finding_iter)
        
        # Update the mode. Use no_grad here to ensure this update is not tracked.
        with torch.no_grad():
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

        # 3. Calculate the Hessian A = -∇∇ log p(f|y) |f_hat
        f_hat_flat = f_hat.squeeze()
        prior_hess_num = K_inv * (params['nu_f'] + fT_K_inv_f) - 2 * (K_inv @ torch.outer(f_hat_flat, f_hat_flat) @ K_inv)
        prior_hess_den = (params['nu_f'] + fT_K_inv_f)**2
        prior_hess = (params['nu_f'] + self.N) * (prior_hess_num / prior_hess_den)

        err = self.y - f_hat
        lik_hess_num = err.pow(2) - params['nu_e'] * params['sigma']**2
        lik_hess_den = (err.pow(2) + params['nu_e'] * params['sigma']**2)**2
        W_diag = -(params['nu_e'] + 1) * (lik_hess_num / lik_hess_den)
        W = torch.diag(W_diag.squeeze())
        # Note: A is the *negative* Hessian of the log posterior
        A = -(prior_hess + W)

        # 4. Calculate log determinant term from Laplace approximation
        # The approx. log marginal likelihood is log p(y) ≈ log p(y, f_hat) - 0.5*log|A|
        # The paper's formula (Eq. 18) simplifies to: ln_Q_at_f_hat + 0.5*log|K| - 0.5*log|A| + consts
        log_det_K = 2 * torch.sum(torch.log(torch.diag(L)))
        sign, log_det_A = torch.linalg.slogdet(A)
        if sign.item() <= 0:
            return torch.tensor(float('inf'), device=self.X.device)
        
        log_det_term = 0.5 * log_det_K - 0.5 * log_det_A

        # 5. Calculate constant terms (which depend on hyperparameters)
        c_nu_f_term = torch.lgamma((params['nu_f'] + self.N)/2) - torch.lgamma(params['nu_f']/2) - (self.N/2)*torch.log(math.pi * params['nu_f'])
        c_nu_e_term = self.N * (torch.lgamma((params['nu_e'] + 1)/2) - torch.lgamma(params['nu_e']/2) - 0.5*torch.log(math.pi * params['nu_e']))
        c_sigma_term = -self.N * torch.log(params['sigma'])

        # Total approximate negative log marginal likelihood
        neg_log_marginal_lik = -(ln_Q_at_f_hat + log_det_term + c_nu_f_term + c_nu_e_term + c_sigma_term)
        
        return neg_log_marginal_lik

    def _m_step(self, optimizer):
        """Performs one M-Step update using the chosen optimizer."""
        def closure():
            optimizer.zero_grad()
            loss = self._calculate_neg_log_marginal_likelihood()
            if not (torch.isinf(loss) or torch.isnan(loss)):
                loss.backward()
            else:
                print("Warning: Loss is inf or NaN, skipping gradient calculation.")
            return loss

        if self.hyper_optimizer_name == 'adam':
            loss = closure()
            if not (torch.isinf(loss) or torch.isnan(loss)):
                optimizer.step()
            return loss.item()
        elif self.hyper_optimizer_name == 'lbfgs':
            loss = optimizer.step(closure)
            return loss.item()
        else:
            raise ValueError(f"Unknown hyper optimizer: '{self.hyper_optimizer_name}'. Choose 'lbfgs' or 'adam'.")


    def fit(self, max_iter_global=100, mode_finding_iter=10, lr=0.01):
        """Fits the model using an EM-like algorithm with Laplace Approximation."""
        params_to_optimize = [
            self.log_kernel_lengthscale, self.log_kernel_variance,
            self.log_likelihood_sigma, self.log_nu_f, self.log_nu_e
        ]
        
        if self.hyper_optimizer_name == 'adam':
            optimizer = torch.optim.Adam(params_to_optimize, lr=lr)
        elif self.hyper_optimizer_name == 'lbfgs':
            optimizer = torch.optim.LBFGS(params_to_optimize, lr=0.1, max_iter=10)
        else:
            raise ValueError(f"Unknown hyper optimizer: '{self.hyper_optimizer_name}'. Choose 'lbfgs' or 'adam'.")

        loss_history = []
        print(f"Starting optimization with mode_optimizer='{self.mode_optimizer_name}' and hyper_optimizer='{self.hyper_optimizer_name}'...")
        
        pbar = tqdm.tqdm(range(max_iter_global))
        for i in pbar:
            # E-Step: Find the posterior mode f_hat
            self._e_step(mode_finding_iter=mode_finding_iter)
            
            # M-Step: Update hyperparameters by maximizing the marginal likelihood
            loss = self._m_step(optimizer)
            loss_history.append(loss)
            pbar.set_description(f"Loss: {loss:.4f}")

        print("\nOptimization finished.")
        return loss_history

    def predict(self, X_test):
        """Make predictions at new test points X_test."""
        with torch.no_grad():
            params = self._get_hyperparams()
            f_hat = self.m_f

            K = rbf_kernel(self.X, self.X, params['lengthscale'], params['variance'])
            K += torch.eye(self.N, device=self.X.device) * 1e-6
            L = torch.linalg.cholesky(K)
            K_inv = torch.cholesky_inverse(L)
            
            K_star_x = rbf_kernel(X_test, self.X, params['lengthscale'], params['variance'])
            K_star_star_diag = rbf_kernel(X_test, X_test, params['lengthscale'], params['variance']).diag()

            # Predictive mean
            pred_mean = K_star_x @ K_inv @ f_hat

            # --- Predictive variance (approximated as Gaussian) ---
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
            A = -(prior_hess + W)
            
            try:
                L_A = torch.linalg.cholesky(A)
                # This term is needed for the predictive variance
                v = torch.cholesky_solve(K_inv @ K_star_x.T, L)
                posterior_uncertainty = (v.T @ torch.cholesky_solve(v, L_A)).diag()
            except torch.linalg.LinAlgError:
                print("Warning: Hessian not positive definite during prediction. Variance may be inaccurate.")
                posterior_uncertainty = torch.zeros(X_test.shape[0], device=self.X.device)

            prior_var_reduction = (K_star_x * (K_inv @ K_star_x.T).T).sum(dim=1)
            pred_var = K_star_star_diag - prior_var_reduction + posterior_uncertainty
            pred_var = pred_var.clamp(min=1e-9)
            
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

#     # 2. Setup and fit the model with chosen optimizers
#     # --- Experiment with different optimizers ---
#     # Common choice: L-BFGS for mode-finding, Adam for hyperparameters
#     # L-BFGS is often better for the inner loop as it's a deterministic subproblem.
#     model = TPRTFullBatch_Tang(
#         X=X_train,
#         y=y_train,
#         mode_optimizer='lbfgs',
#         hyper_optimizer='adam'
#     )
    
#     loss_history = model.fit(max_iter_global=90, mode_finding_iter=20, lr=0.01)

#     # 3. Make predictions
#     X_test = torch.linspace(-6, 6, 200).unsqueeze(1)
#     pred_mean, pred_var, pred_nu = model.predict(X_test)

#     # 4. Visualize the results
#     plt.figure(figsize=(12, 8))
    
#     pred_std = torch.sqrt(pred_var)
#     lower_ci = pred_mean - 1.96 * pred_std
#     upper_ci = pred_mean + 1.96 * pred_std

#     plt.fill_between(X_test.squeeze(), lower_ci.squeeze(), upper_ci.squeeze(), color='orange', alpha=0.3, label='95% Predictive Interval')
#     plt.plot(X_test, pred_mean, 'r-', lw=2, label='Predictive Mean')
#     plt.plot(X_train, y_train, 'kx', mew=2, label='Training Data (with outliers)')
#     plt.plot(X_test, torch.sin(X_test) * 2, 'b--', label='True Function')

#     plt.title(f"TPRT (Mode: {model.mode_optimizer_name.upper()}, Hyper: {model.hyper_optimizer_name.upper()})", fontsize=16)
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
