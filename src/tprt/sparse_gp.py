

def rbf_kernel(X1, X2, lengthscale, variance=1.0):
    """
    Computes the RBF kernel matrix with ARD support.

    Args:
        X1 (torch.Tensor): A tensor of size (N, D).
        X2 (torch.Tensor): A tensor of size (M, D).
        lengthscale (torch.Tensor): A tensor of size (D,) representing the lengthscale for each dimension.
        variance (float): The kernel variance.
    """
    # Ensure variance is a tensor on the correct device
    variance = torch.as_tensor(variance, dtype=X1.dtype, device=X1.device)
    
    # Scale each dimension of X1 and X2 by the corresponding lengthscale
    # This uses broadcasting to efficiently perform the operation
    X1_scaled = X1 / lengthscale
    X2_scaled = X2 / lengthscale
    
    # Compute the squared Euclidean distance in the scaled space
    sqdist = torch.cdist(X1_scaled, X2_scaled, p=2).pow(2)
    
    return variance * torch.exp(-0.5 * sqdist)


class SparseGP(nn.Module):
    """
    Original SVIGP implementation with a single, comprehensive evaluate_model method
    that handles both training and periodic evaluation logging.
    """
    def __init__(self, X, y, M, nu_f=None, nu_e=None,
                 kernel_lengthscale=None, kernel_variance=1.0,
                 likelihood_sigma=0.1, device=None):
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

        self.Z = nn.Parameter(self._initialize_inducing_points())
        if kernel_lengthscale is None:
            kernel_lengthscale = torch.ones(self.D, dtype=dtype)
        else:
            kernel_lengthscale = torch.as_tensor(kernel_lengthscale, dtype=dtype)
        if kernel_lengthscale.ndim == 0:
            kernel_lengthscale = kernel_lengthscale.repeat(self.D)
        self.log_kernel_lengthscale = nn.Parameter(torch.log(kernel_lengthscale))
        self.log_kernel_variance = nn.Parameter(torch.log(torch.tensor(kernel_variance, dtype=dtype)))
        self.log_beta = nn.Parameter(torch.log(torch.tensor(1.0 / likelihood_sigma**2, dtype=dtype)))

        self.register_buffer('m_u', torch.zeros(self.M, 1, dtype=dtype))
        self.register_buffer('L_u', torch.eye(self.M, dtype=dtype))

        self.kernel = rbf_kernel
        self.to(self.device)

    def _initialize_inducing_points(self):
        if self.N > self.M:
            X_np = self.X_full.cpu().numpy()
            kmeans = KMeans(n_clusters=self.M, random_state=0, n_init='auto').fit(X_np)
            Z_init = torch.from_numpy(kmeans.cluster_centers_)
        else:
            indices = np.random.choice(self.N, self.M, replace=False)
            Z_init = self.X_full[indices].clone()
        return Z_init.to(dtype=self.X_full.dtype, device=self.device)

    def _get_params(self):
        return {
            "lengthscale": torch.exp(self.log_kernel_lengthscale),
            "variance": torch.exp(self.log_kernel_variance),
            "beta": torch.exp(self.log_beta)
        }

    def _calculate_elbo(self, X_batch, y_batch):
        B = X_batch.shape[0]
        params = self._get_params()
        jitter = torch.eye(self.M, device=self.device, dtype=self.X_full.dtype) * 1e-6
        K_mm = self.kernel(self.Z, self.Z, params['lengthscale'], params['variance'])
        K_mb = self.kernel(self.Z, X_batch, params['lengthscale'], params['variance'])
        K_bb_diag = params['variance'] * torch.ones(B, device=self.device, dtype=self.X_full.dtype)
        L_Kmm = torch.linalg.cholesky(K_mm + jitter)
        A = torch.cholesky_solve(K_mb, L_Kmm)
        S_u = self.L_u @ self.L_u.T
        mu_b = A.T @ self.m_u
        log_lik_term = (
            0.5 * B * torch.log(params['beta']) - 0.5 * B * math.log(2 * math.pi) -
            0.5 * params['beta'] * ((y_batch - mu_b)**2).sum()
        )
        psi_trace_term = -0.5 * params['beta'] * (K_bb_diag.sum() - (A * K_mb).sum())
        S_trace_term = -0.5 * params['beta'] * (A.T @ S_u @ A).trace()
        scaled_log_lik = (log_lik_term + psi_trace_term + S_trace_term) * (self.N / B)
        K_mm_inv_S = torch.cholesky_solve(S_u, L_Kmm)
        trace_kl = K_mm_inv_S.trace()
        mahalanobis_kl = (self.m_u.T @ torch.cholesky_solve(self.m_u, L_Kmm)).squeeze()
        log_det_kl = 2 * torch.log(L_Kmm.diag()).sum() - 2 * torch.log(self.L_u.diag()).sum()
        kl_div = 0.5 * (trace_kl + mahalanobis_kl - self.M + log_det_kl)
        elbo = scaled_log_lik - kl_div
        return elbo

    def predict(self, X_test):
        X_test_dev = X_test.to(self.device)
        if X_test_dev.ndim == 1: X_test_dev = X_test_dev.unsqueeze(1)
        with torch.no_grad():
            params = self._get_params()
            jitter = torch.eye(self.M, device=self.device, dtype=self.X_full.dtype) * 1e-6
            K_mm = self.kernel(self.Z, self.Z, params['lengthscale'], params['variance'])
            K_sm = self.kernel(X_test_dev, self.Z, params['lengthscale'], params['variance'])
            K_ss_diag = params['variance'] * torch.ones(X_test_dev.shape[0], device=self.device, dtype=self.X_full.dtype)
            L_Kmm = torch.linalg.cholesky(K_mm + jitter)
            A = K_sm @ torch.inverse(K_mm + jitter)
            pred_mean = A @ self.m_u
            S_u = self.L_u @ self.L_u.T
            var_term1 = K_ss_diag
            var_term2 = -(A * K_sm).sum(dim=1)
            var_term3 = (A.T * (S_u @ A.T)).sum(dim=0)
            pred_var = var_term1 + var_term2 + var_term3
            return pred_mean.squeeze(), pred_var.squeeze(), None


    # --- New `evaluate_model` with integrated training and logging ---
    def evaluate_model(self, epochs=100, batch_size=128, lr=0.01,
                       X_test=None, y_test=None, eval_interval=10,
                       result_path=None, **kwargs): # Absorb unused kwargs

        # Use a fixed rho for SVI, as it's a model-specific hyperparameter
        rho = 0.01

        # NOTE: Unlike the T-PRT model, SVIGP optimizes all parameters together.
        optimizer = optim.Adam(self.parameters(), lr=lr)
        dataset = TensorDataset(self.X_full, self.y_full)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Setup for evaluation logging
        can_evaluate = X_test is not None and y_test is not None and result_path is not None
        if can_evaluate:
            result_path = Path(result_path)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            with open(result_path, 'w') as f:
                f.write("epoch,rmse,elbo\n")

        print(f"Starting SVIGP optimization for {epochs} epochs with evaluation...")
        
        jitter = torch.eye(self.M, device=self.device, dtype=self.X_full.dtype) * 1e-6
        elbo_history = []
        
        for epoch in range(epochs):
            self.train() # Set model to training mode
            elbo_val = 0.0

            for X_batch, y_batch in dataloader:
                B = X_batch.shape[0]
                params = self._get_params()
                
                # --- This is the original training logic from the SVIGP fit method ---
                with torch.no_grad():
                    K_mm = self.kernel(self.Z, self.Z, params['lengthscale'], params['variance'])
                    K_mb = self.kernel(self.Z, X_batch, params['lengthscale'], params['variance'])
                    K_mm_inv = torch.inverse(K_mm + jitter)
                    T = K_mm_inv @ K_mb
                    theta2_hat = -0.5 * (K_mm_inv + params["beta"] * (self.N / B) * (T @ T.T))
                    theta1_hat = params["beta"] * (self.N / B) * (T @ y_batch)
                    S_u_inv = torch.inverse(self.L_u @ self.L_u.T + jitter)
                    theta2_old = -0.5 * S_u_inv
                    theta1_old = S_u_inv @ self.m_u
                    theta2_new = (1 - rho) * theta2_old + rho * theta2_hat
                    theta1_new = (1 - rho) * theta1_old + rho * theta1_hat
                    S_u_inv_new = -2 * theta2_new
                    S_u_new = torch.inverse(S_u_inv_new + jitter)
                    L_u_new = torch.linalg.cholesky(S_u_new + jitter)
                    m_u_new = S_u_new @ theta1_new
                    self.m_u.data.copy_(m_u_new)
                    self.L_u.data.copy_(L_u_new)
                
                optimizer.zero_grad()
                elbo = self._calculate_elbo(X_batch, y_batch)
                loss = -elbo
                loss.backward()
                optimizer.step()
                elbo_val = elbo.item() # Keep track of the last elbo in the epoch
                elbo_history.append(elbo_val)
            
            # --- This is the periodic evaluation and logging logic ---
            if can_evaluate and (epoch + 1) % eval_interval == 0:
                self.eval() # Set model to evaluation mode
                with torch.no_grad():
                    pred_mean, _, _ = self.predict(X_test)
                    rmse = torch.sqrt(torch.mean((y_test.to(self.device).view(-1) - pred_mean.view(-1))**2)).item()
                
                log_msg = f"Epoch {epoch+1}/{epochs}, ELBO: {elbo_val:.4f}, Test RMSE: {rmse:.4f}"
                logging.info(log_msg)
                print(log_msg)
                
                with open(result_path, 'a') as f:
                    f.write(f"{epoch+1},{rmse},{elbo_val}\n")

        # Final evaluation after the last epoch
        if can_evaluate and epochs > 0 and epochs % eval_interval != 0:
            self.eval()
            with torch.no_grad():
                pred_mean, _, _ = self.predict(X_test)
                rmse = torch.sqrt(torch.mean((y_test.to(self.device).view(-1) - pred_mean.view(-1))**2)).item()
                print(f"Final RMSE after {epochs} epochs: {rmse:.4f}")
            with open(result_path, 'a') as f:
                f.write(f"{epochs},{rmse},{elbo_val}\n")
        
        print("\nOptimization finished.")
        return elbo_history







import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
import numpy as np
import gpytorch
from pathlib import Path
import logging

class SparseGP(nn.Module):
    """
    A wrapper for a standard GPyTorch SVGP model that provides an interface
    compatible with the SparseTPRTMiniBatch and custom SVIGP models.
    """

    # --- 1. The GPyTorch model is defined as a nested class ---
    class _GPModel(gpytorch.models.ApproximateGP):
        def __init__(self, inducing_points):
            variational_distribution = gpytorch.variational.NaturalVariationalDistribution(inducing_points.size(0))
            variational_strategy = gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            )
            super().__init__(variational_strategy)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # --- 2. The __init__ method matches the required interface ---
    def __init__(self, X, y, M, nu_f=None, nu_e=None, # Dummy args for compatibility
                 kernel_lengthscale=None, kernel_variance=None, # Unused, GPyTorch learns these
                 likelihood_sigma=0.1, device=None):
        super().__init__()

        if device is None:
            self.device = X.device if isinstance(X, torch.Tensor) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.register_buffer('X_full', X.to(self.device))
        # GPyTorch expects a 1D target tensor
        self.register_buffer('y_full', y.view(-1).to(self.device))

        if self.X_full.ndim == 1: self.X_full = self.X_full.unsqueeze(1)

        self.N, self.D = self.X_full.shape
        self.M = M

        # Initialize inducing points
        initial_inducing_points = self.X_full[torch.randperm(self.N)[:self.M]]

        # Initialize the GPyTorch model and likelihood
        self.model = self._GPModel(inducing_points=initial_inducing_points)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        
        # Set initial likelihood noise from likelihood_sigma
        self.likelihood.noise_covar.noise = torch.tensor(likelihood_sigma**2)

        self.to(self.device)

    # --- 3. The `predict` method matches the required interface ---
    def predict(self, X_test):
        X_test_dev = X_test.to(self.device)
        if X_test_dev.ndim == 1: X_test_dev = X_test_dev.unsqueeze(1)

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            latent_pred = self.model(X_test_dev)
            # The observed predictions include the likelihood noise
            # observed_pred = self.likelihood(latent_pred) 
            pred_mean = latent_pred.mean
            pred_var = latent_pred.variance

        return pred_mean.squeeze(), pred_var.squeeze(), None

    # --- 4. The `evaluate_model` method handles training and logging ---
    def evaluate_model(self, epochs=100, batch_size=128, lr=0.01,
                       X_test=None, y_test=None, eval_interval=10,
                       result_path=None, **kwargs): # Absorb unused kwargs
        
        self.train()
        self.likelihood.train()

        # Create DataLoader
        train_dataset = TensorDataset(self.X_full, self.y_full)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Set up optimizers
        variational_ngd_optimizer = gpytorch.optim.NGD(self.model.variational_parameters(), num_data=self.y_full.size(0), lr=0.1)
        hyperparameter_optimizer = torch.optim.Adam([
            {'params': self.model.hyperparameters()},
            {'params': self.likelihood.parameters()},
        ], lr=lr)

        # Objective function (ELBO)
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=self.y_full.size(0))
        
        # Setup for evaluation logging
        can_evaluate = X_test is not None and y_test is not None and result_path is not None
        if can_evaluate:
            result_path = Path(result_path)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            with open(result_path, 'w') as f:
                f.write("epoch,rmse,elbo\n")

        print(f"Starting GPyTorch SVGP optimization for {epochs} epochs with evaluation...")

        for epoch in range(epochs):
            elbo_val = 0.0
            for x_batch, y_batch in train_loader:
                variational_ngd_optimizer.zero_grad()
                hyperparameter_optimizer.zero_grad()

                output = self.model(x_batch)
                loss = -mll(output, y_batch)
                
                loss.backward()
                variational_ngd_optimizer.step()
                hyperparameter_optimizer.step()
                elbo_val = -loss.item()
            
            # --- Periodic evaluation and logging logic ---
            if can_evaluate and (epoch + 1) % eval_interval == 0:
                # Set to eval mode for prediction
                self.model.eval()
                self.likelihood.eval()
                with torch.no_grad():
                    pred_mean, _, _ = self.predict(X_test)
                    rmse = torch.sqrt(torch.mean((y_test.to(self.device).view(-1) - pred_mean.view(-1))**2)).item()
                
                log_msg = f"Epoch {epoch+1}/{epochs}, ELBO: {elbo_val:.4f}, Test RMSE: {rmse:.4f}"
                logging.info(log_msg)
                print(log_msg)
                
                with open(result_path, 'a') as f:
                    f.write(f"{epoch+1},{rmse},{elbo_val}\n")
                
                # Set back to train mode
                self.model.train()
                self.likelihood.train()

        # Final evaluation after the last epoch
        if can_evaluate and epochs > 0 and epochs % eval_interval != 0:
            self.model.eval()
            self.likelihood.eval()
            with torch.no_grad():
                pred_mean, _, _ = self.predict(X_test)
                rmse = torch.sqrt(torch.mean((y_test.to(self.device).view(-1) - pred_mean.view(-1))**2)).item()
                print(f"Final RMSE after {epochs} epochs: {rmse:.4f}")
            with open(result_path, 'a') as f:
                f.write(f"{epochs},{rmse},{elbo_val}\n")
        
        print("\nOptimization finished.")