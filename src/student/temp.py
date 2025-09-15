import logging
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from student.constants import EPSILON, JITTER
from student.kernels import rbf_kernel, matern52_kernel
from student.priors import GammaPrior, LogNormalPrior


# --- Set up basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# =============================================================================
# Refactored SparseGPR Class
# =============================================================================
class SparseGPR(nn.Module):
    """
    Sparse Gaussian Process Regression (GPR) trained with Stochastic Variational Inference (SVI).

    This implementation refactors the original SVIGP logic into a more structured
    class format, separating hyperparameter initialization, E-step, M-step,
    and the main training loop for improved clarity and extensibility.
    """
    def __init__(
        self,
        X, y, M,
        hyper_settings=None,
        kernel="rbf",
        inducing_init_method="kmeans",
        device=None
    ):
        super().__init__()

        if device is None:
            self.device = X.device if isinstance(X, torch.Tensor) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Register data as non-trainable buffers
        self.register_buffer('X_full', X.to(self.device))
        self.register_buffer('y_full', y.view(-1, 1).to(self.device))

        if self.X_full.ndim == 1: self.X_full = self.X_full.unsqueeze(1)
        if self.y_full.ndim == 1: self.y_full = self.y_full.unsqueeze(1)

        self.N, self.D = self.X_full.shape
        self.M = M
        self.dtype = self.X_full.dtype

        # Priors for hyperparameters
        self.lengthscale_prior = GammaPrior(1.0, 1.0)
        self.outputscale_prior = GammaPrior(2.0, 2.0)
        self.noisescale_prior = GammaPrior(2.0, 2.0)

        # Initialize hyperparameters
        hyperparameters = self._initialize_hyperparameters(hyper_settings)
        lengthscale = hyperparameters['lengthscale']
        outputscale = hyperparameters['outputscale']
        noisescale = hyperparameters['noisescale']

        # Register hyperparameters as learnable parameters
        self.log_lengthscale = nn.Parameter(torch.log(lengthscale))
        self.log_outputscale = nn.Parameter(torch.log(outputscale))
        self.log_noisescale = nn.Parameter(torch.log(noisescale))

        # Initialize inducing points as learnable parameters
        self.Z = nn.Parameter(self._initialize_inducing_points(method=inducing_init_method))

        # Register variational parameters for q(u) ~ N(m_u, S_u)
        self.register_buffer('m_u', torch.zeros(self.M, 1, dtype=self.dtype))
        # Storing L_u (Cholesky of S_u) for stability, as in the original logic
        self.register_buffer('L_u', torch.eye(self.M, dtype=self.dtype))

        # Set kernel function
        if kernel == "rbf":
            self.kernel = rbf_kernel
        else:
            raise NotImplementedError(f"Kernel '{kernel}' is not implemented.")

        self.to(self.device)
        logging.info(f"SparseGPR model initialized on device: {self.device}")

    def _initialize_hyperparameters(self, hyper_settings=None):
        self.hyper_optim_mode = {}
        dtype = self.dtype

        param_configs = {
            'lengthscale': {'prior': self.lengthscale_prior, 'is_vector': True},
            'outputscale': {'prior': self.outputscale_prior, 'is_vector': False},
            'noisescale': {'prior': self.noisescale_prior, 'is_vector': False},
        }
        initialized_params = {}

        for name, config in param_configs.items():
            settings = (hyper_settings or {}).get(name, {})
            mode = settings.get("optim", "MLE")
            init_val = settings.get("init", None)
            if mode not in ['MLE', 'MAP', 'FIX']: raise ValueError(f"Invalid mode '{mode}' for '{name}'.")
            self.hyper_optim_mode[name] = mode

            if init_val is None:
                sample_shape = (self.D,) if config['is_vector'] else torch.Size([])
                final_value = config['prior'].sample(sample_shape=sample_shape).to(self.device, dtype=dtype)
            else:
                final_value = init_val
            initialized_params[name] = torch.as_tensor(final_value, dtype=dtype, device=self.device)
            logging.info(f"Initialized {name} (Optim mode: {mode}): {initialized_params[name].cpu().numpy()}")

        ls = initialized_params['lengthscale']
        if ls.ndim == 0: ls = ls.repeat(self.D)
        if ls.shape[0] != self.D: raise ValueError("lengthscale must be scalar or vector of length D")
        initialized_params['lengthscale'] = ls
        return initialized_params

    def _initialize_inducing_points(self, method="kmeans"):
        if self.N >= self.M:
            if method == "kmeans":
                X_np = self.X_full.cpu().numpy()
                kmeans = KMeans(n_clusters=self.M, random_state=42, n_init='auto').fit(X_np)
                Z_init = torch.from_numpy(kmeans.cluster_centers_)
            elif method == "random":
                indices = np.random.choice(self.N, self.M, replace=False)
                Z_init = self.X_full[indices].clone()
            else: raise ValueError(f"Unknown init method: {method}")
        else:
            indices = np.random.choice(self.N, self.M, replace=True)
            Z_init = self.X_full[indices].clone()
        return Z_init.to(dtype=self.dtype, device=self.device)

    def _get_hyperparams(self):
        return {
            "lengthscale": torch.exp(self.log_lengthscale),
            "outputscale": torch.exp(self.log_outputscale),
            "noisescale": torch.exp(self.log_noisescale)
        }

    def _calculate_log_prior(self, params):
        log_prior = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        if self.hyper_optim_mode['lengthscale'] == 'MAP':
            log_prior += self.lengthscale_prior.log_prob(params['lengthscale']).sum()
        if self.hyper_optim_mode['outputscale'] == 'MAP':
            log_prior += self.outputscale_prior.log_prob(params['outputscale'])
        if self.hyper_optim_mode['noisescale'] == 'MAP':
            log_prior += self.noisescale_prior.log_prob(params['noisescale'])
        return log_prior

    def _calculate_elbo(self, X_batch, y_batch):
        B = X_batch.shape[0]
        params = self._get_hyperparams()
        beta = 1.0 / params['noisescale'].clamp(min=EPSILON)

        # --- Kernel Matrices ---
        K_mm = self.kernel(self.Z, self.Z, params['lengthscale'], params['outputscale'])
        K_mm += torch.eye(self.M, device=self.device, dtype=self.dtype) * JITTER
        K_mb = self.kernel(self.Z, X_batch, params['lengthscale'], params['outputscale'])
        L_Kmm = torch.linalg.cholesky(K_mm)

        # --- 1. Expected Log Likelihood Term ---
        S_u = self.L_u @ self.L_u.T
        A = torch.cholesky_solve(K_mb, L_Kmm)  # A = K_mm^-1 @ K_mb
        mu_b = A.T @ self.m_u # Expected value of f_batch

        # E[(y-f)^2] = (y - E[f])^2 + Var(f)
        E_sq_err = torch.sum((y_batch - mu_b)**2)
        
        k_bb_diag_sum = B * params['outputscale']
        # tr(K_bm @ K_mm^-1 @ K_mb)
        psi_trace = torch.sum(A * K_mb)
        # tr(K_bm @ K_mm^-1 @ S_u @ K_mm^-1 @ K_mb)
        S_trace = torch.trace(A.T @ S_u @ A)
        
        var_f_sum = k_bb_diag_sum - psi_trace + S_trace
        
        log_lik_const = -0.5 * B * (math.log(2 * math.pi) - torch.log(beta))
        expected_log_likelihood = (log_lik_const - 0.5 * beta * (E_sq_err + var_f_sum)) * (self.N / B)

        # --- 2. KL Divergence KL(q(u) || p(u)) ---
        K_mm_inv_S = torch.cholesky_solve(S_u, L_Kmm)
        trace_kl = torch.trace(K_mm_inv_S)
        mahalanobis_kl = (self.m_u.T @ torch.cholesky_solve(self.m_u, L_Kmm)).squeeze()
        log_det_kl = 2 * torch.sum(torch.log(L_Kmm.diag())) - 2 * torch.sum(torch.log(self.L_u.diag()))
        kl_div = 0.5 * (trace_kl + mahalanobis_kl - self.M + log_det_kl)

        return expected_log_likelihood - kl_div

    @torch.no_grad()
    def _e_step(self, X_batch, y_batch, var_lr):
        B = X_batch.shape[0]
        params = self._get_hyperparams()
        beta = 1.0 / params['noisescale'].clamp(min=EPSILON)

        K_mm = self.kernel(self.Z, self.Z, params['lengthscale'], params['outputscale'])
        K_mm += torch.eye(self.M, device=self.device, dtype=self.dtype) * JITTER
        K_mb = self.kernel(self.Z, X_batch, params['lengthscale'], params['outputscale'])

        L_Kmm = torch.linalg.cholesky(K_mm)
        K_mm_inv = torch.cholesky_solve(torch.eye(self.M, device=self.device, dtype=self.dtype), L_Kmm)
        T = K_mm_inv @ K_mb

        # Calculate optimal natural parameters for the batch
        scaling = self.N / B
        theta2_hat = -0.5 * (K_mm_inv + beta * scaling * (T @ T.T))
        theta1_hat = beta * scaling * (T @ y_batch)

        # Get current natural parameters
        S_u = self.L_u @ self.L_u.T
        S_u_inv = torch.inverse(S_u + JITTER * torch.eye(self.M, device=self.device, dtype=self.dtype))
        theta2_old = -0.5 * S_u_inv
        theta1_old = S_u_inv @ self.m_u

        # Polyak averaging
        theta2_new = (1 - var_lr) * theta2_old + var_lr * theta2_hat
        theta1_new = (1 - var_lr) * theta1_old + var_lr * theta1_hat

        # Convert back to standard parameters and update
        S_u_inv_new = -2 * theta2_new
        try:
            S_u_new = torch.inverse(S_u_inv_new + JITTER * torch.eye(self.M, device=self.device, dtype=self.dtype))
            m_u_new = S_u_new @ theta1_new
            L_u_new = torch.linalg.cholesky(S_u_new + JITTER * torch.eye(self.M, device=self.device, dtype=self.dtype))
            self.m_u.data.copy_(m_u_new)
            self.L_u.data.copy_(L_u_new)
        except torch.linalg.LinAlgError:
            logging.warning("Skipping E-step update due to non-positive definite matrix.")

    def _m_step(self, optimizer, loss):
        if optimizer is None: return
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def fit(self, epochs=100, batch_size=128, hyper_lr=0.01, var_lr=0.1,
            X_test=None, y_test=None, eval_interval=10):
        
        parameters_to_optimize = [
            p for name, p in self.named_parameters()
            if self.hyper_optim_mode.get(name.replace("log_", ""), "MLE") != 'FIX'
        ]
        
        optimizer = optim.Adam(parameters_to_optimize, lr=hyper_lr) if parameters_to_optimize else None
        dataset = TensorDataset(self.X_full, self.y_full)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        history = {'elbo': [], 'eval_epochs': [], 'eval_metrics': []}
        logging.info(f"Starting SVI optimization for {epochs} epochs...")

        for epoch in range(epochs):
            self.train()
            for X_batch, y_batch in dataloader:
                # M-Step: Update Hyperparameters and Inducing Points
                elbo = self._calculate_elbo(X_batch, y_batch)
                log_prior = self._calculate_log_prior(self._get_hyperparams())
                loss = -(elbo + log_prior)
                self._m_step(optimizer, loss)
                
                # E-Step: Update Variational Parameters
                self._e_step(X_batch, y_batch, var_lr=var_lr)

            history['elbo'].append(elbo.item())
            
            # Periodic evaluation and logging
            if (epoch + 1) % eval_interval == 0:
                log_msg = f"Epoch {epoch+1:4d}/{epochs} | Last ELBO: {elbo.item():.3f}"
                if X_test is not None and y_test is not None:
                    metrics = self._evaluate(X_test, y_test)
                    history['eval_epochs'].append(epoch + 1)
                    history['eval_metrics'].append(metrics)
                    log_msg += f" | Test RMSE: {metrics['rmse']:.4f}"
                logging.info(log_msg)
        
        logging.info("Optimization finished.")
        return history

    def predict(self, X_test):
        self.eval()
        X_test_dev = torch.as_tensor(X_test, dtype=self.dtype, device=self.device)
        if X_test_dev.ndim == 1: X_test_dev = X_test_dev.unsqueeze(1)
        with torch.no_grad():
            params = self._get_hyperparams()
            JITTER = 1e-6
            K_mm = self.kernel(self.Z, self.Z, params['lengthscale'], params['outputscale'])
            K_mm += torch.eye(self.M, device=self.device, dtype=self.dtype) * JITTER
            K_sm = self.kernel(X_test_dev, self.Z, params['lengthscale'], params['outputscale'])
            K_ss_diag = params['outputscale'] * torch.ones(X_test_dev.shape[0], device=self.device, dtype=self.dtype)
            
            L_Kmm = torch.linalg.cholesky(K_mm)
            A = K_sm @ torch.cholesky_solve(torch.eye(self.M, device=self.device, dtype=self.dtype), L_Kmm)
            
            pred_mean = A @ self.m_u
            
            S_u = self.L_u @ self.L_u.T
            var_term1 = K_ss_diag
            var_term2 = -torch.sum(A * K_sm, dim=1)
            var_term3 = torch.sum((A @ S_u) * A, dim=1)
            
            pred_var = var_term1 + var_term2 + var_term3
            return {
                'loc': pred_mean.cpu().squeeze(),
                'variance': pred_var.cpu().squeeze()
            }

    def _evaluate(self, X_test, y_test):
        self.eval()
        with torch.no_grad():
            pred_dict = self.predict(X_test)
            pred_mean = pred_dict['loc']
            y_true = y_test.cpu().view(-1)
            rmse = torch.sqrt(torch.mean((y_true - pred_mean)**2)).item()
        self.train()
        return {'rmse': rmse}

# =============================================================================
# Main Execution Block
# =============================================================================
def main():
    """Main execution block for the Sparse GPR example."""
    # 1. Generate Synthetic Data
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_default_dtype(torch.float64)

    def true_function(x):
        return np.sin(x * 2.5) * np.cos(x * 0.8) * 1.5

    N_train = 1000
    X_train_np = np.random.uniform(-5.0, 5.0, N_train)
    y_train_np = true_function(X_train_np) + np.random.normal(scale=0.2, size=N_train)
    
    N_test = 400
    X_test_np = np.linspace(-6.0, 6.0, N_test)
    y_test_np = true_function(X_test_np)

    X_train = torch.from_numpy(X_train_np)
    y_train = torch.from_numpy(y_train_np)
    X_test = torch.from_numpy(X_test_np)
    y_test = torch.from_numpy(y_test_np)

    # 2. Instantiate and Configure the Sparse GP Model
    M = 30  # Number of inducing points

    # Hyperparameter settings can be customized here
    hyper_config = {
        'lengthscale': {'init': 1.0, 'optim': 'MLE'},
        'outputscale': {'init': 1.0, 'optim': 'MLE'},
        'noisescale': {'init': 0.1, 'optim': 'MLE'},
    }

    model = SparseGPR(X_train, y_train, M=M, hyper_settings=hyper_config)

    # 3. Train the Model
    history = model.fit(
        epochs=150,
        batch_size=256,
        hyper_lr=0.01,
        var_lr=0.1,
        X_test=X_test,
        y_test=y_test,
        eval_interval=10
    )

    # 4. Make Predictions
    pred_dict = model.predict(X_test)
    mu_pred = pred_dict['loc'].numpy()
    var_pred = pred_dict['variance'].numpy()
    std_pred = np.sqrt(var_pred.clip(min=0))
    inducing_points = model.Z.detach().cpu().numpy()

    # 5. Print Final Hyperparameters
    print("\n--- Final Optimized Hyperparameters ---")
    final_params = model._get_hyperparams()
    for name, val in final_params.items():
        print(f"{name}: {val.detach().cpu().numpy()}")
    
    # 6. Plot the Results
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    ax.fill_between(
        X_test_np, mu_pred - 1.96 * std_pred, mu_pred + 1.96 * std_pred,
        color="skyblue", alpha=0.5, label="95% Confidence Interval"
    )
    ax.plot(X_test_np, mu_pred, color="dodgerblue", lw=2, label="GP Mean Prediction")
    ax.plot(X_test_np, y_test_np, 'r--', lw=2, label="True Function")
    
    subset_indices = np.random.choice(N_train, 200, replace=False)
    ax.plot(
        X_train_np[subset_indices], y_train_np[subset_indices],
        'o', color='black', ms=3, alpha=0.4, label="Training Data (Subset)"
    )

    ax.plot(
        inducing_points, -2.5 * np.ones_like(inducing_points),
        'k^', ms=10, label=f"Inducing Points (M={M})"
    )

    ax.set_title("Sparse Gaussian Process Regression (SVIGP)", fontsize=16)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-3, 3)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()