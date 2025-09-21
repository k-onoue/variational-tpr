import logging
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.quasirandom import SobolEngine
from linear_operator import to_linear_operator
from linear_operator.operators import to_linear_operator
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

from student.constants import EPSILON, JITTER
from student.kernels import matern52_kernel, rbf_kernel
from student.priors import GammaPrior, LogNormalPrior


class GPR(nn.Module):
    """
    Exact Gaussian Process Regression.
    Structured in the same format as the XuTPR model for consistency.
    """
    def __init__(self, X, y, kernel='rbf', hyper_settings=None, device=None):
        super().__init__()

        if device is None:
            self.device = X.device if isinstance(X, torch.Tensor) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.register_buffer('X_train', X.to(self.device))
        self.register_buffer('y_train', y.view(-1, 1).to(self.device))
        self.dtype = self.X_train.dtype

        if self.X_train.ndim == 1: self.X_train = self.X_train.unsqueeze(1)
        if self.y_train.ndim == 1: self.y_train = self.y_train.unsqueeze(1)

        self.N, self.D = self.X_train.shape

        self.lengthscale_prior = GammaPrior(3.0, 6.0)
        self.outputscale_prior = GammaPrior(2.0, 0.15)
        self.noisescale_prior = LogNormalPrior(loc=-4.0, scale=1.0)

        hyperparameters = self._initialize_hyperparameters(hyper_settings)
        
        self.log_lengthscale = nn.Parameter(torch.log(hyperparameters['lengthscale']))
        self.log_outputscale = nn.Parameter(torch.log(hyperparameters['outputscale']))
        self.log_noisescale = nn.Parameter(torch.log(hyperparameters['noisescale']))

        if kernel in (None, "rbf"): self.kernel = rbf_kernel
        elif kernel == "matern52": self.kernel = matern52_kernel
        else:
            logging.warning(f"Unknown kernel '{kernel}' specified. Defaulting to RBF kernel.")
            self.kernel = rbf_kernel

        self.to(self.device)

    def _initialize_hyperparameters(self, hyper_settings=None):
        self.hyper_optim_mode = {}
        
        param_configs = {
            'lengthscale': {'prior': self.lengthscale_prior, 'is_vector': True},
            'outputscale': {'prior': self.outputscale_prior, 'is_vector': False},
            'noisescale': {'prior': self.noisescale_prior, 'is_vector': False}
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
                final_value = config['prior'].sample(sample_shape=sample_shape).to(self.device, dtype=self.dtype)
                logging.info(f"Sampled initial {name} (Optim mode: {mode}): {final_value.cpu().numpy()}")
            else:
                final_value = torch.as_tensor(init_val, dtype=self.dtype, device=self.device)
                logging.info(f"Using provided initial {name} (Optim mode: {mode}): {final_value.cpu().numpy()}")
            
            initialized_params[name] = final_value

        ls = initialized_params['lengthscale']
        if ls.ndim == 0: ls = ls.repeat(self.D)
        if ls.shape[0] != self.D: raise ValueError("lengthscale must be scalar or vector of length D")
        initialized_params['lengthscale'] = ls
        
        return initialized_params

    def _get_hyperparams(self):
        return {
            "lengthscale": torch.exp(self.log_lengthscale).clamp(min=EPSILON),
            "outputscale": torch.exp(self.log_outputscale).clamp(min=EPSILON),
            "noisescale": torch.exp(self.log_noisescale).clamp(min=EPSILON*100),
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
    
    def calculate_marginal_log_likelihood(self):
        params = self._get_hyperparams()
        
        K_XX = self.kernel(self.X_train, self.X_train, params['lengthscale'], params['outputscale'])
        noise_diag = torch.eye(self.N, device=self.device, dtype=self.dtype) * params['noisescale'].pow(2)
        K_y_op = to_linear_operator(K_XX + noise_diag).add_jitter(JITTER)

        y_train_casted = self.y_train.to(K_y_op.dtype)
        
        inv_quad_form = y_train_casted.T @ K_y_op.solve(y_train_casted)
        data_fit = -0.5 * inv_quad_form.squeeze()
        complexity_penalty = -0.5 * K_y_op.logdet()
        constant = -0.5 * self.N * np.log(2 * np.pi)
        
        log_lik = data_fit + complexity_penalty + constant
        
        return log_lik

    def fit(self, epochs=200, lr=0.01, X_test=None, y_test=None, eval_interval=10):
        params_to_optimize = [p for name, p in self.named_parameters() if self.hyper_optim_mode.get(name.replace("log_", ""), "MLE") != 'FIX']

        if not params_to_optimize:
            logging.warning("No parameters to optimize. All hyperparameters are set to 'FIX'.")
            return {}

        optimizer = optim.Adam(params_to_optimize, lr=lr)
        
        history = {'elbo': [], 'log_prior': [], 'loss': [], 'hyperparams': [], 'eval_epochs': [], 'eval_metrics': [], 'fit_times': []}
        logging.info(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            fit_start_time = time.time()
            optimizer.zero_grad()
            
            mll = self.calculate_marginal_log_likelihood()
            log_prior = self._calculate_log_prior(self._get_hyperparams())
            loss = -(mll + log_prior)
            
            loss.backward()
            optimizer.step()

            fit_end_time = time.time()

            history['elbo'].append(mll.item())
            history['log_prior'].append(log_prior.item())
            history['loss'].append(loss.item())
            history['hyperparams'].append({k: v.detach().cpu().numpy() for k, v in self._get_hyperparams().items()})
            history['fit_times'].append(fit_end_time - fit_start_time)

            if (epoch + 1) % 10 == 0:
                logging.info(f"Epoch {epoch+1:4d}/{epochs} | Fit Time: {fit_end_time - fit_start_time:.3f}s | Loss: {loss.item():.3f} | MLL: {mll.item():.3f}")

            if X_test is not None and y_test is not None and (epoch + 1) % eval_interval == 0:
                metrics = self._evaluate(X_test, y_test)
                history['eval_epochs'].append(epoch + 1)
                history['eval_metrics'].append(metrics)
                logging.info(f"Epoch {epoch+1:4d} | Test RMSE: {metrics['rmse']:.4f}")

        logging.info("Training finished.")
        return history

    def predict(self, X_test):
        self.eval()
        X_test = torch.as_tensor(X_test, dtype=self.dtype, device=self.device)
        if X_test.ndim == 1: X_test = X_test.unsqueeze(1)
        
        with torch.no_grad():
            params = self._get_hyperparams()
            
            K_XX = self.kernel(self.X_train, self.X_train, params['lengthscale'], params['outputscale'])
            noise_diag = torch.eye(self.N, device=self.device, dtype=self.dtype) * params['noisescale'].pow(2)
            K_y_op = to_linear_operator(K_XX + noise_diag).add_jitter(JITTER)

            K_star_X = self.kernel(X_test, self.X_train, params['lengthscale'], params['outputscale'])
            K_star_star = self.kernel(X_test, X_test, params['lengthscale'], params['outputscale'])
            
            y_train_casted = self.y_train.to(K_y_op.dtype)
            alpha = K_y_op.solve(y_train_casted)

            predictive_mean = K_star_X @ alpha
            V = K_y_op.solve(K_star_X.T)
            predictive_cov = K_star_star - K_star_X @ V

        self.train()

        predictive_mean_np = predictive_mean.squeeze(-1).cpu().numpy()
        predictive_var_np = predictive_cov.diag().cpu().numpy()

        return predictive_mean_np, predictive_var_np

    def _evaluate(self, X_test, y_test):
        mu_pred, _ = self.predict(X_test)
        y_true = y_test.cpu().numpy().squeeze()
        rmse = np.sqrt(mean_squared_error(y_true, mu_pred))
        return {'rmse': rmse}


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

        self.lengthscale_prior = GammaPrior(3.0, 6.0)
        self.outputscale_prior = GammaPrior(2.0, 0.15)
        self.noisescale_prior = LogNormalPrior(loc=-4.0, scale=1.0)

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
            # <<< ADDED START: Sobol sequence initialization >>>
            elif method == "sobol":
                logging.info("Initializing inducing points with Sobol sequence.")
                sobol_engine = SobolEngine(dimension=self.D, scramble=True, seed=42)
                # Generate M points in the unit hypercube [0, 1]^D
                sobol_points = sobol_engine.draw(self.M).to(self.device, dtype=self.X_full.dtype)

                # Scale points to the bounding box of the training data
                X_min = self.X_full.min(dim=0).values
                X_max = self.X_full.max(dim=0).values
                Z_init = X_min + sobol_points * (X_max - X_min)
            # <<< ADDED END >>>
            else: raise ValueError(f"Unknown init method: {method}")
        else:
            indices = np.random.choice(self.N, self.M, replace=True)
            Z_init = self.X_full[indices].clone()
        return Z_init.to(dtype=self.X_full.dtype, device=self.device)

    def _get_hyperparams(self):
        return {
            "lengthscale": torch.exp(self.log_lengthscale).clamp(min=EPSILON),
            "outputscale": torch.exp(self.log_outputscale).clamp(min=EPSILON),
            "noisescale": torch.exp(self.log_noisescale).clamp(min=EPSILON*100),
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
        # --- MODIFIED ---
        # Convert to LinearOperator and add jitter for stability
        K_mm_op = to_linear_operator(K_mm).add_jitter(JITTER)
        K_mb = self.kernel(self.Z, X_batch, params['lengthscale'], params['outputscale'])

        # --- 1. Expected Log Likelihood Term ---
        S_u = self.L_u @ self.L_u.T
        # --- MODIFIED ---
        # A = K_mm^-1 @ K_mb using efficient solve
        A = K_mm_op.solve(K_mb)
        mu_b = A.T @ self.m_u # Expected value of f_batch

        E_sq_err = torch.sum((y_batch - mu_b)**2)
        k_bb_diag_sum = B * params['outputscale']
        psi_trace = torch.sum(A * K_mb)
        S_trace = torch.trace(A.T @ S_u @ A)
        var_f_sum = k_bb_diag_sum - psi_trace + S_trace
        
        log_lik_const = -0.5 * B * (math.log(2 * math.pi) - torch.log(beta))
        expected_log_likelihood = (log_lik_const - 0.5 * beta * (E_sq_err + var_f_sum)) * (self.N / B)

        # --- 2. KL Divergence KL(q(u) || p(u)) ---
        # --- MODIFIED ---
        # Using efficient methods from linear_operator for KL terms
        trace_kl = torch.trace(K_mm_op.solve(S_u))
        mahalanobis_kl = K_mm_op.inv_quad(self.m_u.squeeze(-1))
        log_det_Kmm = K_mm_op.logdet()
        log_det_Su = 2 * torch.sum(torch.log(self.L_u.diag().clamp(min=EPSILON)))
        log_det_kl = log_det_Kmm - log_det_Su

        kl_div = 0.5 * (trace_kl + mahalanobis_kl - self.M + log_det_kl)
        return expected_log_likelihood - kl_div

    @torch.no_grad()
    def _e_step(self, X_batch, y_batch, var_lr):
        B = X_batch.shape[0]
        params = self._get_hyperparams()
        beta = 1.0 / params['noisescale'].clamp(min=EPSILON)

        K_mm = self.kernel(self.Z, self.Z, params['lengthscale'], params['outputscale'])
        # --- MODIFIED ---
        K_mm_op = to_linear_operator(K_mm).add_jitter(JITTER)
        K_mb = self.kernel(self.Z, X_batch, params['lengthscale'], params['outputscale'])

        # --- MODIFIED ---
        # Explicitly compute K_mm_inv tensor as it's needed for theta2_hat
        identity_M = torch.eye(self.M, device=self.device, dtype=self.dtype)
        K_mm_inv = K_mm_op.solve(identity_M)
        T = K_mm_inv @ K_mb

        # Calculate optimal natural parameters for the batch
        scaling = self.N / B
        theta2_hat = -0.5 * (K_mm_inv + beta * scaling * (T @ T.T))
        theta1_hat = beta * scaling * (T @ y_batch)

        # Get current natural parameters
        S_u = self.L_u @ self.L_u.T
        # --- MODIFIED ---
        # Invert S_u using linear_operator
        S_u_op = to_linear_operator(S_u).add_jitter(JITTER)
        S_u_inv = S_u_op.solve(identity_M)
        theta2_old = -0.5 * S_u_inv
        theta1_old = S_u_inv @ self.m_u

        # Polyak averaging
        theta2_new = (1 - var_lr) * theta2_old + var_lr * theta2_hat
        theta1_new = (1 - var_lr) * theta1_old + var_lr * theta1_hat

        # Convert back to standard parameters and update
        S_u_inv_new = -2 * theta2_new
        try:
            # --- MODIFIED ---
            # Invert S_u_inv_new using linear_operator
            S_u_inv_new_op = to_linear_operator(S_u_inv_new).add_jitter(JITTER)
            S_u_new = S_u_inv_new_op.solve(identity_M)
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
        
        logging.info(f"Starting SVI optimization for {epochs} epochs...")

        for epoch in range(epochs):
            self.train()
            
            epoch_loss = 0.0
            epoch_elbo = 0.0
            epoch_log_prior = 0.0
            epoch_fit_time = 0.0
            num_batches = 0

            for X_batch, y_batch in dataloader:
                fit_start_time = time.time()

                self._e_step(X_batch, y_batch, var_lr=var_lr)
                elbo = self._calculate_elbo(X_batch, y_batch)
                log_prior = self._calculate_log_prior(self._get_hyperparams())
                loss = -(elbo + log_prior)
                self._m_step(optimizer, loss)

                fit_end_time = time.time()

                epoch_loss += loss.item()
                epoch_elbo += elbo.item()
                epoch_log_prior += log_prior.item()
                epoch_fit_time += (fit_end_time - fit_start_time)
                num_batches += 1

            # --- Yield results for the completed epoch ---
            avg_loss = epoch_loss / num_batches
            avg_elbo = epoch_elbo / num_batches

            epoch_results = {
                'epoch': epoch + 1,
                'loss': avg_loss,
                'elbo': avg_elbo,
                'log_prior': epoch_log_prior / num_batches,
                'time': epoch_fit_time,
            }

            if (epoch + 1) % eval_interval == 0:
                log_msg = f"Epoch {epoch+1:4d}/{epochs} | ELBO: {avg_elbo:.3f}"
                if X_test is not None and y_test is not None:
                    metrics = self._evaluate(X_test, y_test)
                    epoch_results.update(metrics) # Add RMSE etc. to dict
                    log_msg += f" | Test RMSE: {metrics['rmse']:.4f}"
                logging.info(log_msg)
            
            yield epoch_results
        
        logging.info("Optimization finished.")

    def predict(self, X_test):
        self.eval()
        X_test_dev = torch.as_tensor(X_test, dtype=self.dtype, device=self.device)
        if X_test_dev.ndim == 1: X_test_dev = X_test_dev.unsqueeze(1)
        with torch.no_grad():
            params = self._get_hyperparams()
            K_mm = self.kernel(self.Z, self.Z, params['lengthscale'], params['outputscale'])
            # --- MODIFIED ---
            K_mm_op = to_linear_operator(K_mm).add_jitter(JITTER)
            K_sm = self.kernel(X_test_dev, self.Z, params['lengthscale'], params['outputscale'])
            K_ss_diag = params['outputscale'] * torch.ones(X_test_dev.shape[0], device=self.device, dtype=self.dtype)
            
            # --- MODIFIED ---
            # A = K_sm @ K_mm^-1. More stable to compute via solve: (K_mm^-1 @ K_ms)^T
            A = K_mm_op.solve(K_sm.T).T
            
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