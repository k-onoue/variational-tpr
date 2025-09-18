import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from linear_operator.operators import to_linear_operator

from .constants import EPSILON, JITTER
from .kernels import rbf_kernel, matern52_kernel
from .priors import GammaPrior, LogNormalPrior
from .utils import (
    kl_gamma,
    kl_gaussian_gamma_covariance_param,
    get_optimal_gaussian_gamma,
    gaussian_gamma_standard_to_natural_covariance_param,
    gaussian_gamma_natural_to_standard_covariance_param
)


class TPR(nn.Module):
    def __init__(
        self,
        X, y,
        hyper_settings=None,
        kernel="rbf",
        device=None
    ):
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
        dtype = self.X_full.dtype

        # Priors
        self.lengthscale_prior = GammaPrior(3.0, 6.0)
        self.outputscale_prior = GammaPrior(2.0, 0.15)
        self.nu_prior = LogNormalPrior(loc=1.0, scale=1.0)
        self.noise_prior = LogNormalPrior(loc=-4.0, scale=1.0)

        # Initialize hyperparameters
        hyperparameters = self._initialize_hyperparameters(hyper_settings)
        lengthscale = hyperparameters['lengthscale']
        outputscale = hyperparameters['outputscale']
        dof_func = hyperparameters['dof_func']
        dof_lik = hyperparameters['dof_lik']
        noisescale = hyperparameters['noisescale']

        # Set nn.Parameters for learnable hyperparameters
        self.log_lengthscale = nn.Parameter(torch.log(lengthscale))
        self.log_outputscale = nn.Parameter(torch.log(outputscale))
        self.log_dof_func = nn.Parameter(torch.log(dof_func))
        self.log_dof_lik = nn.Parameter(torch.log(dof_lik))
        self.log_noisescale = nn.Parameter(torch.log(noisescale))
        
        # Register non-trainable buffers for the Normal-Gamma distribution q(f,r)
        self.register_buffer('m_f', torch.zeros(self.N, 1, dtype=dtype))
        self.register_buffer('S_f', torch.eye(self.N, dtype=dtype))
        self.register_buffer('alpha_r', dof_func / 2.0)
        self.register_buffer('beta_r', dof_func / 2.0)

        # Set kernel function
        if kernel in (None, "rbf"):
            self.kernel = rbf_kernel
        elif kernel == "matern52":
            self.kernel = matern52_kernel
        else:
            logging.info("Unknown kernel specified. Defaulting to RBF kernel.")
            self.kernel = rbf_kernel

        self.to(self.device)

    def _initialize_hyperparameters(self, hyper_settings=None):
        self.hyper_optim_mode = {}
        dtype = self.X_full.dtype
        
        param_configs = {
            'lengthscale': {'prior': self.lengthscale_prior, 'is_vector': True},
            'outputscale': {'prior': self.outputscale_prior, 'is_vector': False},
            'dof_func': {'prior': self.nu_prior, 'is_vector': False},
            'dof_lik': {'prior': self.nu_prior, 'is_vector': False},
            'noisescale': {'prior': self.noise_prior, 'is_vector': False}
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
                if name.startswith('dof'): final_value = final_value.clamp(min=2.0)
                logging.info(f"Sampled initial {name} (Optim mode: {mode}): {final_value.cpu().numpy()}")
            else:
                final_value = torch.as_tensor(init_val, device=self.device)
                logging.info(f"Using provided initial {name} (Optim mode: {mode}): {final_value.cpu().numpy()}")
            
            initialized_params[name] = torch.as_tensor(final_value, dtype=dtype, device=self.device)

        ls = initialized_params['lengthscale']
        if ls.ndim == 0: ls = ls.repeat(self.D)
        if ls.shape[0] != self.D: raise ValueError("lengthscale must be scalar or vector of length D")
        initialized_params['lengthscale'] = ls
        
        return initialized_params

    def _get_hyperparams(self):
        return {
            "lengthscale": torch.exp(self.log_lengthscale).clamp(min=EPSILON),
            "outputscale": torch.exp(self.log_outputscale).clamp(min=EPSILON),
            "dof_func": torch.exp(self.log_dof_func).clamp(min=EPSILON + 2.0),
            "dof_lik": torch.exp(self.log_dof_lik).clamp(min=EPSILON + 2.0),
            "noisescale": torch.exp(self.log_noisescale).clamp(min=EPSILON),
        }

    def _calculate_log_prior(self, params):
        log_prior = torch.tensor(0.0, device=self.device, dtype=params['lengthscale'].dtype)
        if self.hyper_optim_mode['lengthscale'] == 'MAP':
            log_prior += self.lengthscale_prior.log_prob(params['lengthscale']).sum()
        if self.hyper_optim_mode['outputscale'] == 'MAP':
            log_prior += self.outputscale_prior.log_prob(params['outputscale'])
        if self.hyper_optim_mode['dof_func'] == 'MAP':
            log_prior += self.nu_prior.log_prob(params['dof_func'])
        if self.hyper_optim_mode['dof_lik'] == 'MAP':
            log_prior += self.nu_prior.log_prob(params['dof_lik'])
        if self.hyper_optim_mode['noisescale'] == 'MAP':
            log_prior += self.noise_prior.log_prob(params['noisescale'])
        return log_prior

    def _calculate_elbo(self, K_XX, local_params):
        alpha_lambda, beta_lambda = local_params
        params = self._get_hyperparams()
        
        # 1. Expected Log-Likelihood
        E_lambda = alpha_lambda / beta_lambda.clamp(min=EPSILON)
        E_log_lambda = torch.digamma(alpha_lambda) - torch.log(beta_lambda.clamp(min=EPSILON))
        
        E_r_inv = self.beta_r / (self.alpha_r - 1.0).clamp(min=EPSILON)
        var_f = E_r_inv * torch.diag(self.S_f)
        E_sq_err = (self.y_full.squeeze() - self.m_f.squeeze())**2 + var_f
        
        noise_var = params["noisescale"]
        exp_log_lik = torch.sum(
            -0.5 * torch.log(2 * torch.pi * noise_var) + 0.5 * E_log_lambda - 0.5 * E_lambda * E_sq_err / noise_var
        )
        
        # 2. KL Divergence (Gamma)
        p_alpha_lambda, p_beta_lambda = params['dof_lik'] / 2.0, params['dof_lik'] / 2.0
        kl_lambda = kl_gamma(alpha_lambda, beta_lambda, p_alpha_lambda, p_beta_lambda).sum()

        # 2. KL Divergence (Gaussian-Gamma)
        p_alpha_r, p_beta_r = params['dof_func'] / 2.0, params['dof_func'] / 2.0
        prior_mean_f = torch.zeros_like(self.m_f.squeeze())
        kl_f_r = kl_gaussian_gamma_covariance_param(
            mu_q=self.m_f.squeeze(), S_q=self.S_f, alpha_q=self.alpha_r, beta_q=self.beta_r,
            mu_p=prior_mean_f, K_p=K_XX.to_dense(), alpha_p=p_alpha_r, beta_p=p_beta_r
        )
        
        return exp_log_lik - kl_lambda - kl_f_r


    def _e_step(self, K_XX_op, params):
        with torch.no_grad():
            identity = torch.eye(self.N, device=self.device, dtype=self.X_full.dtype)
            sigma2 = params['noisescale']

            # Update q(λ) parameters
            E_r_inv = self.beta_r / (self.alpha_r - 1.0).clamp(min=EPSILON)
            var_f = E_r_inv * torch.diag(self.S_f)
            E_sq_err = (self.y_full.squeeze() - self.m_f.squeeze())**2 + var_f
            alpha_lambda = (params['dof_lik'] / 2.0 + 0.5).expand(self.N)
            beta_lambda = params['dof_lik'] / 2.0 + 0.5 * E_sq_err / sigma2
            
            # Update q(f), q(r) parameters independently
            E_lambda = alpha_lambda / beta_lambda.clamp(min=EPSILON)
            E_r = self.alpha_r / self.beta_r.clamp(min=EPSILON)
            K_XX_inv = K_XX_op.solve(identity)
            scaled_K_XX_inv = K_XX_inv * E_r

            target_S_f_inv = scaled_K_XX_inv + torch.diag(E_lambda / sigma2)
            target_S_f_inv_op = to_linear_operator(target_S_f_inv)
            target_S_f = target_S_f_inv_op.solve(identity)
            target_m_f_term = (E_lambda * self.y_full.squeeze()) / sigma2
            target_m_f = target_S_f @ target_m_f_term.unsqueeze(1)
            
            target_alpha_r = params['dof_func'] / 2.0 + self.N / 2.0
            trace_term = torch.trace(K_XX_inv @ target_S_f)
            K_inv_m = K_XX_inv @ target_m_f
            quad_term = (target_m_f.T @ K_inv_m).squeeze()
            target_beta_r = params['dof_func'] / 2.0 + (trace_term + quad_term) / 2.0

            # Update q(f, r) via projection
            _, S_f_proj, _, _ = get_optimal_gaussian_gamma(
                target_m_f, target_S_f, target_alpha_r, target_beta_r
            )

            self.m_f.data      = target_m_f
            self.S_f.data      = S_f_proj
            self.alpha_r.data  = target_alpha_r
            self.beta_r.data   = target_beta_r

            return alpha_lambda, beta_lambda

    def _m_step(self, optimizer, loss):
        if optimizer is None: return
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    def fit(
        self, 
        epochs=100,
        hyper_lr=0.01,
        X_test=None, y_test=None, eval_interval=10
    ):
        parameters_to_optimize = [p for name, p in self.named_parameters() if self.hyper_optim_mode.get(name.replace("log_",""), "MLE") != 'FIX']
        optimizer = optim.Adam(parameters_to_optimize, lr=hyper_lr) if parameters_to_optimize else None

        history = {
            'elbo': [], 'log_prior': [], 'loss': [], 'lengthscale': [], 'outputscale': [],
            'dof_func': [], 'dof_lik': [], 'noisescale': [], 'eval_epochs': [],
            'eval_metrics': [], 'fit_times': []
        }
        logging.info(f"Starting SVI optimization for {epochs} epochs...")

        for epoch in range(epochs):
            self.train()
            fit_start_time = time.time()

            params = self._get_hyperparams()
            K_XX_base = to_linear_operator(self.kernel(self.X_full, self.X_full, params['lengthscale'], params['outputscale']))
            K_XX_op = K_XX_base.add_jitter(JITTER)
            local_params = self._e_step(K_XX_op, params)
            elbo = self._calculate_elbo(K_XX_op, local_params)
            log_prior = self._calculate_log_prior(params)
            loss = - (elbo + log_prior)
            
            self._m_step(optimizer, loss)
            fit_end_time = time.time()

            with torch.no_grad():
                params_final = self._get_hyperparams()
                history['elbo'].append(elbo.item())
                history['log_prior'].append(log_prior.item())
                history['loss'].append(loss.item())
                history['lengthscale'].append(params_final['lengthscale'].detach().cpu().numpy())
                history['outputscale'].append(params_final['outputscale'].item())
                history['noisescale'].append(params_final['noisescale'].item())
                history['dof_func'].append(params_final['dof_func'].item())
                history['dof_lik'].append(params_final['dof_lik'].item())
                history['fit_times'].append(fit_end_time - fit_start_time)
                
            if (epoch + 1) % eval_interval == 0:
                ls_str = ", ".join([f"{l:.3f}" for l in params_final['lengthscale']])
                logging.info(f"Epoch {epoch+1:4d}/{epochs} | Fit Time: {fit_end_time - fit_start_time:.3f}s | ELBO: {elbo.item():8.2f} | l: [{ls_str}] | var: {params_final['outputscale']:.3f} | noise_var: {params_final['noisescale']:.3f} | dof_f: {params_final['dof_func']:.2f} | dof_l: {params_final['dof_lik']:.2f}")

            if X_test is not None and y_test is not None and (epoch + 1) % eval_interval == 0:
                metrics = self._evaluate(X_test, y_test)
                history['eval_epochs'].append(epoch + 1)
                history['eval_metrics'].append(metrics)
                logging.info(f"Epoch {epoch+1:4d}/{epochs} | Test RMSE: {metrics['rmse']:.4f}")

        logging.info("Optimization finished.")
        return history

    def predict(self, X_test):
        X_test = torch.as_tensor(X_test, dtype=self.X_full.dtype, device=self.device)
        if X_test.ndim == 1: X_test = X_test.unsqueeze(1)
            
        self.eval()
        with torch.no_grad():
            params = self._get_hyperparams()
            K_XX_base = to_linear_operator(self.kernel(self.X_full, self.X_full, params['lengthscale'], params['outputscale']))
            K_XX_op = K_XX_base.add_jitter(JITTER)
            K_star_X = self.kernel(X_test, self.X_full, params['lengthscale'], params['outputscale'])
            k_star_star = self.kernel(X_test, X_test, params['lengthscale'], params['outputscale']).diag()

            # Predictive Location (mean): mu_star = K_*X @ K_XX^-1 @ m_f
            K_XX_inv_mf = K_XX_op.solve(self.m_f)
            mu_star = K_star_X @ K_XX_inv_mf
            
            # Predictive Degrees of Freedom
            dof_star = 2 * self.alpha_r
            
            # Predictive Scale
            A = K_XX_op.solve(K_star_X.T).T
            
            term1 = k_star_star - torch.sum(A * K_star_X, dim=1)
            term2 = torch.sum((A @ self.S_f) * A, dim=1)
            scale_sq_star_f = (self.beta_r / self.alpha_r.clamp(min=EPSILON)) * (term1 + term2)
            
            dof_lik = params['dof_lik']
            noise_var = params['noisescale']
            expected_noise_var = noise_var * dof_lik / (dof_lik - 2).clamp(min=EPSILON)
            
            return {
                'loc': mu_star.squeeze(), 
                'scale_sq': (scale_sq_star_f + expected_noise_var).clamp(min=EPSILON), 
                'dof': dof_star.clamp(min=EPSILON)
            }

    def _evaluate(self, X_test, y_test):
        f_pred_tensor = self.predict(X_test)
        f_pred_numpy = f_pred_tensor['loc'].cpu().numpy()
        y_true_numpy = y_test.cpu().numpy().squeeze()
        metrics = {'rmse': np.sqrt(mean_squared_error(y_true_numpy, f_pred_numpy))}
        return metrics



class SparseTPR(nn.Module):
    """
    Sparse Student-t Process Regression (TPR) with a Student-t Likelihood.
    The model is trained using Structured Stochastic Variational Inference (SVI).
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

        # Register data as non-trainable buffers (y is continuous)
        self.register_buffer('X_full', X.to(self.device))
        self.register_buffer('y_full', y.view(-1, 1).to(self.device))

        if self.X_full.ndim == 1: self.X_full = self.X_full.unsqueeze(1)
        if self.y_full.ndim == 1: self.y_full = self.y_full.unsqueeze(1)

        self.N, self.D = self.X_full.shape
        self.M = M
        dtype = self.X_full.dtype

        # Priors for hyperparameters
        self.lengthscale_prior = GammaPrior(3.0, 6.0)
        self.outputscale_prior = GammaPrior(2.0, 0.15)
        self.nu_prior = LogNormalPrior(loc=1.0, scale=1.0)
        self.noise_prior = LogNormalPrior(loc=-4.0, scale=1.0)

        # Initialize hyperparameters
        hyperparameters = self._initialize_hyperparameters(hyper_settings)
        lengthscale = hyperparameters['lengthscale']
        outputscale = hyperparameters['outputscale']
        dof_func = hyperparameters['dof_func']
        dof_lik = hyperparameters['dof_lik']
        noisescale = hyperparameters['noisescale']

        # Register hyperparameters as learnable parameters
        self.log_lengthscale = nn.Parameter(torch.log(lengthscale))
        self.log_outputscale = nn.Parameter(torch.log(outputscale))
        self.log_dof_func = nn.Parameter(torch.log(dof_func))
        self.log_dof_lik = nn.Parameter(torch.log(dof_lik))
        self.log_noisescale = nn.Parameter(torch.log(noisescale))
        
        # Initialize inducing points as learnable parameters
        self.Z = nn.Parameter(self._initialize_inducing_points(method=inducing_init_method))

        # Register variational parameters for q(u, r) ~ Normal-Gamma(m_u, S_u, alpha_r, beta_r)
        self.register_buffer('m_u', torch.zeros(self.M, 1, dtype=dtype))
        self.register_buffer('S_u', torch.eye(self.M, dtype=dtype))
        self.register_buffer('alpha_r', dof_func / 2.0)
        self.register_buffer('beta_r', dof_func / 2.0)

        # Set kernel function
        if kernel in (None, "rbf"): self.kernel = rbf_kernel
        elif kernel == "matern52": self.kernel = matern52_kernel
        else:
            logging.info("Unknown kernel specified. Defaulting to RBF kernel.")
            self.kernel = rbf_kernel

        self.to(self.device)

    def _initialize_hyperparameters(self, hyper_settings=None):
        self.hyper_optim_mode = {}
        dtype = self.X_full.dtype
        
        param_configs = {
            'lengthscale': {'prior': self.lengthscale_prior, 'is_vector': True},
            'outputscale': {'prior': self.outputscale_prior, 'is_vector': False},
            'dof_func': {'prior': self.nu_prior, 'is_vector': False},
            'dof_lik': {'prior': self.nu_prior, 'is_vector': False},
            'noisescale': {'prior': self.noise_prior, 'is_vector': False}
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
                if name.startswith('dof'): final_value = final_value.clamp(min=2.0)
                logging.info(f"Sampled initial {name} (Optim mode: {mode}): {final_value.cpu().numpy()}")
            else:
                final_value = init_val
                logging.info(f"Using provided initial {name} (Optim mode: {mode}): {final_value}")
            
            initialized_params[name] = torch.as_tensor(final_value, dtype=dtype, device=self.device)

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
        return Z_init.to(dtype=self.X_full.dtype, device=self.device)

    def _get_hyperparams(self):
        return {
            "lengthscale": torch.exp(self.log_lengthscale).clamp(min=EPSILON),
            "outputscale": torch.exp(self.log_outputscale).clamp(min=EPSILON),
            "dof_func": torch.exp(self.log_dof_func).clamp(min=EPSILON + 2.0),
            "dof_lik": torch.exp(self.log_dof_lik).clamp(min=EPSILON + 2.0),
            "noisescale": torch.exp(self.log_noisescale).clamp(min=EPSILON),
        }

    def _calculate_log_prior(self, params):
        log_prior = torch.tensor(0.0, device=self.device, dtype=params['lengthscale'].dtype)
        if self.hyper_optim_mode['lengthscale'] == 'MAP':
            log_prior += self.lengthscale_prior.log_prob(params['lengthscale']).sum()
        if self.hyper_optim_mode['outputscale'] == 'MAP':
            log_prior += self.outputscale_prior.log_prob(params['outputscale'])
        if self.hyper_optim_mode['dof_func'] == 'MAP':
            log_prior += self.nu_prior.log_prob(params['dof_func'])
        if self.hyper_optim_mode['dof_lik'] == 'MAP':
            log_prior += self.nu_prior.log_prob(params['dof_lik'])
        if self.hyper_optim_mode['noisescale'] == 'MAP':
            log_prior += self.noise_prior.log_prob(params['noisescale'])
        return log_prior

    def _calculate_elbo(self, X_batch, y_batch, K_XZ_batch, K_ZZ, local_params):
        alpha_lambda_batch, beta_lambda_batch, E_lambda_batch = local_params
        params = self._get_hyperparams()
        batch_size = X_batch.shape[0]
        scaling_factor = self.N / batch_size
        
        # --- 1. Expected Log Likelihood Term ---
        E_log_lambda_batch = torch.digamma(alpha_lambda_batch) - torch.log(beta_lambda_batch.clamp(min=EPSILON))
        
        A = K_XZ_batch @ K_ZZ.solve(torch.eye(self.M, device=self.device))
        mu_f_batch = A @ self.m_u
        E_r_inv = self.beta_r / (self.alpha_r - 1.0).clamp(min=EPSILON)
        K_tilde_diag = params['outputscale'] - torch.sum(A * K_XZ_batch, dim=1)
        var_f_batch = E_r_inv * (K_tilde_diag + torch.sum((A @ self.S_u) * A, dim=1))
        E_sq_err = (y_batch.squeeze() - mu_f_batch.squeeze())**2 + var_f_batch

        log_2pi = torch.log(torch.tensor(2 * torch.pi, device=self.device))
        log_lik_per_item = -0.5 * log_2pi - 0.5 * torch.log(params['noisescale']) + 0.5 * E_log_lambda_batch - 0.5 * E_lambda_batch * E_sq_err / params['noisescale']
        exp_log_lik = torch.sum(log_lik_per_item) * scaling_factor

        # --- 2. KL Divergence KL(q(lambda) || p(lambda)) ---
        p_alpha_lambda, p_beta_lambda = params['dof_lik'] / 2.0, params['dof_lik'] / 2.0
        kl_lambda = kl_gamma(alpha_lambda_batch, beta_lambda_batch, p_alpha_lambda, p_beta_lambda).sum() * scaling_factor

        # --- 3. KL Divergence KL(q(u,r) || p(u,r)) ---
        p_alpha_r, p_beta_r = params['dof_func'] / 2.0, params['dof_func'] / 2.0
        prior_mean_u = torch.zeros_like(self.m_u.squeeze())
        kl_u_r = kl_gaussian_gamma_covariance_param(
            mu_q=self.m_u.squeeze(), S_q=self.S_u, alpha_q=self.alpha_r, beta_q=self.beta_r,
            mu_p=prior_mean_u, K_p=K_ZZ.to_dense(), alpha_p=p_alpha_r, beta_p=p_beta_r
        )
        
        return exp_log_lik - kl_lambda - kl_u_r

    def _e_step_local(self, X_batch, y_batch, K_XZ_batch, K_ZZ, params):
        """E-Step for local parameters q(lambda_i) for the mini-batch."""
        with torch.no_grad():
            A_batch = K_XZ_batch @ K_ZZ.solve(torch.eye(self.M, device=self.device, dtype=X_batch.dtype))
            
            # --- Calculate E[(y_i - f_i)^2] ---
            # 1. E[f_i]
            mu_f_batch = A_batch @ self.m_u
            
            # 2. Var(f_i)
            E_r_inv = self.beta_r / (self.alpha_r - 1.0).clamp(min=EPSILON)
            K_tilde_diag = params['outputscale'] - torch.sum(A_batch * K_XZ_batch, dim=1)
            var_f_batch = E_r_inv * (K_tilde_diag + torch.sum((A_batch @ self.S_u) * A_batch, dim=1))
            
            # 3. Combine terms
            E_sq_err = (y_batch.squeeze() - mu_f_batch.squeeze())**2 + var_f_batch

            # --- Update local parameters ---
            alpha_lambda_batch = params['dof_lik'] / 2.0 + 0.5
            beta_lambda_batch = params['dof_lik'] / 2.0 + 0.5 * E_sq_err / params['noisescale']

            E_lambda_batch = alpha_lambda_batch / beta_lambda_batch.clamp(min=EPSILON)

            return alpha_lambda_batch, beta_lambda_batch, E_lambda_batch

    def _e_step_global(self, X_batch, y_batch, K_XZ_batch, K_ZZ, local_params, params, var_lr):
        # E-Step (Global)
        with torch.no_grad():
            _, _, E_lambda_batch = local_params
            scaling_factor = self.N / X_batch.shape[0]

            identity_M = torch.eye(self.M, device=self.device, dtype=X_batch.dtype)

            K_ZZ_inv = K_ZZ.solve(identity_M)
            K_ZZ_inv = to_linear_operator(K_ZZ_inv)
            A_batch = K_XZ_batch @ K_ZZ_inv
            
            # --- Target Parameters for q(u, r) ---
            # Target for q(u)
            E_r = self.alpha_r / self.beta_r.clamp(min=EPSILON)
            S_u_inv_data_term = (A_batch.T * E_lambda_batch) @ A_batch / params['noisescale'] * scaling_factor
            target_S_u_inv = E_r * K_ZZ_inv + S_u_inv_data_term
            target_S_u_inv = target_S_u_inv.add_jitter(JITTER)
            target_S_u = target_S_u_inv.solve(identity_M)
            m_u_data_term = A_batch.T @ torch.diag(E_lambda_batch) @ y_batch / params['noisescale'] * scaling_factor
            target_m_u = target_S_u @ m_u_data_term
            
            # Target for q(r) - using simplified update
            target_alpha_r = params['dof_func'] / 2.0 + self.M / 2.0
            E_quad_u = torch.trace(K_ZZ_inv @ (target_S_u + target_m_u @ target_m_u.T))
            target_beta_r = params['dof_func'] / 2.0 + E_quad_u / 2.0

            # Solve another variational problem argmin KL(q(u,r)||q(u)q(r))
            _, target_S_u, _, _ = get_optimal_gaussian_gamma(target_m_u, target_S_u, target_alpha_r, target_beta_r)

            # Convert current and target parameters to natural form for q(u, r)
            eta1_curr, eta2_curr, eta3_curr, eta4_curr = gaussian_gamma_standard_to_natural_covariance_param(self.m_u, self.S_u, self.alpha_r, self.beta_r)
            eta1_targ, eta2_targ, eta3_targ, eta4_targ = gaussian_gamma_standard_to_natural_covariance_param(target_m_u, target_S_u, target_alpha_r, target_beta_r)

            # Polyak averaging step
            eta1_new = (1 - var_lr) * eta1_curr + var_lr * eta1_targ
            eta2_new = (1 - var_lr) * eta2_curr + var_lr * eta2_targ
            eta3_new = (1 - var_lr) * eta3_curr + var_lr * eta3_targ
            eta4_new = (1 - var_lr) * eta4_curr + var_lr * eta4_targ

            # Convert back to standard parameters
            m_u_new, S_u_new, alpha_r_new, beta_r_new = gaussian_gamma_natural_to_standard_covariance_param(eta1_new, eta2_new, eta3_new, eta4_new)

            # Update model state
            self.m_u.data = m_u_new
            self.S_u.data = S_u_new
            self.alpha_r.data = alpha_r_new
            self.beta_r.data = beta_r_new

    def _m_step(self, optimizer, loss):
        if optimizer is None: return
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # def fit(
    #     self, 
    #     epochs=100, batch_size=128, 
    #     hyper_lr=0.01, var_lr=0.1,
    #     X_test=None, y_test=None, eval_interval=10
    # ):
    #     parameters_to_optimize = [p for name, p in self.named_parameters() if self.hyper_optim_mode.get(name.replace("log_",""), "MLE") != 'FIX']
        
    
    #     optimizer = optim.Adam(parameters_to_optimize, lr=hyper_lr) if parameters_to_optimize else None
    #     dataset = TensorDataset(self.X_full, self.y_full)
    #     generator = torch.Generator(device='cpu')
    #     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    #     history = {
    #         'elbo': [], 'log_prior': [], 'loss': [],
    #         'lengthscale': [], 'outputscale': [], 'dof_func': [], 'dof_lik': [], 'noisescale': [],
    #         'eval_epochs': [], 'eval_metrics': [], 'fit_times': []
    #     }
    #     logging.info(f"Starting SVI optimization for {epochs} epochs...")

    #     for epoch in range(epochs):
    #         for X_batch, y_batch in dataloader:

    #             fit_start_time = time.time()

    #             params = self._get_hyperparams()
    #             K_ZZ_base = to_linear_operator(self.kernel(self.Z, self.Z, params['lengthscale'], params['outputscale']))
    #             K_ZZ = K_ZZ_base.add_jitter(JITTER)
    #             K_XZ_batch = self.kernel(X_batch, self.Z, params['lengthscale'], params['outputscale'])

    #             # E-Step (Local)
    #             local_params = self._e_step_local(X_batch, y_batch, K_XZ_batch, K_ZZ, params)
    #             self._e_step_global(X_batch, y_batch, K_XZ_batch, K_ZZ, local_params, params, var_lr)

    #             # M-Step
    #             elbo = self._calculate_elbo(X_batch, y_batch, K_XZ_batch, K_ZZ, local_params)
    #             log_prior = self._calculate_log_prior(params)
    #             loss = - (elbo + log_prior)
                
    #             self._m_step(optimizer, loss)

    #             fit_end_time = time.time()

    #             # --- Store history ---
    #             params_final = self._get_hyperparams()
    #             history['elbo'].append(elbo.item())
    #             history['log_prior'].append(log_prior.item())
    #             history['loss'].append(loss.item())
    #             history['lengthscale'].append(params_final['lengthscale'].detach().cpu().numpy())
    #             history['outputscale'].append(params_final['outputscale'].item())
    #             history['noisescale'].append(params_final['noisescale'].item())
    #             history['dof_func'].append(params_final['dof_func'].item())
    #             history['dof_lik'].append(params_final['dof_lik'].item())
    #             history['fit_times'].append(fit_end_time - fit_start_time)

                
    #         if (epoch + 1) % eval_interval == 0:
    #             ls_str = ", ".join([f"{l:.3f}" for l in params_final['lengthscale']])
    #             logging.info(f"Epoch {epoch+1:3d}/{epochs} | Fit Time: {fit_end_time - fit_start_time:.3f}s | ELBO: {elbo.item():8.2f} | l: [{ls_str}] | var: {params_final['outputscale']:.3f} | noise2: {params_final['noisescale']:3f} | dof_func: {params_final['dof_func']:.2f} | dof_lik: {params_final['dof_lik']:.2f}")

    #         # --- Evaluation Step ---
    #         if X_test is not None and y_test is not None and (epoch + 1) % eval_interval == 0:
    #             metrics = self._evaluate(X_test, y_test)
    #             history['eval_epochs'].append(epoch + 1)
    #             history['eval_metrics'].append(metrics)
    #             logging.info(
    #                 f"Epoch {epoch+1:3d}/{epochs} | Test Metrics: "
    #                 f"RMSE: {metrics['rmse']:.3f}"
    #             )

    #     logging.info("Optimization finished.")
    #     return history

    def fit(
        self,
        epochs=100, batch_size=128,
        hyper_lr=0.01, var_lr=0.1,
        X_test=None, y_test=None, eval_interval=10
    ):
        parameters_to_optimize = [p for name, p in self.named_parameters() if self.hyper_optim_mode.get(name.replace("log_",""), "MLE") != 'FIX']


        optimizer = optim.Adam(parameters_to_optimize, lr=hyper_lr) if parameters_to_optimize else None
        dataset = TensorDataset(self.X_full, self.y_full)
        generator = torch.Generator(device='cpu')
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

        logging.info(f"Starting SVI optimization for {epochs} epochs...")

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_elbo = 0.0
            epoch_log_prior = 0.0
            epoch_fit_time = 0.0
            num_batches = 0

            for X_batch, y_batch in dataloader:
                fit_start_time = time.time()

                params = self._get_hyperparams()
                K_ZZ_base = to_linear_operator(self.kernel(self.Z, self.Z, params['lengthscale'], params['outputscale']))
                K_ZZ = K_ZZ_base.add_jitter(JITTER)
                K_XZ_batch = self.kernel(X_batch, self.Z, params['lengthscale'], params['outputscale'])

                local_params = self._e_step_local(X_batch, y_batch, K_XZ_batch, K_ZZ, params)
                self._e_step_global(X_batch, y_batch, K_XZ_batch, K_ZZ, local_params, params, var_lr)

                elbo = self._calculate_elbo(X_batch, y_batch, K_XZ_batch, K_ZZ, local_params)
                log_prior = self._calculate_log_prior(params)
                loss = - (elbo + log_prior)

                self._m_step(optimizer, loss)

                fit_end_time = time.time()

                epoch_loss += loss.item()
                epoch_elbo += elbo.item()
                epoch_log_prior += log_prior.item()
                epoch_fit_time += (fit_end_time - fit_start_time)
                num_batches += 1

            # --- Yield results for the completed epoch ---
            epoch_results = {
                'epoch': epoch + 1,
                'loss': epoch_loss / num_batches,
                'elbo': epoch_elbo / num_batches,
                'log_prior': epoch_log_prior / num_batches,
                'time': epoch_fit_time,
            }

            if (epoch + 1) % eval_interval == 0:
                params_final = self._get_hyperparams()
                ls_str = ", ".join([f"{l:.3f}" for l in params_final['lengthscale']])
                logging.info(f"Epoch {epoch+1:3d}/{epochs} | Fit Time: {epoch_fit_time:.3f}s | ELBO: {epoch_results['elbo']:8.2f} | l: [{ls_str}] | var: {params_final['outputscale']:.3f} | noise2: {params_final['noisescale']:3f} | dof_func: {params_final['dof_func']:.2f} | dof_lik: {params_final['dof_lik']:.2f}")

            if X_test is not None and y_test is not None and (epoch + 1) % eval_interval == 0:
                metrics = self._evaluate(X_test, y_test)
                epoch_results.update(metrics) # Add RMSE etc. to the results dict
                logging.info(
                    f"Epoch {epoch+1:3d}/{epochs} | Test Metrics: "
                    f"RMSE: {metrics['rmse']:.3f}"
                )

            yield epoch_results

        logging.info("Optimization finished.")

    def predict(self, X_test):
        """
        Calculates the parameters of the predictive distribution q(f_*) for new test data.
        The predictive distribution is a Student-t distribution.
        
        Args:
            X_test (torch.Tensor or np.ndarray): The test data points.
            
        Returns:
            dict: A dictionary containing the parameters of the Student-t distribution:
                  'loc' (mean), 'scale_sq' (scale squared), and 'dof' (degrees of freedom).
        """
        X_test = torch.as_tensor(X_test, dtype=self.X_full.dtype, device=self.device)
        with torch.no_grad():
            params = self._get_hyperparams()

            K_ZZ_base = to_linear_operator(self.kernel(self.Z, self.Z, params['lengthscale'], params['outputscale']))
            K_ZZ = K_ZZ_base.add_jitter(JITTER)
            K_star_Z = self.kernel(X_test, self.Z, params['lengthscale'], params['outputscale'])
            k_star_star = params['outputscale']
            
            identity_M = torch.eye(self.M, device=self.device, dtype=X_test.dtype)
            A_star = K_star_Z @ K_ZZ.solve(identity_M)
            
            # --- Parameters of the Student-t predictive distribution q(f_*) ---
            # Location (mean)
            mu_star = A_star @ self.m_u
            
            # Degrees of Freedom
            dof_star = 2 * self.alpha_r
            
            # Scale-squared
            scale_sq_star = (self.beta_r / self.alpha_r.clamp(min=EPSILON)) * \
                            (k_star_star - torch.sum(A_star * K_star_Z, dim=1) + torch.sum((A_star @ self.S_u) * A_star, dim=1))
            
            return {
                'loc': mu_star.squeeze(),
                'scale_sq': scale_sq_star.clamp(min=EPSILON),
                'dof': dof_star.clamp(min=EPSILON)
            }

    def _evaluate(self, X_test, y_test):
        """Evaluates the model on test data and returns a dictionary of metrics."""
        self.eval()
        with torch.no_grad():
            f_pred_tensor = self.predict(X_test)
            f_pred_numpy = f_pred_tensor['loc'].cpu().numpy()
            y_true_numpy = y_test.cpu().numpy()

            # metrics = {
            #     'rmse': np.sqrt(np.mean((y_true_numpy - f_pred_numpy)**2))
            # }
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_true_numpy, f_pred_numpy))
            }
        self.train()
        return metrics

