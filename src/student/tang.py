import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from linear_operator.operators import to_linear_operator
from sklearn.metrics import mean_squared_error

from .constants import EPSILON, JITTER
from .kernels import matern52_kernel, rbf_kernel
from .priors import GammaPrior, LogNormalPrior



class TangTPR(nn.Module):
    """
    Student-t Process Regression with Student-t Likelihood (Tang et al., 2017)
    
    This implementation uses the Laplace Approximation method described in the paper.
    """
    def __init__(self, X, y, kernel='rbf', hyper_settings=None, device=None):
        super().__init__()

        if device is None:
            self.device = X.device if isinstance(X, torch.Tensor) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.register_buffer('X_train', X.to(self.device))
        self.register_buffer('y_train', y.view(-1, 1).to(self.device))

        if self.X_train.ndim == 1: self.X_train = self.X_train.unsqueeze(1)
        if self.y_train.ndim == 1: self.y_train = self.y_train.unsqueeze(1)

        self.N, self.D = self.X_train.shape
        dtype = self.X_train.dtype

        # --- Priors for Hyperparameters (for MAP estimation) ---
        self.lengthscale_prior = GammaPrior(3.0, 6.0)
        self.outputscale_prior = GammaPrior(2.0, 0.15)
        self.dof_func_prior = LogNormalPrior(loc=1.0, scale=1.0)  # Prior for ν₁, degrees of freedom of the TP
        self.dof_lik_prior = LogNormalPrior(loc=1.0, scale=1.0)   # Prior for ν₂, degrees of freedom of the likelihood
        self.noisescale_prior = LogNormalPrior(loc=-4.0, scale=1.0)


        # --- Initialize Hyperparameters ---
        hyperparameters = self._initialize_hyperparameters(hyper_settings)
        
        # Register hyperparameters as learnable parameters in log-space
        self.log_lengthscale = nn.Parameter(torch.log(hyperparameters['lengthscale']))
        self.log_outputscale = nn.Parameter(torch.log(hyperparameters['outputscale']))
        self.log_dof_func = nn.Parameter(torch.log(hyperparameters['dof_func']))    # ν₁
        self.log_dof_lik = nn.Parameter(torch.log(hyperparameters['dof_lik']))     # ν₂
        self.log_noisescale = nn.Parameter(torch.log(hyperparameters['noisescale'])) # σ

        # --- Latent function mode f̂ (the central parameter of Laplace Approx) ---
        self.f_hat = nn.Parameter(torch.zeros_like(self.y_train))

        # Set kernel function
        if kernel in (None, "rbf"): self.kernel = rbf_kernel
        elif kernel == "matern52": self.kernel = matern52_kernel
        else:
            logging.info(f"Unknown kernel '{kernel}'. Defaulting to RBF kernel.")
            self.kernel = rbf_kernel

        self.to(self.device)

    def _initialize_hyperparameters(self, hyper_settings=None):
        """Initializes hyperparameters from settings or by sampling from priors."""
        self.hyper_optim_mode = {}
        dtype = self.X_train.dtype
        
        param_configs = {
            'lengthscale': {'prior': self.lengthscale_prior, 'is_vector': True},
            'outputscale': {'prior': self.outputscale_prior, 'is_vector': False},
            'dof_func': {'prior': self.dof_func_prior, 'is_vector': False},
            'dof_lik': {'prior': self.dof_lik_prior, 'is_vector': False},
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
                final_value = config['prior'].sample(sample_shape=sample_shape).to(self.device, dtype=dtype)
                if name.startswith('dof'): final_value = final_value.clamp(min=2.0)
            else:
                final_value = torch.as_tensor(init_val, dtype=dtype, device=self.device)
            
            logging.info(f"Initialized {name} = {final_value.cpu().numpy()} (Optim mode: {mode})")
            initialized_params[name] = final_value

        ls = initialized_params['lengthscale']
        if ls.ndim == 0: ls = ls.repeat(self.D)
        if ls.shape[0] != self.D: raise ValueError("lengthscale must be scalar or vector of length D")
        initialized_params['lengthscale'] = ls
        
        return initialized_params
    
    def _get_hyperparams(self):
        """Returns transformed (positive) hyperparameters from their log-space storage."""
        return {
            "lengthscale": torch.exp(self.log_lengthscale).clamp(min=EPSILON),
            "outputscale": torch.exp(self.log_outputscale).clamp(min=EPSILON),
            "dof_func": torch.exp(self.log_dof_func).clamp(min=EPSILON), 
            "dof_lik": torch.exp(self.log_dof_lik).clamp(min=EPSILON),
            "noisescale": torch.exp(self.log_noisescale).clamp(min=EPSILON*100), 
        }

    def _calculate_log_prior(self, params):
        """Calculates the log prior probability of the hyperparameters for MAP estimation."""
        log_prior = torch.tensor(0.0, device=self.device, dtype=params['lengthscale'].dtype)
        if self.hyper_optim_mode['lengthscale'] == 'MAP':
            log_prior += self.lengthscale_prior.log_prob(params['lengthscale']).sum()
        if self.hyper_optim_mode['outputscale'] == 'MAP':
            log_prior += self.outputscale_prior.log_prob(params['outputscale'])
        if self.hyper_optim_mode['dof_func'] == 'MAP':
            log_prior += self.dof_func_prior.log_prob(params['dof_func'])
        if self.hyper_optim_mode['dof_lik'] == 'MAP':
            log_prior += self.dof_lik_prior.log_prob(params['dof_lik'])
        if self.hyper_optim_mode['noisescale'] == 'MAP':
            log_prior += self.noisescale_prior.log_prob(params['noisescale'])
        return log_prior

    def _calculate_ln_Q(self, f, params, K_op):
        """Calculates ln Q, the negative log unnormalized posterior from formula (12)."""
        nu1, nu2, sigma = params['dof_func'], params['dof_lik'], params['noisescale']

        # Log-likelihood term
        lik_term_inner = ((self.y_train - f) / sigma).pow(2)
        log_lik_term = 0.5 * (nu2 + 1) * torch.log(1 + lik_term_inner / nu2).sum()

        # Log-prior term for f
        K_inv_f = K_op.solve(f)
        f_K_inv_f = (f * K_inv_f).sum()
        log_prior_term = 0.5 * (nu1 + self.N) * torch.log(1 + f_K_inv_f / nu1)

        return log_lik_term + log_prior_term

    def _calculate_approx_nll(self, f_hat, params, K_op):
        """Calculates the approximate negative log marginal likelihood from formula (18)."""
        # We use the standard form: NLL ≈ ln Q(f̂) + 0.5 * log|Hessian(ln Q)|
        ln_Q_at_f_hat = self._calculate_ln_Q(f_hat, params, K_op)
        
        # Calculate Hessian of ln Q at f_hat
        def func_to_hess(f_vec):
            # The hessian function needs a function mapping a 1D vector to a scalar
            return self._calculate_ln_Q(f_vec.view(-1, 1), params, K_op)

        hessian_matrix = torch.autograd.functional.hessian(func_to_hess, f_hat.flatten())
        
        # Add jitter for numerical stability before taking the determinant
        hessian_stable = hessian_matrix + torch.eye(self.N, device=self.device) * JITTER
        log_det_hessian = torch.linalg.slogdet(hessian_stable).logabsdet
        
        return ln_Q_at_f_hat + 0.5 * log_det_hessian
    
    # def fit(
    #     self, 
    #     epochs=100, 
    #     lr_hyper=0.01, 
    #     lr_f=0.1, f_steps=10, 
    #     X_test=None, y_test=None, eval_interval=10
    # ):
    #     """
    #     Trains the model by minimizing the approximate negative log marginal likelihood.
    #     This involves a nested optimization and records training history.
    #     1. Inner loop: Find the posterior mode f_hat using LBFGS.
    #     2. Outer loop: Update hyperparameters using Adam.
        
    #     Args:
    #         epochs (int): Number of training epochs.
    #         lr_hyper (float): Learning rate for the hyperparameter optimizer (Adam).
    #         lr_f (float): Learning rate for the latent mode optimizer (L-BFGS).
    #         f_steps (int): Number of optimization steps for finding f_hat in each epoch.
    #         X_test (torch.Tensor, optional): Test data for periodic evaluation.
    #         y_test (torch.Tensor, optional): Test labels for periodic evaluation.
    #         eval_interval (int): How often (in epochs) to perform evaluation.

    #     Returns:
    #         dict: A history dictionary containing training metrics.
    #     """
    #     hyper_params_to_opt = []
    #     for name, p in self.named_parameters():
    #         if name != 'f_hat' and self.hyper_optim_mode.get(name.replace("log_", ""), "MLE") != 'FIX':
    #             hyper_params_to_opt.append(p)
        
    #     optimizer_hyper = optim.Adam(hyper_params_to_opt, lr=lr_hyper) if hyper_params_to_opt else None
    #     optimizer_f = optim.LBFGS([self.f_hat], lr=lr_f)
        
    #     # --- History Recording Initialization ---
    #     history = {
    #         'elbo': [], 
    #         'log_prior': [], 
    #         'loss': [], 
    #         'hyperparams': [],
    #         'eval_epochs': [], 
    #         'eval_metrics': [], 
    #         'fit_times': []
    #     }
        
    #     logging.info(f"Starting training for {epochs} epochs...")
    #     for epoch in range(epochs):
    #         fit_start_time = time.time()
            
    #         # --- Step 1: Find posterior mode f_hat (inner optimization) ---
    #         params = self._get_hyperparams()
    #         K_XX_base = self.kernel(self.X_train, self.X_train, params['lengthscale'], params['outputscale'])
    #         K_XX_op = to_linear_operator(K_XX_base).add_jitter(JITTER)

    #         def closure_f():
    #             optimizer_f.zero_grad()
    #             loss_f = self._calculate_ln_Q(self.f_hat, params, K_XX_op.detach())
    #             loss_f.backward(retain_graph=True)
    #             return loss_f
            
    #         for _ in range(f_steps):
    #             optimizer_f.step(closure_f)
            
    #         f_hat_detached = self.f_hat.detach().clone()

    #         # --- Step 2: Update hyperparameters (outer optimization) ---
    #         if optimizer_hyper:
    #             optimizer_hyper.zero_grad()
                
    #             params = self._get_hyperparams()
    #             K_XX_base = self.kernel(self.X_train, self.X_train, params['lengthscale'], params['outputscale'])
    #             K_XX_op = to_linear_operator(K_XX_base).add_jitter(JITTER)

    #             approx_nll = self._calculate_approx_nll(f_hat_detached, params, K_XX_op)
    #             log_prior = self._calculate_log_prior(params)
                
    #             loss_hyper = approx_nll - log_prior
    #             loss_hyper.backward()
    #             optimizer_hyper.step()

    #             fit_end_time = time.time()

    #             # --- Store history for this epoch ---
    #             history['elbo'].append(approx_nll.item())
    #             history['log_prior'].append(log_prior.item())
    #             history['loss'].append(loss_hyper.item())
    #             history['hyperparams'].append({k: v.detach().cpu().numpy() for k, v in self._get_hyperparams().items()})
    #             history['fit_times'].append(fit_end_time - fit_start_time)

    #             if (epoch + 1) % 10 == 0:
    #                 logging.info(f"Epoch {epoch+1:4d}/{epochs} | Fit Time: {fit_end_time - fit_start_time:.3f}s | NLL: {approx_nll.item():.3f} | Loss: {loss_hyper.item():.3f}")
    #         else:
    #             # If no hyperparams to optimize, just log time
    #             fit_end_time = time.time()
    #             history['fit_times'].append(fit_end_time - fit_start_time)


    #         # --- Evaluation Step ---
    #         if X_test is not None and y_test is not None and (epoch + 1) % eval_interval == 0:
    #             metrics = self._evaluate(X_test, y_test)
    #             history['eval_epochs'].append(epoch + 1)
    #             history['eval_metrics'].append(metrics)
    #             logging.info(f"Epoch {epoch+1:4d} | Test RMSE: {metrics['rmse']:.4f}")

    #     logging.info("Training finished.")
    #     return history

    def fit(
        self, 
        epochs=100, 
        lr_hyper=0.01, 
        lr_f=0.1, f_steps=10, 
        X_test=None, y_test=None, eval_interval=10
    ):
        """
        Trains the model by minimizing the approximate negative log marginal likelihood.
        This involves a nested optimization and records training history.
        1. Inner loop: Find the posterior mode f_hat using Adam.
        2. Outer loop: Update hyperparameters using Adam.
        
        Args:
            epochs (int): Number of training epochs.
            lr_hyper (float): Learning rate for the hyperparameter optimizer (Adam).
            lr_f (float): Learning rate for the latent mode optimizer (Adam).
            f_steps (int): Number of optimization steps for finding f_hat in each epoch.
            X_test (torch.Tensor, optional): Test data for periodic evaluation.
            y_test (torch.Tensor, optional): Test labels for periodic evaluation.
            eval_interval (int): How often (in epochs) to perform evaluation.

        Returns:
            dict: A history dictionary containing training metrics.
        """
        hyper_params_to_opt = []
        for name, p in self.named_parameters():
            if name != 'f_hat' and self.hyper_optim_mode.get(name.replace("log_", ""), "MLE") != 'FIX':
                hyper_params_to_opt.append(p)
        
        optimizer_hyper = optim.Adam(hyper_params_to_opt, lr=lr_hyper) if hyper_params_to_opt else None
        # --- MODIFIED: Changed LBFGS to Adam ---
        optimizer_f = optim.Adam([self.f_hat], lr=lr_f)
        
        # --- History Recording Initialization ---
        history = {
            'elbo': [], 
            'log_prior': [], 
            'loss': [], 
            'hyperparams': [],
            'eval_epochs': [], 
            'eval_metrics': [], 
            'fit_times': []
        }
        
        logging.info(f"Starting training for {epochs} epochs...")
        for epoch in range(epochs):
            fit_start_time = time.time()
            
            # --- Step 1: Find posterior mode f_hat (inner optimization) ---
            params = self._get_hyperparams()
            K_XX_base = self.kernel(self.X_train, self.X_train, params['lengthscale'], params['outputscale'])
            K_XX_op = to_linear_operator(K_XX_base).add_jitter(JITTER)

            # --- MODIFIED: Replaced LBFGS loop with standard Adam loop ---
            for _ in range(f_steps):
                optimizer_f.zero_grad()
                loss_f = self._calculate_ln_Q(self.f_hat, params, K_XX_op.detach())
                loss_f.backward(retain_graph=True)
                optimizer_f.step()
            
            f_hat_detached = self.f_hat.detach().clone()

            # --- Step 2: Update hyperparameters (outer optimization) ---
            if optimizer_hyper:
                optimizer_hyper.zero_grad()
                
                params = self._get_hyperparams()
                K_XX_base = self.kernel(self.X_train, self.X_train, params['lengthscale'], params['outputscale'])
                K_XX_op = to_linear_operator(K_XX_base).add_jitter(JITTER)

                approx_nll = self._calculate_approx_nll(f_hat_detached, params, K_XX_op)
                log_prior = self._calculate_log_prior(params)
                
                loss_hyper = approx_nll - log_prior
                loss_hyper.backward()
                optimizer_hyper.step()

                fit_end_time = time.time()

                # --- Store history for this epoch ---
                # history['elbo'].append(approx_nll.item())
                history['elbo'].append(-approx_nll.item())
                history['log_prior'].append(log_prior.item())
                history['loss'].append(loss_hyper.item())
                history['hyperparams'].append({k: v.detach().cpu().numpy() for k, v in self._get_hyperparams().items()})
                history['fit_times'].append(fit_end_time - fit_start_time)

                if (epoch + 1) % 10 == 0:
                    logging.info(f"Epoch {epoch+1:4d}/{epochs} | Fit Time: {fit_end_time - fit_start_time:.3f}s | NLL: {approx_nll.item():.3f} | Loss: {loss_hyper.item():.3f}")
            else:
                # If no hyperparams to optimize, just log time
                fit_end_time = time.time()
                history['fit_times'].append(fit_end_time - fit_start_time)


            # --- Evaluation Step ---
            if X_test is not None and y_test is not None and (epoch + 1) % eval_interval == 0:
                metrics = self._evaluate(X_test, y_test)
                history['eval_epochs'].append(epoch + 1)
                history['eval_metrics'].append(metrics)
                logging.info(f"Epoch {epoch+1:4d} | Test RMSE: {metrics['rmse']:.4f}")

        logging.info("Training finished.")
        return history

    def predict(self, X_test):
        """
        Computes the predictive mean for new data X_test using formula (20).
        Returns the mean of the predictive distribution.
        """
        self.eval()
        X_test = torch.as_tensor(X_test, dtype=self.X_train.dtype, device=self.device)
        if X_test.ndim == 1: X_test = X_test.unsqueeze(1)
        
        with torch.no_grad():
            params = self._get_hyperparams()
            K_XX_base = self.kernel(self.X_train, self.X_train, params['lengthscale'], params['outputscale'])
            K_XX_op = to_linear_operator(K_XX_base).add_jitter(JITTER)
            
            K_star_X = self.kernel(X_test, self.X_train, params['lengthscale'], params['outputscale'])
            
            # Predictive mean: k*^T K_inv f_hat
            K_inv_f_hat = K_XX_op.solve(self.f_hat)
            mu_pred = K_star_X @ K_inv_f_hat
            
        self.train()
        return mu_pred.cpu().numpy()

    def _evaluate(self, X_test, y_test):
        """Evaluates the model on test data and returns a dictionary of metrics."""
        mu_pred = self.predict(X_test)
        y_true = y_test.cpu().numpy().squeeze()
        rmse = np.sqrt(mean_squared_error(y_true, mu_pred))
        return {'rmse': rmse}