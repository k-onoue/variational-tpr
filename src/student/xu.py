import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error
from linear_operator.operators import to_linear_operator

from .constants import EPSILON, JITTER
from .kernels import rbf_kernel, matern52_kernel
from .priors import GammaPrior, LogNormalPrior
from .utils import sample_mvt, kl_mvt_empirical


class XuTPR(nn.Module):
    """
    Non-Sparse Variational Student-t Process Regression.
    Refactored to support flexible hyperparameter optimization and evaluation.
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

        # --- Priors for Hyperparameters ---
        self.lengthscale_prior = GammaPrior(2.0, 1.0)
        self.outputscale_prior = GammaPrior(2.0, 1.0)
        self.dof_func_prior = LogNormalPrior(loc=1.0, scale=1.0) # Prior for dof_func
        self.dof_lik_prior = LogNormalPrior(loc=1.0, scale=1.0)   # Prior for dof_lik
        self.noisescale_prior = LogNormalPrior(loc=-2.0, scale=1.0) # Centered around exp(-2) ~ 0.13

        # --- Initialize Hyperparameters ---
        hyperparameters = self._initialize_hyperparameters(hyper_settings)
        
        # Register hyperparameters as learnable parameters
        self.log_lengthscale = nn.Parameter(torch.log(hyperparameters['lengthscale']))
        self.log_outputscale = nn.Parameter(torch.log(hyperparameters['outputscale']))
        self.log_dof_func = nn.Parameter(torch.log(hyperparameters['dof_func']))
        self.log_dof_lik = nn.Parameter(torch.log(hyperparameters['dof_lik']))
        self.log_noisescale = nn.Parameter(torch.log(hyperparameters['noisescale']))

        # --- Variational Parameters for q(f) ---
        self.m = nn.Parameter(torch.zeros(self.N, 1, device=self.device, dtype=dtype))
        self.chol_S = nn.Parameter(torch.eye(self.N, device=self.device, dtype=dtype))
        self.log_dof_q = nn.Parameter(torch.log(torch.tensor(4.0, device=self.device, dtype=dtype)))

        # Set kernel function
        if kernel in (None, "rbf"): self.kernel = rbf_kernel
        elif kernel == "matern52": self.kernel = matern52_kernel
        else:
            logging.info("Unknown kernel specified. Defaulting to RBF kernel.")
            self.kernel = rbf_kernel

        self.to(self.device)

    def _initialize_hyperparameters(self, hyper_settings=None):
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
                logging.info(f"Sampled initial {name} (Optim mode: {mode}): {final_value.cpu().numpy()}")
            else:
                final_value = torch.as_tensor(init_val, dtype=dtype, device=self.device)
                logging.info(f"Using provided initial {name} (Optim mode: {mode}): {final_value.cpu().numpy()}")
            
            initialized_params[name] = final_value

        ls = initialized_params['lengthscale']
        if ls.ndim == 0: ls = ls.repeat(self.D)
        if ls.shape[0] != self.D: raise ValueError("lengthscale must be scalar or vector of length D")
        initialized_params['lengthscale'] = ls
        
        return initialized_params

    def _get_hyperparams(self):
        """Returns transformed (positive) parameters from their log-space storage."""
        return {
            "lengthscale": torch.exp(self.log_lengthscale).clamp(min=EPSILON),
            "outputscale": torch.exp(self.log_outputscale).clamp(min=EPSILON),
            "dof_func": torch.exp(self.log_dof_func).clamp(min=EPSILON),
            "dof_lik": torch.exp(self.log_dof_lik).clamp(min=EPSILON),
            "noisescale": torch.exp(self.log_noisescale).clamp(min=EPSILON),
            "dof_q": torch.exp(self.log_dof_q).clamp(min=EPSILON),
        }

    def _calculate_log_prior(self, params):
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
    
    def calculate_elbo(self, num_samples=1):
        params = self._get_hyperparams()

        scale_tril_q = torch.tril(self.chol_S)
        f_samples = sample_mvt(self.m, scale_tril_q, params['dof_q'], num_samples)
        lik_dist = torch.distributions.StudentT(
            df=params['dof_lik'], loc=f_samples, scale=params['noisescale'] 
        )
        expected_log_lik = lik_dist.log_prob(self.y_train).sum(0).mean()

        K_XX_base = self.kernel(self.X_train, self.X_train, params['lengthscale'], params['outputscale'])
        K_XX_op = to_linear_operator(K_XX_base).add_jitter(JITTER)
        K_XX_chol = K_XX_op.cholesky()
        
        kl_div = kl_mvt_empirical(
            mu_q=self.m,
            scale_tril_q=scale_tril_q,
            dof_q=params['dof_q'],
            mu_p=torch.zeros_like(self.m),
            scale_tril_p=K_XX_chol,
            dof_p=params['dof_func'],
            num_samples=num_samples
        )
        
        return expected_log_lik - kl_div

    def fit(self, epochs=200, lr=0.01, num_mc_samples=8, X_test=None, y_test=None, eval_interval=10):
        """Trains the model by maximizing the ELBO (plus log prior for MAP)."""
        
        # Select parameters to optimize based on the specified mode
        params_to_optimize = []
        for name, p in self.named_parameters():
            # Variational parameters are always optimized
            if name in ['m', 'chol_S', 'log_dof_q']:
                params_to_optimize.append(p)
            # Hyperparameters are optimized if not 'FIX'
            elif self.hyper_optim_mode.get(name.replace("log_", ""), "MLE") != 'FIX':
                params_to_optimize.append(p)

        optimizer = optim.Adam(params_to_optimize, lr=lr) if params_to_optimize else None
        
        history = {
            'elbo': [], 'log_prior': [], 'loss': [], 'hyperparams': [],
            'eval_epochs': [], 'eval_metrics': [], 'fit_times': []
        }
        logging.info(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):

            fit_start_time = time.time()

            optimizer.zero_grad()
            
            elbo = self.calculate_elbo(num_samples=num_mc_samples)
            log_prior = self._calculate_log_prior(self._get_hyperparams())
            loss = -(elbo + log_prior)
            
            loss.backward()
            optimizer.step()

            fit_end_time = time.time()

            # Store history
            history['elbo'].append(elbo.item())
            history['log_prior'].append(log_prior.item())
            history['loss'].append(loss.item())
            history['hyperparams'].append({k: v.detach().cpu().numpy() for k, v in self._get_hyperparams().items()})
            history['fit_times'].append(fit_end_time - fit_start_time)


            if (epoch + 1) % 10 == 0:
                logging.info(f"Epoch {epoch+1:4d}/{epochs} | Fit Time: {fit_end_time - fit_start_time:.3f}s | Loss: {loss.item():.3f} | ELBO: {elbo.item():.3f}")

            # Evaluation Step
            if X_test is not None and y_test is not None and (epoch + 1) % eval_interval == 0:
                metrics = self._evaluate(X_test, y_test)
                history['eval_epochs'].append(epoch + 1)
                history['eval_metrics'].append(metrics)
                logging.info(f"Epoch {epoch+1:4d} | Test RMSE: {metrics['rmse']:.4f}")

        logging.info("Training finished.")
        return history

    def predict(self, X_test, num_samples=1000):
        """Generates samples from the predictive distribution q(f*)."""
        self.eval()
        X_test = torch.as_tensor(X_test, dtype=self.X_train.dtype, device=self.device)
        if X_test.ndim == 1: X_test = X_test.unsqueeze(1)
        
        with torch.no_grad():
            params = self._get_hyperparams()
            f_samples_posterior = sample_mvt(self.m, self.chol_S, params['dof_q'], num_samples)
            
            K_XX_base = self.kernel(self.X_train, self.X_train, params['lengthscale'], params['outputscale'])
            K_XX_op = to_linear_operator(K_XX_base).add_jitter(JITTER)
            K_star_X = self.kernel(X_test, self.X_train, params['lengthscale'], params['outputscale'])
            k_star_star_diag = self.kernel(X_test, X_test, params['lengthscale'], params['outputscale']).diag()

            K_inv_f_samples = K_XX_op.solve(f_samples_posterior)
            predictive_loc_per_sample = K_star_X @ K_inv_f_samples
            beta_per_sample = (f_samples_posterior * K_inv_f_samples).sum(0)
            dof_pred = params['dof_func'] + self.N
            
            K_star_X_K_inv = K_XX_op.solve(K_star_X.T).T
            term2 = (K_star_X_K_inv * K_star_X).sum(1)
            
            scale_factor = (params['dof_func'] + beta_per_sample) / dof_pred
            scale_base = k_star_star_diag - term2
            predictive_scale_sq_per_sample = scale_base.unsqueeze(1) * scale_factor.unsqueeze(0)
            
            pred_dist = torch.distributions.StudentT(
                df=dof_pred,
                loc=predictive_loc_per_sample,
                scale=torch.sqrt(predictive_scale_sq_per_sample.clamp(min=EPSILON))
            )
            predictive_samples = pred_dist.sample()
            
        self.train()
        return predictive_samples.cpu().numpy()

    def _evaluate(self, X_test, y_test):
        """Evaluates the model on test data and returns a dictionary of metrics."""
        predictive_samples = self.predict(X_test, num_samples=1000)
        mu_pred = np.mean(predictive_samples, axis=1)
        y_true = y_test.cpu().numpy().squeeze()

        rmse = np.sqrt(mean_squared_error(y_true, mu_pred))
        return {'rmse': rmse}



# # --- Main execution block for testing ---
# if __name__ == '__main__':
#     from sklearn.model_selection import train_test_split
#     import matplotlib.pyplot as plt
#     from scipy.stats import t as scipy_t

#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
#     # 1. Generate synthetic data
#     torch.manual_seed(42); np.random.seed(42)
#     X_data = torch.linspace(-6, 6, 200).unsqueeze(1)
#     y_true = torch.sin(X_data * 1.5) + torch.cos(X_data * 0.5)
#     noise = torch.from_numpy(scipy_t.rvs(df=3, size=200)).unsqueeze(1) * 0.25
#     y_data = y_true + noise
#     y_data[[20, 80, 150]] += torch.tensor([[6.0], [-5.0], [5.5]])

#     X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)

#     # 2. Initialize and train the model with hyperparameter settings
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     # Example: Fix the noise scale, use MAP for lengthscale, and MLE for others
#     hyper_settings = {
#         'lengthscale': {'optim': 'MAP'},
#         'outputscale': {'optim': 'FIX', 'init': 1.0},
#         'noisescale':  {'optim': 'MAP'},
#         'dof_func':    {'optim': 'MAP'},
#         'dof_lik':     {'optim': 'MAP'},
#         # 'outputscale', 'dof_prior', 'dof_lik' will default to 'MLE'
#     }

#     model = XuTPR(X_train, y_train, hyper_settings=hyper_settings, device=device)
    
#     history = model.fit(
#         epochs=200, lr=0.01, num_mc_samples=10000,
#         X_test=X_test, y_test=y_test, eval_interval=10
#     )
    
#     # 3. Plot optimization history
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(history['loss'], label='- (ELBO + log Prior)')
#     plt.title("Loss During Training")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.grid(True); plt.legend()

#     plt.subplot(1, 2, 2)
#     plt.plot(history['eval_epochs'], [m['rmse'] for m in history['eval_metrics']], 'o-')
#     plt.title("Test RMSE During Training")
#     plt.xlabel("Epoch")
#     plt.ylabel("RMSE")
#     plt.grid(True); plt.tight_layout(); plt.show()
    
#     # 4. Make final predictions and plot the result
#     X_plot = torch.linspace(-8, 8, 400).unsqueeze(1)
#     predictive_samples = model.predict(X_plot, num_samples=10000)
    
#     mu_pred = np.mean(predictive_samples, axis=1)
#     lower, upper = np.quantile(predictive_samples, [0.025, 0.975], axis=1)

#     plt.figure(figsize=(12, 7))
#     plt.plot(X_train.cpu(), y_train.cpu(), 'rx', label='Training Data w/ Outliers', alpha=0.6)
#     plt.plot(X_test.cpu(), y_test.cpu(), 'ko', mfc='none', label='Test Data')
#     plt.plot(X_plot.cpu(), mu_pred, 'b-', lw=2, label='Predictive Mean')
#     plt.fill_between(X_plot.squeeze().cpu(), lower, upper, color='blue', alpha=0.2, label='95% Credible Interval')
#     plt.title('Non-Sparse Variational Student-t Process', fontsize=16)
#     plt.xlabel('Input X'); plt.ylabel('Output Y'); plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout(); plt.ylim(-8, 8); plt.show()