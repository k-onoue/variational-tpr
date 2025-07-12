import torch

# --- Helper class for Priors ---
class GammaPrior:
    def __init__(self, concentration, rate):
        self.concentration = concentration
        self.rate = rate

    def log_prob(self, x):
        # Log probability density of Gamma distribution (constants omitted)
        return (self.concentration - 1.0) * torch.log(x.clamp(min=1e-9)) - self.rate * x

class LogNormalPrior:
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.var = scale**2

    def log_prob(self, x):
        # Log probability density of LogNormal distribution (constants omitted)
        log_x = torch.log(x.clamp(min=1e-9))
        return -log_x - (log_x - self.loc)**2 / (2 * self.var)