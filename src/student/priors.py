import torch
from torch.distributions import Gamma, LogNormal

from .constants import EPSILON


class GammaPrior:
    def __init__(self, concentration, rate):
        self.concentration = torch.tensor(float(concentration))
        self.rate = torch.tensor(float(rate))
        self.dist = Gamma(self.concentration, self.rate)

    # Log probability density of Gamma distribution (constants omitted)
    def log_prob(self, x):
        return (self.concentration - 1.0) * torch.log(x.clamp(min=EPSILON)) - self.rate * x

    def sample(self, sample_shape=torch.Size()):
        return self.dist.sample(sample_shape)


class LogNormalPrior:
    def __init__(self, loc, scale):
        self.loc = torch.tensor(float(loc))
        self.scale = torch.tensor(float(scale))
        self.var = self.scale**2
        self.dist = LogNormal(self.loc, self.scale)

    # Log probability density of Gamma distribution (constants omitted)
    def log_prob(self, x):
        log_x = torch.log(x.clamp(min=EPSILON))
        return -log_x - (log_x - self.loc)**2 / (2 * self.var)

    def sample(self, sample_shape=torch.Size()):
        return self.dist.sample(sample_shape)