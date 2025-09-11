import torch


def rbf_kernel(X1, X2, lengthscale, outputscale=1.0):
    """
    Computes the RBF kernel matrix with ARD support.

    Args:
        X1 (torch.Tensor): A tensor of size (N, D).
        X2 (torch.Tensor): A tensor of size (M, D).
        lengthscale (torch.Tensor): A tensor of size (D,) representing the lengthscale for each dimension.
        outputscale (float): The kernel outputscale.
    """
    # Ensure outputscale is a tensor on the correct device
    outputscale = torch.as_tensor(outputscale, dtype=X1.dtype, device=X1.device)
    
    # Scale each dimension of X1 and X2 by the corresponding lengthscale
    # This uses broadcasting to efficiently perform the operation
    X1_scaled = X1 / lengthscale
    X2_scaled = X2 / lengthscale
    
    # Compute the squared Euclidean distance in the scaled space
    sqdist = torch.cdist(X1_scaled, X2_scaled, p=2).pow(2)
    
    return outputscale * torch.exp(-0.5 * sqdist)


def matern52_kernel(X1, X2, lengthscale, outputscale=1.0):
    _sqrt_5 = torch.sqrt(torch.tensor(5.0))
    sqdist = torch.cdist(X1 / lengthscale, X2 / lengthscale, p=2)
    term1 = 1 + _sqrt_5 * sqdist + (5/3) * sqdist**2
    term2 = torch.exp(_sqrt_5 * sqdist)
    return outputscale * term1 * term2