import torch

# --- Original Functions (Now with added robustness) ---

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPSILON = 1e-12


def kl_gamma(alpha1, beta1, alpha2, beta2):
    """
    Calculates the KL divergence KL(P1 || P2) between two Gamma distributions.
    P1 = Gamma(alpha1, beta1), P2 = Gamma(alpha2, beta2).
    """
    return (
        (alpha1 - alpha2) * torch.digamma(alpha1)
        - torch.lgamma(alpha1)
        + torch.lgamma(alpha2)
        + alpha2 * (torch.log(beta1) - torch.log(beta2))
        + alpha1 * (beta2 - beta1) / beta1
    )


def kl_gaussian(mu1, Sigma1, mu2, Sigma2):
    """
    Calculates the KL divergence KL(P1 || P2) between two D-dimensional
    multivariate Gaussian distributions.
    P1 = N(mu1, Sigma1), P2 = N(mu2, Sigma2).
    """
    D = mu1.shape[0]
    trace_term = torch.trace(torch.linalg.solve(Sigma2, Sigma1))
    diff_mu = mu2 - mu1
    mahalanobis_term = torch.dot(diff_mu, torch.linalg.solve(Sigma2, diff_mu))
    _, logdet_Sigma1 = torch.linalg.slogdet(Sigma1)
    _, logdet_Sigma2 = torch.linalg.slogdet(Sigma2)
    log_det_term = logdet_Sigma2 - logdet_Sigma1
    return 0.5 * (trace_term + mahalanobis_term - D + log_det_term)


def kl_gaussian_gamma_covariance_param(mu_q, S_q, alpha_q, beta_q, mu_p, K_p, alpha_p, beta_p):
    """
    Calculates KL divergence for the COVARIANCE parameterization.
    This is the CORRECT version for your model.

    Assumes the distributions are parameterized as:
    P(f, r) = N(f | mu, r^-1 * S) * Gamma(r | alpha, beta)
    
    The KL divergence is E_q(r)[KL(q(f|r) || p(f|r))] + KL(q(r) || p(r)).
    """
    D = mu_q.shape[0]

    # KL divergence between the Gamma marginals
    kl_gamma_term = kl_gamma(alpha_q, beta_q, alpha_p, beta_p)

    # Expectation of the KL divergence between the conditional Gaussians
    _, logdet_S = torch.linalg.slogdet(S_q)
    _, logdet_K = torch.linalg.slogdet(K_p)
    trace_term = torch.trace(torch.linalg.solve(K_p, S_q))
    
    diff_mu = mu_q - mu_p
    mahalanobis_term = torch.dot(diff_mu, torch.linalg.solve(K_p, diff_mu))

    # E[r] from the q distribution is alpha_q / beta_q
    E_r = alpha_q / beta_q

    # Assemble the final formula where E[r] ONLY multiplies the Mahalanobis term
    kl_gaussian_expected_term = 0.5 * (
        (logdet_K - logdet_S) - D + trace_term + E_r * mahalanobis_term
    )

    return kl_gamma_term + kl_gaussian_expected_term


def kl_gaussian_gamma_precision_param(mu1, Sigma1, alpha1, beta1, mu2, Sigma2, alpha2, beta2):
    """
    Calculates KL divergence for the PRECISION parameterization.
    This was the OLD version. Retained for reference.

    Assumes the distributions are parameterized as:
    P(mu, tau) = N(mu | mu_0, (tau * Sigma_0)^-1) * Gamma(tau | alpha, beta)
    """
    D = mu1.shape[0]
    kl_gamma_term = kl_gamma(alpha1, beta1, alpha2, beta2)

    diff_mu = mu1 - mu2
    mahalanobis_term = torch.dot(diff_mu, torch.linalg.solve(Sigma2, diff_mu))
    trace_term = torch.trace(torch.linalg.solve(Sigma2, Sigma1))
    
    # In this incorrect formulation, E[tau] multiplies the trace and -D terms
    E_tau = alpha1 / beta1
    expectation_term = 0.5 * E_tau * (mahalanobis_term + trace_term - D)
    
    _, logdet_Sigma1 = torch.linalg.slogdet(Sigma1)
    _, logdet_Sigma2 = torch.linalg.slogdet(Sigma2)
    log_det_term = 0.5 * (logdet_Sigma2 - logdet_Sigma1)

    return kl_gamma_term + expectation_term + log_det_term


def get_optimal_gaussian_gamma(m_q, S_q, alpha_q, beta_q):
    """
    Calculates the parameters of a Gaussian-Gamma distribution P that minimize KL(Q || P).

    This function implements the result of minimizing the Kullback-Leibler divergence
    KL(Q || P) where:
    - Q(f, r) = Normal(f | m_q, S_q) * Gamma(r | alpha_q, beta_q) (a factorized distribution)
    - P(f, r) = Normal(f | m_p, r⁻¹ * S_p) * Gamma(r | alpha_p, beta_p) (a coupled distribution)

    The minimization yields the following optimal parameters for P.

    Args:
        m_q (torch.Tensor): The mean vector of the Normal distribution in Q. Shape (D,).
        S_q (torch.Tensor): The covariance matrix of the Normal distribution in Q. Shape (D, D).
        alpha_q (torch.Tensor): The shape parameter of the Gamma distribution in Q. Scalar.
        beta_q (torch.Tensor): The rate parameter of the Gamma distribution in Q. Scalar.

    Returns:
        tuple[torch.Tensor]: A tuple containing the optimal parameters for P:
                             (m_p, S_p, alpha_p, beta_p).
    """
    
    # 1. Optimal mean parameter for P is the mean from Q.
    # m_p = E_Q(f)[f]
    m_p = m_q
    
    # 2. Optimal Gamma parameters for P are the parameters from Q.
    # This results from minimizing KL(Q(r) || P(r)).
    alpha_p = alpha_q
    beta_p = beta_q
    
    # 3. Optimal covariance-like parameter S_p is the covariance from Q
    #    scaled by the expected precision E[r] from Q.
    # E_Q(r)[r] = alpha_q / beta_q
    expected_r = (alpha_q / beta_q).clamp(min=EPSILON)
    S_p = expected_r * S_q
    
    return m_p, S_p, alpha_p, beta_p


def gaussian_gamma_standard_to_natural_covariance_param(m, S, alpha, beta):
    """
    Converts standard parameters of a Normal-Gamma distribution to natural parameters
    (covariance version).
    """
    identity = torch.eye(S.shape[0], device=S.device, dtype=S.dtype)
    S_inv = torch.linalg.solve(S, identity)
    eta1 = S_inv @ m
    eta2 = -0.5 * S_inv
    eta3 = alpha - 0.5
    # --- FIX: Use torch.dot for the quadratic form to avoid the warning ---
    # eta4 = -beta - 0.5 * torch.dot(m, S_inv @ m)
    eta4 = -beta - 0.5 * m.T @ S_inv @ m
    return eta1, eta2, eta3, eta4


def gaussian_gamma_standard_to_natural_precision_param(m, P, alpha, beta):
    """
    Converts standard parameters of a Normal-Gamma distribution to natural parameters
    (precision version).
    """
    eta1 = P @ m
    eta2 = -0.5 * P
    eta3 = alpha - 0.5
    # --- FIX: Use torch.dot for the quadratic form to avoid the warning ---
    # eta4 = -beta - 0.5 * torch.dot(m, P @ m)
    eta4 = -beta - 0.5 * m.T @ P @ m
    return eta1, eta2, eta3, eta4


def gaussian_gamma_natural_to_standard_covariance_param(eta1, eta2, eta3, eta4):
    """
    Converts natural parameters of a Normal-Gamma distribution back to standard
    parameters (covariance version).
    """
    identity = torch.eye(eta2.shape[0], device=eta2.device, dtype=eta2.dtype)
    S_inv = -2 * eta2
    S = torch.linalg.solve(S_inv, identity)
    m = S @ eta1
    alpha = (eta3 + 0.5).clamp(min=EPSILON)
    # --- FIX: Use torch.dot for the quadratic form to avoid the warning ---
    # beta = (-eta4 - 0.5 * torch.dot(m, S_inv @ m)).squeeze().clamp(min=EPSILON)
    beta = (-eta4 - 0.5 * m.T @ S_inv @ m).squeeze().clamp(min=EPSILON)
    return m, S, alpha, beta


def gaussian_gamma_natural_to_standard_precision_param(eta1, eta2, eta3, eta4):
    """
    Converts natural parameters of a Normal-Gamma distribution back to standard
    parameters (precision version).
    """
    P = -2 * eta2
    m = torch.linalg.solve(P, eta1)
    alpha = (eta3 + 0.5).clamp(min=EPSILON)
    # --- FIX: Use torch.dot for the quadratic form to avoid the warning ---
    # beta = (-eta4 - 0.5 * torch.dot(m, P @ m)).squeeze().clamp(min=EPSILON)
    beta = (-eta4 - 0.5 * m.T @ P @ m).squeeze().clamp(min=EPSILON)
    return m, P, alpha, beta