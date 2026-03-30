"""
Sub-Gaussian Alpha-Stable Noise Model for Blind-MIMOSC.

Implements the noise model N = A^{1/2} G where:
  - G ~ CN(0, sigma_n^2 I)  (complex Gaussian)
  - A ~ PositiveStable(alpha/2)  (positive alpha/2-stable random variable)
  - A and G are independent

This is a Gaussian scale mixture. Setting alpha=2 recovers pure AWGN.

Reference (Taqqu's definition):
  Samoradnitsky & Taqqu (1994). "Stable Non-Gaussian Random Processes."
"""
import math
import warnings
import numpy as np
import torch
from scipy.stats import levy_stable
from typing import Optional


def sample_positive_stable(
    alpha_half: float,
    n_samples: int,
    device: torch.device = torch.device("cpu"),
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Sample from the positive stable distribution with index alpha/2.

    Uses the Chambers-Mallows-Stuck (CMS) method via scipy.stats.levy_stable
    in its parameterization S(alpha/2, 1, scale, 0; 1).

    Args:
        alpha_half: Stability index alpha/2, must be in (0, 1).
        n_samples:  Number of samples to draw.
        device:     Output device.
        seed:       Optional RNG seed.

    Returns:
        samples: (n_samples,) tensor of positive stable samples (all > 0).
    """
    if not 0 < alpha_half < 1:
        raise ValueError(f"alpha_half must be in (0, 1), got {alpha_half}")

    rng = np.random.default_rng(seed)
    # scipy levy_stable: S(alpha, beta=1, loc=0, scale) gives one-sided stable
    # For alpha/2-stable: scale = cos(pi*alpha/4)^(2/alpha)  (unit scale in Nolan's form)
    scale = math.cos(math.pi * alpha_half / 2) ** (2.0 / alpha_half)
    samples = levy_stable.rvs(
        alpha=alpha_half,
        beta=1.0,
        loc=0.0,
        scale=scale,
        size=n_samples,
        random_state=rng,
    )
    samples = np.clip(samples, 1e-8, None)  # positive stable: all > 0
    return torch.from_numpy(samples.astype(np.float32)).to(device)


def stable_log_density(
    a: torch.Tensor,
    alpha_half: float,
    n_terms: int = 100,
) -> torch.Tensor:
    """
    Compute log p_{stable}(a; alpha/2) via series expansion of the Laplace transform.

    For a positive alpha/2-stable variable A, we use the Laplace transform:
        E[exp(-s A)] = exp(-s^{alpha/2})

    The density is evaluated numerically via inverse Laplace (Euler/Bromwich).
    For efficiency, we use a precomputed lookup table approach based on
    scipy's PDF.

    Args:
        a:          (N,) tensor of positive values.
        alpha_half: Stability index.

    Returns:
        log_p: (N,) tensor of log densities.
    """
    a_np = a.cpu().numpy()
    scale = math.cos(math.pi * alpha_half / 2) ** (2.0 / alpha_half)
    log_p = levy_stable.logpdf(a_np, alpha=alpha_half, beta=1.0, loc=0.0, scale=scale)
    return torch.from_numpy(log_p.astype(np.float32)).to(a.device)


class SubGaussianStableNoise:
    """
    Sub-Gaussian Alpha-Stable noise N = A^{1/2} G.

    Provides:
      - Noise generation
      - Log-likelihood computation
      - Posterior sampling over A given (Y, H0, D0)

    Args:
        alpha:   Stability index in (1, 2]. alpha=2 is pure AWGN.
        sigma_n: Scale of the Gaussian component.
    """

    def __init__(self, alpha: float = 1.5, sigma_n: float = 1.0):
        if not 1 < alpha <= 2:
            raise ValueError(f"alpha must be in (1, 2], got {alpha}")
        self.alpha = alpha
        self.alpha_half = alpha / 2.0
        self.sigma_n = sigma_n

        if alpha == 2.0:
            self._is_gaussian = True
        else:
            self._is_gaussian = False

    def sample_noise(
        self,
        shape: tuple,
        device: torch.device = torch.device("cpu"),
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample N = A^{1/2} G of given shape (complex).

        Returns:
            N: Complex tensor of given shape.
        """
        if self._is_gaussian:
            scale = self.sigma_n / math.sqrt(2)
            return torch.complex(
                torch.randn(shape, device=device) * scale,
                torch.randn(shape, device=device) * scale,
            )

        total = int(np.prod(shape))
        A = sample_positive_stable(self.alpha_half, total, device=device, seed=seed)
        A = A.view(shape)

        # G ~ CN(0, sigma_n^2 I)
        g_scale = self.sigma_n / math.sqrt(2)
        G = torch.complex(
            torch.randn(shape, device=device) * g_scale,
            torch.randn(shape, device=device) * g_scale,
        )
        return A.sqrt() * G

    def log_likelihood(
        self,
        Y: torch.Tensor,
        mean: torch.Tensor,
        A: Optional[torch.Tensor] = None,
        L_A: int = 20,
    ) -> torch.Tensor:
        """
        Compute log p(Y | mean) = log E_A[ CN(Y; mean, A*sigma_n^2*I) ].

        Uses Monte Carlo integration over A if A is not provided.

        Args:
            Y:    (B, NrK, T) complex received signal.
            mean: (B, NrK, T) complex signal mean H @ X.
            A:    (L_A,) pre-sampled values of A (optional).
            L_A:  Number of MC samples.

        Returns:
            log_lik: (B,) log-likelihood per batch element.
        """
        B, NrK, T = Y.shape
        residual = Y - mean  # (B, NrK, T)
        res_sq = (residual.abs() ** 2).sum(dim=(1, 2))  # (B,)

        if self._is_gaussian:
            var = self.sigma_n ** 2
            log_lik = -res_sq / var - NrK * T * math.log(math.pi * var)
            return log_lik

        # MC integration over A
        device = Y.device
        if A is None:
            A_samples = sample_positive_stable(self.alpha_half, L_A, device=device)
        else:
            A_samples = A

        # For each a, compute log CN(Y; mean, a * sigma_n^2 * I)
        # log CN = -n*log(pi*a*sigma_n^2) - ||Y-mean||^2 / (a*sigma_n^2)
        n_dim = NrK * T
        var_base = self.sigma_n ** 2
        log_probs = []
        for a in A_samples:
            a_val = a.item()
            log_cn = -n_dim * math.log(math.pi * a_val * var_base) - res_sq / (a_val * var_base)
            log_probs.append(log_cn)

        # log E_A[...] ≈ log mean_j exp(log_cn_j) (log-sum-exp)
        log_probs_stack = torch.stack(log_probs, dim=1)  # (B, L_A)
        log_lik = torch.logsumexp(log_probs_stack, dim=1) - math.log(L_A)
        return log_lik

    def sample_A_posterior(
        self,
        residual_sq: torch.Tensor,
        L_A: int = 20,
        n_mcmc: int = 100,
    ) -> torch.Tensor:
        """
        Draw samples from p(A | Y, H0, D0) via importance sampling.

        p(A | residual) ∝ CN(residual; 0, A*sigma_n^2*I) * p_stable(A)

        Args:
            residual_sq: (B,) squared Frobenius norm ||Y - H@X||^2_F.
            L_A:         Number of posterior samples to return.
            n_mcmc:      Number of importance sampling proposals.

        Returns:
            A_posterior: (B, L_A) tensor of posterior samples.
        """
        B = residual_sq.shape[0]
        device = residual_sq.device

        # Proposal: prior p_stable(A)
        A_prop = sample_positive_stable(self.alpha_half, n_mcmc, device=device)  # (n_mcmc,)

        A_posterior = torch.zeros(B, L_A, device=device)
        var_base = self.sigma_n ** 2

        for b in range(B):
            rsq = residual_sq[b].item()
            # Log weights: log CN(y; 0, A*sigma_n^2*I) evaluated at each A
            log_w = -rsq / (A_prop * var_base + 1e-8)
            # Normalize
            w = torch.softmax(log_w, dim=0)
            # Resample
            idx = torch.multinomial(w, L_A, replacement=True)
            A_posterior[b] = A_prop[idx]

        return A_posterior
