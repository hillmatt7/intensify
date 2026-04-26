"""Power-law kernel for Hawkes processes."""

import numpy as np

from .base import Kernel



class PowerLawKernel(Kernel):
    r"""
    Power-law kernel: φ(t) = α (t + c)^{-(1+β)}.

    Parameters
    ----------
    alpha : float
        Amplitude. Must be positive.
    beta : float
        Tail exponent (> 0). Larger beta => heavier tail? Actually larger beta makes decay faster.
        For heavy tail, beta small.
    c : float
        Offset parameter to avoid singularity at t=0. Must be positive (c > 0).

    Notes
    -----
    L1 norm must be computed numerically and may be >= 1 for heavy tails.
    This kernel does NOT have recursive form; uses O(N²) general likelihood.

    Examples
    --------
    >>> kernel = PowerLawKernel(alpha=0.5, beta=0.8, c=0.1)
    """

    def __init__(self, alpha: float, beta: float, c: float = 1.0):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if beta <= 0:
            raise ValueError("beta must be positive")
        if c <= 0:
            raise ValueError("c must be positive to avoid singularity at t=0")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.c = float(c)

    def evaluate(self, t: np.array) -> np.array:
        """
        φ(t) = α (t + c)^{-(1+β)}
        """
        t = np.asarray(t)
        return self.alpha * np.power(t + self.c, -(1.0 + self.beta))

    def integrate(self, t: float) -> float:
        """
        ∫_0^t φ(τ) dτ.

        For power-law, indefinite integral: ∫ (τ+c)^{-(1+β)} dτ = -(1/β) (τ+c)^{-β}.
        So definite integral from 0 to t: (α/β)[ c^{-β} - (t+c)^{-β} ].
        """
        t_val = float(t)
        term0 = self.c ** (-self.beta)
        termt = (t_val + self.c) ** (-self.beta)
        return self.alpha / self.beta * (term0 - termt)

    def integrate_vec(self, t: np.array) -> np.array:
        t = np.asarray(t)
        term0 = self.c ** (-self.beta)
        termt = np.power(t + self.c, -self.beta)
        return (self.alpha / self.beta) * (term0 - termt)

    def l1_norm(self) -> float:
        """
        ∫_0^∞ φ(t) dt = α / β * c^{-β}.
        """
        return self.alpha / self.beta * (self.c ** (-self.beta))

    def has_recursive_form(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"PowerLawKernel(alpha={self.alpha}, beta={self.beta}, c={self.c})"