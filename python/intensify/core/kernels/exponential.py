"""Exponential kernel for Hawkes processes."""

import numpy as np

from .base import Kernel


class ExponentialKernel(Kernel):
    r"""
    Exponential kernel: φ(t) = α * β * exp(-β * t) for t >= 0.

    Parameters
    ----------
    alpha : float
        Jump size (amplitude). Must be positive and < 1 for stationarity.
    beta : float
        Decay rate. Must be positive.

    Properties
    ----------
    L1 norm = α (branching ratio). Must be < 1 for stationarity.
    Recursive form: R_i = exp(-β * dt) * (1 + R_{i-1})

    Examples
    --------
    >>> kernel = ExponentialKernel(alpha=0.3, beta=1.5)
    >>> t = jnp.linspace(0, 5, 100)
    >>> vals = kernel.evaluate(t)
    >>> integral = kernel.integrate(10.0)
    >>> norm = kernel.l1_norm()
    """

    def __init__(self, alpha: float, beta: float, *, allow_signed: bool = False):
        if beta <= 0:
            raise ValueError("beta must be positive")
        if not allow_signed and alpha <= 0:
            raise ValueError("alpha must be positive")
        if allow_signed and alpha == 0:
            raise ValueError("alpha must be non-zero when allow_signed=True")
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.allow_signed = bool(allow_signed)

    def evaluate(self, t: np.array) -> np.array:
        """
        Evaluate kernel at time lags t.

        φ(t) = α β exp(-β t)
        """
        t = np.asarray(t)
        return self.alpha * self.beta * np.exp(-self.beta * t)

    def integrate(self, t: float) -> float:
        """
        Compute ∫_0^t φ(τ) dτ = α (1 - exp(-β t)).
        """
        t_val = float(t)
        return self.alpha * (1.0 - np.exp(-self.beta * t_val))

    def integrate_vec(self, t: np.array) -> np.array:
        t = np.asarray(t)
        return self.alpha * (1.0 - np.exp(-self.beta * t))

    def l1_norm(self) -> float:
        """
        ∫_0^∞ φ(t) dt = α.
        """
        return abs(self.alpha) if self.allow_signed else self.alpha

    def has_recursive_form(self) -> bool:
        # Signed / inhibitory kernels need the general O(N^2) path for correctness
        # with arbitrary pre-intensity before a nonlinear link (see NonlinearHawkes).
        return not self.allow_signed

    def recursive_state_update(self, state: np.array, dt: float) -> np.array:
        """
        Update recursive sufficient statistic.

        R_i = exp(-β * dt) * (1 + R_{i-1})
        """
        exp_factor = np.exp(-self.beta * dt)
        return exp_factor * (1.0 + state)

    def recursive_init_state(self) -> np.array:
        return np.asarray(0.0)

    def recursive_intensity_excitation(self, state: np.array) -> np.array:
        """Contribution α β R to λ - μ (R is pre-interval state)."""
        return self.alpha * self.beta * state

    def recursive_decay(self, state: np.array, dt: float) -> np.array:
        return np.exp(-self.beta * dt) * state

    def recursive_absorb(self, state: np.array) -> np.array:
        return state + 1.0

    def __repr__(self) -> str:
        sig = ", allow_signed=True" if self.allow_signed else ""
        return f"ExponentialKernel(alpha={self.alpha}, beta={self.beta}{sig})"
