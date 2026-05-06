"""Sum-of-exponentials kernel for Hawkes processes."""

import numpy as np

from .base import Kernel


class SumExponentialKernel(Kernel):
    r"""
    Sum-of-exponentials kernel: φ(t) = Σ_k α_k β_k exp(-β_k t).

    Parameters
    ----------
    alphas : list or array of floats
        Amplitudes α_k for each component. Must be positive.
    betas : list or array of floats
        Decay rates β_k for each component. Must be positive.
        Must have same length as alphas.

    Properties
    ----------
    L1 norm = Σ_k α_k.
    Recursive: For each component k, R_i^k = exp(-β_k * dt) * (1 + R_{i-1}^k).

    Examples
    --------
    >>> kernel = SumExponentialKernel(alphas=[0.2, 0.1], betas=[1.0, 5.0])
    >>> t = jnp.linspace(0, 5, 100)
    >>> vals = kernel.evaluate(t)
    """

    def __init__(self, alphas, betas):
        if len(alphas) != len(betas):
            raise ValueError("alphas and betas must have same length")
        if any(a <= 0 for a in alphas):
            raise ValueError("all alphas must be positive")
        if any(b <= 0 for b in betas):
            raise ValueError("all betas must be positive")
        self.alphas = [float(a) for a in alphas]
        self.betas = [float(b) for b in betas]
        self.n_components = len(alphas)

    def evaluate(self, t: np.array) -> np.array:
        """
        φ(t) = Σ_k α_k β_k exp(-β_k t)
        """
        t = np.asarray(t)
        result = np.zeros_like(t)
        for a, b in zip(self.alphas, self.betas):
            result += a * b * np.exp(-b * t)
        return result

    def integrate(self, t: float) -> float:
        """∫_0^t φ(τ) dτ = Σ_k α_k (1 - exp(-β_k t))."""
        t_val = float(t)
        integral = 0.0
        for a, b in zip(self.alphas, self.betas):
            integral += a * (1 - np.exp(-b * t_val))
        return float(integral)

    def integrate_vec(self, t: np.array) -> np.array:
        t = np.asarray(t)
        result = np.zeros_like(t)
        for a, b in zip(self.alphas, self.betas):
            result = result + a * (1.0 - np.exp(-b * t))
        return result

    def l1_norm(self) -> float:
        """∫_0^∞ φ(t) dt = Σ_k α_k."""
        return float(sum(self.alphas))

    def scale(self, factor: float) -> None:
        """Scale all component weights so L1 norm becomes factor * current."""
        f = float(factor)
        self.alphas = [a * f for a in self.alphas]

    def has_recursive_form(self) -> bool:
        return True

    def recursive_state_update(self, state: np.array, dt: float) -> np.array:
        """
        state: vector of R_{i-1}^k for each component k.
        R_i^k = exp(-β_k * dt) * (1 + R_{i-1}^k)

        Parameters
        ----------
        state : jnp.ndarray or np.ndarray, shape (n_components,)
        dt : float

        Returns
        -------
        new_state : same shape as state
        """
        new_state = []
        for k in range(self.n_components):
            R_prev = state[k] if hasattr(state, "__getitem__") else state
            R_new = np.exp(-self.betas[k] * dt) * (1.0 + R_prev)
            new_state.append(R_new)
        return np.asarray(new_state)

    def recursive_init_state(self) -> np.array:
        return np.zeros(self.n_components)

    def recursive_intensity_excitation(self, state: np.array) -> np.array:
        acc = np.asarray(0.0)
        for k in range(self.n_components):
            acc = acc + self.alphas[k] * self.betas[k] * state[k]
        return acc

    def recursive_decay(self, state: np.array, dt: float) -> np.array:
        decayed = []
        for k in range(self.n_components):
            R_k = state[k] if hasattr(state, "__getitem__") else state
            decayed.append(np.exp(-self.betas[k] * dt) * R_k)
        return np.asarray(decayed)

    def recursive_absorb(self, state: np.array) -> np.array:
        return state + 1.0

    def __repr__(self) -> str:
        return f"SumExponentialKernel(alphas={self.alphas}, betas={self.betas})"
