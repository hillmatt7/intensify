"""Approximate Power-Law kernel via geometric sum of exponentials (Bacry-Muzy)."""

import numpy as np

from .base import Kernel


class ApproxPowerLawKernel(Kernel):
    r"""
    Approximate power-law kernel using a sum of exponentials.

    φ(t) = α Σ_{k=1}^K w_k β_k exp(-β_k t)

    where β_k are geometrically spaced: β_k = β_min * r^(k-1)
    and weights w_k are set to approximate a power-law with exponent β_pow.

    Parameters
    ----------
    alpha : float
        Overall amplitude.
    beta_pow : float
        Desired power-law exponent (similar to PowerLawKernel's beta).
        The kernel's L1 norm = α (independent of spacing).
    beta_min : float
        Smallest decay rate (controls long memory).
    r : float
        Geometric ratio (> 1) between successive β's. Default 1.5.
    n_components : int
        Number of exponential components K. Default 10.
    """

    def __init__(
        self,
        alpha: float,
        beta_pow: float,
        beta_min: float,
        r: float = 1.5,
        n_components: int = 10,
    ):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if beta_pow <= 0:
            raise ValueError("beta_pow must be positive")
        if beta_min <= 0:
            raise ValueError("beta_min must be positive")
        if r <= 1.0:
            raise ValueError("r must be > 1 for geometric spacing")
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        self.alpha = float(alpha)
        self.beta_pow = float(beta_pow)
        self.beta_min = float(beta_min)
        self.r = float(r)
        self.n_components = int(n_components)
        # Compute components: betas and weights
        self.betas = [beta_min * (r**k) for k in range(self.n_components)]
        # Weights: w_k = β_k^{β_pow - 1} / Σ_j β_j^{β_pow - 1}  (normalized to sum to 1)
        # Actually the Bacry-Muzy construction: to approximate φ(t) ~ t^{-(1+β_pow)}
        # Use weights w_k ∝ β_k^{β_pow - 1}
        weights_raw = [b ** (beta_pow - 1) for b in self.betas]
        sum_weights = sum(weights_raw)
        self.weights = [w / sum_weights for w in weights_raw]
        # L1 norm of approximation is α Σ_k w_k = α

    def evaluate(self, t: np.array) -> np.array:
        """
        φ(t) = α Σ_k w_k β_k exp(-β_k t)
        """
        t = np.asarray(t)
        result = np.zeros_like(t)
        for w, b in zip(self.weights, self.betas):
            result += self.alpha * w * b * np.exp(-b * t)
        return result

    def integrate(self, t: float) -> float:
        """
        ∫_0^t φ(τ) dτ = α Σ_k w_k (1 - exp(-β_k t))
        """
        t_val = float(t)
        integral = 0.0
        for w, b in zip(self.weights, self.betas):
            integral += w * (1 - np.exp(-b * t_val))
        return self.alpha * integral

    def integrate_vec(self, t: np.array) -> np.array:
        t = np.asarray(t)
        result = np.zeros_like(t)
        for w, b in zip(self.weights, self.betas):
            result = result + w * (1.0 - np.exp(-b * t))
        return self.alpha * result

    def l1_norm(self) -> float:
        """∫_0^∞ φ(t) dt = α."""
        return self.alpha

    def has_recursive_form(self) -> bool:
        return True

    def recursive_state_update(self, state: np.array, dt: float) -> np.array:
        """
        state: vector of R_{i-1}^k for each component k.
        R_i^k = exp(-β_k * dt) * (1 + R_{i-1}^k)
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
            acc = acc + self.alpha * self.weights[k] * self.betas[k] * state[k]
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
        return f"ApproxPowerLawKernel(alpha={self.alpha}, beta_pow={self.beta_pow}, n_components={self.n_components})"
