"""Nonparametric piecewise-constant kernel for Hawkes processes."""

from __future__ import annotations

import numpy as np

from .base import Kernel


class NonparametricKernel(Kernel):
    r"""
    Nonparametric piecewise-constant kernel.

    The kernel is defined on bins: [τ_0, τ_1), [τ_1, τ_2), ..., [τ_{K-1}, τ_K)
    with τ_0 = 0 and τ_K = ∞ (implicitly). The value on bin i is a_i.

    φ(t) = a_i for t ∈ [τ_i, τ_{i+1}), and 0 for t >= τ_K.

    Parameters
    ----------
    edges : array-like
        Bin edges τ_0=0, τ_1, ..., τ_K. Must be increasing, with τ_0=0.
    values : array-like
        Bin heights a_i for each bin (length K). Must be non-negative.
        Alternatively, if values is None, initialize uniformly or via heuristic.

    Notes
    -----
    - Does NOT have recursive form.
    - L1 norm = Σ_i a_i * (τ_{i+1} - τ_i). Compute via trapezoidal-like sum.
    - The last bin extends to infinity, so its contribution is a_{K-1} * (∞ - τ_{K-1})? That's infinite unless a_{K-1}=0.
      To avoid infinite mass, we must enforce a_{K-1}=0. In practice, we treat the kernel as having finite support: τ_K is a cutoff
      beyond which the kernel is zero. So we need a finite τ_K and set the last bin to be [τ_{K-1}, τ_K) with height a_{K-1}, and for t≥τ_K, φ(t)=0.
    - The bin edges should include an upper cutoff τ_K.
    """

    def __init__(self, edges: list[float], values: list[float]):
        if edges[0] != 0.0:
            raise ValueError("edges must start with 0.0")
        if any(edges[i] >= edges[i + 1] for i in range(len(edges) - 1)):
            raise ValueError("edges must be strictly increasing")
        if len(edges) != len(values) + 1:
            raise ValueError("len(edges) must be len(values)+1")
        if any(v < 0 for v in values):
            raise ValueError("kernel values must be non-negative")
        self.edges = [float(e) for e in edges]
        self.values = [float(v) for v in values]
        self.n_bins = len(values)

    def evaluate(self, t: np.array) -> np.array:
        """
        Piecewise constant: find which bin t falls into and return corresponding value.
        """
        t = np.asarray(t)
        # Vectorized lookup: for each t, find bin index
        # In JAX we could use jnp.searchsorted; in NumPy, np.searchsorted.
        # We'll do a Python loop for simplicity.
        # For scalar t or small arrays, this is fine.
        # For large arrays, we could vectorize.
        if t.shape == ():  # scalar
            # Find last edge <= t
            idx = 0
            while idx < self.n_bins and t >= self.edges[idx + 1]:
                idx += 1
            if idx >= self.n_bins or self.edges[idx] > t:
                return np.asarray(0.0)
            return np.asarray(self.values[idx])
        else:
            # Array case: loop
            results = []
            for ti in t:
                idx = 0
                while idx < self.n_bins and ti >= self.edges[idx + 1]:
                    idx += 1
                if idx >= self.n_bins or self.edges[idx] > ti:
                    val = 0.0
                else:
                    val = self.values[idx]
                results.append(val)
            return np.asarray(results)

    def integrate(self, t: float) -> float:
        """
        Compute ∫_0^t φ(τ) dτ.

        Integrate over bins that lie within [0, t].
        """
        t_val = float(t)
        integral = 0.0
        for i in range(self.n_bins):
            a = self.edges[i]
            b = self.edges[i + 1]
            if a >= t_val:
                break
            if b <= t_val:
                # full bin
                integral += self.values[i] * (b - a)
            else:
                # partial bin
                integral += self.values[i] * (t_val - a)
        return integral

    def l1_norm(self) -> float:
        """
        Total integral over all bins (assuming kernel is zero after last edge).
        """
        integral = 0.0
        for i in range(self.n_bins):
            integral += self.values[i] * (self.edges[i + 1] - self.edges[i])
        return integral

    def scale(self, factor: float) -> None:
        """Scale all bin heights so L1 norm becomes factor * current."""
        f = float(factor)
        self.values = [v * f for v in self.values]

    @property
    def jit_compatible(self) -> bool:
        return False

    def has_recursive_form(self) -> bool:
        return False

    @classmethod
    def select_bin_count_aic(
        cls,
        events: np.ndarray,
        T: float,
        k_min: int = 4,
        k_max: int = 18,
        max_lag: float | None = None,
    ) -> tuple[int, NonparametricKernel]:
        """
        Choose a bin count K on [0, max_lag] by AIC on a histogram of inter-event times
        (proxy for kernel support; coarse but cheap).

        Returns
        -------
        best_k : int
        kernel : NonparametricKernel
            Piecewise-constant heights from the MLE of a homogeneous Poisson-per-bin model
            on inter-arrival counts (ignoring self-excitation).
        """
        ev = np.sort(np.asarray(events, dtype=float).ravel())
        ev = ev[(ev >= 0) & (ev <= T)]
        dts = np.diff(np.concatenate([[0.0], ev]))
        dts = dts[dts > 0]
        if len(dts) == 0:
            K = k_min
            upper = max(float(T), 1.0)
            edges = np.linspace(0.0, upper, K + 1).tolist()
            values = [0.01] * K
            return K, cls(edges=edges, values=values)
        upper = (
            float(max_lag) if max_lag is not None else float(np.quantile(dts, 0.995))
        )
        upper = max(upper, float(np.max(dts)) * 1.01, 1e-6)

        best_score = np.inf
        best_k = k_min
        best_kernel = None
        n_obs = len(dts)
        for K in range(k_min, min(k_max, n_obs // 2 + 1) + 1):
            edges = np.linspace(0.0, upper, K + 1)
            counts, _ = np.histogram(dts, bins=edges)
            widths = np.diff(edges)
            ll = 0.0
            for i in range(K):
                w = float(widths[i])
                c = int(counts[i])
                lam = (c + 1e-6) / (w + 1e-12)
                ll += c * np.log(lam + 1e-300) - lam * w
                for k in range(1, c + 1):
                    ll -= np.log(k)
            n_params = K + 1
            aic = 2 * n_params - 2 * ll
            if aic < best_score:
                best_score = aic
                best_k = K
                values = [(counts[i] + 1e-6) / (widths[i] + 1e-12) for i in range(K)]
                best_kernel = cls(edges=edges.tolist(), values=values)
        if best_kernel is None:
            raise ValueError(
                "auto_select_bins could not fit any kernel; check that events "
                "contain at least 2 sorted non-negative samples."
            )
        return best_k, best_kernel

    def __repr__(self) -> str:
        return f"NonparametricKernel(edges={self.edges}, values={self.values})"
