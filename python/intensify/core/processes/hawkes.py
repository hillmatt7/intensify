"""Hawkes process models (univariate and multivariate)."""

from __future__ import annotations

import warnings

from ...backends import get_backend
from ...core.base import PointProcessBase
from ...core.kernels import Kernel

bt = get_backend()


class UnivariateHawkes(PointProcessBase):
    """
    Univariate Hawkes process: λ*(t) = μ + Σ_{t_i < t} φ(t - t_i).

    Parameters
    ----------
    mu : float
        Baseline intensity (background rate). Must be non-negative.
    kernel : Kernel
        Excitation kernel φ. Must implement Kernel ABC.
    """

    def __init__(self, mu: float, kernel: Kernel):
        if mu < 0:
            raise ValueError("mu must be non-negative")
        self.mu = float(mu)
        self.kernel = kernel

    def simulate(self, T: float, seed: int = None) -> bt.array:
        """
        Simulate using Ogata thinning.

        Parameters
        ----------
        T : float
            End of observation window.
        seed : int, optional
            Random seed.

        Returns
        -------
        events : jnp.ndarray or np.ndarray
            Sorted event timestamps.
        """
        # Phase 3 port: ExponentialKernel routes through the Rust simulator.
        from ..._rust import _ext  # noqa: PLC0415
        from ..kernels.exponential import ExponentialKernel

        if isinstance(self.kernel, ExponentialKernel) and not self.kernel.allow_signed:
            seed_u64 = int(seed) if seed is not None else 0
            arr = _ext.simulation.simulate_uni_exp_hawkes(
                float(T),
                float(self.mu),
                float(self.kernel.alpha),
                float(self.kernel.beta),
                seed_u64,
            )
            return bt.asarray(arr)

        from ..simulation.thinning import ogata_thinning

        key = bt.random.PRNGKey(seed) if seed is not None else None
        return ogata_thinning(self, T, key=key, seed=seed)

    def intensity(self, t, history: bt.array):
        """
        Compute conditional intensity λ(t | history).

        λ(t) = μ + Σ_{t_i < t} φ(t - t_i)

        Accepts scalar or array *t*.  When *t* is an array, uses vectorized
        broadcasting: ``lags[i, j] = t[i] - history[j]``.
        """
        history = bt.asarray(history)
        if history.size == 0:
            t_arr = bt.asarray(t).ravel()
            if t_arr.size == 1:
                return float(self.mu)
            return bt.full(t_arr.shape, self.mu)

        t_flat = bt.asarray(t).ravel()
        if t_flat.size == 1:
            tv = float(t_flat[0])
            lags = tv - history
            causal = lags > 0
            if not bt.any(causal):
                return float(self.mu)
            kernel_vals = self.kernel.evaluate(lags[causal])
            return float(self.mu + bt.sum(kernel_vals))

        # Vectorised path: t shape (T,), history shape (N,)
        lags = t_flat[:, None] - history[None, :]  # (T, N)
        causal = lags > 0
        kernel_vals = bt.where(causal, self.kernel.evaluate(lags), 0.0)
        return self.mu + bt.sum(kernel_vals, axis=1)

    def log_likelihood(self, events: bt.array, T: float) -> float:
        """
        Compute log-likelihood of observed events.

        Uses recursive O(N) path if kernel has_recursive_form(), else O(N^2) general path.

        Parameters
        ----------
        events : jnp.ndarray or np.ndarray
            Sorted event timestamps in ascending order.
        T : float
            Observation window end.

        Returns
        -------
        ll : float
            Log-likelihood.
        """
        from ...backends._backend import get_backend_name
        from ..inference.mle import (
            _general_likelihood,
            _general_likelihood_numpy,
            _recursive_likelihood,
            _recursive_likelihood_numpy,
        )

        if len(events) == 0:
            return 0.0

        use_numpy = get_backend_name() == "numpy"
        if self.kernel.has_recursive_form():
            if use_numpy:
                return float(_recursive_likelihood_numpy(self, events, T))
            return _recursive_likelihood(self, events, T)
        if use_numpy:
            return float(_general_likelihood_numpy(self, events, T))
        return _general_likelihood(self, events, T)

    def _calc_compensator(self, events: bt.array, T: float) -> float:
        """
        Compute compensator: ∫_0^T λ(t) dt = μ T + Σ ∫_0^{T-t_i} φ(s) ds.
        This is used in likelihood computation.
        """
        n = len(events)
        s = 0.0
        for i in range(n):
            s += self.kernel.integrate(float(T - events[i]))
        return self.mu * T + s

    def get_params(self) -> dict:
        return {"mu": self.mu, "kernel": self.kernel}

    def set_params(self, params: dict) -> None:
        if "mu" in params:
            self.mu = float(params["mu"])
        if "kernel" in params:
            self.kernel = params["kernel"]

    def project_params(self) -> None:
        """Project parameters to ensure stationarity (branching ratio < 1)."""
        norm = float(self.kernel.l1_norm())
        if norm >= 1.0:
            warnings.warn(
                f"Kernel L1 norm {norm:.4f} >= 1 is non-stationary. "
                "Projecting to 0.99.",
                UserWarning,
            )
            try:
                self.kernel.scale(0.99 / norm)
            except NotImplementedError:
                warnings.warn(
                    f"{type(self.kernel).__name__} does not support in-place "
                    "projection; manual parameter adjustment required.",
                    UserWarning,
                )


class MultivariateHawkes(PointProcessBase):
    """
    Multivariate Hawkes process: λ*_m(t) = μ_m + Σ_k Σ_{t_i^k < t} φ_{mk}(t - t_i^k).

    Parameters
    ----------
    n_dims : int
        Number of dimensions (M).
    mu : array-like or float
        Baseline intensity vector of length n_dims (or scalar for all same).
    kernel : Kernel or list of Kernel
        Either a single shared kernel or a list of n_dims*n_dims kernels.
        For full flexibility, provide a 2D array or list-of-lists of kernels
        representing φ_{mk} from dimension k to dimension m.
    """

    def __init__(
        self,
        n_dims: int,
        mu: float | bt.array,
        kernel: Kernel | list[list[Kernel]],
    ):
        self.n_dims = int(n_dims)
        if self.n_dims <= 0:
            raise ValueError("n_dims must be positive")

        mu_a = bt.asarray(mu)
        if mu_a.shape == () or mu_a.size == 1:
            self.mu = bt.full(self.n_dims, float(mu_a))
        elif mu_a.shape == (self.n_dims,):
            self.mu = mu_a
        else:
            raise ValueError(f"mu must be scalar or length-{self.n_dims} vector")

        # Process kernel matrix
        if isinstance(kernel, Kernel):
            # Shared single kernel for all pairs
            self.kernel_matrix = [
                [kernel for _ in range(self.n_dims)] for _ in range(self.n_dims)
            ]
        else:
            # kernel is list-of-lists
            if len(kernel) != self.n_dims or any(len(row) != self.n_dims for row in kernel):
                raise ValueError(
                    f"kernel matrix must be {self.n_dims}x{self.n_dims}"
                )
            self.kernel_matrix = kernel

    def simulate(self, T: float, seed: int = None) -> list[bt.array]:
        """
        Simulate multivariate events. Returns a list of event arrays, one per dimension.

        Uses Ogata thinning over all dimensions (multivariate thinning).

        Parameters
        ----------
        T : float
            End of observation window.
        seed : int, optional
            Random seed.

        Returns
        -------
        events_by_dim : list of jnp.ndarray
            Sorted events for each dimension.
        """
        # Phase 3 port: ExponentialKernel + shared β routes through Rust simulator.
        from ..._rust import _ext, mv_shared_beta  # noqa: PLC0415

        shared_beta = mv_shared_beta(self)
        if shared_beta is not None:
            import numpy as np  # noqa: PLC0415
            seed_u64 = int(seed) if seed is not None else 0
            mu = np.ascontiguousarray(np.asarray(self.mu, dtype=np.float64).ravel())
            alpha = np.empty(self.n_dims * self.n_dims, dtype=np.float64)
            for i in range(self.n_dims):
                for j in range(self.n_dims):
                    alpha[i * self.n_dims + j] = float(self.kernel_matrix[i][j].alpha)
            histories = _ext.simulation.simulate_mv_exp_hawkes(
                float(T), mu, alpha, float(shared_beta), seed_u64,
            )
            return [bt.asarray(h) for h in histories]

        from ..simulation.thinning import ogata_thinning_multivariate

        key = bt.random.PRNGKey(seed) if seed is not None else None
        return ogata_thinning_multivariate(self, T, key=key, seed=seed)

    def intensity(self, t: float, history: list[bt.array]) -> bt.array:
        """
        Compute conditional intensity vector λ(t) given histories for each dimension.

        Parameters
        ----------
        t : float
            Time at which to evaluate.
        history : list of arrays
            history[d] contains events for dimension d before t.

        Returns
        -------
        lambda_vec : jnp.ndarray or np.ndarray
            Intensity for each dimension, shape (n_dims,).
        """
        lam = bt.asarray(self.mu, dtype=float).copy()
        for m in range(self.n_dims):
            for k in range(self.n_dims):
                hist_k = history[k]
                if len(hist_k) == 0:
                    continue
                lags = t - hist_k
                causal = lags > 0
                if bt.any(causal):
                    kernel_vals = self.kernel_matrix[m][k].evaluate(lags[causal])
                    inc = bt.sum(kernel_vals)
                    if hasattr(lam, "at"):
                        lam = lam.at[m].add(inc)
                    else:
                        lam[m] = float(lam[m]) + float(inc)
        return lam

    def log_likelihood(self, events: list[bt.array], T: float) -> float:
        """
        Compute total log-likelihood across all dimensions.

        For multivariate Hawkes, the likelihood factorizes across dimensions:
        L = Π_m L_m(λ*_m(t)) where λ*_m depends on all dimensions' history.

        Parameters
        ----------
        events : list of arrays
            Event timestamps per dimension.
        T : float
            Observation window end.

        Returns
        -------
        ll : float
            Total log-likelihood.
        """
        total_ll = 0.0
        for m in range(self.n_dims):
            # Build-intensity function specific to dimension m given all histories
            # For efficiency, we could use recursive path if all kernels are recursive.
            # Here we do general O(N^2) per dimension, where N = total events across all dims
            total_ll += self._log_likelihood_dim(m, events, T)
        return total_ll

    def _log_likelihood_dim(self, m: int, events: list[bt.array], T: float) -> float:
        """
        Log-likelihood contribution for dimension *m*:

            L_m = Σ_{i: src(i)=m} log λ_m(t_i)  -  Λ_m(T)

        The log-intensity sum is taken only over events whose source is
        *m*; the compensator Λ_m integrates λ_m over [0, T]. Summing over
        all m gives the full multivariate-Hawkes log-likelihood
        Σ_n log λ_{k_n}(t_n) − Σ_m Λ_m(T).
        """
        ev_m = bt.asarray(events[m], dtype=float)
        n_m = int(ev_m.size) if hasattr(ev_m, "size") else len(ev_m)

        if n_m == 0:
            sum_log = 0.0
        else:
            lam_m = bt.full(n_m, float(self.mu[m]))
            for k in range(self.n_dims):
                ev_k = bt.asarray(events[k], dtype=float)
                if ev_k.size == 0:
                    continue
                lags = ev_m[:, None] - ev_k[None, :]  # (n_m, n_k)
                causal = lags > 0
                phi = bt.where(causal, self.kernel_matrix[m][k].evaluate(lags), 0.0)
                lam_m = lam_m + bt.sum(phi, axis=1)
            sum_log = float(bt.sum(bt.log(lam_m)))

        # Compensator: μ_m T + Σ_k Σ_{t_i^k} ∫_0^{T-t_i^k} φ_{mk}(s) ds
        comp = float(self.mu[m]) * T
        for k in range(self.n_dims):
            ev_k = bt.asarray(events[k], dtype=float)
            if ev_k.size == 0:
                continue
            tails = T - ev_k
            comp += float(bt.sum(self.kernel_matrix[m][k].integrate_vec(tails)))
        return float(sum_log - comp)

    def get_params(self) -> dict:
        mu = self.mu
        kernel_matrix = self.kernel_matrix
        return {"mu": mu, "kernel_matrix": kernel_matrix}

    def set_params(self, params: dict) -> None:
        if "mu" in params:
            self.mu = bt.asarray(params["mu"])
        if "kernel_matrix" in params:
            self.kernel_matrix = params["kernel_matrix"]

    def project_params(self) -> None:
        """Project parameters to ensure stationarity (row L1 norms < 1).

        Row-wise L1 norm < 1 is a Gershgorin-sufficient condition for the
        spectral radius of the L1-norm matrix to be < 1.
        """
        for m in range(self.n_dims):
            total_norm = 0.0
            for k in range(self.n_dims):
                total_norm += self.kernel_matrix[m][k].l1_norm()
            if total_norm >= 1.0:
                scale = 0.99 / total_norm
                failed: list[str] = []
                for k in range(self.n_dims):
                    kern = self.kernel_matrix[m][k]
                    try:
                        kern.scale(scale)
                    except NotImplementedError:
                        failed.append(type(kern).__name__)
                if failed:
                    warnings.warn(
                        f"Projection partial for dim {m}: "
                        f"{set(failed)} do not support scale().",
                        UserWarning,
                    )
                else:
                    warnings.warn(
                        f"Projected dim {m} kernel row L1 norm from "
                        f"{total_norm:.4f} to 0.99.",
                        UserWarning,
                    )