"""Nonlinear (link) Hawkes: nonnegative intensity from signed pre-intensity."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np

from ...core.base import PointProcessBase
from ...core.kernels import Kernel
from ..simulation.thinning import ogata_thinning_multivariate



def _softplus(x: float) -> float:
    if x > 35:
        return float(x)
    return float(np.log1p(np.exp(x)))


def _relu(x: float) -> float:
    return float(max(0.0, x))


def _sigmoid_scaled(x: float, scale: float = 5.0) -> float:
    return float(scale / (1.0 + np.exp(-np.clip(x, -40.0, 40.0))))


def _identity_pos(x: float) -> float:
    return float(max(x, 1e-12))


def _make_link_fn(kind: str, sigmoid_scale: float) -> Callable[[float], float]:
    if kind == "softplus":
        return _softplus
    if kind == "relu":
        return _relu
    if kind == "sigmoid":
        return partial(_sigmoid_scaled, scale=sigmoid_scale)
    if kind == "identity":
        return _identity_pos
    raise ValueError(kind)


class NonlinearHawkes(PointProcessBase):
    r"""
    Hawkes process with a link on the pre-intensity:

    .. math::

        \lambda^*(t) = f\left(\mu + \sum_{t_i<t}\phi(t-t_i)\right)

    where :math:`f` is ``softplus``, ``relu``, ``sigmoid``, or a custom nonnegative callable.
    Use ``ExponentialKernel(..., allow_signed=True)`` for inhibition in the pre-sum.
    """

    def __init__(
        self,
        mu: float,
        kernel: Kernel,
        link_function: str | Callable[[float], float] = "softplus",
        *,
        sigmoid_scale: float = 5.0,
    ):
        self.mu = float(mu)
        self.kernel = kernel
        self.sigmoid_scale = float(sigmoid_scale)
        if link_function == "softplus":
            self._link = _softplus
        elif link_function == "relu":
            self._link = _relu
        elif link_function == "sigmoid":
            self._link = partial(_sigmoid_scaled, scale=self.sigmoid_scale)
        elif link_function == "identity":
            self._link = _identity_pos
        elif callable(link_function):
            self._link = link_function
        else:
            raise ValueError(
                "link_function must be 'softplus', 'relu', 'sigmoid', 'identity', or callable"
            )

    def _pre_intensity(self, t: float, events: np.ndarray) -> float:
        ev = np.asarray(events, dtype=float).ravel()
        if ev.size == 0:
            return float(self.mu)
        tv = float(t)
        lags = tv - ev
        causal = lags > 0
        if not np.any(causal):
            return float(self.mu)
        kernel_vals = self.kernel.evaluate(np.asarray(lags[causal]))
        return float(self.mu) + float(np.sum(np.asarray(kernel_vals)))

    def intensity(self, t: float, history: np.array) -> float:
        ev = np.asarray(np.asarray(history), dtype=float).ravel()
        return float(self._link(self._pre_intensity(float(t), ev)))

    def log_likelihood(self, events: np.array, T: float, *, n_quad: int = 512) -> float:
        """Likelihood with compensator approximated by trapezoid rule on ``n_quad`` points."""
        events = np.asarray(np.asarray(events), dtype=float).ravel()
        if len(events) == 0:
            return float(-self._compensator_numerical(events, float(T), n_quad))
        n = len(events)
        # Vectorized lag matrix
        lags = events[:, None] - events[None, :]  # (n, n)
        causal = lags > 0
        kernel_vals = np.where(
            causal,
            np.asarray(self.kernel.evaluate(np.asarray(lags))),
            0.0,
        )
        pre_arr = float(self.mu) + np.sum(kernel_vals, axis=1)
        ll = 0.0
        for i in range(n):
            lam = float(self._link(float(pre_arr[i])))
            ll += float(np.log(max(lam, 1e-300)))
        comp = self._compensator_numerical(events, float(T), n_quad)
        return float(ll - comp)

    def _compensator_numerical(self, events: np.ndarray, T: float, n_quad: int) -> float:
        if T <= 0:
            return 0.0
        grid = np.linspace(0.0, T, max(8, int(n_quad)))
        ev = np.asarray(events, dtype=float).ravel()
        if ev.size == 0:
            vals = np.full_like(grid, self._link(float(self.mu)))
        else:
            # Vectorized pre-intensity over the quadrature grid
            # lags shape: (n_quad, N_events)
            lags = grid[:, None] - ev[None, :]
            causal = lags > 0
            kernel_vals = np.where(
                causal,
                np.asarray(self.kernel.evaluate(np.asarray(lags))),
                0.0,
            )
            pre = float(self.mu) + np.sum(kernel_vals, axis=1)
            vals = np.array([self._link(float(p)) for p in pre], dtype=float)
        return float(np.trapezoid(vals, grid))

    def simulate(self, T: float, seed: int | None = None) -> np.array:
        """Ogata thinning with adaptive upper bound on linked intensity."""
        rng = np.random.default_rng(seed)
        times: list[float] = []
        t = 0.0
        hist: list[float] = []
        lam_max = max(self.mu + 10.0, 20.0)
        while t < float(T):
            dt = float(rng.exponential(1.0 / lam_max))
            t += dt
            if t >= float(T):
                break
            h = np.asarray(hist, dtype=float) if hist else np.zeros(0)
            lam = float(self.intensity(t, h))
            lam_max = max(lam_max, lam * 1.3 + 0.05)
            if rng.uniform(0, 1) < lam / lam_max:
                times.append(t)
                hist.append(t)
        return np.asarray(times) if times else np.zeros(0)

    def get_params(self) -> dict:
        return {"mu": self.mu, "kernel": self.kernel, "sigmoid_scale": self.sigmoid_scale}

    def set_params(self, params: dict) -> None:
        if "mu" in params:
            self.mu = float(params["mu"])
        if "kernel" in params:
            self.kernel = params["kernel"]
        if "sigmoid_scale" in params:
            self.sigmoid_scale = float(params["sigmoid_scale"])

    def project_params(self) -> None:
        pass


class MultivariateNonlinearHawkes(PointProcessBase):
    r"""
    Multivariate Hawkes with per-dimension link on pre-intensity:

    .. math::

        \lambda^*_m(t) = f_m\left(\mu_m + \sum_k \sum_{t_i^k < t} \phi_{mk}(t-t_i^k)\right)
    """

    def __init__(
        self,
        n_dims: int,
        mu: float | np.array,
        kernel: Kernel | list[list[Kernel]],
        link_function: str | list[str] = "softplus",
        *,
        sigmoid_scale: float = 5.0,
    ):
        from .hawkes import MultivariateHawkes

        self._mv = MultivariateHawkes(n_dims, mu, kernel)
        self.sigmoid_scale = float(sigmoid_scale)
        self.n_dims = int(self._mv.n_dims)
        self.mu = self._mv.mu
        self.kernel_matrix = self._mv.kernel_matrix
        kinds = [link_function] * self.n_dims if isinstance(link_function, str) else list(link_function)
        if len(kinds) != self.n_dims:
            raise ValueError("link_function list must have length n_dims")
        self._link_kinds = kinds
        self._links = [_make_link_fn(k, self.sigmoid_scale) for k in kinds]

    def _pre_intensity_dim(self, m: int, t: float, history: list[np.array]) -> float:
        mu_m = float(np.asarray(self.mu, dtype=float).ravel()[m])
        lam = mu_m
        for k in range(self.n_dims):
            hist_k = np.asarray(history[k], dtype=float).ravel()
            for t_i in hist_k:
                if float(t_i) >= float(t):
                    continue
                lag = float(t) - float(t_i)
                lam += float(self.kernel_matrix[m][k].evaluate(np.asarray([lag]))[0])
        return lam

    def intensity(self, t: float, history: list[np.array]) -> np.array:
        out = [self._links[m](self._pre_intensity_dim(m, float(t), history)) for m in range(self.n_dims)]
        return np.asarray(out, dtype=float)

    def simulate(self, T: float, seed: int | None = None) -> list[np.array]:
        return ogata_thinning_multivariate(self, T, seed=seed)

    def log_likelihood(self, events: list[np.array], T: float, *, n_quad: int = 128) -> float:
        evs = [np.asarray(np.asarray(e), dtype=float).ravel() for e in events]
        T = float(T)
        all_ev: list[tuple[float, int]] = []
        for k in range(self.n_dims):
            for t_i in evs[k]:
                all_ev.append((float(t_i), k))
        all_ev.sort(key=lambda x: x[0])
        hists: list[list[float]] = [[] for _ in range(self.n_dims)]
        ll = 0.0
        for t_i, src in all_ev:
            hist_bt = [np.asarray(hists[k]) if hists[k] else np.zeros(0) for k in range(self.n_dims)]
            pre = self._pre_intensity_dim(src, t_i, hist_bt)
            lam = float(self._links[src](pre))
            ll += float(np.log(max(lam, 1e-300)))
            hists[src].append(t_i)
        comp = 0.0
        for m in range(self.n_dims):
            comp += self._compensator_dim(m, evs, T, n_quad)
        return float(ll - comp)

    def _compensator_dim(self, m: int, evs: list[np.ndarray], T: float, n_quad: int) -> float:
        t = np.linspace(0.0, T, max(8, int(n_quad)))
        vals = []
        for ti in t:
            hist_bt = [
                np.asarray(evs[k][evs[k] < float(ti)]) if np.any(evs[k] < float(ti)) else np.zeros(0)
                for k in range(self.n_dims)
            ]
            pre = self._pre_intensity_dim(m, float(ti), hist_bt)
            vals.append(self._links[m](pre))
        return float(np.trapezoid(np.asarray(vals, dtype=float), t))

    def get_params(self) -> dict:
        return {
            "mu": self.mu,
            "kernel_matrix": self.kernel_matrix,
            "link_kinds": list(self._link_kinds),
        }

    def set_params(self, params: dict) -> None:
        if "mu" in params:
            self.mu = np.asarray(params["mu"])
            self._mv.mu = self.mu
        if "kernel_matrix" in params:
            self.kernel_matrix = params["kernel_matrix"]
            self._mv.kernel_matrix = self.kernel_matrix

    def project_params(self) -> None:
        self._mv.project_params()
