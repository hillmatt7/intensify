"""Marked Hawkes process: mark-weighted self-excitation."""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np

from ...core.base import PointProcessBase
from ...core.inference import FitResult, get_inference_engine
from ...core.kernels import ExponentialKernel, Kernel



def _default_mark_distribution(rng: np.random.Generator, n: int) -> np.ndarray:
    """IID Exponential(1) marks for simulation."""
    return rng.exponential(1.0, size=n)


class MarkedHawkes(PointProcessBase):
    r"""
    Marked Hawkes process

    .. math::

        \lambda^*(t) = \mu + \sum_{t_i < t} g(m_i)\,\phi(t - t_i)

    Event times :math:`t_i` are observed together with scalar marks :math:`m_i \ge 0`.
    """

    def __init__(
        self,
        mu: float,
        kernel: Kernel,
        mark_influence: str | Callable[[float], float] = "linear",
        *,
        mark_power: float = 1.0,
        simulate_marks: Callable[[np.random.Generator, int], np.ndarray] | None = None,
    ):
        if mu < 0:
            raise ValueError("mu must be non-negative")
        self.mu = float(mu)
        self.kernel = kernel
        self.mark_power = float(mark_power)
        self._simulate_marks = simulate_marks or _default_mark_distribution
        if mark_influence == "linear":
            self._mark_influence_kind = "linear"
            self._mark_fn: Callable[[float], float] | None = None
        elif mark_influence == "log":
            self._mark_influence_kind = "log"
            self._mark_fn = None
        elif mark_influence == "power":
            self._mark_influence_kind = "power"
            self._mark_fn = None
        elif callable(mark_influence):
            self._mark_influence_kind = "callable"
            self._mark_fn = mark_influence
        else:
            raise ValueError("mark_influence must be 'linear', 'log', 'power', or callable")

    def _g(self, m: float | np.floating) -> float:
        m = float(m)
        if self._mark_influence_kind == "linear":
            return m
        if self._mark_influence_kind == "log":
            return float(np.log1p(m))
        if self._mark_influence_kind == "power":
            return float(np.maximum(m, 0.0) ** self.mark_power)
        if self._mark_fn is None:
            raise RuntimeError(
                "MarkedHawkes mark_influence='callable' requires _mark_fn to be set"
            )
        return float(self._mark_fn(m))

    def _normalize_marks(self, marks: np.ndarray) -> np.ndarray:
        marks = np.asarray(marks, dtype=float).ravel()
        if marks.size == 0:
            return marks
        sd = float(np.std(marks))
        if sd > 1e3:
            warnings.warn(
                "Marks have very large scale (std>1000); scaling for identifiability.",
                UserWarning,
            )
            marks = marks / sd
        return marks

    def simulate(self, T: float, seed: int | None = None) -> tuple[np.array, np.ndarray]:
        """Simulate via Ogata thinning; marks drawn at each accepted event."""
        rng = np.random.default_rng(seed)
        times: list[float] = []
        marks_list: list[float] = []
        t = 0.0
        hist_t: list[float] = []
        hist_m: list[float] = []
        lam_max = max(self.mu + 10.0, self.mu + 5.0 * max(float(self.kernel.l1_norm()), 0.1))

        while t < float(T):
            dt = float(rng.exponential(1.0 / lam_max))
            t += dt
            if t >= float(T):
                break
            hist_bt = np.asarray(hist_t, dtype=float) if hist_t else np.zeros(0)
            mhist = np.asarray(hist_m, dtype=float) if hist_m else np.zeros(0)
            lam = float(self.intensity(t, hist_bt, mhist))
            lam_max = max(lam_max, lam * 1.2 + 0.05)
            if rng.uniform(0, 1) < lam / lam_max:
                m = float(self._simulate_marks(rng, 1)[0])
                times.append(t)
                marks_list.append(m)
                hist_t.append(t)
                hist_m.append(m)
        if not times:
            return np.zeros(0), np.zeros(0)
        return np.asarray(times), np.asarray(marks_list, dtype=float)

    def intensity(self, t: float, history: np.array, marks_history: np.ndarray) -> float:
        """Conditional intensity at ``t``; ``marks_history`` aligned with ``history``."""
        history = np.asarray(history, dtype=float).ravel()
        marks_history = np.asarray(marks_history, dtype=float).ravel()
        if history.size == 0:
            return float(self.mu)
        if marks_history.size != history.size:
            raise ValueError("marks_history must match history length")
        lam = float(self.mu)
        tv = float(t)
        for t_i, m_i in zip(history.tolist(), marks_history.tolist(), strict=True):
            if float(t_i) >= tv:
                continue
            lag = tv - float(t_i)
            lam += self._g(m_i) * float(self.kernel.evaluate(np.asarray([lag]))[0])
        return lam

    def log_likelihood(self, events: np.array, marks: np.array, T: float) -> float:
        events = np.asarray(np.asarray(events), dtype=float).ravel()
        marks = np.asarray(np.asarray(marks), dtype=float).ravel()
        if events.size != marks.size:
            raise ValueError("events and marks must have the same length")
        if events.size == 0:
            return 0.0
        if isinstance(self.kernel, ExponentialKernel) and not self.kernel.allow_signed:
            return self._loglik_exponential_recursive(events, marks, float(T))
        return self._loglik_general(events, marks, float(T))

    def _loglik_general(self, events: np.ndarray, marks: np.ndarray, T: float) -> float:
        n = len(events)
        ll = 0.0
        for i in range(n):
            lam = float(self.mu)
            for j in range(i):
                lag = events[i] - events[j]
                if lag > 0:
                    lam += self._g(marks[j]) * float(self.kernel.evaluate(np.asarray([lag]))[0])
            ll += float(np.log(max(lam, 1e-300)))
        comp = float(self.mu) * T
        for j in range(n):
            comp += self._g(marks[j]) * float(self.kernel.integrate(float(T - events[j])))
        return float(ll - comp)

    def _loglik_exponential_recursive(self, events: np.ndarray, marks: np.ndarray, T: float) -> float:
        mu = float(self.mu)
        alpha = float(self.kernel.alpha)
        beta = float(self.kernel.beta)
        R = 0.0
        t_prev = 0.0
        ll = 0.0
        for i, t_i in enumerate(events):
            dt = float(t_i - t_prev)
            R = float(np.exp(-beta * dt)) * R
            lam = mu + alpha * beta * R
            ll += float(np.log(max(lam, 1e-300)))
            R += self._g(marks[i])
            t_prev = float(t_i)
        comp = mu * T
        for j in range(len(events)):
            comp += self._g(marks[j]) * float(self.kernel.integrate(float(T - events[j])))
        return float(ll - comp)

    def get_params(self) -> dict:
        return {"mu": self.mu, "kernel": self.kernel, "mark_power": self.mark_power}

    def set_params(self, params: dict) -> None:
        if "mu" in params:
            self.mu = float(params["mu"])
        if "kernel" in params:
            self.kernel = params["kernel"]
        if "mark_power" in params:
            self.mark_power = float(params["mark_power"])

    def project_params(self) -> None:
        if self.kernel.l1_norm() >= 1.0 and not getattr(self.kernel, "allow_signed", False):
            warnings.warn(
                "Kernel L1 norm >= 1; projecting alpha for stationarity.",
                UserWarning,
            )
            if hasattr(self.kernel, "alpha"):
                self.kernel.alpha = min(0.99, float(self.kernel.alpha))

    def fit(
        self,
        events,
        marks=None,
        T: float | None = None,
        method: str = "mle",
    ) -> FitResult:
        """Fit marks and times.

        Accepts either ``model.fit(events, marks, T=T)`` or
        ``model.fit((events, marks), T=T)`` for convenience.
        """
        if isinstance(events, tuple) and marks is None:
            events, marks = events
        if marks is None:
            raise TypeError(
                "MarkedHawkes.fit() requires marks. "
                "Use model.fit(events, marks, T=T) or model.fit((events, marks), T=T)"
            )
        events_np = np.asarray(np.asarray(events), dtype=float).ravel()
        marks_np = self._normalize_marks(np.asarray(np.asarray(marks), dtype=float).ravel())
        if events_np.size != marks_np.size:
            raise ValueError("events and marks must have the same length")
        if T is None:
            warnings.warn(
                "T not specified; inferring T = max(events).",
                UserWarning,
            )
            T = float(events_np.max()) if events_np.size else 0.0
        T = float(T)
        engine = get_inference_engine(method)
        return engine.fit(self, (events_np, marks_np), T)
