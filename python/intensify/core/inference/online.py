"""Online (streaming) approximate inference for recursive Hawkes kernels."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from . import FitResult, InferenceEngine, compute_information_criteria
from .univariate_hawkes_mle_params import (
    hawkes_mle_apply_vector,
    hawkes_mle_bounds,
    hawkes_mle_initial_vector,
)


class OnlineInference(InferenceEngine):
    """
    Sliding-window stochastic gradient updates on recursive-form Hawkes log-likelihood.

    Each ``update`` step performs one SGD step using the most recent inter-arrival
    contribution. Intended for univariate Hawkes with ``has_recursive_form()`` kernels.
    """

    def __init__(
        self,
        lr: float = 0.01,
        window: int = 10_000,
        forgetting_factor: float = 0.999,
        min_events: int = 20,
    ):
        self.lr = float(lr)
        self.window = int(window)
        self.forgetting_factor = float(forgetting_factor)
        self.min_events = int(min_events)
        self._times: deque[float] = deque(maxlen=max(window, 16))
        self._n_updates = 0
        self._x: np.ndarray | None = None

    def reset(self) -> None:
        self._times.clear()
        self._n_updates = 0
        self._x = None

    def update(self, process: Any, event_time: float) -> None:
        """Ingest one event time (monotone increasing recommended)."""
        from ..processes.hawkes import UnivariateHawkes

        if not isinstance(process, UnivariateHawkes):
            raise TypeError("OnlineInference.update supports UnivariateHawkes only.")
        if not process.kernel.has_recursive_form():
            raise ValueError(
                "Kernel must declare has_recursive_form() for online inference."
            )
        t = float(event_time)
        self._times.append(t)
        if len(self._times) < 2:
            return
        ev = np.array(list(self._times), dtype=float)
        T = float(ev[-1])
        if len(ev) < self.min_events:
            self._n_updates += 1
            return

        if self._x is None:
            self._x = hawkes_mle_initial_vector(process)
        x = self._x
        bounds = hawkes_mle_bounds(process)
        lo = np.array([b[0] if b[0] is not None else -np.inf for b in bounds])
        hi = np.array([b[1] if b[1] is not None else np.inf for b in bounds])

        def nll(v: np.ndarray) -> float:
            hawkes_mle_apply_vector(process, v)
            return -float(process.log_likelihood(np.asarray(ev), T))

        grad = np.zeros_like(x)
        eps = 1e-5
        f0 = nll(x)
        for i in range(len(x)):
            xp = x.copy()
            xp[i] += eps
            grad[i] = (nll(xp) - f0) / eps
        x_new = x - self.lr * grad * (self.forgetting_factor**self._n_updates)
        x_new = np.clip(x_new, lo, hi)
        self._x = x_new
        hawkes_mle_apply_vector(process, x_new)
        self._n_updates += 1

    def fit(self, process: Any, events: Any, T: float, **kwargs: Any) -> FitResult:
        """Replay a batch through ``update`` and return a :class:`FitResult` snapshot."""
        from ..processes.hawkes import UnivariateHawkes

        if not isinstance(process, UnivariateHawkes):
            raise TypeError("OnlineInference.fit supports UnivariateHawkes only.")
        self.reset()
        ev = np.asarray(events, dtype=float).ravel()
        for t in ev:
            self.update(process, float(t))
        T = float(T) if T is not None else float(ev.max()) if len(ev) else 0.0
        ll = float(process.log_likelihood(np.asarray(ev), T))
        params = process.get_params()
        aic, bic = compute_information_criteria(ll, params, len(ev))
        warm = len(ev) < self.min_events
        result = FitResult(
            params=params,
            log_likelihood=ll,
            aic=aic,
            bic=bic,
            std_errors=None,
            convergence_info={
                "method": "online_sgd",
                "n_updates": self._n_updates,
                "warm_up": warm,
            },
        )
        result.process = process
        result.events = ev
        result.T = T
        result.branching_ratio_ = process.kernel.l1_norm()
        return result

    def current_params(self, process: Any) -> dict:
        """Return parameters from ``process`` after streaming updates."""
        return process.get_params()


from . import register_engine

register_engine("online", OnlineInference())
