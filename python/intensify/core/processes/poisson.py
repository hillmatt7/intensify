"""Poisson point process models."""

import warnings
from collections.abc import Callable

import numpy as np

from ...core.base import PointProcessBase
from ...core.inference import FitResult



class HomogeneousPoisson(PointProcessBase):
    """
    Homogeneous Poisson process with constant rate λ.

    The intensity is λ(t) = λ everywhere.

    Parameters
    ----------
    rate : float, optional
        Event rate λ. If not provided, it is estimated from data on fit.
    """

    def __init__(self, rate: float | None = None):
        if rate is not None:
            if rate <= 0:
                raise ValueError("rate must be positive")
            self.rate = float(rate)
        else:
            self.rate = None

    def simulate(self, T: float, seed: int = None) -> np.array:
        """
        Simulate events on [0, T] using exponential inter-arrivals.

        Parameters
        ----------
        T : float
            End of observation window.
        seed : int, optional
            Random seed.

        Returns
        -------
        events : jnp.ndarray or np.ndarray
            Sorted event timestamps in [0, T].
        """

        events: list[float] = []
        t = 0.0
        rate = self.rate if self.rate is not None else 1.0

        rng = np.random.default_rng(seed)
        while t < T:
            dt = float(rng.exponential(1.0 / rate))
            t += dt
            if t <= T:
                events.append(t)
        return np.asarray(events) if events else np.zeros(0)

    def intensity(self, t: float, history: np.array) -> float:
        """
        Constant intensity λ(t) = λ.

        Parameters
        ----------
        t : float
            Time (unused).
        history : array
            Past events (unused).

        Returns
        -------
        lambda : float
            Current intensity (constant).
        """
        if self.rate is None:
            raise ValueError("rate must be set before computing intensity")
        return float(self.rate)

    def log_likelihood(self, events: np.array, T: float) -> float:
        """
        Log-likelihood: sum log(λ) - λ T.

        Parameters
        ----------
        events : array
            Event timestamps.
        T : float
            Observation window end.

        Returns
        -------
        ll : float
            Log-likelihood value.
        """
        if self.rate is None:
            raise ValueError("rate must be set before computing log-likelihood")
        n = len(events)
        return n * np.log(self.rate) - self.rate * T

    def get_params(self) -> dict:
        """Return parameters as dict."""
        return {"rate": self.rate} if self.rate is not None else {}

    def set_params(self, params: dict) -> None:
        """Set parameters from dict."""
        if "rate" in params:
            self.rate = float(params["rate"])

    def fit(self, events, T: float | None = None) -> FitResult:
        """
        Fit homogeneous Poisson rate via MLE: λ̂ = N(T) / T.

        Override base .fit() because this is analytic, no engine needed.

        Parameters
        ----------
        events : array
            Event timestamps.
        T : float, optional
            End of observation window. Inferred from events if not given.

        Returns
        -------
        result : FitResult
        """
        events = np.asarray(events)
        if T is None:
            T = float(events.max())
        n = len(events)
        rate_hat = float(n) / T
        self.rate = rate_hat
        ll = self.log_likelihood(events, T)
        # AIC/BIC with n = number of events
        aic = 2 * 1 - 2 * ll  # 1 parameter
        bic = 1 * np.log(n) - 2 * ll
        result = FitResult(
            params={"rate": rate_hat},
            log_likelihood=ll,
            aic=aic,
            bic=bic,
            std_errors={"rate": np.sqrt(rate_hat / n) if n > 0 else float("inf")},
            convergence_info={"method": "analytic MLE"},
        )
        result.process = self
        result.events = events
        result.T = T
        return result


class InhomogeneousPoisson(PointProcessBase):
    """
    Inhomogeneous Poisson process with time-varying intensity λ(t).

    Supports either a piecewise-constant intensity or a callable function.

    Parameters
    ----------
    intensity_func : callable, optional
        Function λ(t) that returns intensity at time t (0 <= t <= T).
        Must be non-negative.
    rates : dict, optional
        Mapping {interval_start: rate} for piecewise-constant segments.
        Intervals are [t_i, t_{i+1}) with final segment to T. Must cover [0, T].
        Example: {0.0: 1.0, 2.5: 3.0} means rate 1.0 from 0 to 2.5, then 3.0 from 2.5 onward.
    """

    def __init__(
        self,
        intensity_func: Callable[[float], float] | None = None,
        rates: dict | None = None,
    ):
        if intensity_func is None and rates is None:
            raise ValueError("Must provide either intensity_func or rates")
        if intensity_func is not None and rates is not None:
            raise ValueError("Provide only one of intensity_func or rates")
        self.intensity_func = intensity_func
        self.rates = rates  # Dict[float, float]
        self._validate()

    def _validate(self):
        if self.rates is not None:
            # Ensure sorted keys and non-negative rates
            keys = sorted(self.rates.keys())
            if keys[0] != 0.0:
                raise ValueError("rates must start with 0.0")
            for k in keys:
                if self.rates[k] < 0:
                    raise ValueError("rates must be non-negative")

    def simulate(self, T: float, seed: int = None) -> np.array:
        """
        Simulate using Ogata thinning (approximate for arbitrary λ(t)).

        Requires an upper bound on intensity.

        Parameters
        ----------
        T : float
            Observation window end.
        seed : int, optional
            Random seed.

        Returns
        -------
        events : jnp.ndarray or np.ndarray
            Sorted event timestamps.
        """
        if self.intensity_func is not None:
            # For piecewise constant, we can compute exact max; for general func, we approximate
            warnings.warn(
                "Simulation for arbitrary intensity function uses adaptive thinning. "
                "Consider piecewise-constant rates for efficiency.",
                UserWarning,
            )
            return self._simulate_thinning(T, seed)
        else:
            # piecewise-constant: we can compute max rate
            max_rate = max(self.rates.values())
            return self._simulate_thinning(T, seed, lambda_max=max_rate)

    def _simulate_thinning(self, T: float, seed: int = None, lambda_max: float | None = None) -> np.array:
        """Ogata thinning algorithm."""

        events: list[float] = []
        t = 0.0

        if lambda_max is None:
            times = np.linspace(0, T, 1000)
            lam_vals = np.asarray(
                [self.intensity_func(float(ti)) for ti in times]  # type: ignore[misc]
            )
            lambda_max = float(lam_vals.max()) * 1.1
        if lambda_max <= 0:
            lambda_max = 1.0

        rng = np.random.default_rng(seed)
        while t < T:
            dt = float(rng.exponential(1.0 / lambda_max))
            t += dt
            if t >= T:
                break
            lam_t = (
                float(self.intensity_func(t))  # type: ignore[misc]
                if self.intensity_func
                else self._piecewise_intensity(t)
            )
            if rng.uniform(0.0, 1.0) < lam_t / lambda_max:
                events.append(t)
        return np.asarray(events) if events else np.zeros(0)

    def _piecewise_intensity(self, t: float) -> float:
        """Lookup rate from piecewise table."""
        if self.rates is None:
            raise ValueError("rates not set")
        sorted_starts = sorted(self.rates.keys())
        rate = None
        for start in reversed(sorted_starts):
            if t >= start:
                rate = self.rates[start]
                break
        if rate is None:
            raise ValueError(f"Could not determine intensity at t={t}")
        return rate

    def intensity(self, t: float, history: np.array) -> float:
        """Return λ(t)."""
        if self.intensity_func is not None:
            return float(self.intensity_func(t))
        else:
            return float(self._piecewise_intensity(t))

    def log_likelihood(self, events: np.array, T: float) -> float:
        """
        Log-likelihood for inhomogeneous Poisson: ∑ log λ(t_i) - ∫_0^T λ(t) dt.

        Parameters
        ----------
        events : array
            Event timestamps.
        T : float
            Observation window end.

        Returns
        -------
        ll : float
        """
        if self.intensity_func is not None:
            # Sum log λ(t_i)
            sum_log = np.sum(np.log(np.asarray([self.intensity_func(float(t)) for t in events])))
            # Integral via numerical quadrature (simple Riemann sum)
            n_grid = 10000
            times = np.linspace(0, T, n_grid)
            dts = T / n_grid
            integral = np.sum(np.asarray([self.intensity_func(float(t)) for t in times])) * dts
            return float(sum_log - integral)
        else:
            # Piecewise: sum over events plus exact integral per piece
            sum_log = 0.0
            for t in events:
                sum_log += np.log(self._piecewise_intensity(float(t)))
            # Integral: sum over intervals not fully covered? Compute total integral
            integral = 0.0
            starts = sorted(self.rates.keys())
            for i, start in enumerate(starts):
                end = T if i == len(starts) - 1 else starts[i + 1]
                if start >= T:
                    break
                end_clip = min(end, T)
                rate = self.rates[start]
                integral += rate * (end_clip - start)
            return float(sum_log - integral)

    def get_params(self) -> dict:
        if self.intensity_func is not None:
            return {"intensity_func": self.intensity_func}
        else:
            return {"rates": self.rates}

    def set_params(self, params: dict) -> None:
        if "intensity_func" in params:
            self.intensity_func = params["intensity_func"]
        if "rates" in params:
            self.rates = params["rates"]
            self._validate()