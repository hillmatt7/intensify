"""Abstract base classes for point process models."""

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np




@runtime_checkable
class PointProcess(Protocol):
    """Protocol defining the interface for point process models."""

    @abstractmethod
    def simulate(self, T: float, seed: int = None) -> np.array:
        """Generate event times on interval [0, T].

        Parameters
        ----------
        T : float
            End of observation window.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        events : jnp.ndarray or np.ndarray
            Sorted array of event timestamps in [0, T].
        """
        pass

    @abstractmethod
    def intensity(self, t: float, history: np.array) -> float:
        """Evaluate conditional intensity function λ(t | history).

        Parameters
        ----------
        t : float
            Time at which to evaluate intensity.
        history : jnp.ndarray or np.ndarray
            Past event times before t.

        Returns
        -------
        intensity : float
            Conditional intensity value at time t.
        """
        pass

    @abstractmethod
    def log_likelihood(self, events: np.array, T: float) -> float:
        """Compute log-likelihood of observed event sequence.

        Parameters
        ----------
        events : jnp.ndarray or np.ndarray
            Event timestamps on [0, T].
        T : float
            End of observation window.

        Returns
        -------
        ll : float
            Log-likelihood value.
        """
        pass

    def fit(self, events, T: float = None, method: str = "mle", **kwargs):
        """Fit process parameters to observed event data.

        Parameters
        ----------
        events : array-like or domain-specific data object
            Event timestamps. May also accept domain objects (SpikeTrainData, OrderBookStream).
        T : float, optional
            Observation window end time. Inferred from events if not provided.
        method : str
            Inference method: 'mle', 'em', 'bayesian' (bayesian not yet implemented).

        Returns
        -------
        result : FitResult
            Standardized container with fitted parameters and diagnostics.
        """
        from .inference import get_inference_engine

        # Infer T if not provided
        if T is None:
            import warnings
            warnings.warn(
                "T not specified; inferring T = max(events). "
                "This may be incorrect if the observation window extends past the last event.",
                UserWarning,
            )
            events_array = np.asarray(events)
            T = float(events_array.max())

        engine = get_inference_engine(method)
        return engine.fit(self, events, T, **kwargs)

    def get_params(self) -> dict:
        """Return model parameters as a dict (for optimization)."""
        raise NotImplementedError

    def set_params(self, params: dict) -> None:
        """Set model parameters from a dict."""
        raise NotImplementedError

    def project_params(self) -> None:
        """Project parameters onto feasible set (e.g., enforce stationarity)."""
        pass  # default: nothing to project


class PointProcessBase(ABC, PointProcess):
    """Convenience base class that implements get/set_params stubs and other helpers."""

    def get_params(self) -> dict:
        raise NotImplementedError

    def set_params(self, params: dict) -> None:
        raise NotImplementedError