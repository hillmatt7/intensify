"""Abstract base class for Hawkes excitation kernels."""

from abc import ABC, abstractmethod

import numpy as np




class Kernel(ABC):
    """
    Abstract base class for Hawkes excitation kernels.

    All kernels must be:
    - Non-negative: phi(t) >= 0 for all t >= 0
    - Causal: phi(t) = 0 for t < 0
    - Backend-agnostic (JAX/NumPy) and differentiable (JAX) where applicable

    Computation path is selected automatically based on has_recursive_form():
    - True  → O(N) recursive likelihood via jax.lax.scan (or equivalent)
    - False → O(N²) general likelihood via pairwise evaluation

    Kernels with recursive form (ExponentialKernel, SumExponentialKernel,
    ApproxPowerLawKernel) override has_recursive_form() and
    recursive_state_update(). All other kernels use the general path
    automatically — no changes required when adding new kernels.
    """

    # Abstract: must implement by all subclasses
    @abstractmethod
    def evaluate(self, t: np.array) -> np.array:
        """Evaluate kernel at time lags t (can be vectorized).

        Parameters
        ----------
        t : jnp.ndarray or np.ndarray
            Array of time lags (t >= 0).

        Returns
        -------
        phi_t : jnp.ndarray or np.ndarray
            Kernel values at given lags, same shape as t.
        """
        pass

    @abstractmethod
    def integrate(self, t: float) -> float:
        """Compute integral of kernel from 0 to t (compensator term).

        Parameters
        ----------
        t : float
            Upper integration limit (t >= 0).

        Returns
        -------
        integral : float
            ∫_0^t phi(τ) dτ.
        """
        pass

    @abstractmethod
    def l1_norm(self) -> float:
        """Compute integral from 0 to infinity (total excitation mass).

        Returns
        -------
        norm : float
            ∫_0^∞ phi(t) dt. Must be finite. For stationarity in Hawkes,
            this should be < 1 for background rate μ > 0 (branching ratio < 1).
        """
        pass

    def integrate_vec(self, t: np.array) -> np.array:
        """Vectorized integral: apply integrate() to each element of *t*.

        Subclasses with closed-form integrals should override this with a
        pure-backend implementation for JIT traceability.
        """
        t = np.asarray(t)
        return np.asarray([self.integrate(float(ti)) for ti in t.ravel()]).reshape(t.shape)

    @property
    def jit_compatible(self) -> bool:
        """Whether this kernel's evaluate/integrate_vec are JAX-JIT traceable."""
        return True

    # Default (can override by subclasses)
    def is_stationary(self) -> bool:
        """Check if kernel's L1 norm is less than 1 (stationarity condition)."""
        return self.l1_norm() < 1.0

    def scale(self, factor: float) -> None:
        """Multiply the kernel's L1 norm by *factor* in place.

        Default implementation scales ``self.alpha`` if present. Kernels whose
        mass lives in other attributes (``alphas`` list, ``values`` array)
        must override this method.
        """
        if hasattr(self, "alpha"):
            self.alpha = float(self.alpha) * float(factor)
            return
        raise NotImplementedError(
            f"{self.__class__.__name__} must override scale() — L1 mass does "
            "not live in a single 'alpha' attribute."
        )

    # --- Recursive dispatch interface ---
    # Default: general O(N²) path. Fast kernels override both methods below.

    def has_recursive_form(self) -> bool:
        """
        Return True if this kernel admits O(N) recursive likelihood computation.
        Only exponential-family kernels can return True.
        Default is False — safe for any new kernel added to the library.
        """
        return False

    def recursive_state_update(self, state: np.array, dt: float) -> np.array:
        """
        Update the recursive sufficient statistic R given time elapsed dt
        since the last event. Used by the recursive likelihood computation
        when has_recursive_form() is True.

        Parameters
        ----------
        state : jnp.ndarray or np.ndarray
            Previous sufficient statistic R_{i-1}.
        dt : float
            Time elapsed since last event (t_i - t_{i-1}).

        Returns
        -------
        new_state : jnp.ndarray or np.ndarray
            Updated sufficient statistic R_i.

        Notes
        -----
        For ExponentialKernel (scalar):
            R_i = exp(-β * dt) * (1 + R_{i-1})

        For SumExponentialKernel (per component):
            R_i^k = exp(-β_k * dt) * (1 + R_{i-1}^k)

        Raises
        ------
        NotImplementedError
            If called when has_recursive_form() is False.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} declared has_recursive_form()=True "
            f"but did not implement recursive_state_update()."
        )

    def recursive_init_state(self) -> np.array:
        """
        Initial recursive state R before the first event (typically zero scalar).
        Vector-valued for multi-component exponential kernels.
        """
        if not self.has_recursive_form():
            raise NotImplementedError(
                f"{self.__class__.__name__} has no recursive form for init state."
            )
        return np.asarray(0.0)

    def recursive_intensity_excitation(self, state: np.array) -> np.array:
        """
        Excitation from past events (λ_exc with λ = μ + λ_exc) **before** advancing
        state by dt to the next event time. Used by O(N) likelihood scans.
        """
        if not self.has_recursive_form():
            raise NotImplementedError(
                f"{self.__class__.__name__} has no recursive intensity excitation."
            )
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement recursive_intensity_excitation()."
        )

    def recursive_decay(self, state: np.array, dt: float) -> np.array:
        """
        Decay the recursive state by elapsed time *dt* without absorbing a new event.

        For ExponentialKernel: ``exp(-beta * dt) * state``.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement recursive_decay()."
        )

    def recursive_absorb(self, state: np.array) -> np.array:
        """
        Absorb a new event at lag 0 into the recursive state.

        For ExponentialKernel: ``state + 1.0``.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement recursive_absorb()."
        )