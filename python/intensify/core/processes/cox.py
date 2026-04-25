"""Cox (doubly stochastic Poisson) process models."""


import numpy as np

from ...backends import get_backend
from ...core.base import PointProcessBase

bt = get_backend()


class LogGaussianCoxProcess(PointProcessBase):
    """
    Log-Gaussian Cox Process (LGCP).

    Intensity driven by an exponentiated Gaussian process:
        λ(t) = exp(GP(t)), where GP ~ Gaussian process.

    Since the GP path is infinite-dimensional, practical implementations
    discretize time into a grid and treat λ(t) as piecewise-constant with
    Gaussian log-intensity.

    This is a simplified implementation: we assume a piecewise-constant
    intensity with Gaussian-distributed log-rate in each bin.

    Parameters
    ----------
    n_bins : int
        Number of time bins for discretization.
    mu_prior : float
        Mean of the Gaussian process on log-scale.
    sigma_prior : float
        Standard deviation of the Gaussian process.
    """

    def __init__(self, n_bins: int = 100, mu_prior: float = 0.0, sigma_prior: float = 1.0):
        if n_bins < 1:
            raise ValueError("n_bins must be positive")
        self.n_bins = int(n_bins)
        self.mu_prior = float(mu_prior)
        self.sigma_prior = float(sigma_prior)
        # The latent log-intensities for each bin will be inferred via MCMC or Laplace
        self.log_lambda = None
        # Last observation window (set by simulate / log_likelihood / user)
        self.last_T = None

    def _make_bins(self, T: float):
        """Return bin edges and midpoints."""
        edges = np.linspace(0, T, self.n_bins + 1)
        mids = (edges[:-1] + edges[1:]) / 2
        return edges, mids

    def simulate(self, T: float, seed: int = None) -> bt.array:
        """
        Simulate LGCP via thinning with piecewise-constant intensity from a sample of log rates.

        Requires that self.log_lambda is set (e.g., from sampling or from a mean function).

        If not set, simulate from the prior: sample log-lambda from Gaussian, then Poisson.
        """
        npr = np.random.default_rng(seed)

        # Sample latent log rates if not already present
        if self.log_lambda is None:
            # Sample from prior: N(mu_prior, sigma_prior^2) i.i.d. per bin
            self.log_lambda = npr.normal(
                loc=self.mu_prior, scale=self.sigma_prior, size=self.n_bins
            )

        edges, _ = self._make_bins(T)
        # Convert to rates per bin
        rates = np.exp(np.asarray(self.log_lambda))  # shape (n_bins,)
        events = []
        for i in range(self.n_bins):
            rate = rates[i]
            bin_start = edges[i]
            bin_end = edges[i + 1]
            bin_length = bin_end - bin_start
            Ni = int(npr.poisson(float(rate * bin_length)))
            # Uniformly scatter within bin
            if Ni > 0:
                ti = npr.uniform(low=bin_start, high=bin_end, size=Ni)
                events.extend(sorted(ti))
        self.last_T = float(T)
        return bt.array(events) if events else bt.zeros(0)

    def intensity(self, t: float, history: bt.array) -> float:
        if self.log_lambda is None:
            raise ValueError("log_lambda not set; intensities unknown")
        if self.last_T is None:
            raise ValueError(
                "last_T unknown; call simulate(T) or set_last_window(T) before intensity()"
            )
        edges, _ = self._make_bins(self.last_T)
        t = float(t)
        if t < 0 or t > self.last_T:
            return 0.0
        for i in range(self.n_bins):
            if edges[i] <= t < edges[i + 1]:
                ll = self.log_lambda[i]
                return float(np.exp(np.asarray(ll)))
        # t == last_T falls in last bin upper edge
        ll = self.log_lambda[-1]
        return float(np.exp(np.asarray(ll)))

    def set_last_window(self, T: float) -> None:
        """Set observation window [0, T] for bin mapping (piecewise-constant intensity)."""
        self.last_T = float(T)

    def log_likelihood(self, events: bt.array, T: float) -> float:
        """
        Conditional log-likelihood given latent ``log_lambda`` (complete-data Poisson on bins).

        Uses independent Poisson counts per bin: N_i ~ Poi(λ_i Δ_i) with λ_i = exp(log_lambda_i).

        Notes
        -----
        The marginal likelihood integrating over the log-Gaussian field requires MCMC / Laplace;
        that path is not implemented.
        """
        if self.log_lambda is None:
            raise ValueError(
                "log_lambda must be set for conditional LGCP likelihood. "
                "Simulate or assign latent log-rates first."
            )
        self.last_T = float(T)
        events = np.asarray(events, dtype=float)
        edges, _ = self._make_bins(T)
        rates = np.exp(np.asarray(self.log_lambda, dtype=float))
        ll = 0.0
        for i in range(self.n_bins):
            a, b = edges[i], edges[i + 1]
            dt = b - a
            if i == self.n_bins - 1:
                mask = (events >= a) & (events <= b)
            else:
                mask = (events >= a) & (events < b)
            n = int(np.sum(mask))
            mu_bin = rates[i] * dt
            ll += n * np.log(rates[i] + 1e-300) - mu_bin
            for k in range(1, n + 1):
                ll -= np.log(k)
        return float(ll)

    def get_params(self) -> dict:
        return {
            "n_bins": self.n_bins,
            "mu_prior": self.mu_prior,
            "sigma_prior": self.sigma_prior,
            "log_lambda": self.log_lambda,
            "last_T": self.last_T,
        }

    def set_params(self, params: dict) -> None:
        for k, v in params.items():
            setattr(self, k, v)


class ShotNoiseCoxProcess(PointProcessBase):
    """
    Shot-noise Cox process (also called Poisson cluster process).

    Intensity driven by a shot noise process: λ(t) = Σ_i h(t - τ_i)
    where τ_i are a Poisson process of "shots" and h is a decay kernel (e.g., exponential).

    This is very close to a Hawkes process but with exogenous shot times. The difference
    is that in a Hawkes process, the intensity drives its own future events (endogenous).
    In a shot-noise Cox process, the shots are exogenous (Poisson with rate ν), and
    each shot generates a cluster of events according to a kernel h.

    In practice, the observed events are a Poisson process with intensity λ(t) given by the sum of shots.

    Parameters
    ----------
    shot_rate : float
        Rate of exogenous shots (τ_i).
    shot_kernel : Kernel
        Decay kernel h(t) for each shot's contribution to intensity.
    nu : float
        Expected number of offspring per shot? Actually the intensity contribution integral of shot_kernel.
    """

    def __init__(self, shot_rate: float, shot_kernel):
        if shot_rate <= 0:
            raise ValueError("shot_rate must be positive")
        self.shot_rate = float(shot_rate)
        self.shot_kernel = shot_kernel
        self.shot_times = None
        self._last_T = None

    def simulate(self, T: float, seed: int = None) -> bt.array:
        """
        Simulate shot-noise Cox process.

        Steps:
        1. Generate shot times τ_i from Poisson(rate=shot_rate) on [0, T].
        2. For each shot at τ_i, generate a cluster of daughter events.
           The number of daughters for a shot could be Poisson with mean equal to integral of shot_kernel?
           But the intensity is the sum of h(t-τ_i). Actually, cluster size distribution is typically infinite
           because each shot's contribution is continuous (infinite events). Wait: The standard shot-noise process:
           λ(t) = Σ_i h(t-τ_i), where each shot produces an infinite number of events densely? No,
           the intensity is a function; actual events are generated from a Poisson process with that intensity.
           So the end result is still a Poisson process with that intensity. So we can just compute λ(t) and thin.
           However, that defeats the purpose of having cluster representation.
        Alternative view: The Cox process is defined by intensity λ(t) = Σ_i h(t-τ_i). Given shot times τ_i, the process is an inhomogeneous Poisson with that λ(t). We can simulate by thinning using that intensity.
        """
        if bt.random is None:
            raise RuntimeError("Random backend not available")
        npr = np.random.default_rng(seed)
        n_shots = int(npr.poisson(float(self.shot_rate * T)))
        shot_times = npr.uniform(0.0, T, size=n_shots) if n_shots > 0 else np.array([])
        self.shot_times = np.sort(np.asarray(shot_times, dtype=float))
        self._last_T = float(T)
        # Now we need to simulate a Poisson process with intensity λ(t) = Σ_{i} h(t-τ_i)
        # Use Ogata thinning with that intensity function. We'll need to compute intensity on the fly.
        # For efficiency, we'd precompute λ(t) on a grid and interpolate.
        # But for moderate N, we can compute directly each time.
        # Implement a thinning loop similar to ogata_thinning but with custom intensity.
        events = []
        t = 0.0
        # Upper bound: sum over all shots influence at t could be max. Each h(t-τ_i) max is h(0)=αβ (for exponential). So λ_max = n_shots * h0? But n_shots variable.
        # Better: as we progress, only shots with τ_i < t contribute, and maximum total intensity occurs at time just after all shots have occurred? Not trivial.
        # For simplicity, use a very loose bound: shot_rate * ∫ h(t) dt = shot_rate * L1 norm.
        L1 = self.shot_kernel.l1_norm()
        lambda_max = self.shot_rate * L1 * 1.5 + 0.01

        while t < T:
            dt = float(npr.exponential(1.0 / lambda_max))
            t += dt
            if t >= T:
                break
            # Compute λ(t) = Σ_{τ_i < t} h(t-τ_i)
            # Use history_shots that are < t
            active = shot_times[shot_times < t]
            if len(active) == 0:
                lam = 0.0
            else:
                lags = t - active
                lam = bt.sum(self.shot_kernel.evaluate(lags))
            # Accept
            u = float(npr.uniform())
            if u <= lam / lambda_max:
                events.append(t)
        ev = bt.array(events) if events else bt.zeros(0)
        return ev

    def intensity(self, t: float, history: bt.array) -> float:
        if self.shot_times is None:
            raise ValueError(
                "shot_times not set; simulate first or assign shot_times for this Cox process"
            )
        tv = float(t)
        active = self.shot_times[self.shot_times < tv]
        if len(active) == 0:
            return 0.0
        lags = tv - active
        return float(bt.sum(self.shot_kernel.evaluate(bt.asarray(lags))))

    def log_likelihood(self, events: bt.array, T: float) -> float:
        """
        Log-likelihood of events given **fixed** shot times (latent Cox field observed).

        Treats N(t) as inhomogeneous Poisson with λ(t) = Σ_k h(t - τ_k) for τ_k < t.
        """
        if self.shot_times is None:
            raise ValueError(
                "shot_times required; run simulate(T) or set shot_times before log_likelihood."
            )
        self._last_T = float(T)
        events = np.asarray(events, dtype=float)
        events = events[(events >= 0) & (events <= T)]
        events.sort()
        if len(events) == 0:
            compensator = sum(
                float(self.shot_kernel.integrate(T - float(tau)))
                for tau in self.shot_times
                if tau < T
            )
            return float(-compensator)
        ll = 0.0
        for t_i in events:
            active = self.shot_times[self.shot_times < t_i]
            if len(active) == 0:
                raise ValueError("Shot-noise intensity zero at an event time; invalid data?")
            lags = t_i - active
            lam = float(bt.sum(self.shot_kernel.evaluate(bt.asarray(lags))))
            ll += np.log(max(lam, 1e-300))
        compensator = sum(
            float(self.shot_kernel.integrate(T - float(tau)))
            for tau in self.shot_times
            if tau < T
        )
        return float(ll - compensator)

    def get_params(self) -> dict:
        return {
            "shot_rate": self.shot_rate,
            "shot_kernel": self.shot_kernel,
            "shot_times": self.shot_times,
        }

    def set_params(self, params: dict) -> None:
        for k, v in params.items():
            setattr(self, k, v)