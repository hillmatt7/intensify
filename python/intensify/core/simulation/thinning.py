"""Simulation algorithms: Ogata thinning (univariate and multivariate).

Numpy-only fallback. The Rust simulators (`simulate_uni_exp_hawkes`,
`simulate_mv_exp_hawkes`) cover the ExponentialKernel cases and are
called directly from `Hawkes.simulate` / `MultivariateHawkes.simulate`.
This module only handles non-exp kernels and signed-exp.
"""

import numpy as np

from ...core.processes.hawkes import MultivariateHawkes, UnivariateHawkes


def _estimate_capacity(mu: float, kernel_l1: float, T: float) -> int:
    """Rough upper bound on event count for buffer pre-allocation."""
    rate = mu / max(1.0 - kernel_l1, 0.05)
    return max(int(rate * T * 3) + 64, 256)


def ogata_thinning(
    process: UnivariateHawkes,
    T: float,
    seed: int | None = None,
) -> np.ndarray:
    """Ogata thinning for a univariate point process. Pure-numpy fallback;
    the Rust simulator handles ExponentialKernel directly via the
    `Hawkes.simulate()` dispatch."""
    return _ogata_thinning_numpy(process, T, seed)


def _ogata_thinning_numpy(process, T, seed):
    events: list[float] = []
    history: list[float] = []
    t = 0.0
    lambda_max = float(process.mu) * 1.5 + 1e-6

    if seed is not None:
        np.random.seed(seed)
    while t < T:
        dt = float(np.random.exponential(1.0 / lambda_max))
        t += dt
        if t >= T:
            break
        hist = np.asarray(history) if history else np.zeros(0)
        current_lambda = float(process.intensity(t, hist))
        if current_lambda > lambda_max:
            lambda_max = current_lambda * 1.5
        u = float(np.random.uniform(0.0, 1.0))
        if u <= current_lambda / lambda_max:
            events.append(t)
            history.append(t)
    if not events:
        return np.zeros(0)
    return np.asarray(events, dtype=np.float64)


def ogata_thinning_multivariate(
    process: MultivariateHawkes,
    T: float,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Multivariate Ogata thinning. Pure-numpy fallback."""
    M = process.n_dims
    histories: list[list[float]] = [[] for _ in range(M)]
    t = 0.0
    lambda_max = float(np.sum(process.mu)) * 1.5 + 1e-6

    if seed is not None:
        np.random.seed(seed)
    while t < T:
        dt = float(np.random.exponential(1.0 / lambda_max))
        t += dt
        if t >= T:
            break
        lambda_vec = process.intensity(
            t, [np.asarray(h) if h else np.zeros(0) for h in histories]
        )
        total_lambda = float(np.sum(lambda_vec))
        if total_lambda > lambda_max:
            lambda_max = total_lambda * 1.5
        u = float(np.random.uniform(0.0, 1.0))
        if u > total_lambda / lambda_max:
            continue
        probs = (
            lambda_vec / total_lambda
            if total_lambda > 0
            else np.ones(M) / M
        )
        cum_probs = np.cumsum(probs)
        u2 = float(np.random.uniform(0.0, 1.0))
        chosen_dim = 0
        while chosen_dim < M - 1 and u2 > float(cum_probs[chosen_dim]):
            chosen_dim += 1
        histories[chosen_dim].append(t)
    return [np.asarray(h, dtype=np.float64) if h else np.zeros(0) for h in histories]
