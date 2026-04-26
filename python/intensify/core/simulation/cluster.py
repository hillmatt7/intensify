"""Cluster (branching) simulation for Hawkes processes."""

import warnings

import numpy as np

from ...backends import get_backend
from ...core.processes.hawkes import MultivariateHawkes, UnivariateHawkes

bt = get_backend()


def branching_simulation(process: UnivariateHawkes, T: float, seed: int = None) -> bt.array:
    """
    Simulate univariate Hawkes using the branching (Galton-Watson) representation.

    More efficient than thinning when branching ratio (||φ||_1) is close to 1,
    because it avoids many rejections. However, it can be memory-intensive for
    near-critical processes (branching ratio -> 1) because the simulated tree
    can become very large.

    Parameters
    ----------
    process : UnivariateHawkes
        The Hawkes process to simulate.
    T : float
        End of observation window.
    seed : int, optional
        Random seed.

    Returns
    -------
    events : jnp.ndarray or np.ndarray
        Sorted event timestamps in [0, T].
    """
    # Phase 3: ExponentialKernel routes through Rust branching simulator.
    from ...core.kernels.exponential import ExponentialKernel
    if isinstance(process.kernel, ExponentialKernel) and not process.kernel.allow_signed:
        from ..._rust import _ext
        seed_u64 = int(seed) if seed is not None else 0
        arr = _ext.simulation.simulate_uni_exp_branching(
            float(T),
            float(process.mu),
            float(process.kernel.alpha),
            float(process.kernel.beta),
            seed_u64,
        )
        return bt.asarray(arr)

    npr = np.random.default_rng(seed)

    # Generate immigrants: Poisson with rate μ (NumPy: JAX poisson needs a PRNG key)
    n_immigrant = int(npr.poisson(float(process.mu * T)))
    immigrant_times = npr.uniform(0.0, T, size=n_immigrant) if n_immigrant > 0 else np.array([])
    # Sort immigrants
    if n_immigrant > 0:
        immigrant_times = bt.asarray(np.sort(np.asarray(immigrant_times, dtype=float)))
    else:
        immigrant_times = bt.zeros(0)

    # Each immigrant can produce offspring via a branching process
    events = []
    queue = list(immigrant_times) if hasattr(immigrant_times, "__iter__") else []
    if hasattr(immigrant_times, "tolist"):
        queue = immigrant_times.tolist()

    while queue:
        parent_time = queue.pop(0)
        events.append(parent_time)
        # Number of offspring produced by this event: Poisson(||φ||_1) i.e., mean = L1 norm
        n_offspring = int(npr.poisson(float(process.kernel.l1_norm())))
        # Offspring times are parent_time + inter-arrival distribution: but for Hawkes with general kernel, the offspring distribution is not a simple exponential; it's distributed according to the kernel density scaled by L1.
        # For an exponential kernel, the offspring times are drawn from an exponential distribution with rate β, independent between offspring.
        # For general kernels, we'd need to sample from a distribution with density φ(t)/||φ||_1.
        # Simplify: only support ExponentialKernel for this representation; others use rejection.
        if hasattr(process.kernel, "beta") and hasattr(process.kernel, "alpha"):
            # Exponential: φ(t)=αβ exp(-β t); normalized distribution: Exponential(rate=β)
            beta = process.kernel.beta
            offspring_times = parent_time + npr.exponential(
                scale=1.0 / float(beta), size=n_offspring
            )
            # Keep only those <= T
            for ti in offspring_times:
                if ti <= T:
                    queue.append(ti)
        else:
            # For non-exponential kernels, naive rejection: sample uniform candidate times and accept with probability φ(dt)/max φ?
            # Better: use thinning for offspring within bounded interval after parent.
            # But that would re-introduce thinning complexity. For now, skip; warn.
            warnings.warn("Branching simulation for non-exponential kernels not implemented", UserWarning)
            break

    if not events:
        return bt.zeros(0)
    return bt.array(sorted(events))


def branching_simulation_multivariate(process: MultivariateHawkes, T: float, seed: int = None) -> list[bt.array]:
    """
    Multivariate branching simulation. Immigrants appear in each dimension
    according to background rate μ_m. Each event (immigrant or offspring) can
    produce offspring in any dimension m' with rate given by the kernel φ_{m', m}.

    Currently only supports ExponentialKernel shared across all pairs.

    Returns
    -------
    events_by_dim : list of arrays, one per dimension
    """
    # Phase 3: shared-β ExponentialKernel matrix routes through Rust.
    from ..._rust import _ext, mv_shared_beta
    shared = mv_shared_beta(process)
    if shared is not None:
        seed_u64 = int(seed) if seed is not None else 0
        M = process.n_dims
        mu = np.ascontiguousarray(np.asarray(process.mu, dtype=np.float64).ravel())
        alpha = np.empty(M * M, dtype=np.float64)
        for i in range(M):
            for j in range(M):
                alpha[i * M + j] = float(process.kernel_matrix[i][j].alpha)
        histories = _ext.simulation.simulate_mv_exp_branching(
            float(T), mu, alpha, float(shared), seed_u64,
        )
        return [bt.asarray(h) for h in histories]

    npr = np.random.default_rng(seed)
    M = process.n_dims
    # Immigrant events per dimension
    all_events = [[] for _ in range(M)]
    queue = []  # list of (time, parent_dim)

    # Generate immigrants for each dimension: Poisson(μ_m)
    for m in range(M):
        mu_m = float(process.mu[m]) if hasattr(process.mu, "__getitem__") else float(process.mu)
        n_imm = int(npr.poisson(mu_m * T))
        times = npr.uniform(0, T, size=n_imm) if n_imm > 0 else np.array([])
        for ti in times:
            queue.append((float(ti), m, None))  # (time, dim, parent_time)
        all_events[m].extend(times)

    # Process queue
    while queue:
        t, dim, _ = queue.pop(0)
        if t > T:
            continue
        # For this event in dimension dim, offspring in dimension m' come from kernels φ_{m',dim}
        for m_prime in range(M):
            kernel = process.kernel_matrix[m_prime][dim]
            # Expected number of offspring to dim m': L1 norm of that kernel
            L1 = kernel.l1_norm()
            n_off = int(npr.poisson(float(L1)))
            if n_off == 0:
                continue
            # For exponential kernels, we can sample offspring times as exponential delays
            if hasattr(kernel, "beta"):
                beta = kernel.beta
                delays = npr.exponential(scale=1.0 / float(beta), size=n_off)
                for d in delays:
                    offspring_t = t + d
                    if offspring_t <= T:
                        queue.append((offspring_t, m_prime, t))
                        all_events[m_prime].append(offspring_t)
            else:
                warnings.warn("Multivariate branching simulation only supports ExponentialKernel", UserWarning)
                break

    # Sort each dimension's events and convert to backend arrays
    result = []
    for m in range(M):
        ev = sorted(all_events[m])
        result.append(bt.array(ev))
    return result