"""Simulation algorithms: Ogata thinning (univariate and multivariate)."""


import numpy as np

from ...backends import get_backend
from ...backends._backend import get_backend_name
from ...core.processes.hawkes import MultivariateHawkes, UnivariateHawkes

bt = get_backend()


def _estimate_capacity(mu: float, kernel_l1: float, T: float) -> int:
    """Rough upper bound on event count for buffer pre-allocation."""
    rate = mu / max(1.0 - kernel_l1, 0.05)
    return max(int(rate * T * 3) + 64, 256)


def ogata_thinning(
    process: UnivariateHawkes,
    T: float,
    key: object | None = None,
    seed: int | None = None,
) -> object:
    """
    Ogata thinning algorithm for univariate point processes.

    Parameters
    ----------
    process : UnivariateHawkes
        The point process to simulate.
    T : float
        End of observation window.
    key : optional
        JAX PRNG key when using the JAX backend.
    seed : int, optional
        Seed for the NumPy backend.

    Returns
    -------
    events : array
        Sorted event timestamps.
    """
    if get_backend_name() == "numpy":
        return _ogata_thinning_numpy(process, T, seed)

    jit_ok = getattr(process.kernel, "jit_compatible", False)
    if jit_ok:
        return _ogata_thinning_jax_while(process, T, key, seed)
    return _ogata_thinning_jax_python(process, T, key, seed)


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
        hist = bt.array(history) if history else bt.zeros(0)
        current_lambda = float(process.intensity(t, hist))
        if current_lambda > lambda_max:
            lambda_max = current_lambda * 1.5
        u = float(np.random.uniform(0.0, 1.0))
        if u <= current_lambda / lambda_max:
            events.append(t)
            history.append(t)
    if not events:
        return bt.zeros(0)
    return bt.array(events)


def _ogata_thinning_jax_python(process, T, key, seed):
    """JAX backend but Python-loop (for JIT-incompatible kernels)."""
    import jax.random as jr

    events: list[float] = []
    history: list[float] = []
    t = 0.0
    lambda_max = float(process.mu) * 1.5 + 1e-6
    k = key if key is not None else jr.PRNGKey(int(seed) if seed is not None else 0)
    while t < T:
        k, sk = jr.split(k)
        dt = float(jr.exponential(sk) / lambda_max)
        t += dt
        if t >= T:
            break
        hist = bt.array(history) if history else bt.zeros(0)
        current_lambda = float(process.intensity(t, hist))
        if current_lambda > lambda_max:
            lambda_max = current_lambda * 1.5
        k, sk = jr.split(k)
        u = float(jr.uniform(sk))
        if u <= current_lambda / lambda_max:
            events.append(t)
            history.append(t)
    if not events:
        return bt.zeros(0)
    return bt.array(events)


def _ogata_thinning_jax_while(process, T, key, seed):
    """JAX thinning using jax.lax.while_loop with a fixed-capacity buffer.

    Only valid when ``process.kernel.jit_compatible`` is True (parametric
    kernels whose ``evaluate`` is traceable).
    """
    import jax
    import jax.numpy as jnp
    import jax.random as jr

    from ..kernels.exponential import ExponentialKernel

    mu = jnp.asarray(float(process.mu), dtype=jnp.float64)
    T_val = jnp.asarray(float(T), dtype=jnp.float64)
    capacity = _estimate_capacity(float(mu), float(process.kernel.l1_norm()), float(T))
    rng = key if key is not None else jr.PRNGKey(int(seed) if seed is not None else 0)

    kernel = process.kernel

    if isinstance(kernel, ExponentialKernel) and not kernel.allow_signed:
        alpha = jnp.asarray(kernel.alpha, dtype=jnp.float64)
        beta = jnp.asarray(kernel.beta, dtype=jnp.float64)

        def _intensity_from_buf(t, buf, n):
            history = jax.lax.dynamic_slice(buf, (0,), (capacity,))
            lags = t - history
            valid = (lags > 0) & (jnp.arange(capacity) < n)
            kernel_vals = jnp.where(valid, alpha * beta * jnp.exp(-beta * lags), 0.0)
            return mu + jnp.sum(kernel_vals)
    else:
        def _intensity_from_buf(t, buf, n):
            history = jax.lax.dynamic_slice(buf, (0,), (capacity,))
            lags = t - history
            valid = (lags > 0) & (jnp.arange(capacity) < n)
            kernel_vals = jnp.where(valid, kernel.evaluate(lags), 0.0)
            return mu + jnp.sum(kernel_vals)

    init_buf = jnp.zeros(capacity, dtype=jnp.float64)
    lambda_max_init = mu * 1.5 + 1e-6
    cap = jnp.array(capacity, dtype=jnp.int32)

    # carry: (t, buf, n_events, lambda_max, rng_key)
    init_carry = (jnp.array(0.0), init_buf, jnp.array(0, dtype=jnp.int32), lambda_max_init, rng)

    def cond_fn(carry):
        t, _buf, n, _lm, _k = carry
        return (t < T_val) & (n < cap)

    def body_fn(carry):
        t, buf, n, lam_max, k = carry
        k, sk1, sk2 = jr.split(k, 3)
        dt = jr.exponential(sk1) / lam_max
        t_new = t + dt

        cur_lam = _intensity_from_buf(t_new, buf, n)
        lam_max_new = jnp.maximum(lam_max, cur_lam * 1.5)
        u = jr.uniform(sk2)
        accept = (u <= cur_lam / lam_max_new) & (t_new < T_val)

        buf_new = jnp.where(accept, buf.at[n].set(t_new), buf)
        n_new = jnp.where(accept, n + 1, n)
        return (t_new, buf_new, n_new, lam_max_new, k)

    _t_f, buf_f, n_f, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    n_out = int(n_f)
    if n_out == 0:
        return jnp.zeros(0, dtype=jnp.float64)
    return buf_f[:n_out]


def ogata_thinning_multivariate(
    process: MultivariateHawkes,
    T: float,
    key: object | None = None,
    seed: int | None = None,
) -> list[object]:
    """Multivariate Ogata thinning."""
    M = process.n_dims
    histories: list[list[float]] = [[] for _ in range(M)]
    t = 0.0
    lambda_max = float(bt.sum(process.mu)) * 1.5 + 1e-6

    if get_backend_name() == "numpy":
        if seed is not None:
            np.random.seed(seed)
        while t < T:
            dt = float(np.random.exponential(1.0 / lambda_max))
            t += dt
            if t >= T:
                break
            lambda_vec = process.intensity(
                t, [bt.array(h) if h else bt.zeros(0) for h in histories]
            )
            total_lambda = float(bt.sum(lambda_vec))
            if total_lambda > lambda_max:
                lambda_max = total_lambda * 1.5
            u = float(np.random.uniform(0.0, 1.0))
            if u > total_lambda / lambda_max:
                continue
            probs = (
                lambda_vec / total_lambda
                if total_lambda > 0
                else bt.ones(M) / M
            )
            cum_probs = bt.cumsum(probs)
            u2 = float(np.random.uniform(0.0, 1.0))
            chosen_dim = 0
            while chosen_dim < M - 1 and u2 > float(cum_probs[chosen_dim]):
                chosen_dim += 1
            histories[chosen_dim].append(t)
        return [bt.array(h) if h else bt.zeros(0) for h in histories]

    import jax.random as jr

    k = key if key is not None else jr.PRNGKey(int(seed) if seed is not None else 0)
    while t < T:
        k, sk = jr.split(k)
        dt = float(jr.exponential(sk) / lambda_max)
        t += dt
        if t >= T:
            break
        lambda_vec = process.intensity(
            t, [bt.array(h) if h else bt.zeros(0) for h in histories]
        )
        total_lambda = float(bt.sum(lambda_vec))
        if total_lambda > lambda_max:
            lambda_max = total_lambda * 1.5
        k, sk = jr.split(k)
        u = float(jr.uniform(sk))
        if u > total_lambda / lambda_max:
            continue
        probs = (
            lambda_vec / total_lambda if total_lambda > 0 else bt.ones(M) / M
        )
        cum_probs = bt.cumsum(probs)
        k, sk = jr.split(k)
        u2 = float(jr.uniform(sk))
        chosen_dim = 0
        while chosen_dim < M - 1 and u2 > float(cum_probs[chosen_dim]):
            chosen_dim += 1
        histories[chosen_dim].append(t)
    return [bt.array(h) if h else bt.zeros(0) for h in histories]
