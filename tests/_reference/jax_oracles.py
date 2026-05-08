"""Frozen JAX implementations of multivariate exp Hawkes neg-log-likelihood
functions. Used as cross-validation oracles for the Rust port.

These were copied verbatim out of `python/intensify/core/inference/mle.py`
when the JAX paths were removed in the 0.3.0 Rust port. They are
**dev-only** — not packaged in any wheel, never imported at runtime by
intensify itself. See `tests/_reference/README.md`.
"""

from __future__ import annotations

# JAX defaults to float32; force float64 globally for cross-val precision.
# (Pre-port the intensify backends/jax_backend.py did this; with the
# backend abstraction gone, the oracle has to set it itself.)
from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)


def neg_ll_mv_exp(params, times_all, sources_all, T_jax, M):
    """Vectorized multivariate exp-Hawkes neg-log-likelihood (JAX, dense).

    Parameters layout (flat): [μ (M), (α_{m,k}, β_{m,k}) interleaved per cell
    row-major]. Uses a dense N×N lag matrix with strict lower-triangular
    causal mask. ``times_all`` and ``sources_all`` are sorted by time.
    """
    import jax.numpy as jnp

    mu = params[:M]
    rest = params[M:].reshape(M * M, 2)
    alpha_mat = rest[:, 0].reshape(M, M)
    beta_mat = rest[:, 1].reshape(M, M)

    lags = times_all[:, None] - times_all[None, :]
    causal = lags > 0

    src_i = sources_all[:, None]
    src_j = sources_all[None, :]
    alpha_ij = alpha_mat[src_i, src_j]
    beta_ij = beta_mat[src_i, src_j]

    safe_lags = jnp.where(causal, lags, 0.0)
    phi = jnp.where(
        causal,
        alpha_ij * beta_ij * jnp.exp(-beta_ij * safe_lags),
        0.0,
    )
    intensities = mu[sources_all] + jnp.sum(phi, axis=1)
    sum_log = jnp.sum(jnp.log(jnp.maximum(intensities, 1e-30)))

    tails = T_jax - times_all
    alpha_per_m = alpha_mat[:, sources_all]
    beta_per_m = beta_mat[:, sources_all]
    integ = alpha_per_m * (1.0 - jnp.exp(-beta_per_m * tails[None, :]))
    compensator_per_m = mu * T_jax + jnp.sum(integ, axis=1)
    total_comp = jnp.sum(compensator_per_m)

    return -(sum_log - total_comp)


def neg_ll_mv_exp_recursive(params, times_all, sources_all, T_jax, M, beta_scalar):
    """O(N·M) multivariate exp-Hawkes neg-log-likelihood with shared β.

    Uses a Hawkes recursion with an M-vector state R_k tracking the
    exponential-decay sum of past events from source k. Correct only when
    every β_{m,k} is the same scalar.

    Parameters layout: same as `neg_ll_mv_exp` — `[μ (M), (α, β)
    interleaved]`. The β slots are ignored (replaced by `beta_scalar`).
    """
    import jax
    import jax.numpy as jnp

    mu = params[:M]
    rest = params[M:].reshape(M * M, 2)
    alpha_mat = rest[:, 0].reshape(M, M)
    beta = beta_scalar

    dts = jnp.diff(times_all, prepend=jnp.array(0.0, dtype=times_all.dtype))
    one_hot = jax.nn.one_hot(sources_all, M, dtype=params.dtype)

    def step(R, inputs):
        dt, src_onehot = inputs
        R_dec = R * jnp.exp(-beta * dt)
        alpha_row = src_onehot @ alpha_mat
        mu_src = jnp.dot(src_onehot, mu)
        lam = mu_src + beta * jnp.dot(alpha_row, R_dec)
        log_lam = jnp.log(jnp.maximum(lam, 1e-30))
        R_new = R_dec + src_onehot
        return R_new, log_lam

    _, log_terms = jax.lax.scan(step, jnp.zeros(M, dtype=params.dtype), (dts, one_hot))
    sum_log = jnp.sum(log_terms)

    tails = T_jax - times_all
    per_j = 1.0 - jnp.exp(-beta * tails)
    per_k = one_hot.T @ per_j
    integ_per_m = alpha_mat @ per_k
    compensator_per_m = mu * T_jax + integ_per_m
    total_comp = jnp.sum(compensator_per_m)

    return -(sum_log - total_comp)
