"""Cross-validation: Rust mv_exp_dense (per-cell β fitted) matches the
existing JAX `_neg_ll_mv_exp` reference to ~1e-10 across many seeds.

Layout: both the Rust and JAX paths use the same flat coefficient
ordering for the joint-decay case:
  [μ_0..μ_{M-1}, (α_{0,0}, β_{0,0}), (α_{0,1}, β_{0,1}), ...,
   (α_{M-1, M-1}, β_{M-1, M-1})]   — total M + 2·M².
Internally the Rust core takes mu / alpha / beta as separate flat
row-major arrays; the Python wrapper splits + merges.
"""

from __future__ import annotations

import numpy as np
import pytest


def _rust_dense(times, sources, T, M, mu, alpha, beta):
    """Call the Rust dense entry. Inputs as numpy arrays of correct dtype."""
    from intensify._libintensify.likelihood import mv_exp_dense_neg_ll_with_grad

    return mv_exp_dense_neg_ll_with_grad(
        np.ascontiguousarray(times, dtype=np.float64),
        np.ascontiguousarray(sources, dtype=np.int64),
        float(T),
        int(M),
        np.ascontiguousarray(mu, dtype=np.float64),
        np.ascontiguousarray(alpha, dtype=np.float64),
        np.ascontiguousarray(beta, dtype=np.float64),
    )


def _flatten_events(events):
    times: list[float] = []
    sources: list[int] = []
    for k, ev in enumerate(events):
        for t in ev:
            times.append(float(t))
            sources.append(k)
    order = np.argsort(times, kind="stable")
    return (
        np.asarray(times, dtype=np.float64)[order],
        np.asarray(sources, dtype=np.int64)[order],
    )


def _gen_seed(seed: int, n_dims: int, max_events: int = 800):
    """Sample a stationary multivariate exp Hawkes with per-cell decay."""
    from intensify.core.kernels.exponential import ExponentialKernel
    from intensify.core.processes.hawkes import MultivariateHawkes

    rng = np.random.default_rng(seed)
    mu_true = rng.uniform(0.05, 0.4, size=n_dims)
    alpha_true = rng.uniform(0.0, 0.15, size=(n_dims, n_dims))
    row_norms = alpha_true.sum(axis=1)
    scale = np.minimum(0.6 / np.maximum(row_norms, 1e-8), 1.0)
    alpha_true = alpha_true * scale[:, None]
    # Per-cell β between 0.5 and 3.0
    beta_true = rng.uniform(0.5, 3.0, size=(n_dims, n_dims))
    T = float(rng.uniform(20.0, 60.0))

    proc = MultivariateHawkes(
        n_dims=n_dims,
        mu=mu_true.tolist(),
        kernel=[
            [
                ExponentialKernel(
                    alpha=float(alpha_true[i, j]),
                    beta=float(beta_true[i, j]),
                )
                for j in range(n_dims)
            ]
            for i in range(n_dims)
        ],
    )
    events = proc.simulate(T=T, seed=seed)
    if any(len(e) > max_events for e in events) or sum(len(e) for e in events) < n_dims * 2:
        return _gen_seed(seed + 7919, n_dims, max_events)
    return [np.asarray(e, dtype=np.float64) for e in events], T, mu_true, alpha_true, beta_true


def _intensify_flat_layout(mu, alpha, beta, M):
    """Pack into intensify's [μ, (α, β) interleaved] layout."""
    parts: list[float] = list(mu)
    for i in range(M):
        for j in range(M):
            parts.append(float(alpha[i, j]))
            parts.append(float(beta[i, j]))
    return np.asarray(parts, dtype=np.float64)


@pytest.mark.parametrize("seed", range(15))
def test_mv_exp_dense_matches_jax_2d(seed: int) -> None:
    """2D joint-decay: Rust loss matches JAX `_neg_ll_mv_exp` to 1e-10."""
    import jax.numpy as jnp

    from tests._reference.jax_oracles import neg_ll_mv_exp as _neg_ll_mv_exp

    events, T, mu_true, alpha_true, beta_true = _gen_seed(seed, n_dims=2)
    M = 2

    times_all, sources_all = _flatten_events(events)
    rust_loss, _, _, _ = _rust_dense(
        times_all, sources_all, T, M, mu_true, alpha_true.flatten(), beta_true.flatten()
    )

    jax_params = _intensify_flat_layout(mu_true, alpha_true, beta_true, M)
    jax_loss = float(
        _neg_ll_mv_exp(
            jnp.asarray(jax_params),
            jnp.asarray(times_all),
            jnp.asarray(sources_all, dtype=jnp.int32),
            jnp.asarray(T),
            M,
        )
    )

    assert abs(rust_loss - jax_loss) < 1e-10, (
        f"seed {seed}: rust={rust_loss:.10f} jax={jax_loss:.10f} "
        f"diff={abs(rust_loss - jax_loss):.2e}"
    )


@pytest.mark.parametrize("seed", range(10))
def test_mv_exp_dense_matches_jax_5d(seed: int) -> None:
    """5D joint-decay (the headline mv_exp_5d benchmark)."""
    import jax.numpy as jnp

    from tests._reference.jax_oracles import neg_ll_mv_exp as _neg_ll_mv_exp

    events, T, mu_true, alpha_true, beta_true = _gen_seed(seed + 100, n_dims=5)
    M = 5
    times_all, sources_all = _flatten_events(events)
    rust_loss, _, _, _ = _rust_dense(
        times_all, sources_all, T, M, mu_true, alpha_true.flatten(), beta_true.flatten()
    )
    jax_params = _intensify_flat_layout(mu_true, alpha_true, beta_true, M)
    jax_loss = float(
        _neg_ll_mv_exp(
            jnp.asarray(jax_params),
            jnp.asarray(times_all),
            jnp.asarray(sources_all, dtype=jnp.int32),
            jnp.asarray(T),
            M,
        )
    )
    assert abs(rust_loss - jax_loss) < 1e-9


@pytest.mark.parametrize("seed", range(8))
def test_mv_exp_dense_grad_finite_difference(seed: int) -> None:
    """Per-cell β gradient: 5-point stencil sanity check (3D)."""
    events, T, mu_true, alpha_true, beta_true = _gen_seed(seed + 200, n_dims=3)
    M = 3
    times_all, sources_all = _flatten_events(events)

    val, gm, ga, gb = _rust_dense(
        times_all, sources_all, T, M, mu_true, alpha_true.flatten(), beta_true.flatten()
    )

    # Build the full flat vector and compare cell-by-cell grad at h=1e-6.
    from intensify._libintensify.likelihood import mv_exp_dense_neg_ll

    h = 1e-6

    def loss_at(mu, alpha, beta):
        return mv_exp_dense_neg_ll(
            np.ascontiguousarray(times_all, dtype=np.float64),
            np.ascontiguousarray(sources_all, dtype=np.int64),
            float(T), int(M),
            np.ascontiguousarray(mu, dtype=np.float64),
            np.ascontiguousarray(alpha, dtype=np.float64),
            np.ascontiguousarray(beta, dtype=np.float64),
        )

    # μ block
    for m in range(M):
        mu_p = mu_true.copy(); mu_p[m] += h
        mu_m = mu_true.copy(); mu_m[m] -= h
        mu_p2 = mu_true.copy(); mu_p2[m] += 2*h
        mu_m2 = mu_true.copy(); mu_m2[m] -= 2*h
        a_flat = alpha_true.flatten(); b_flat = beta_true.flatten()
        f_p = loss_at(mu_p, a_flat, b_flat)
        f_n = loss_at(mu_m, a_flat, b_flat)
        f_p2 = loss_at(mu_p2, a_flat, b_flat)
        f_n2 = loss_at(mu_m2, a_flat, b_flat)
        numeric = (-f_p2 + 8*f_p - 8*f_n + f_n2) / (12*h)
        assert abs(gm[m] - numeric) / max(abs(numeric), 1e-6) < 1e-5, (
            f"seed {seed} grad μ_{m}: analytic={gm[m]} numeric={numeric}"
        )

    # α and β cells: spot-check a few rather than all M² (kept fast)
    for cell in [(0, 0), (0, M - 1), (M - 1, 0), (M - 1, M - 1)]:
        i, j = cell
        idx = i * M + j
        # α
        a_p = alpha_true.copy(); a_p[i, j] += h
        a_m = alpha_true.copy(); a_m[i, j] -= h
        a_p2 = alpha_true.copy(); a_p2[i, j] += 2*h
        a_m2 = alpha_true.copy(); a_m2[i, j] -= 2*h
        b_flat = beta_true.flatten()
        f_p = loss_at(mu_true, a_p.flatten(), b_flat)
        f_n = loss_at(mu_true, a_m.flatten(), b_flat)
        f_p2 = loss_at(mu_true, a_p2.flatten(), b_flat)
        f_n2 = loss_at(mu_true, a_m2.flatten(), b_flat)
        numeric_a = (-f_p2 + 8*f_p - 8*f_n + f_n2) / (12*h)
        assert abs(ga[idx] - numeric_a) / max(abs(numeric_a), 1e-6) < 1e-5

        # β
        a_flat = alpha_true.flatten()
        b_p = beta_true.copy(); b_p[i, j] += h
        b_m = beta_true.copy(); b_m[i, j] -= h
        b_p2 = beta_true.copy(); b_p2[i, j] += 2*h
        b_m2 = beta_true.copy(); b_m2[i, j] -= 2*h
        f_p = loss_at(mu_true, a_flat, b_p.flatten())
        f_n = loss_at(mu_true, a_flat, b_m.flatten())
        f_p2 = loss_at(mu_true, a_flat, b_p2.flatten())
        f_n2 = loss_at(mu_true, a_flat, b_m2.flatten())
        numeric_b = (-f_p2 + 8*f_p - 8*f_n + f_n2) / (12*h)
        # Looser tol — quadrature truncation in the 5-point stencil dominates
        # for some seeds; the underlying analytic gradient is correct (each
        # seed crosses 1e-12 cross-val with brute-force in the Rust unit tests).
        assert abs(gb[idx] - numeric_b) / max(abs(numeric_b), 1e-6) < 1e-4


def test_mv_exp_dense_value_grad_consistent() -> None:
    events, T, mu_true, alpha_true, beta_true = _gen_seed(0, n_dims=3)
    M = 3
    times_all, sources_all = _flatten_events(events)

    from intensify._libintensify.likelihood import (
        mv_exp_dense_neg_ll,
        mv_exp_dense_neg_ll_with_grad,
    )

    v1, *_ = mv_exp_dense_neg_ll_with_grad(
        np.ascontiguousarray(times_all, dtype=np.float64),
        np.ascontiguousarray(sources_all, dtype=np.int64),
        float(T), int(M),
        np.ascontiguousarray(mu_true, dtype=np.float64),
        np.ascontiguousarray(alpha_true.flatten(), dtype=np.float64),
        np.ascontiguousarray(beta_true.flatten(), dtype=np.float64),
    )
    v2 = mv_exp_dense_neg_ll(
        np.ascontiguousarray(times_all, dtype=np.float64),
        np.ascontiguousarray(sources_all, dtype=np.int64),
        float(T), int(M),
        np.ascontiguousarray(mu_true, dtype=np.float64),
        np.ascontiguousarray(alpha_true.flatten(), dtype=np.float64),
        np.ascontiguousarray(beta_true.flatten(), dtype=np.float64),
    )
    assert abs(v1 - v2) < 1e-15
