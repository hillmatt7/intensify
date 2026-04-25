"""Cross-validation: Rust mv_exp_recursive (β fixed) matches the existing
JAX `_neg_ll_mv_exp_recursive` reference to ~1e-10 across many seeds.

Layout mapping:
  Rust:  coeffs = [μ_0..μ_{M-1}, α_{0,:}, α_{1,:}, ..., α_{M-1,:}]      (M + M²)
  JAX:   params = [μ_0..μ_{M-1}, (α_{i,j}, β_{i,j}) interleaved]        (M + 2M²)

The Rust path uses a fixed shared β; the JAX function takes that β as
`beta_scalar` and ignores the β slots in `params`.
"""

from __future__ import annotations

import numpy as np
import pytest

from intensify._libintensify.likelihood import MvExpRecursiveLogLik


def _rust_coeffs_to_jax_params(mu: np.ndarray, alpha: np.ndarray, beta: float) -> np.ndarray:
    """Pack (μ, α, β) into JAX's flat layout [μ, (α, β) interleaved]."""
    M = len(mu)
    parts: list[float] = list(mu)
    for i in range(M):
        for j in range(M):
            parts.append(float(alpha[i, j]))
            parts.append(float(beta))
    return np.asarray(parts, dtype=np.float64)


def _flatten_events(events: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Sort events across dims into a single (times, sources) pair."""
    times: list[float] = []
    sources: list[int] = []
    for k, ev in enumerate(events):
        for t in ev:
            times.append(float(t))
            sources.append(k)
    order = np.argsort(times, kind="stable")
    return (
        np.asarray(times, dtype=np.float64)[order],
        np.asarray(sources, dtype=np.int32)[order],
    )


def _gen_seed(seed: int, n_dims: int, max_events: int = 800):
    """Sample a stationary multivariate exp Hawkes and return events + ground truth."""
    from intensify.core.kernels.exponential import ExponentialKernel
    from intensify.core.processes.hawkes import MultivariateHawkes

    rng = np.random.default_rng(seed)
    mu_true = rng.uniform(0.05, 0.4, size=n_dims)
    # Pick a row-stochastic alpha with overall spectral radius < ~0.7 for
    # tractable simulation. Random with each row L1 < 0.6.
    alpha_true = rng.uniform(0.0, 0.15, size=(n_dims, n_dims))
    row_norms = alpha_true.sum(axis=1)
    scale = np.minimum(0.6 / np.maximum(row_norms, 1e-8), 1.0)
    alpha_true = alpha_true * scale[:, None]
    beta_true = float(rng.uniform(0.5, 3.0))
    T = float(rng.uniform(20.0, 60.0))

    proc = MultivariateHawkes(
        n_dims=n_dims,
        mu=mu_true.tolist(),
        kernel=[
            [ExponentialKernel(alpha=float(alpha_true[i, j]), beta=beta_true) for j in range(n_dims)]
            for i in range(n_dims)
        ],
    )
    events = proc.simulate(T=T, seed=seed)
    if any(len(e) > max_events for e in events) or sum(len(e) for e in events) < n_dims * 2:
        return _gen_seed(seed + 7919, n_dims, max_events)

    return [np.asarray(e, dtype=np.float64) for e in events], T, mu_true, alpha_true, beta_true


@pytest.mark.parametrize("seed", range(20))
def test_mv_exp_recursive_matches_jax_2d(seed: int) -> None:
    """2D Hawkes — Rust loss matches JAX `_neg_ll_mv_exp_recursive` to 1e-10."""
    import jax.numpy as jnp

    from intensify.core.inference.mle import _neg_ll_mv_exp_recursive

    events, T, mu_true, alpha_true, beta_true = _gen_seed(seed, n_dims=2)
    M = 2

    # Rust path
    rust_model = MvExpRecursiveLogLik(events, T, beta_true)
    rust_coeffs = np.concatenate([mu_true, alpha_true.flatten()])
    rust_out = np.zeros(rust_model.n_coeffs)
    rust_loss = rust_model.loss_and_grad(rust_coeffs, rust_out)

    # JAX reference
    times_all, sources_all = _flatten_events(events)
    jax_params = _rust_coeffs_to_jax_params(mu_true, alpha_true, beta_true)
    jax_loss = float(
        _neg_ll_mv_exp_recursive(
            jnp.asarray(jax_params),
            jnp.asarray(times_all),
            jnp.asarray(sources_all, dtype=jnp.int32),
            jnp.asarray(T),
            M,
            jnp.asarray(beta_true),
        )
    )

    assert abs(rust_loss - jax_loss) < 1e-10, (
        f"seed {seed}: rust={rust_loss:.10f} jax={jax_loss:.10f} "
        f"diff={abs(rust_loss - jax_loss):.2e}"
    )


@pytest.mark.parametrize("seed", range(15))
def test_mv_exp_recursive_matches_jax_5d(seed: int) -> None:
    """5D Hawkes — the headline case (mv_exp_5d benchmark dim)."""
    import jax.numpy as jnp

    from intensify.core.inference.mle import _neg_ll_mv_exp_recursive

    events, T, mu_true, alpha_true, beta_true = _gen_seed(seed + 100, n_dims=5)
    M = 5
    rust_model = MvExpRecursiveLogLik(events, T, beta_true)
    rust_coeffs = np.concatenate([mu_true, alpha_true.flatten()])
    rust_out = np.zeros(rust_model.n_coeffs)
    rust_loss = rust_model.loss_and_grad(rust_coeffs, rust_out)

    times_all, sources_all = _flatten_events(events)
    jax_params = _rust_coeffs_to_jax_params(mu_true, alpha_true, beta_true)
    jax_loss = float(
        _neg_ll_mv_exp_recursive(
            jnp.asarray(jax_params),
            jnp.asarray(times_all),
            jnp.asarray(sources_all, dtype=jnp.int32),
            jnp.asarray(T),
            M,
            jnp.asarray(beta_true),
        )
    )

    assert abs(rust_loss - jax_loss) < 1e-9, (
        f"seed {seed}: rust={rust_loss:.10f} jax={jax_loss:.10f}"
    )


@pytest.mark.parametrize("seed", range(10))
def test_mv_exp_recursive_grad_finite_difference(seed: int) -> None:
    """5-point stencil sanity check on the analytic gradient (5D)."""
    events, T, mu_true, alpha_true, beta_true = _gen_seed(seed + 200, n_dims=3)
    rust_model = MvExpRecursiveLogLik(events, T, beta_true)
    coeffs = np.concatenate([mu_true, alpha_true.flatten()])
    n = len(coeffs)

    grad_a = np.zeros(n)
    rust_model.loss_and_grad(coeffs, grad_a)

    h = 1e-6
    grad_n = np.zeros(n)
    for idx in range(n):
        p_p = coeffs.copy(); p_p[idx] += h
        p_m = coeffs.copy(); p_m[idx] -= h
        p_p2 = coeffs.copy(); p_p2[idx] += 2 * h
        p_m2 = coeffs.copy(); p_m2[idx] -= 2 * h
        f_p = rust_model.loss(p_p)
        f_m = rust_model.loss(p_m)
        f_p2 = rust_model.loss(p_p2)
        f_m2 = rust_model.loss(p_m2)
        grad_n[idx] = (-f_p2 + 8 * f_p - 8 * f_m + f_m2) / (12 * h)

    np.testing.assert_allclose(grad_a, grad_n, rtol=1e-6, atol=1e-9)


def test_mv_exp_recursive_layout_documented() -> None:
    """Sanity: coeffs layout is `[μ, α row-major]` with len M + M²."""
    times = [np.array([0.5, 1.3, 2.1], dtype=np.float64),
             np.array([0.7, 1.5], dtype=np.float64)]
    model = MvExpRecursiveLogLik(times, 3.0, 1.5)
    assert model.n_dims == 2
    assert model.n_coeffs == 2 + 2 * 2  # μ + α
    assert model.n_total_jumps == 5


def test_mv_exp_recursive_input_validation() -> None:
    """Bad inputs raise ValueError with helpful messages."""
    # negative decay
    with pytest.raises(ValueError, match="decay must be positive"):
        MvExpRecursiveLogLik([np.array([0.5])], 1.0, 0.0)
    # event past horizon
    with pytest.raises(ValueError, match="outside"):
        MvExpRecursiveLogLik([np.array([0.5, 2.0])], 1.0, 1.0)
    # wrong coeffs length
    model = MvExpRecursiveLogLik([np.array([0.5])], 1.0, 1.5)
    with pytest.raises(ValueError, match="length mismatch"):
        out = np.zeros(model.n_coeffs)
        model.loss_and_grad(np.array([0.1, 0.2, 0.3]), out)  # too long
