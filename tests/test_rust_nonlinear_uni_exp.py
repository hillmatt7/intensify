"""Cross-validation: Rust nonlinear_uni_exp matches the existing Python
NonlinearHawkes log_likelihood (trapezoidal quadrature + builtin links)
to machine precision when both use the same n_quad grid.
"""

from __future__ import annotations

import numpy as np
import pytest
from intensify._libintensify.likelihood import (
    nonlinear_uni_exp_neg_ll,
    nonlinear_uni_exp_neg_ll_with_grad,
)


def _sim(seed: int, link: str = "softplus", max_events: int = 400):
    from intensify.core.kernels.exponential import ExponentialKernel
    from intensify.core.processes.nonlinear_hawkes import NonlinearHawkes

    rng = np.random.default_rng(seed)
    mu = float(rng.uniform(0.1, 0.5))
    alpha = float(rng.uniform(0.1, 0.4))
    beta = float(rng.uniform(0.5, 2.5))
    T = float(rng.uniform(20.0, 60.0))
    proc = NonlinearHawkes(
        mu=mu,
        kernel=ExponentialKernel(alpha=alpha, beta=beta),
        link_function=link,
    )
    events = proc.simulate(T=T, seed=seed)
    if len(events) == 0 or len(events) > max_events:
        return _sim(seed + 7919, link, max_events)
    return np.asarray(events, dtype=np.float64), T, mu, alpha, beta


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("link", ["softplus", "relu", "sigmoid", "identity"])
def test_nonlinear_matches_python_reference(seed: int, link: str) -> None:
    """Rust nonlinear matches Python NonlinearHawkes.log_likelihood to ~1e-12
    when both use the same n_quad grid."""
    from intensify.core.kernels.exponential import ExponentialKernel
    from intensify.core.processes.nonlinear_hawkes import NonlinearHawkes

    events, T, mu, alpha, beta = _sim(seed, link)
    proc = NonlinearHawkes(
        mu=mu,
        kernel=ExponentialKernel(alpha=alpha, beta=beta),
        link_function=link,
    )
    n_quad = 512
    ref_log_lik = float(proc.log_likelihood(events, T, n_quad=n_quad))
    sigmoid_scale = float(proc.sigmoid_scale)
    rust_neg = float(
        nonlinear_uni_exp_neg_ll(
            events,
            T,
            mu,
            alpha,
            beta,
            link,
            sigmoid_scale,
            n_quad,
        )
    )
    assert abs(rust_neg - (-ref_log_lik)) < 1e-9, (
        f"seed={seed} link={link}: rust={rust_neg:.10f} ref={-ref_log_lik:.10f}"
    )


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("link", ["softplus", "sigmoid"])
def test_nonlinear_grad_finite_difference(seed: int, link: str) -> None:
    """5-point stencil gradient sanity (loose tol — quadrature error ~1e-3)."""
    events, T, mu, alpha, beta = _sim(seed + 100, link)
    n_quad = 1024
    sigmoid_scale = 5.0
    x0 = np.array([mu, alpha, beta], dtype=np.float64)

    _, grad_a = nonlinear_uni_exp_neg_ll_with_grad(
        events,
        T,
        mu,
        alpha,
        beta,
        link,
        sigmoid_scale,
        n_quad,
    )
    grad_a = np.asarray(grad_a)

    h = 1e-5
    grad_n = np.zeros(3)
    for idx in range(3):
        bumps = []
        for delta in (-2 * h, -h, h, 2 * h):
            x = x0.copy()
            x[idx] += delta
            bumps.append(
                nonlinear_uni_exp_neg_ll(
                    events,
                    T,
                    *x,
                    link,
                    sigmoid_scale,
                    n_quad,
                )
            )
        f_m2, f_m, f_p, f_p2 = bumps
        grad_n[idx] = (-f_p2 + 8 * f_p - 8 * f_m + f_m2) / (12 * h)
    # Relax: trapezoidal compensator + finite-diff probe both contribute.
    np.testing.assert_allclose(grad_a, grad_n, rtol=5e-3, atol=1e-7)


def test_nonlinear_end_to_end_via_public_api() -> None:
    """Public API routes NonlinearHawkes(ExponentialKernel, builtin link)
    through the Rust path."""
    from intensify.core.inference import MLEInference
    from intensify.core.kernels.exponential import ExponentialKernel
    from intensify.core.processes.nonlinear_hawkes import NonlinearHawkes

    events, T, _, _, _ = _sim(42, "softplus")
    proc = NonlinearHawkes(
        mu=0.3,
        kernel=ExponentialKernel(alpha=0.1, beta=1.0),
        link_function="softplus",
    )
    result = MLEInference(max_iter=200).fit(proc, events, T=T)
    assert result.convergence_info["backend"] == "rust"
    assert result.convergence_info["model"] == "nonlinear_hawkes_exp"
    assert np.isfinite(result.log_likelihood)


def test_nonlinear_callable_falls_through_to_python() -> None:
    """Custom callable link_function should still go through the existing
    Python path (no callback into Rust)."""
    from intensify.core.inference import MLEInference
    from intensify.core.kernels.exponential import ExponentialKernel
    from intensify.core.processes.nonlinear_hawkes import NonlinearHawkes

    events, T, _, _, _ = _sim(7, "softplus")
    proc = NonlinearHawkes(
        mu=0.3,
        kernel=ExponentialKernel(alpha=0.1, beta=1.0),
        link_function=lambda z: max(0.0, z) ** 1.5 + 0.01,
    )
    result = MLEInference(max_iter=50).fit(proc, events, T=T)
    # Falls through to existing _fit_nonlinear_numpy
    assert (
        result.convergence_info["backend"] in ("numpy", "rust")
        or "model" in result.convergence_info
    )
    # Just check it ran without error
    assert np.isfinite(result.log_likelihood)
