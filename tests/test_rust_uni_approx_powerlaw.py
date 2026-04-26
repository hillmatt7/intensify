"""Cross-validation: Rust uni_approx_powerlaw matches the existing
`_general_likelihood_numpy` reference (which evaluates the kernel via its
Python `evaluate()` method and integrates) to ~1e-12 across many seeds.
"""

from __future__ import annotations

import numpy as np
import pytest

from intensify._libintensify.likelihood import (
    uni_approx_powerlaw_neg_ll,
    uni_approx_powerlaw_neg_ll_with_grad,
)


def _sim_seed(seed: int, n_components: int = 6, max_events: int = 600):
    """Sample a stationary uni-ApproxPL Hawkes."""
    from intensify.core.kernels.approx_power_law import ApproxPowerLawKernel
    from intensify.core.processes.hawkes import UnivariateHawkes

    rng = np.random.default_rng(seed)
    mu = float(rng.uniform(0.05, 0.4))
    alpha = float(rng.uniform(0.1, 0.4))
    beta_pow = float(rng.uniform(0.5, 2.0))
    beta_min = float(rng.uniform(0.2, 0.8))
    T = float(rng.uniform(20.0, 60.0))
    proc = UnivariateHawkes(
        mu=mu,
        kernel=ApproxPowerLawKernel(
            alpha=alpha, beta_pow=beta_pow, beta_min=beta_min,
            r=1.5, n_components=n_components,
        ),
    )
    events = proc.simulate(T=T, seed=seed)
    if len(events) == 0 or len(events) > max_events:
        return _sim_seed(seed + 7919, n_components, max_events)
    return (
        np.asarray(events, dtype=np.float64), T, mu, alpha, beta_pow, beta_min,
    )


@pytest.mark.parametrize("seed", range(15))
def test_uni_approx_pl_matches_general_likelihood(seed: int) -> None:
    """Rust uni_approx_powerlaw matches `_general_likelihood_numpy` to 1e-10."""
    from intensify.core.inference.mle import _general_likelihood_numpy
    from intensify.core.kernels.approx_power_law import ApproxPowerLawKernel
    from intensify.core.processes.hawkes import UnivariateHawkes

    events, T, mu, alpha, beta_pow, beta_min = _sim_seed(seed)
    proc = UnivariateHawkes(
        mu=mu,
        kernel=ApproxPowerLawKernel(
            alpha=alpha, beta_pow=beta_pow, beta_min=beta_min,
            r=1.5, n_components=6,
        ),
    )
    rust_neg = float(uni_approx_powerlaw_neg_ll(
        events, T, mu, alpha, beta_pow, beta_min, 1.5, 6,
    ))
    ref_log_lik = float(_general_likelihood_numpy(proc, events, T))
    assert abs(rust_neg - (-ref_log_lik)) < 1e-9, (
        f"seed={seed}: rust={rust_neg:.10f} ref={-ref_log_lik:.10f}"
    )


@pytest.mark.parametrize("seed", range(8))
def test_uni_approx_pl_grad_finite_difference(seed: int) -> None:
    """5-point stencil sanity check (4 free parameters: μ, α, β_pow, β_min)."""
    events, T, mu, alpha, beta_pow, beta_min = _sim_seed(seed + 100)
    x0 = np.array([mu, alpha, beta_pow, beta_min], dtype=np.float64)

    _, grad_a = uni_approx_powerlaw_neg_ll_with_grad(
        events, T, mu, alpha, beta_pow, beta_min, 1.5, 6,
    )
    grad_a = np.asarray(grad_a)

    h = 1e-6
    grad_n = np.zeros(4)
    for idx in range(4):
        bumps = []
        for delta in (-2*h, -h, h, 2*h):
            x = x0.copy()
            x[idx] += delta
            bumps.append(uni_approx_powerlaw_neg_ll(events, T, *x, 1.5, 6))
        f_m2, f_m, f_p, f_p2 = bumps
        grad_n[idx] = (-f_p2 + 8 * f_p - 8 * f_m + f_m2) / (12 * h)
    np.testing.assert_allclose(grad_a, grad_n, rtol=1e-5, atol=1e-9)


def test_uni_approx_pl_end_to_end_via_public_api() -> None:
    """Public API routes UnivariateHawkes(ApproxPowerLawKernel) to Rust."""
    from intensify.core.inference import MLEInference
    from intensify.core.kernels.approx_power_law import ApproxPowerLawKernel
    from intensify.core.processes.hawkes import UnivariateHawkes

    events, T, _, _, _, _ = _sim_seed(42)
    proc = UnivariateHawkes(
        mu=0.2,
        kernel=ApproxPowerLawKernel(
            alpha=0.2, beta_pow=1.0, beta_min=0.5,
            r=1.5, n_components=6,
        ),
    )
    result = MLEInference(max_iter=200).fit(proc, events, T=T)
    assert result.convergence_info["backend"] == "rust"
    assert result.convergence_info["model"] == "univariate_hawkes_approx_powerlaw"
    assert np.isfinite(result.log_likelihood)
