"""Cross-validation: Rust uni_powerlaw Hawkes neg-log-likelihood matches the
existing JAX/numpy reference (`_general_likelihood_numpy` for a
UnivariateHawkes with PowerLawKernel) to ~1e-12 across many seeds.
"""

from __future__ import annotations

import numpy as np
import pytest

from intensify._libintensify.kernels import PowerLawKernel as RustPowerLawKernel
from intensify._libintensify.likelihood import (
    uni_powerlaw_neg_ll,
    uni_powerlaw_neg_ll_with_grad,
)


def _sim_seed(seed: int, max_events: int = 600):
    """Sample a stationary uni-PowerLaw Hawkes via the existing simulator."""
    from intensify.core.kernels.power_law import PowerLawKernel
    from intensify.core.processes.hawkes import UnivariateHawkes

    rng = np.random.default_rng(seed)
    mu = float(rng.uniform(0.05, 0.4))
    alpha = float(rng.uniform(0.05, 0.3))
    beta = float(rng.uniform(0.5, 2.0))
    c = float(rng.uniform(0.2, 1.0))
    T = float(rng.uniform(20.0, 60.0))
    proc = UnivariateHawkes(
        mu=mu, kernel=PowerLawKernel(alpha=alpha, beta=beta, c=c),
    )
    events = proc.simulate(T=T, seed=seed)
    if len(events) == 0 or len(events) > max_events:
        return _sim_seed(seed + 7919, max_events)
    return np.asarray(events, dtype=np.float64), T, mu, alpha, beta, c


@pytest.mark.parametrize("seed", range(20))
def test_uni_powerlaw_matches_general_likelihood(seed: int) -> None:
    """Rust uni_powerlaw matches the existing _general_likelihood_numpy
    reference (which uses kernel.evaluate / integrate_vec) to 1e-12.
    """
    from intensify.core.inference.mle import _general_likelihood_numpy
    from intensify.core.kernels.power_law import PowerLawKernel
    from intensify.core.processes.hawkes import UnivariateHawkes

    events, T, mu, alpha, beta, c = _sim_seed(seed)
    proc = UnivariateHawkes(
        mu=mu, kernel=PowerLawKernel(alpha=alpha, beta=beta, c=c),
    )
    rust_neg = float(uni_powerlaw_neg_ll(events, T, mu, alpha, beta, c))
    ref_log_lik = float(_general_likelihood_numpy(proc, events, T))
    assert abs(rust_neg - (-ref_log_lik)) < 1e-10, (
        f"seed {seed}: rust={rust_neg:.10f} ref={-ref_log_lik:.10f}"
    )


@pytest.mark.parametrize("seed", range(15))
def test_uni_powerlaw_grad_finite_difference(seed: int) -> None:
    """5-point stencil sanity check on analytic gradient."""
    events, T, mu, alpha, beta, c = _sim_seed(seed + 200)
    x0 = np.array([mu, alpha, beta, c], dtype=np.float64)

    _, grad_a = uni_powerlaw_neg_ll_with_grad(events, T, mu, alpha, beta, c)
    grad_a = np.asarray(grad_a)

    h = 1e-6
    grad_n = np.zeros(4)
    for idx in range(4):
        bumps = []
        for delta in (-2*h, -h, h, 2*h):
            x = x0.copy()
            x[idx] += delta
            bumps.append(uni_powerlaw_neg_ll(events, T, *x))
        f_m2, f_m, f_p, f_p2 = bumps
        grad_n[idx] = (-f_p2 + 8*f_p - 8*f_m + f_m2) / (12*h)
    np.testing.assert_allclose(grad_a, grad_n, rtol=1e-5, atol=1e-9)


def test_uni_powerlaw_rust_kernel_imports_and_works() -> None:
    """Rust PowerLawKernel exposes the same Python API as the existing
    intensify ExponentialKernel: evaluate, integrate, integrate_vec,
    l1_norm, scale, has_recursive_form, repr."""
    k = RustPowerLawKernel(alpha=0.5, beta=0.8, c=0.3)
    assert k.alpha == 0.5
    assert k.beta == 0.8
    assert k.c == 0.3
    np.testing.assert_allclose(
        k.evaluate(np.array([0.0, 1.0, 2.0])),
        [0.5 * 0.3 ** -1.8, 0.5 * 1.3 ** -1.8, 0.5 * 2.3 ** -1.8],
    )
    np.testing.assert_allclose(
        k.integrate(2.0),
        (0.5 / 0.8) * (0.3 ** -0.8 - 2.3 ** -0.8),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        k.l1_norm(),
        (0.5 / 0.8) * 0.3 ** -0.8,
        rtol=1e-12,
    )
    assert k.has_recursive_form() is False
    k.scale(0.5)
    assert k.alpha == 0.25
    assert "PowerLawKernel" in repr(k)


def test_uni_powerlaw_end_to_end_via_public_api() -> None:
    """The MLE dispatch routes UnivariateHawkes(PowerLawKernel) through
    the Rust core post-Phase-3."""
    from intensify.core.inference import MLEInference
    from intensify.core.kernels.power_law import PowerLawKernel
    from intensify.core.processes.hawkes import UnivariateHawkes

    events, T, mu_true, alpha_true, beta_true, c_true = _sim_seed(42)
    proc = UnivariateHawkes(
        mu=0.2, kernel=PowerLawKernel(alpha=0.1, beta=1.0, c=0.5),
    )
    result = MLEInference(max_iter=500).fit(proc, events, T=T)
    assert result.convergence_info["backend"] == "rust"
    assert result.convergence_info["model"] == "univariate_hawkes_powerlaw"
    assert np.isfinite(result.log_likelihood)
