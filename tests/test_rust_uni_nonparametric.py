"""Cross-validation: Rust uni_nonparametric Hawkes neg-log-likelihood matches
the existing `_general_likelihood_numpy` reference to ~1e-12 across many
seeds.

Also stress-tests at N=500 — the existing Python path was effectively
unusable here (ISSUES.md #8: ~7 minutes for N=300, killed at N=500). The
Rust path completes in milliseconds.
"""

from __future__ import annotations

import numpy as np
import pytest

from intensify._libintensify.kernels import NonparametricKernel as RustNonparametricKernel
from intensify._libintensify.likelihood import (
    uni_nonparametric_neg_ll,
    uni_nonparametric_neg_ll_with_grad,
)


def _gen_seed(seed: int, n_target: int = 100, n_bins: int = 5):
    """Sample a small uni-Hawkes via a Poisson-noise proxy + nonparametric
    kernel. We don't need stationary self-excitation here, just a non-trivial
    timestamp distribution to test the likelihood."""
    rng = np.random.default_rng(seed)
    T = float(rng.uniform(20.0, 60.0))
    # Sparse Poisson events
    events = np.sort(rng.uniform(0.0, T, size=n_target))
    mu = float(rng.uniform(0.05, 0.4))
    # Edges: linspace 0 to T/4
    edges = np.linspace(0.0, T / 4, n_bins + 1)
    values = rng.uniform(0.0, 0.3, size=n_bins)
    return events, T, mu, edges, values


@pytest.mark.parametrize("seed", range(15))
def test_uni_nonparametric_matches_general_likelihood(seed: int) -> None:
    """Rust uni_nonparametric matches `_general_likelihood_numpy` to 1e-10."""
    from intensify.core.inference.mle import _general_likelihood_numpy
    from intensify.core.kernels.nonparametric import NonparametricKernel
    from intensify.core.processes.hawkes import UnivariateHawkes

    events, T, mu, edges, values = _gen_seed(seed)
    proc = UnivariateHawkes(
        mu=mu, kernel=NonparametricKernel(edges=edges.tolist(), values=values.tolist()),
    )
    rust_neg = float(uni_nonparametric_neg_ll(events, T, mu, edges, values))
    ref_log_lik = float(_general_likelihood_numpy(proc, events, T))
    assert abs(rust_neg - (-ref_log_lik)) < 1e-10, (
        f"seed {seed}: rust={rust_neg:.10f} ref={-ref_log_lik:.10f}"
    )


@pytest.mark.parametrize("seed", range(10))
def test_uni_nonparametric_grad_finite_difference(seed: int) -> None:
    """5-point stencil sanity check: analytic gradient matches finite-diff."""
    events, T, mu, edges, values = _gen_seed(seed + 100, n_bins=4)
    n_params = 1 + len(values)

    _, gmu_a, gv_a = uni_nonparametric_neg_ll_with_grad(events, T, mu, edges, values)
    grad_a = np.concatenate([[gmu_a], np.asarray(gv_a)])

    h = 1e-6
    grad_n = np.zeros(n_params)
    for idx in range(n_params):
        bumps = []
        for delta in (-2*h, -h, h, 2*h):
            mu_x = mu + (delta if idx == 0 else 0.0)
            v_x = values.copy()
            if idx > 0:
                v_x[idx - 1] += delta
            bumps.append(uni_nonparametric_neg_ll(events, T, mu_x, edges, v_x))
        f_m2, f_m, f_p, f_p2 = bumps
        grad_n[idx] = (-f_p2 + 8 * f_p - 8 * f_m + f_m2) / (12 * h)
    np.testing.assert_allclose(grad_a, grad_n, rtol=1e-5, atol=1e-8)


def test_uni_nonparametric_rust_kernel_api() -> None:
    """Rust NonparametricKernel exposes expected Python API."""
    edges = np.array([0.0, 1.0, 2.0, 4.0])
    values = np.array([0.5, 0.3, 0.1])
    k = RustNonparametricKernel(edges, values)
    assert k.n_bins == 3
    np.testing.assert_allclose(k.edges, edges)
    np.testing.assert_allclose(k.values, values)
    np.testing.assert_allclose(
        k.evaluate(np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.5, 4.0, 10.0])),
        [0.5, 0.5, 0.3, 0.3, 0.1, 0.1, 0.0, 0.0],
    )
    # ∫_0^3 = 0.5*1 + 0.3*1 + 0.1*1 = 0.9
    np.testing.assert_allclose(k.integrate(3.0), 0.9, rtol=1e-12)
    np.testing.assert_allclose(k.l1_norm(), 0.5 + 0.3 + 0.2, rtol=1e-12)
    assert k.has_recursive_form() is False
    k.scale(0.5)
    np.testing.assert_allclose(k.values, values / 2)


def test_uni_nonparametric_at_n500_completes_quickly() -> None:
    """N=500 used to take >7 min in the Python JAX path (ISSUES.md #8).
    Rust should handle this in well under a second."""
    import time

    rng = np.random.default_rng(2026)
    T = 100.0
    events = np.sort(rng.uniform(0.0, T, size=500))
    edges = np.linspace(0.0, 5.0, 8)
    values = rng.uniform(0.05, 0.2, size=7)
    mu = 0.3

    t0 = time.perf_counter()
    val = uni_nonparametric_neg_ll(events, T, mu, edges, values)
    elapsed = time.perf_counter() - t0
    assert np.isfinite(val)
    assert elapsed < 1.0, f"N=500 took {elapsed:.2f}s — should be < 1s"


def test_uni_nonparametric_end_to_end_via_public_api() -> None:
    """Public API routes UnivariateHawkes(NonparametricKernel) to Rust."""
    from intensify.core.inference import MLEInference
    from intensify.core.kernels.nonparametric import NonparametricKernel
    from intensify.core.processes.hawkes import UnivariateHawkes

    events, T, _, edges, _ = _gen_seed(7, n_target=200, n_bins=5)
    proc = UnivariateHawkes(
        mu=0.2,
        kernel=NonparametricKernel(
            edges=edges.tolist(),
            values=[0.1] * 5,
        ),
    )
    result = MLEInference(max_iter=200).fit(proc, events, T=T)
    assert result.convergence_info["backend"] == "rust"
    assert result.convergence_info["model"] == "univariate_hawkes_nonparametric"
    assert np.isfinite(result.log_likelihood)
