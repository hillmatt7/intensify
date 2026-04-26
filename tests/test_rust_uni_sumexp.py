"""Cross-validation: Rust uni_sumexp Hawkes neg-log-likelihood matches the
existing JAX/numpy reference (`_recursive_likelihood_numpy` for a
UnivariateHawkes with SumExponentialKernel) to ~1e-12 across many seeds.
"""

from __future__ import annotations

import numpy as np
import pytest

from intensify._libintensify.likelihood import (
    uni_sumexp_neg_ll,
    uni_sumexp_neg_ll_with_grad,
)


def _sim_seed(seed: int, n_components: int = 2, max_events: int = 600):
    """Sample a stationary uni-SumExp Hawkes."""
    from intensify.core.kernels.sum_exponential import SumExponentialKernel
    from intensify.core.processes.hawkes import UnivariateHawkes

    rng = np.random.default_rng(seed)
    mu = float(rng.uniform(0.05, 0.4))
    alphas = rng.uniform(0.05, 0.3, size=n_components)
    # Constrain L1 norm < 1 for stationarity
    alphas = alphas * (0.7 / max(alphas.sum(), 1e-8))
    betas = rng.uniform(0.5, 4.0, size=n_components)
    T = float(rng.uniform(20.0, 60.0))
    proc = UnivariateHawkes(
        mu=mu,
        kernel=SumExponentialKernel(alphas=alphas.tolist(), betas=betas.tolist()),
    )
    events = proc.simulate(T=T, seed=seed)
    if len(events) == 0 or len(events) > max_events:
        return _sim_seed(seed + 7919, n_components, max_events)
    return np.asarray(events, dtype=np.float64), T, mu, alphas, betas


@pytest.mark.parametrize("seed", range(15))
@pytest.mark.parametrize("K", [2, 3, 5])
def test_uni_sumexp_matches_recursive_likelihood(seed: int, K: int) -> None:
    """Rust uni_sumexp matches `_recursive_likelihood_numpy` to 1e-10."""
    from intensify.core.inference.mle import _recursive_likelihood_numpy
    from intensify.core.kernels.sum_exponential import SumExponentialKernel
    from intensify.core.processes.hawkes import UnivariateHawkes

    events, T, mu, alphas, betas = _sim_seed(seed * 13 + K, n_components=K)
    proc = UnivariateHawkes(
        mu=mu,
        kernel=SumExponentialKernel(alphas=alphas.tolist(), betas=betas.tolist()),
    )
    rust_neg = float(uni_sumexp_neg_ll(events, T, mu, alphas, betas))
    ref_log_lik = float(_recursive_likelihood_numpy(proc, events, T))
    assert abs(rust_neg - (-ref_log_lik)) < 1e-10, (
        f"seed={seed} K={K}: rust={rust_neg:.10f} ref={-ref_log_lik:.10f}"
    )


@pytest.mark.parametrize("seed", range(8))
def test_uni_sumexp_grad_finite_difference(seed: int) -> None:
    """5-point stencil sanity check on analytic gradient."""
    events, T, mu, alphas, betas = _sim_seed(seed + 200, n_components=3)
    K = len(alphas)
    n_params = 1 + 2 * K

    val, gmu_a, ga_a, gb_a = uni_sumexp_neg_ll_with_grad(events, T, mu, alphas, betas)
    grad_a = np.concatenate([[gmu_a], np.asarray(ga_a), np.asarray(gb_a)])

    h = 1e-6
    grad_n = np.zeros(n_params)
    for idx in range(n_params):
        bumps = []
        for delta in (-2*h, -h, h, 2*h):
            mu_x = mu
            a_x = alphas.copy()
            b_x = betas.copy()
            if idx == 0:
                mu_x = mu + delta
            elif idx <= K:
                a_x[idx - 1] += delta
            else:
                b_x[idx - 1 - K] += delta
            bumps.append(uni_sumexp_neg_ll(events, T, mu_x, a_x, b_x))
        f_m2, f_m, f_p, f_p2 = bumps
        grad_n[idx] = (-f_p2 + 8 * f_p - 8 * f_m + f_m2) / (12 * h)

    np.testing.assert_allclose(grad_a, grad_n, rtol=1e-5, atol=1e-9)


def test_uni_sumexp_end_to_end_via_public_api() -> None:
    """Public API routes UnivariateHawkes(SumExponentialKernel) to Rust."""
    from intensify.core.inference import MLEInference
    from intensify.core.kernels.sum_exponential import SumExponentialKernel
    from intensify.core.processes.hawkes import UnivariateHawkes

    events, T, _, alphas, betas = _sim_seed(42, n_components=2)
    proc = UnivariateHawkes(
        mu=0.2,
        kernel=SumExponentialKernel(alphas=[0.1, 0.05], betas=[1.0, 3.0]),
    )
    result = MLEInference(max_iter=300).fit(proc, events, T=T)
    assert result.convergence_info["backend"] == "rust"
    assert result.convergence_info["model"] == "univariate_hawkes_sumexp"
    assert np.isfinite(result.log_likelihood)
