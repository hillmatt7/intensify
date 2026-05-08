"""Cross-validation: Rust marked_uni_exp Hawkes neg-log-likelihood matches
the existing Python `MarkedHawkes._loglik_exponential_recursive` reference
to ~1e-12 across many seeds and across all three builtin mark-influence
kinds (linear, log, power).
"""

from __future__ import annotations

import numpy as np
import pytest
from intensify._libintensify.likelihood import (
    marked_uni_exp_neg_ll,
    marked_uni_exp_neg_ll_with_grad,
)


def _sim_seed(seed: int, max_events: int = 600):
    """Simulate a marked exp Hawkes with linear influence."""
    from intensify.core.kernels.exponential import ExponentialKernel
    from intensify.core.processes.marked_hawkes import MarkedHawkes

    rng = np.random.default_rng(seed)
    mu = float(rng.uniform(0.1, 0.4))
    alpha = float(rng.uniform(0.1, 0.4))
    beta = float(rng.uniform(0.5, 2.5))
    T = float(rng.uniform(20.0, 50.0))
    proc = MarkedHawkes(
        mu=mu,
        kernel=ExponentialKernel(alpha=alpha, beta=beta),
        mark_influence="linear",
    )
    events, marks = proc.simulate(T=T, seed=seed)
    if len(events) == 0 or len(events) > max_events:
        return _sim_seed(seed + 7919, max_events)
    return (
        np.asarray(events, dtype=np.float64),
        np.asarray(marks, dtype=np.float64),
        T,
        mu,
        alpha,
        beta,
    )


def _g_values_for_kind(
    marks: np.ndarray, kind: str, mark_power: float = 1.0
) -> np.ndarray:
    """Replicate the Python wrapper's mark-influence evaluation for tests."""
    if kind == "linear":
        return marks.astype(np.float64).copy()
    if kind == "log":
        return np.log1p(marks)
    if kind == "power":
        return np.maximum(marks, 0.0) ** mark_power
    raise ValueError(kind)


@pytest.mark.parametrize("seed", range(15))
@pytest.mark.parametrize("influence", ["linear", "log", "power"])
def test_marked_uni_exp_matches_python(seed: int, influence: str) -> None:
    """Rust marked_uni_exp matches the Python reference to 1e-10."""
    from intensify.core.kernels.exponential import ExponentialKernel
    from intensify.core.processes.marked_hawkes import MarkedHawkes

    events, marks, T, mu, alpha, beta = _sim_seed(seed)
    kwargs = {"mark_influence": influence}
    if influence == "power":
        kwargs["mark_power"] = 0.7
    proc = MarkedHawkes(
        mu=mu,
        kernel=ExponentialKernel(alpha=alpha, beta=beta),
        **kwargs,
    )
    ref_log_lik = float(proc.log_likelihood(events, marks, T))
    g_vals = _g_values_for_kind(marks, influence, kwargs.get("mark_power", 1.0))
    rust_neg = float(marked_uni_exp_neg_ll(events, g_vals, T, mu, alpha, beta))
    assert abs(rust_neg - (-ref_log_lik)) < 1e-10, (
        f"seed={seed} influence={influence}: rust={rust_neg:.10f} ref={-ref_log_lik:.10f}"
    )


@pytest.mark.parametrize("seed", range(8))
def test_marked_uni_exp_grad_finite_difference(seed: int) -> None:
    """5-point stencil sanity on analytic gradient (linear influence)."""
    events, marks, T, mu, alpha, beta = _sim_seed(seed + 100)
    x0 = np.array([mu, alpha, beta], dtype=np.float64)
    g_vals = _g_values_for_kind(marks, "linear")

    _, grad_a = marked_uni_exp_neg_ll_with_grad(events, g_vals, T, mu, alpha, beta)
    grad_a = np.asarray(grad_a)

    h = 1e-6
    grad_n = np.zeros(3)
    for idx in range(3):
        bumps = []
        for delta in (-2 * h, -h, h, 2 * h):
            x = x0.copy()
            x[idx] += delta
            bumps.append(marked_uni_exp_neg_ll(events, g_vals, T, *x))
        f_m2, f_m, f_p, f_p2 = bumps
        grad_n[idx] = (-f_p2 + 8 * f_p - 8 * f_m + f_m2) / (12 * h)
    np.testing.assert_allclose(grad_a, grad_n, rtol=1e-5, atol=1e-9)


def test_marked_uni_exp_end_to_end_via_public_api() -> None:
    """Public API routes MarkedHawkes(ExponentialKernel) to Rust."""
    from intensify.core.inference import MLEInference
    from intensify.core.kernels.exponential import ExponentialKernel
    from intensify.core.processes.marked_hawkes import MarkedHawkes

    events, marks, T, _, _, _ = _sim_seed(42)
    proc = MarkedHawkes(
        mu=0.2,
        kernel=ExponentialKernel(alpha=0.1, beta=1.0),
        mark_influence="linear",
    )
    result = MLEInference(max_iter=200).fit(proc, (events, marks), T=T)
    assert result.convergence_info["backend"] == "rust"
    assert result.convergence_info["model"] == "marked_hawkes_exp"
    assert np.isfinite(result.log_likelihood)


def test_marked_uni_exp_callable_now_routes_to_rust() -> None:
    """Callable mark_influence is now supported via the Rust path. The
    Python wrapper evaluates g(m_j) once at the start, then the Rust hot
    loop runs without per-pair Python callbacks."""
    from intensify.core.inference import MLEInference
    from intensify.core.kernels.exponential import ExponentialKernel
    from intensify.core.processes.marked_hawkes import MarkedHawkes

    events, marks, T, _, _, _ = _sim_seed(99)
    proc = MarkedHawkes(
        mu=0.2,
        kernel=ExponentialKernel(alpha=0.1, beta=1.0),
        mark_influence=lambda m: float(m) ** 1.5,
    )
    result = MLEInference(max_iter=100).fit(proc, (events, marks), T=T)
    assert result.convergence_info["backend"] == "rust"
    assert result.convergence_info["model"] == "marked_hawkes_exp"
    assert np.isfinite(result.log_likelihood)


def test_marked_uni_exp_callable_matches_python_reference() -> None:
    """Cross-val: callable mark_influence Rust path matches Python at 1e-10."""
    from intensify.core.kernels.exponential import ExponentialKernel
    from intensify.core.processes.marked_hawkes import MarkedHawkes

    events, marks, T, mu, alpha, beta = _sim_seed(123)
    custom_g = lambda m: 0.5 * float(m) + 0.1 * float(m) ** 2  # noqa: E731
    proc = MarkedHawkes(
        mu=mu,
        kernel=ExponentialKernel(alpha=alpha, beta=beta),
        mark_influence=custom_g,
    )
    ref_log_lik = float(proc.log_likelihood(events, marks, T))

    g_vals = np.asarray([custom_g(m) for m in marks], dtype=np.float64)
    rust_neg = float(marked_uni_exp_neg_ll(events, g_vals, T, mu, alpha, beta))
    assert abs(rust_neg - (-ref_log_lik)) < 1e-10
