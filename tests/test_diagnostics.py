"""Diagnostics and goodness-of-fit tests."""

import numpy as np
import pytest
from intensify.backends import get_backend
from intensify.core.diagnostics import (
    pearson_residuals,
    raw_residuals,
    time_rescaling_test,
)
from intensify.core.diagnostics.goodness_of_fit import qq_plot, residual_intensity_plot
from intensify.core.inference import FitResult, MLEInference
from intensify.core.kernels import ExponentialKernel
from intensify.core.processes import UnivariateHawkes

bt = get_backend()


@pytest.fixture
def fitted_hawkes_result():
    np.random.seed(42)
    kernel = ExponentialKernel(alpha=0.25, beta=1.2)
    proc = UnivariateHawkes(mu=0.35, kernel=kernel)
    events = proc.simulate(T=10.0, seed=99)
    proc2 = UnivariateHawkes(mu=0.3, kernel=ExponentialKernel(alpha=0.2, beta=1.0))
    return MLEInference(max_iter=200, tol=1e-4).fit(proc2, events, 10.0)


def test_time_rescaling_returns_stats(fitted_hawkes_result):
    r = fitted_hawkes_result
    stat, p = time_rescaling_test(r)
    assert np.isfinite(stat) and np.isfinite(p)


def test_time_rescaling_correct_model_does_not_reject():
    """Under the true model, the KS test should not reject at alpha=0.01."""
    np.random.seed(123)
    kernel = ExponentialKernel(alpha=0.3, beta=2.0)
    proc = UnivariateHawkes(mu=0.5, kernel=kernel)
    events = proc.simulate(T=50.0, seed=123)
    if len(events) < 10:
        pytest.skip("Too few events for meaningful test")
    result = FitResult(
        params=proc.get_params(),
        log_likelihood=0.0,
    )
    result.process = proc
    result.events = events
    result.T = 50.0
    stat, p = time_rescaling_test(result)
    assert p > 0.01, f"KS test rejected correct model with p={p:.4f}"


def test_qq_plot_returns_figure(fitted_hawkes_result):
    fig = qq_plot(fitted_hawkes_result)
    assert fig is not None
    assert len(fig.axes) >= 1


def test_residual_intensity_plot_smoke(fitted_hawkes_result):
    fig = residual_intensity_plot(fitted_hawkes_result)
    assert fig is not None


def test_raw_residuals_shape():
    events = np.array([0.2, 0.8, 1.5])
    r = raw_residuals(events, 2.0, lambda t: 1.0 + 0.5 * float(t))
    assert len(r) == 3


def test_pearson_residuals_shape():
    events = np.array([0.1, 0.6, 1.2])
    T = 2.0
    r = pearson_residuals(events, T, lambda t: 2.0)
    assert len(r) == 3
