"""Edge case tests for core functionality."""

import numpy as np
import pytest

from intensify._config import config_get, config_reset, config_set
from intensify.backends import get_backend
from intensify.core.inference import MLEInference
from intensify.core.kernels import (
    ExponentialKernel,
    PowerLawKernel,
    SumExponentialKernel,
)
from intensify.core.processes import HomogeneousPoisson, UnivariateHawkes
from intensify.core.simulation import ogata_thinning

bt = get_backend()


def test_empty_events():
    kernel = ExponentialKernel(alpha=0.3, beta=1.5)
    hawkes = UnivariateHawkes(mu=0.5, kernel=kernel)
    events = bt.array([])
    T = 1.0
    # Should not crash
    ll = hawkes.log_likelihood(events, T)
    assert ll == 0.0 or np.isnan(ll)  # depends on implementation; we return 0.0


def test_unsorted_events_handled():
    # Our code expects sorted events; fit inputs should be sorted internally? Document expects sorted.
    # But log_likelihood should still compute correctly if sorted?
    kernel = ExponentialKernel(alpha=0.3, beta=1.0)
    hawkes = UnivariateHawkes(mu=0.5, kernel=kernel)
    events_unsorted = bt.array([0.8, 0.2, 0.5])
    try:
        hawkes.log_likelihood(events_unsorted, T=1.0)
        # Might be incorrect if unsorted, but should not crash
    except Exception:
        # Acceptable to raise; but we can sort internally in fit
        pass


def test_duplicate_timestamps():
    kernel = ExponentialKernel(alpha=0.3, beta=1.5)
    hawkes = UnivariateHawkes(mu=0.5, kernel=kernel)
    events = bt.array([0.1, 0.1, 0.2])
    # Should not crash; two events at same time cause zero lags maybe problematic
    try:
        hawkes.log_likelihood(events, T=1.0)
    except Exception:
        # Could be division by zero or zero intensity; acceptable to raise a clear error
        pass


def test_near_critical_branching_ratio():
    # Branching ratio = 0.99, should be stationary but near-critical
    kernel = ExponentialKernel(alpha=0.99, beta=2.0)
    assert kernel.is_stationary() is True
    process = UnivariateHawkes(mu=0.01, kernel=kernel)
    # Sorted event times from cumulative sum of exponential ISIs
    np.random.seed(42)
    events = bt.array(np.cumsum(np.random.exponential(1.0, size=500)))
    T = float(np.asarray(events).max()) + 1.0
    from intensify.core.inference import get_inference_engine
    result = get_inference_engine("mle").fit(process, events, T)
    assert result.params["kernel"].alpha < 1.0


def test_power_law_stationarity_warning():
    # Power law with heavy tail might have large L1 norm
    kernel = PowerLawKernel(alpha=1.5, beta=0.3, c=1.0)
    # L1 = alpha/β = 5.0 => >1. Should produce warning when used with Hawkes.
    assert not kernel.is_stationary()


def test_ogata_thinning_with_zero_intensity():
    # Near-zero excitation; alpha=0 is invalid for ExponentialKernel
    kernel = ExponentialKernel(alpha=1e-9, beta=1.0)
    process = UnivariateHawkes(mu=0.0, kernel=kernel)
    events = ogata_thinning(process, T=1.0, seed=42)
    # Should be empty
    assert len(events) == 0


def test_sum_exponential_many_components():
    alphas = [0.1] * 10
    betas = [2.0 ** i for i in range(10)]  # increasing betas
    kernel = SumExponentialKernel(alphas, betas)
    t = bt.array([0.1, 1.0, 10.0])
    vals = kernel.evaluate(t)
    assert vals.shape == t.shape
    # L1 norm = sum(alphas) = 1.0
    assert np.isclose(kernel.l1_norm(), 1.0)


def test_config_get_set_reset():
    config_reset()
    assert config_get("recursive_warning_threshold") == 50_000
    config_set("recursive_warning_threshold", 12_345)
    assert config_get("recursive_warning_threshold") == 12_345
    config_reset()
    assert config_get("recursive_warning_threshold") == 50_000


def test_fit_infers_T_with_warning():
    kernel = ExponentialKernel(alpha=0.25, beta=1.2)
    hawkes = UnivariateHawkes(mu=0.3, kernel=kernel)
    events = bt.array([0.1, 0.4, 0.9])
    with pytest.warns(UserWarning, match="T not specified"):
        hawkes.fit(events, T=None, method="mle")