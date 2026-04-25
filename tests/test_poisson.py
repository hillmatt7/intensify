"""Tests for Poisson process models."""

import numpy as np
import pytest

from intensify.backends import get_backend, set_backend
from intensify.core.processes import HomogeneousPoisson

bt = get_backend()


def test_homogeneous_poisson_initialization():
    p = HomogeneousPoisson(rate=2.5)
    assert p.rate == 2.5
    with pytest.raises(ValueError):
        HomogeneousPoisson(rate=-1.0)


def test_homogeneous_poisson_intensity():
    p = HomogeneousPoisson(rate=3.0)
    assert p.intensity(0.5, bt.array([])) == 3.0


def test_homogeneous_poisson_log_likelihood():
    p = HomogeneousPoisson(rate=2.0)
    events = bt.array([0.1, 0.7, 1.3, 2.5])
    T = 3.0
    ll = p.log_likelihood(events, T)
    # Expected: n*log(rate) - rate*T = 4*log(2) - 2*3
    expected = 4 * np.log(2.0) - 2.0 * 3.0
    assert np.isclose(ll, expected)


def test_homogeneous_poisson_fit_mle():
    p = HomogeneousPoisson()
    np.random.seed(42)
    true_rate = 4.0
    gaps = np.random.exponential(1.0 / true_rate, size=200)
    events = np.cumsum(gaps)
    T = float(events.max()) + 1.0
    result = p.fit(events, T)
    assert p.rate == result.params["rate"]
    assert np.abs(p.rate - true_rate) < 0.5  # should be close
    assert result.log_likelihood > 0
    assert "rate" in result.std_errors


def test_homogeneous_poisson_simulate():
    p = HomogeneousPoisson(rate=5.0)
    events = p.simulate(T=10.0, seed=123)
    assert hasattr(events, "shape")
    assert len(events) >= 0
    assert bt.all(events >= 0) and bt.all(events <= 10.0)
    # Rate should be roughly 5
    if len(events) > 0:
        estimated_rate = len(events) / 10.0
        assert np.abs(estimated_rate - 5.0) < 2.0  # rough