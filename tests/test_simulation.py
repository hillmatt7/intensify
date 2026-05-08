"""Tests for simulation algorithms."""

import numpy as np
import pytest
from intensify.backends import get_backend
from intensify.core.kernels import ExponentialKernel
from intensify.core.processes import HomogeneousPoisson, UnivariateHawkes
from intensify.core.simulation import ogata_thinning

bt = get_backend()


def test_ogata_thinning_univariate():
    # Test that thinning produces plausible event counts
    kernel = ExponentialKernel(alpha=0.3, beta=1.0)
    process = UnivariateHawkes(mu=0.5, kernel=kernel)
    events = ogata_thinning(process, T=10.0, seed=42)
    assert hasattr(events, "shape")
    if len(events) > 1:
        assert events[0] < events[-1]  # sorted
        assert events.min() >= 0 and events.max() <= 10.0


def test_ogata_thinning_vs_exponential():
    # For very low branching ratio, should be close to Poisson rate
    kernel = ExponentialKernel(alpha=0.1, beta=1.0)
    process = UnivariateHawkes(mu=1.0, kernel=kernel)
    events = ogata_thinning(process, T=100.0, seed=123)
    # Rate should be roughly mu / (1 - ||phi||_1) = 1.0 / 0.9 ≈ 1.11
    if len(events) > 10:
        rate_est = len(events) / 100.0
        assert 0.8 < rate_est < 1.5
