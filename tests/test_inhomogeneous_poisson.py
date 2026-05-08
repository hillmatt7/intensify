"""Tests for Inhomogeneous Poisson."""

import numpy as np
import pytest
from intensify.backends import get_backend
from intensify.core.processes import InhomogeneousPoisson

bt = get_backend()


def test_inhomogeneous_from_piecewise():
    rates = {0.0: 1.0, 2.0: 3.0, 5.0: 0.5}
    ipp = InhomogeneousPoisson(rates=rates)
    assert ipp.intensity(1.0, bt.array([])) == 1.0
    assert ipp.intensity(3.0, bt.array([])) == 3.0
    assert ipp.intensity(6.0, bt.array([])) == 0.5


def test_inhomogeneous_log_likelihood():
    rates = {0.0: 2.0, 3.0: 1.0}
    ipp = InhomogeneousPoisson(rates=rates)
    events = bt.array([0.5, 1.5, 3.5, 4.0])
    T = 5.0
    ll = ipp.log_likelihood(events, T)
    # sum log rates + integral: events at 0.5,1.5 under rate 2; 3.5,4.0 under rate 1.
    # sum_log = 2*log(2) + 2*log(1)=2*log2
    # integral: from 0-3: 2*3=6; 3-5: 1*2=2; total 8
    # ll = 2*log2 - 8
    expected = 2 * np.log(2.0) - 8.0
    assert np.isclose(ll, expected)


def test_inhomogeneous_simulate_piecewise():
    rates = {0.0: 5.0, 2.0: 10.0}
    ipp = InhomogeneousPoisson(rates=rates)
    events = ipp.simulate(T=5.0, seed=42)
    # Rate should be higher in second half
    assert len(events) > 0
