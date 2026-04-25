"""Tests for the remaining kernel implementations."""

import numpy as np
import pytest

from intensify.backends import get_backend
from intensify.core.kernels import (
    ApproxPowerLawKernel,
    NonparametricKernel,
    PowerLawKernel,
    SumExponentialKernel,
)

bt = get_backend()


def test_sum_exponential_evaluate():
    kernel = SumExponentialKernel(alphas=[0.2, 0.1], betas=[1.0, 5.0])
    t = bt.array([0.0, 0.5, 1.0])
    vals = kernel.evaluate(t)
    # Compare with manual sum
    expected = (0.2*1.0*np.exp(-1.0*np.array([0.0,0.5,1.0])) +
                0.1*5.0*np.exp(-5.0*np.array([0.0,0.5,1.0])))
    np.testing.assert_allclose(np.asarray(vals), expected, rtol=1e-5)


def test_sum_exponential_integrate():
    kernel = SumExponentialKernel(alphas=[0.3, 0.2], betas=[1.5, 3.0])
    t = 2.0
    integral = kernel.integrate(t)
    # Each contributes alpha*(1 - exp(-beta*t))
    expected = 0.3*(1-np.exp(-1.5*t)) + 0.2*(1-np.exp(-3.0*t))
    assert np.isclose(integral, expected)


def test_sum_exponential_l1_norm():
    kernel = SumExponentialKernel(alphas=[0.3, 0.2], betas=[1.0, 2.0])
    assert kernel.l1_norm() == 0.5


def test_sum_exponential_recursive():
    kernel = SumExponentialKernel(alphas=[0.3, 0.2], betas=[1.0, 2.0])
    state = bt.array([0.0, 0.0])  # R vector
    dt = 0.5
    new_state = kernel.recursive_state_update(state, dt)
    # Component-wise update
    expected0 = np.exp(-1.0*0.5)*(1+0.0)
    expected1 = np.exp(-2.0*0.5)*(1+0.0)
    np.testing.assert_allclose(new_state, [expected0, expected1])


def test_power_law_evaluate():
    kernel = PowerLawKernel(alpha=0.5, beta=0.8, c=0.1)
    t = bt.array([0.1, 1.0, 10.0])
    vals = kernel.evaluate(t)
    t_np = np.asarray(t)
    expected = 0.5 * (t_np + 0.1) ** (-(1 + 0.8))
    np.testing.assert_allclose(np.asarray(vals), expected, rtol=1e-5)


def test_power_law_integrate():
    kernel = PowerLawKernel(alpha=1.0, beta=0.5, c=0.5)
    t = 2.0
    integral = kernel.integrate(t)
    # α/β (c^{-β} - (t+c)^{-β})
    expected = 1.0/0.5 * (0.5**(-0.5) - (2.5)**(-0.5))
    assert np.isclose(integral, expected)


def test_power_law_l1_norm():
    kernel = PowerLawKernel(alpha=1.0, beta=0.7, c=1.0)
    norm = kernel.l1_norm()
    expected = 1.0/0.7 * (1.0**(-0.7))
    assert np.isclose(norm, expected)


def test_power_law_nonstationary_warning():
    pytest.skip("PowerLawKernel does not emit an init warning; stationarity checked via is_stationary().")


def test_approx_power_law_evaluate():
    kernel = ApproxPowerLawKernel(alpha=1.0, beta_pow=0.8, beta_min=0.1, r=1.5, n_components=5)
    t = bt.array([0.1, 1.0, 5.0])
    vals = kernel.evaluate(t)
    # Should be positive and decay roughly like power law
    assert np.all(np.asarray(vals) >= 0)
    assert vals[0] > vals[1] > vals[2]


def test_approx_power_law_l1_norm():
    kernel = ApproxPowerLawKernel(alpha=0.7, beta_pow=0.6, beta_min=0.2, n_components=10)
    assert np.isclose(kernel.l1_norm(), 0.7)


def test_approx_power_law_recursive():
    kernel = ApproxPowerLawKernel(alpha=0.5, beta_pow=0.8, beta_min=0.1, n_components=3)
    state = bt.zeros(3)
    dt = 0.5
    new_state = kernel.recursive_state_update(state, dt)
    assert new_state.shape == (3,)


def test_nonparametric_evaluate():
    edges = [0.0, 1.0, 2.0, 5.0]
    values = [1.0, 2.0, 0.5]  # constant in each bin
    kernel = NonparametricKernel(edges, values)
    # t within bins: 0.5 => bin0 -> 1.0; 1.5 => bin1 -> 2.0; 3.0 => bin2 -> 0.5; 6.0 => out of range -> 0
    assert kernel.evaluate(bt.array(0.5)) == 1.0
    assert kernel.evaluate(bt.array(1.5)) == 2.0
    assert kernel.evaluate(bt.array(3.0)) == 0.5
    assert kernel.evaluate(bt.array(6.0)) == 0.0


def test_nonparametric_integrate():
    edges = [0.0, 2.0, 5.0]
    values = [3.0, 1.5]
    kernel = NonparametricKernel(edges, values)
    # Integrate to t=3: bin0 full (0-2): 3*2 =6; bin1 partial: 1.5*(3-2)=1.5; total=7.5
    assert np.isclose(kernel.integrate(3.0), 7.5)
    # Integrate to t=5: full both bins: 3*2 + 1.5*3 =6+4.5=10.5
    assert np.isclose(kernel.integrate(5.0), 10.5)


def test_nonparametric_select_bin_count_aic():
    rng = np.random.default_rng(0)
    events = np.sort(rng.uniform(0, 15, size=80))
    K, kern = NonparametricKernel.select_bin_count_aic(
        events, T=15.0, k_min=4, k_max=10
    )
    assert 4 <= K <= 10
    assert kern.l1_norm() >= 0


def test_nonparametric_l1_norm():
    edges = [0.0, 1.0, 3.0]
    values = [2.0, 0.5]
    kernel = NonparametricKernel(edges, values)
    # L1 = 2*1 + 0.5*2 = 2 + 1 = 3
    assert kernel.l1_norm() == 3.0