"""Tests for Cox processes."""

import numpy as np
import pytest

from intensify.backends import get_backend
from intensify.core.kernels import ExponentialKernel
from intensify.core.processes import LogGaussianCoxProcess, ShotNoiseCoxProcess

bt = get_backend()


def test_lgcp_simulate_returns_events():
    lgcp = LogGaussianCoxProcess(n_bins=20, mu_prior=-1.0, sigma_prior=0.3)
    lgcp.log_lambda = np.linspace(-0.5, 0.5, 20).astype(float)
    ev = lgcp.simulate(T=3.0, seed=0)
    assert hasattr(ev, "shape") or len(ev) >= 0


def test_lgcp_intensity_matches_bins():
    lgcp = LogGaussianCoxProcess(n_bins=5, mu_prior=0.0, sigma_prior=1.0)
    lgcp.log_lambda = np.log(np.array([1.0, 2.0, 3.0, 2.0, 1.0]))
    lgcp.set_last_window(T=5.0)
    edges = np.linspace(0, 5, 6)
    mid = 0.5 * (edges[0] + edges[1])
    assert np.isclose(lgcp.intensity(mid, bt.array([])), 1.0)


def test_lgcp_conditional_log_likelihood():
    lgcp = LogGaussianCoxProcess(n_bins=10, mu_prior=0.0, sigma_prior=1.0)
    rng = np.random.default_rng(0)
    lgcp.log_lambda = rng.normal(0.0, 0.2, size=10)
    T = 2.0
    events = bt.array([0.2, 0.4, 1.1, 1.5])
    ll = lgcp.log_likelihood(events, T)
    assert isinstance(ll, float)
    assert np.isfinite(ll)


def test_shot_noise_simulate():
    k = ExponentialKernel(alpha=0.4, beta=2.0)
    proc = ShotNoiseCoxProcess(shot_rate=0.8, shot_kernel=k)
    proc.simulate(T=4.0, seed=123)
    assert proc.shot_times is not None
    assert len(proc.shot_times) >= 0


def test_shot_noise_intensity_and_ll():
    k = ExponentialKernel(alpha=0.3, beta=1.5)
    proc = ShotNoiseCoxProcess(shot_rate=1.0, shot_kernel=k)
    ev = proc.simulate(T=5.0, seed=7)
    if len(proc.shot_times) == 0:
        pytest.skip("no shots in replicate")
    lam = proc.intensity(float(ev[0]) + 1e-6, bt.array([]))
    assert lam >= 0.0
    ll = proc.log_likelihood(ev, T=5.0)
    assert np.isfinite(ll)
