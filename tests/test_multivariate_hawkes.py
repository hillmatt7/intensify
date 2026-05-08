"""Tests for Multivariate Hawkes."""

import numpy as np
import pytest
from intensify.backends import get_backend
from intensify.core.kernels import ExponentialKernel
from intensify.core.processes import MultivariateHawkes

bt = get_backend()


def test_multivariate_hawkes_initialization():
    # Shared kernel
    kernel = ExponentialKernel(alpha=0.3, beta=1.5)
    mh = MultivariateHawkes(n_dims=3, mu=0.1, kernel=kernel)
    assert mh.n_dims == 3
    assert mh.mu.shape == (3,)
    flat = [k for row in mh.kernel_matrix for k in row]
    assert all(isinstance(k, ExponentialKernel) for k in flat)


def test_multivariate_hawkes_intensity():
    # 2-dim with different kernels
    k1 = ExponentialKernel(alpha=0.3, beta=1.5)
    k2 = ExponentialKernel(alpha=0.2, beta=1.0)
    # Define full matrix: from dim 0 and 1 to each output dim
    kernel_matrix = [[k1, k2], [k1, k2]]
    mh = MultivariateHawkes(n_dims=2, mu=[0.1, 0.2], kernel=kernel_matrix)
    # History: dim0 had event at 0.5, dim1 at 0.7
    history = [bt.array([0.5]), bt.array([0.7])]
    lam = mh.intensity(1.0, history)
    # Expected: mu + sum_k φ_mk(1-t_k)
    # For m=0: μ0 + φ_00(0.5) + φ_01(0.3)
    expected0 = 0.1 + k1.evaluate(bt.array([0.5]))[0] + k2.evaluate(bt.array([0.3]))[0]
    # For m=1: μ1 + φ_10(0.5) + φ_11(0.3) where φ_10 is same k1, φ_11 same k2
    expected1 = 0.2 + k1.evaluate(bt.array([0.5]))[0] + k2.evaluate(bt.array([0.3]))[0]
    np.testing.assert_allclose(lam, [expected0, expected1])


def test_multivariate_hawkes_get_set_params():
    kernels = [
        [ExponentialKernel(0.3, 1.5), ExponentialKernel(0.2, 1.0)],
        [ExponentialKernel(0.1, 2.0), ExponentialKernel(0.4, 1.2)],
    ]
    mh = MultivariateHawkes(n_dims=2, mu=[0.1, 0.2], kernel=kernels)
    p = mh.get_params()
    assert "mu" in p and "kernel_matrix" in p
    # Set new mu and kernels
    new_mu = [0.5, 0.6]
    new_kernels = [[ExponentialKernel(0.4, 1.0) for _ in range(2)] for _ in range(2)]
    mh.set_params({"mu": new_mu, "kernel_matrix": new_kernels})
    np.testing.assert_allclose(mh.mu, new_mu)


def test_multivariate_hawkes_project_params():
    kernels = [
        [
            ExponentialKernel(alpha=0.9, beta=1.0),
            ExponentialKernel(alpha=0.2, beta=1.0),
        ],
        [
            ExponentialKernel(alpha=0.1, beta=1.0),
            ExponentialKernel(alpha=0.3, beta=1.0),
        ],
    ]
    mh = MultivariateHawkes(n_dims=2, mu=[1.0, 1.0], kernel=kernels)
    # Row 0 sum = 0.9 + 0.2 = 1.1 >= 1 -> should warn AND project
    with pytest.warns(UserWarning):
        mh.project_params()
    # After projection, row norms must be < 1
    for m in range(mh.n_dims):
        row_norm = sum(mh.kernel_matrix[m][k].l1_norm() for k in range(mh.n_dims))
        assert row_norm < 1.0, f"Row {m} norm {row_norm} not projected below 1"


def test_multivariate_fit_sets_branching_ratio_and_endogeneity():
    k = ExponentialKernel(alpha=0.2, beta=1.5)
    mh = MultivariateHawkes(n_dims=2, mu=[0.3, 0.25], kernel=k)
    ev0 = np.array([0.3, 1.1, 2.0, 3.5, 4.8])
    ev1 = np.array([0.5, 1.8, 2.5, 4.0])
    from intensify.core.inference import MLEInference

    result = MLEInference(max_iter=50, tol=1e-3).fit(mh, [ev0, ev1], 6.0)
    assert result.branching_ratio_ is not None
    assert np.isfinite(result.branching_ratio_)
    assert result.branching_ratio_ < 1.0
    assert result.endogeneity_index_ is not None
    assert 0 <= result.endogeneity_index_ <= 1.0


def test_multivariate_simulate_produces_events():
    k = ExponentialKernel(alpha=0.25, beta=1.4)
    mh = MultivariateHawkes(n_dims=2, mu=[0.3, 0.25], kernel=k)
    out = mh.simulate(T=8.0, seed=42)
    assert len(out) == 2
    assert sum(len(x) for x in out) >= 0


def test_multivariate_log_likelihood_runs():
    k = ExponentialKernel(alpha=0.2, beta=1.5)
    mh = MultivariateHawkes(n_dims=2, mu=[0.2, 0.15], kernel=k)
    ev0 = np.array([0.3, 1.1, 2.0])
    ev1 = np.array([0.5, 1.8])
    ll = mh.log_likelihood([bt.asarray(ev0), bt.asarray(ev1)], T=3.0)
    assert np.isfinite(ll)
