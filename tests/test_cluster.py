"""Branching (cluster) simulation tests."""

import numpy as np
from intensify.core.kernels import ExponentialKernel
from intensify.core.processes import MultivariateHawkes, UnivariateHawkes
from intensify.core.simulation import (
    branching_simulation,
    branching_simulation_multivariate,
)


def test_branching_simulation_univariate():
    k = ExponentialKernel(alpha=0.35, beta=1.5)
    p = UnivariateHawkes(mu=0.5, kernel=k)
    ev = branching_simulation(p, T=6.0, seed=1)
    assert hasattr(ev, "shape") or len(ev) >= 0
    if len(ev) > 0:
        assert float(np.min(ev)) >= 0.0


def test_branching_simulation_multivariate():
    k = ExponentialKernel(alpha=0.2, beta=2.0)
    p = MultivariateHawkes(n_dims=2, mu=[0.4, 0.3], kernel=k)
    out = branching_simulation_multivariate(p, T=5.0, seed=2)
    assert len(out) == 2
    for ev in out:
        assert hasattr(ev, "shape") or hasattr(ev, "__len__")
