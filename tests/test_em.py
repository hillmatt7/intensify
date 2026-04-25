"""Tests for EM inference."""

import numpy as np
import pytest

from intensify.backends import get_backend
from intensify.core.inference import get_inference_engine
from intensify.core.kernels import ExponentialKernel
from intensify.core.processes import UnivariateHawkes

bt = get_backend()


def test_em_inference_basic():
    # Simple test that EM engine runs on univariate Hawkes
    kernel = ExponentialKernel(alpha=0.3, beta=1.5)
    process = UnivariateHawkes(mu=0.4, kernel=kernel)
    np.random.seed(42)
    events = np.random.uniform(0, 10, size=50)
    events.sort()
    T = 10.0

    engine = get_inference_engine("em")
    result = engine.fit(process, events, T)
    assert result.log_likelihood is not None
    assert result.aic is not None
    assert result.bic is not None
    # Params should be positive
    assert result.params["mu"] >= 0
    assert result.params["kernel"].alpha >= 0
    assert result.params["kernel"].beta >= 0