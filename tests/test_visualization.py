"""Visualization smoke tests."""

import numpy as np

from intensify.backends import get_backend
from intensify.core.inference import FitResult
from intensify.core.kernels import ExponentialKernel
from intensify.core.processes import UnivariateHawkes
from intensify.visualization import plot_intensity, plot_kernel

bt = get_backend()


def test_plot_intensity_smoke():
    kernel = ExponentialKernel(alpha=0.3, beta=1.2)
    proc = UnivariateHawkes(mu=0.4, kernel=kernel)
    events = bt.array([0.1, 0.5, 1.0, 1.8])
    result = FitResult(
        params=proc.get_params(),
        log_likelihood=-2.0,
        aic=5.0,
        bic=6.0,
    )
    result.process = proc
    result.events = events
    result.T = 2.0
    fig = plot_intensity(result)
    assert fig is not None


def test_plot_kernel_smoke():
    k = ExponentialKernel(alpha=0.35, beta=1.5)
    fig = plot_kernel(k)
    assert fig is not None


def test_plot_diagnostics_smoke():
    kernel = ExponentialKernel(alpha=0.25, beta=1.0)
    proc = UnivariateHawkes(mu=0.3, kernel=kernel)
    events = bt.array([0.15, 0.6, 1.2])
    r = FitResult(
        params=proc.get_params(),
        log_likelihood=-1.5,
        aic=4.0,
        bic=5.0,
    )
    r.process = proc
    r.events = events
    r.T = 2.0
    fig = r.plot_diagnostics()
    assert fig is not None
