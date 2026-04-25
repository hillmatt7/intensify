"""Residual analysis for point process models."""

import numpy as np

from ...backends import get_backend

bt = get_backend()


def raw_residuals(events: bt.array, T: float, intensity_func) -> np.ndarray:
    """
    Raw residuals: r_i = n_i - ∫_{t_{i-1}}^{t_i} λ(t) dt, where n_i is 1 for events.

    For univariate processes, residuals should be roughly zero-mean.

    Parameters
    ----------
    events : array
        Sorted event timestamps.
    T : float
        End of observation window.
    intensity_func : callable
        Function λ(t) that gives intensity at time t (may use full history? Not exactly; for residual definition we need conditional intensity given past).
        We'll compute increment using intensity evaluated along a path? The proper definition for point processes: the compensated process N(t) - Λ(t).
        The residuals at event times are 1 minus the integral of intensity over the inter-arrival.

    Returns
    -------
    residuals : np.ndarray
        Residual for each interval (len = n_events).
    """
    n = len(events)
    if n == 0:
        return np.array([])
    # Intervals: Δt_i = t_i - t_{i-1} with t_0=0
    t_prev = 0.0
    residuals = []
    for i in range(n):
        t_i = events[i]
        dt = t_i - t_prev
        # Need ∫_{t_prev}^{t_i} λ(t) dt.
        # Since λ(t) conditional on past history, it's not stationary. We can approximate integral numerically.
        # For each ts, we need λ(ts) based on history up to ts.
        # This is tricky because we need the history before each ts.
        # One could simulate forward, building history gradually.
        # We'll approximate: average intensity over the interval times dt.
        # Not perfect but gives a rough estimate.
        lam_avg = intensity_func((t_prev + t_i) / 2)  # crude
        integral_approx = lam_avg * dt
        residual = 1.0 - integral_approx
        residuals.append(residual)
        t_prev = t_i
    return np.array(residuals)


def pearson_residuals(events: bt.array, T: float, intensity_func) -> np.ndarray:
    """
    Pearson residuals: r_i = (n_i - μ_i) / sqrt(μ_i), where μ_i = ∫_{t_{i-1}}^{t_i} λ(t) dt.
    """
    n = len(events)
    if n == 0:
        return np.array([])
    t_prev = 0.0
    residuals = []
    for i in range(n):
        t_i = events[i]
        dt = t_i - t_prev
        lam_avg = intensity_func((t_prev + t_i) / 2)
        mu_i = lam_avg * dt
        if mu_i <= 0:
            r = 0.0
        else:
            r = (1.0 - mu_i) / np.sqrt(mu_i)
        residuals.append(r)
        t_prev = t_i
    return np.array(residuals)


__all__ = ["raw_residuals", "pearson_residuals"]