"""Goodness-of-fit tests for point process models."""


import numpy as np
from scipy import stats

from ...backends import get_backend
from ...core.inference import FitResult


def _compute_compensators(process, events: np.ndarray) -> np.ndarray:
    """Compute cumulative intensity Lambda(t_i) = integral_0^{t_i} lambda(s) ds at each event.

    Uses O(N) recursive path when the kernel supports it (exponential family),
    falls back to O(N^2) general computation otherwise.
    """
    kernel = process.kernel if hasattr(process, "kernel") else None
    if kernel is not None and hasattr(kernel, "has_recursive_form") and kernel.has_recursive_form():
        return _recursive_compensators(process, events)
    return _general_compensators(process, events)


def _recursive_compensators(process, events: np.ndarray) -> np.ndarray:
    """O(N) compensator Lambda(t_i) for exponential-family kernels.

    For phi(t) = alpha * beta * exp(-beta t), the compensator increment
    between events is

        tau_i = mu * dt + alpha * R(t_{i-1}^+) * (1 - exp(-beta * dt))

    where R tracks the recursive state that is also used by the O(N)
    likelihood path. This is mathematically equivalent to the pairwise
    integral in ``_general_compensators`` but runs in O(N).

    For SumExponentialKernel (vector of components) the same recursion
    applies per component with shared state.
    """
    t_arr = np.asarray(events, dtype=float)
    n = len(t_arr)
    Lambda_i = np.zeros(n)
    mu = float(process.mu) if hasattr(process, "mu") else 0.0
    kernel = process.kernel

    # Dispatch by the actual parametric form of the kernel.
    from ..kernels.exponential import ExponentialKernel
    from ..kernels.sum_exponential import SumExponentialKernel

    if isinstance(kernel, ExponentialKernel):
        alpha = float(kernel.alpha)
        beta = float(kernel.beta)
        if getattr(kernel, "allow_signed", False):
            alpha = abs(alpha)  # compensator uses |phi| magnitude
        R = 0.0
        t_prev = 0.0
        Lambda_prev = 0.0
        for i in range(n):
            dt = float(t_arr[i] - t_prev)
            decay = np.exp(-beta * dt) if beta > 0 else 1.0
            tau = mu * dt + alpha * R * (1.0 - decay)
            Lambda_prev += tau
            Lambda_i[i] = Lambda_prev
            R = R * decay + 1.0
            t_prev = float(t_arr[i])
        return Lambda_i

    if isinstance(kernel, SumExponentialKernel):
        alphas = np.asarray(kernel.alphas, dtype=float)
        betas = np.asarray(kernel.betas, dtype=float)
        R = np.zeros_like(alphas)
        t_prev = 0.0
        Lambda_prev = 0.0
        for i in range(n):
            dt = float(t_arr[i] - t_prev)
            decay = np.exp(-betas * dt)
            tau = mu * dt + float(np.sum(alphas * R * (1.0 - decay)))
            Lambda_prev += tau
            Lambda_i[i] = Lambda_prev
            R = R * decay + 1.0
            t_prev = float(t_arr[i])
        return Lambda_i

    # Unknown recursive kernel — fall back to the verified general path.
    return _general_compensators(process, events)


def _general_compensators(process, events: np.ndarray) -> np.ndarray:
    """O(N^2) compensator computation for general kernels."""
    t_arr = np.asarray(events, dtype=float)
    n = len(t_arr)
    Lambda_i = np.zeros(n)
    mu = float(process.mu) if hasattr(process, "mu") else 0.0
    kernel = process.kernel if hasattr(process, "kernel") else None

    for i in range(n):
        t_i = t_arr[i]
        comp = mu * t_i
        if kernel is not None:
            for j in range(i):
                lag = t_i - t_arr[j]
                comp += kernel.integrate(lag)
        Lambda_i[i] = comp
    return Lambda_i


def _compensator_intervals(process, events: np.ndarray) -> np.ndarray:
    """Inter-compensator intervals Delta_tau_i = Lambda(t_i) - Lambda(t_{i-1}).

    Under a correct model these should be i.i.d. Exp(1) by the time-rescaling theorem.
    """
    Lambda_i = _compute_compensators(process, events)
    return np.diff(Lambda_i, prepend=0.0)


def time_rescaling_test(result: FitResult, events=None, T: float = None) -> tuple[float, float]:
    """
    Time-rescaling theorem test (KS test).

    Under the correct model the inter-compensator intervals
    Delta_tau_i = Lambda(t_i) - Lambda(t_{i-1}) are i.i.d. Exp(1).

    Parameters
    ----------
    result : FitResult
        Fitted model. Should have process, events, T.
    events : array, optional
        Event timestamps. If None, uses result.events.
    T : float, optional
        Observation window end. Should match model.

    Returns
    -------
    ks_stat : float
        Kolmogorov-Smirnov statistic.
    p_value : float
        p-value for the test. Large p-value indicates model is adequate.
    """
    if events is None:
        events = result.events
    if T is None:
        T = result.T
    process = result.process
    if process is None:
        raise ValueError("result.process is required for time_rescaling_test")

    tau_intervals = _compensator_intervals(process, np.asarray(events))

    # p-value computed assuming known parameters; may be anti-conservative
    # when parameters are estimated from the same data.
    ks_stat, p_value = stats.kstest(tau_intervals, stats.expon().cdf)

    return ks_stat, p_value


def qq_plot(result: FitResult, events=None, T: float = None, ax=None):
    """
    QQ plot of inter-compensator intervals against theoretical Exp(1) quantiles.
    """
    import matplotlib.pyplot as plt

    if events is None:
        events = result.events
    if T is None:
        T = result.T
    process = result.process

    tau_intervals = _compensator_intervals(process, np.asarray(events))
    n = len(tau_intervals)

    tau_sorted = np.sort(tau_intervals)
    p_i = (np.arange(1, n + 1) - 0.5) / n
    theoretical_quantiles = -np.log(1 - p_i)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()
    ax.plot(theoretical_quantiles, tau_sorted, "o", markersize=4)
    lims = [min(theoretical_quantiles.min(), tau_sorted.min()), max(theoretical_quantiles.max(), tau_sorted.max())]
    ax.plot(lims, lims, "r--", alpha=0.7)
    ax.set_xlabel("Theoretical Exp(1) quantiles")
    ax.set_ylabel("Inter-compensator intervals")
    ax.set_title("Time-Rescaling QQ Plot")
    fig.tight_layout()
    return fig


def residual_intensity_plot(result: FitResult, events=None, T: float = None, ax=None, **kwargs):
    """
    Plot residual intensity over time: λ*(t) - λ_0(t), where λ_0(t) is the estimated intensity.
    Or simply show the intensity with events to visually check match.
    This is essentially a wrapper around plot_intensity.
    """
    from ...visualization import plot_intensity
    return plot_intensity(result, events=events, T=T, ax=ax, **kwargs)


__all__ = ["time_rescaling_test", "qq_plot", "residual_intensity_plot"]