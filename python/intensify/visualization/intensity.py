"""Plotting functions for intensity profiles."""


import matplotlib.pyplot as plt
import numpy as np



def plot_intensity(
    result,
    events=None,
    T: float | None = None,
    ax: plt.Axes | None = None,
    t_grid=None,
    **kwargs,
) -> plt.Figure:
    """
    Plot fitted conditional intensity function over time, with events marked.

    Parameters
    ----------
    result : FitResult
        Result returned by a fit() method. Should have `process`, `events`, `T` attributes.
    events : array, optional
        Event timestamps to overlay. If None, uses result.events.
    T : float, optional
        End of observation window. If None, uses result.T.
    ax : matplotlib.axes.Axes, optional
        Axes to plot into.
    t_grid : array, optional
        Custom time grid for intensity evaluation.
    **kwargs
        Additional arguments passed to `ax.plot` for intensity line.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    """
    # Use result's stored references if not provided
    if events is None and result.events is not None:
        events = result.events
    if T is None and result.T is not None:
        T = result.T
    if result.process is None:
        raise ValueError("result.process is required for plotting intensity")

    process = result.process

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    # Determine time grid
    if t_grid is None:
        t_grid = np.linspace(0, T, max(1000, int(T) * 100))

    # Evaluate intensity on grid
    # For multivariate, plot intensities for all dimensions? Here we handle univariate.
    # The process.intensity(t, history) expects full history up to t, but we can
    # simulate the intensity by using the fitted intensity function directly.
    # Simpler: use process.intensity with full event set? That would be incorrect because
    # intensity(t) should use only events before t. We need to compute forward.
    # We'll recompute intensity by scanning through time and gradually adding events.
    # For efficiency, we could precompute the closed-form expression if available.
    # For general case, we'll do a naive loop: for each t, use history = events[events < t].
    # Not super efficient but works for small N.
    intensities = []
    events_array = np.asarray(events)
    for ti in t_grid:
        mask = events_array < ti
        hist = events_array[mask]
        if hist.size > 0:
            hist_bt = np.asarray(hist)
        else:
            hist_bt = np.zeros(0)
        lam = process.intensity(ti, hist_bt)
        # Convert to Python float for plotting
        if hasattr(lam, "item"):
            lam = lam.item()
        intensities.append(float(lam))
    intensities = np.array(intensities)

    # Plot intensity line
    ax.plot(t_grid, intensities, color="steelblue", **kwargs)

    # Mark events as vertical lines or ticks
    if events_array.size > 0:
        ax.scatter(events_array, np.full_like(events_array, intensities.max() * 0.05), color="red", marker="|", s=40, label="events", zorder=5)
        # Alternatively use vlines
        # for ev in events_array:
        #     ax.axvline(ev, color='red', alpha=0.3, linewidth=0.8)

    ax.set_xlabel("Time")
    ax.set_ylabel("Intensity λ(t)")
    ax.set_title("Conditional Intensity")
    ax.legend()
    fig.tight_layout()
    return fig