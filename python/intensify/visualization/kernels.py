"""Plotting functions for kernels."""

import matplotlib.pyplot as plt
import numpy as np


def plot_kernel(
    kernel,
    t_max: float | None = None,
    ax: plt.Axes | None = None,
    log_scale: bool = False,
    **kwargs,
) -> plt.Figure:
    """
    Plot the shape of a kernel function.

    Parameters
    ----------
    kernel : Kernel
        The kernel to plot.
    t_max : float, optional
        Maximum time to plot. If None, inferred from kernel's effective support.
    ax : matplotlib.axes.Axes, optional
        Axes to plot into.
    log_scale : bool, default False
        If True, use log scale on y-axis.
    **kwargs
        Additional arguments passed to `ax.plot`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    # Determine t_max
    if t_max is None:
        # Heuristic: go out to where kernel is < 0.01 of max
        t_test = np.logspace(-2, 2, 1000)
        vals = kernel.evaluate(np.asarray(t_test))
        vals_np = np.asarray(vals) if hasattr(vals, "__array__") else np.array(vals)
        idx = np.where(vals_np > vals_np.max() * 0.01)[0]
        t_max = float(t_test[idx[-1]]) if len(idx) > 0 else 5.0

    t_grid = np.linspace(0, t_max, 1000)
    phi_vals = kernel.evaluate(np.asarray(t_grid))
    phi_np = (
        np.asarray(phi_vals) if hasattr(phi_vals, "__array__") else np.array(phi_vals)
    )

    ax.plot(t_grid, phi_np, **kwargs)

    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("φ(t) (log scale)")
    else:
        ax.set_ylabel("φ(t)")

    ax.set_xlabel("t")
    ax.set_title(f"Kernel: {kernel.__class__.__name__}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
