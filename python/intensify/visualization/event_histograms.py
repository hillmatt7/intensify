"""General-purpose event-time histograms."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_inter_event_intervals(
    events: np.ndarray,
    ax: plt.Axes | None = None,
    bins: int = 40,
    **kwargs: object,
) -> plt.Figure:
    """Histogram of inter-event intervals with exponential reference (by mean ISI)."""
    ev = np.asarray(events, dtype=float).ravel()
    if ev.size < 2:
        raise ValueError("Need at least two events for ISI histogram")
    isi = np.diff(np.sort(ev))
    lam_hat = 1.0 / float(np.mean(isi))
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.get_figure()
    ax.hist(isi, bins=bins, density=True, color="steelblue", alpha=0.75, **kwargs)
    xs = np.linspace(0.0, float(np.max(isi)), 200)
    ax.plot(
        xs, lam_hat * np.exp(-lam_hat * xs), color="black", lw=2, label="Exp reference"
    )
    ax.set_xlabel("Inter-event interval")
    ax.set_ylabel("Density")
    ax.legend()
    ax.set_title("Inter-event intervals")
    fig.tight_layout()
    return fig


def plot_event_aligned_histogram(
    events: np.ndarray,
    reference_times: np.ndarray,
    window: tuple[float, float],
    bin_width: float = 0.05,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Average event-aligned histogram relative to ``reference_times`` (PSTH-style).

    Parameters
    ----------
    events : array
        Full array of event timestamps.
    reference_times : array
        Anchor timestamps (e.g. stimulus onsets or spike times from a
        reference neuron). For each reference time, the function counts
        events within ``window`` and averages across all references.
    window : tuple[float, float]
        ``(lo, hi)`` window around each reference time, in the same
        units as the event timestamps.  For example ``(-0.1, 0.5)``
        means 100 ms before to 500 ms after.
    bin_width : float, default 0.05
        Histogram bin width (same time units).
    ax : matplotlib Axes, optional
        Axes to draw into; a new figure is created if *None*.

    Returns
    -------
    fig : matplotlib.figure.Figure

    Examples
    --------
    >>> stim_times = np.array([1.0, 5.0, 10.0])
    >>> plot_event_aligned_histogram(spike_times, stim_times, window=(-0.1, 0.5))
    """
    ev = np.asarray(events, dtype=float).ravel()
    ref = np.asarray(reference_times, dtype=float).ravel()
    lo, hi = float(window[0]), float(window[1])
    edges = np.arange(lo, hi + bin_width, bin_width)
    sums = np.zeros(len(edges) - 1)
    count = 0
    for r in ref:
        rel = ev - r
        rel = rel[(rel >= lo) & (rel < hi)]
        if rel.size == 0:
            continue
        h, _ = np.histogram(rel, bins=edges)
        sums += h
        count += 1
    centers = (edges[:-1] + edges[1:]) / 2
    if count == 0:
        rate = np.zeros_like(centers, dtype=float)
    else:
        rate = sums / (count * bin_width)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.get_figure()
    ax.bar(
        centers,
        rate,
        width=bin_width * 0.9,
        color="darkgreen",
        alpha=0.75,
    )
    ax.axvline(0.0, color="k", lw=1.0, alpha=0.5)
    ax.set_xlabel("Time relative to reference")
    ax.set_ylabel("Rate (events / s)")
    ax.set_title("Event-aligned histogram")
    fig.tight_layout()
    return fig
