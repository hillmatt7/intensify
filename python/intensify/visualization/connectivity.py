"""Plot connectivity / weight matrices as directed graphs (matplotlib only)."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np


def plot_connectivity(
    matrix_or_result,
    labels: list[str] | None = None,
    threshold: float = 0.0,
    layout: str = "circular",
    ax: plt.Axes | None = None,
    **kwargs: object,
) -> plt.Figure:
    """
    Plot a directed graph for a connectivity matrix or a fitted multivariate result.

    Accepts either a raw ``np.ndarray`` weight matrix or a
    :class:`~intensify.core.inference.FitResult` (from which the connectivity
    matrix is extracted automatically).

    Convention: arrow ``k -> m`` when ``matrix[m, k]`` exceeds ``threshold``.
    """
    from ..core.inference import FitResult

    if isinstance(matrix_or_result, FitResult):
        W = np.asarray(matrix_or_result.connectivity_matrix(), dtype=float)
    else:
        W = np.asarray(matrix_or_result, dtype=float)
    M = W.shape[0]
    if labels is None:
        labels = [str(i) for i in range(M)]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    pos: dict[int, tuple[float, float]]
    if layout == "grid":
        side = int(math.ceil(math.sqrt(M)))
        pos = {}
        idx = 0
        for r in range(side):
            for c in range(side):
                if idx < M:
                    pos[idx] = (c, -r)
                    idx += 1
    else:
        pos = {
            i: (0.5 + 0.45 * math.cos(2 * math.pi * i / M), 0.5 + 0.45 * math.sin(2 * math.pi * i / M))
            for i in range(M)
        }

    max_abs = float(np.max(np.abs(W))) or 1.0
    for m in range(M):
        for k in range(M):
            w = float(W[m, k])
            if abs(w) <= threshold:
                continue
            x0, y0 = pos[k]
            x1, y1 = pos[m]
            color = "#c0392b" if w > 0 else "#2980b9"
            lw = 0.5 + 3.5 * abs(w) / max_abs
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops={"arrowstyle": "->", "color": color, "lw": lw, "alpha": 0.85},
            )

    for i in range(M):
        x, y = pos[i]
        ax.scatter([x], [y], s=320, c="w", edgecolors="#333", zorder=3)
        ax.text(x, y, labels[i], ha="center", va="center", fontsize=9, zorder=4)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Connectivity")
    fig.tight_layout()
    return fig
