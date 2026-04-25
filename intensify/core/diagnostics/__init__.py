"""Diagnostics and goodness-of-fit tests."""

from .metrics import branching_ratio, endogeneity_index

try:
    from .goodness_of_fit import (
        qq_plot,
        residual_intensity_plot,
        time_rescaling_test,
    )
except ImportError:
    time_rescaling_test = None  # type: ignore[assignment]
    qq_plot = None  # type: ignore[assignment]
    residual_intensity_plot = None  # type: ignore[assignment]

try:
    from .residuals import pearson_residuals, raw_residuals
except ImportError:
    raw_residuals = None  # type: ignore[assignment]
    pearson_residuals = None  # type: ignore[assignment]

from ..inference import FitResult, compute_information_criteria

__all__ = [
    "time_rescaling_test",
    "qq_plot",
    "residual_intensity_plot",
    "raw_residuals",
    "pearson_residuals",
    "compute_information_criteria",
    "FitResult",
    "branching_ratio",
    "endogeneity_index",
]
