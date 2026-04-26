"""
Intensify — A modern library for point process modeling with Hawkes specialization.
"""

__version__ = "0.3.0-alpha.0"

# Core abstractions
# Config API
from ._config import config_get, config_reset, config_set
from .core.base import PointProcess
from .core.inference import (
    BayesianInference,
    FitResult,
    MLEInference,
    OnlineInference,
    get_inference_engine,
)
from .core.kernels import (
    ApproxPowerLawKernel,
    ExponentialKernel,
    Kernel,
    NonparametricKernel,
    PowerLawKernel,
    SumExponentialKernel,
)
from .core.regularizers import ElasticNet, L1
from .core.processes import (
    Hawkes,  # alias to UnivariateHawkes
    HomogeneousPoisson,
    InhomogeneousPoisson,
    LogGaussianCoxProcess,
    MarkedHawkes,
    MultivariateHawkes,
    MultivariateNonlinearHawkes,
    NonlinearHawkes,
    Poisson,  # alias to HomogeneousPoisson
    ShotNoiseCoxProcess,
    UnivariateHawkes,
)
from .visualization import (
    plot_connectivity,
    plot_event_aligned_histogram,
    plot_intensity,
    plot_inter_event_intervals,
    plot_kernel,
)

__all__ = [
    # Metadata
    "__version__",
    # Kernels
    "Kernel",
    "ExponentialKernel",
    "SumExponentialKernel",
    "PowerLawKernel",
    "ApproxPowerLawKernel",
    "NonparametricKernel",
    # Processes
    "PointProcess",
    "HomogeneousPoisson",
    "InhomogeneousPoisson",
    "Poisson",
    "UnivariateHawkes",
    "Hawkes",
    "MultivariateHawkes",
    "MarkedHawkes",
    "NonlinearHawkes",
    "MultivariateNonlinearHawkes",
    "LogGaussianCoxProcess",
    "ShotNoiseCoxProcess",
    # Regularizers
    "L1",
    "ElasticNet",
    # Inference
    "FitResult",
    "get_inference_engine",
    "MLEInference",
    "OnlineInference",
    "BayesianInference",
    # Visualization
    "plot_intensity",
    "plot_kernel",
    "plot_connectivity",
    "plot_inter_event_intervals",
    "plot_event_aligned_histogram",
    # Config
    "config_get",
    "config_set",
    "config_reset",
]