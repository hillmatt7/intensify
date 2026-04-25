"""Point process models."""

from .hawkes import MultivariateHawkes, UnivariateHawkes
from .marked_hawkes import MarkedHawkes
from .nonlinear_hawkes import MultivariateNonlinearHawkes, NonlinearHawkes
from .poisson import HomogeneousPoisson, InhomogeneousPoisson

# Aliases for progressive disclosure
Hawkes = UnivariateHawkes  # Simple univariate Hawkes
Poisson = HomogeneousPoisson  # Simple Poisson

# Placeholder for Phase 1
try:
    from .cox import LogGaussianCoxProcess, ShotNoiseCoxProcess
except ImportError:
    LogGaussianCoxProcess = None
    ShotNoiseCoxProcess = None

__all__ = [
    "HomogeneousPoisson",
    "InhomogeneousPoisson",
    "Poisson",
    "Hawkes",
    "UnivariateHawkes",
    "MultivariateHawkes",
    "MarkedHawkes",
    "NonlinearHawkes",
    "MultivariateNonlinearHawkes",
    "LogGaussianCoxProcess",
    "ShotNoiseCoxProcess",
]