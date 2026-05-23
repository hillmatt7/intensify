"""Point process models."""

from .hawkes import MultivariateHawkes, UnivariateHawkes
from .marked_hawkes import MarkedHawkes
from .nonlinear_hawkes import MultivariateNonlinearHawkes, NonlinearHawkes
from .poisson import HomogeneousPoisson, InhomogeneousPoisson

# Aliases for progressive disclosure
Hawkes = UnivariateHawkes  # Simple univariate Hawkes
Poisson = HomogeneousPoisson  # Simple Poisson

from .cox import LogGaussianCoxProcess, ShotNoiseCoxProcess

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
