"""Kernel implementations."""

from .base import Kernel
from .exponential import ExponentialKernel

# Placeholder for Phase 1: these will be implemented later
try:
    from .sum_exponential import SumExponentialKernel
except ImportError:
    SumExponentialKernel = None

try:
    from .power_law import PowerLawKernel
except ImportError:
    PowerLawKernel = None

try:
    from .approx_power_law import ApproxPowerLawKernel
except ImportError:
    ApproxPowerLawKernel = None

try:
    from .nonparametric import NonparametricKernel
except ImportError:
    NonparametricKernel = None

__all__ = [
    "Kernel",
    "ExponentialKernel",
    "SumExponentialKernel",
    "PowerLawKernel",
    "ApproxPowerLawKernel",
    "NonparametricKernel",
]