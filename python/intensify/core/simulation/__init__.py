"""Simulation algorithms."""

from .thinning import ogata_thinning, ogata_thinning_multivariate

# Placeholder for Phase 1
try:
    from .cluster import branching_simulation, branching_simulation_multivariate
except ImportError:
    branching_simulation = None  # type: ignore[misc]
    branching_simulation_multivariate = None  # type: ignore[misc]

__all__ = [
    "ogata_thinning",
    "ogata_thinning_multivariate",
    "branching_simulation",
    "branching_simulation_multivariate",
]
