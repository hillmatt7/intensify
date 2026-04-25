"""Scalar summary metrics for fitted Hawkes models."""

from __future__ import annotations

from typing import Any


def branching_ratio(process_or_result: Any) -> float:
    """
    Branching ratio / :math:`L^1` norm of the kernel for standard Hawkes.

    Accepts a :class:`~intensify.core.processes.hawkes.UnivariateHawkes`,
    a :class:`~intensify.core.inference.FitResult` with ``branching_ratio_`` or a process
    exposing ``kernel``.
    """
    if hasattr(process_or_result, "branching_ratio_"):
        br = process_or_result.branching_ratio_
        if br is not None:
            return float(br)
    proc = (
        process_or_result.process
        if hasattr(process_or_result, "process")
        else process_or_result
    )
    k = getattr(proc, "kernel", None)
    if k is not None and hasattr(k, "l1_norm"):
        return float(k.l1_norm())
    raise TypeError("Cannot infer branching ratio from object")


def endogeneity_index(process_or_result: Any) -> float:
    """
    Fraction of events attributable to self-excitation: :math:`n/(1+n)` where
    *n* is the branching ratio.  Bounded in [0, 1).
    """
    n = branching_ratio(process_or_result)
    if n >= 1.0:
        return 1.0
    return float(n / (1.0 + n))
