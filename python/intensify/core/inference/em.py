"""Expectation-Maximization inference for Hawkes processes.

Pre-0.3.0 this module had a JAX-jit autodiff M-step. With JAX removed
in the Rust port, EM falls through to MLE — both are equivalent in
distribution for the supported (univariate exp Hawkes) case, and the
Rust MLE path is now faster than the original JAX-EM ever was.
"""

from __future__ import annotations

import warnings

import numpy as np

from . import FitResult, InferenceEngine


class EMInference(InferenceEngine):
    """EM algorithm for UnivariateHawkes with ExponentialKernel.

    Routes to MLE post-Rust-port (the JAX autodiff M-step is gone; MLE
    via the Rust core is now both faster and equivalent in distribution
    for this conjugate-pair model).
    """

    def __init__(
        self,
        max_iter: int = 100,
        mstep_iter: int = 30,
        lr: float = 0.01,
        tol: float = 1e-4,
    ):
        self.max_iter = int(max_iter)
        self.mstep_iter = int(mstep_iter)
        self.lr = float(lr)
        self.tol = float(tol)

    def fit(self, process, events: np.ndarray, T: float, **kwargs) -> FitResult:
        from ..kernels.exponential import ExponentialKernel
        from ..processes.hawkes import UnivariateHawkes

        if not isinstance(process, UnivariateHawkes):
            raise NotImplementedError(
                "EMInference currently only supports UnivariateHawkes"
            )
        if not isinstance(process.kernel, ExponentialKernel):
            raise NotImplementedError(
                "EMInference currently only supports ExponentialKernel"
            )

        warnings.warn(
            "EM inference now falls through to MLE in the 0.3.0 Rust port "
            "(JAX-jit autodiff M-step removed). Use MLEInference directly "
            "for clarity.",
            DeprecationWarning,
            stacklevel=2,
        )
        from .mle import MLEInference

        return MLEInference(max_iter=self.max_iter, tol=self.tol).fit(
            process,
            events,
            T,
            **kwargs,
        )


# Auto-register EM engine
from . import register_engine

register_engine("em", EMInference())
