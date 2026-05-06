"""Inference engines for point process models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _flatten_params(params: Any) -> int:
    """Count scalar parameters for information criteria."""
    if isinstance(params, dict):
        return sum(_flatten_params(v) for v in params.values())
    if hasattr(params, "shape") and hasattr(params, "size"):
        return int(params.size)
    return 1


def compute_information_criteria(
    log_likelihood: float, params: dict, n_obs: int
) -> tuple[float, float]:
    """Compute AIC and BIC given log-likelihood, parameters, and observation count."""
    n_params = _flatten_params(params)
    aic = 2 * n_params - 2 * log_likelihood
    if n_obs <= 0:
        bic = float("nan")
    else:
        import math

        bic = n_params * math.log(n_obs) - 2 * log_likelihood
    return aic, bic


@dataclass
class FitResult:
    """Standardized container for inference results."""

    params: dict[str, Any]
    log_likelihood: float
    std_errors: dict[str, float] | None = None
    convergence_info: dict[str, Any] = field(default_factory=dict)

    branching_ratio_: float | None = None
    endogeneity_index_: float | None = None

    process: Any | None = None
    events: Any | None = None
    T: float | None = None

    aic: float | None = None
    bic: float | None = None

    marks_: Any | None = None
    posterior_samples_: dict[str, Any] | None = None
    credible_intervals_: dict[str, tuple[float, float]] | None = None
    effective_sample_size_: dict[str, float] | None = None
    r_hat_: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.aic is None or self.bic is None:
            n_obs = len(self.events) if self.events is not None else 1
            aic, bic = compute_information_criteria(
                self.log_likelihood, self.params, n_obs
            )
            if self.aic is None:
                self.aic = aic
            if self.bic is None:
                self.bic = bic

    def summary(self) -> str:
        lines = [
            "FitResult:",
            f"  Log-likelihood: {self.log_likelihood:.4f}",
            f"  AIC: {self.aic:.4f}",
            f"  BIC: {self.bic:.4f}",
            "  Parameters:",
        ]
        for k, v in self.params.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    lines.append(f"    {k}.{kk}: {vv}")
            elif (
                hasattr(v, "shape") and hasattr(v, "size") and getattr(v, "size", 0) > 1
            ):
                lines.append(f"    {k}: array shape {v.shape}")
            else:
                lines.append(f"    {k}: {v}")
        if self.std_errors:
            lines.append("  Standard errors:")
            for k, se in self.std_errors.items():
                lines.append(f"    {k}: {se:.4f}")
        if self.branching_ratio_ is not None:
            lines.append(f"  Branching ratio: {self.branching_ratio_:.4f}")
        if self.endogeneity_index_ is not None:
            lines.append(f"  Endogeneity index: {self.endogeneity_index_:.4f}")
        return "\n".join(lines)

    def flat_params(self) -> dict[str, float]:
        """Return all fitted parameters as scalar name->value pairs.

        Walks the process object (if available) and extracts numeric
        attributes so users don't need to inspect kernel objects directly.
        """
        result: dict[str, float] = {}
        proc = self.process
        if proc is None:
            for k, v in self.params.items():
                try:
                    result[k] = float(v)
                except (TypeError, ValueError):
                    pass
            return result

        if hasattr(proc, "mu"):
            mu = proc.mu
            if hasattr(mu, "__len__"):
                import numpy as _np

                for i, val in enumerate(_np.asarray(mu).ravel()):
                    result[f"mu_{i}"] = float(val)
            else:
                result["mu"] = float(mu)

        kern = getattr(proc, "kernel", None)
        if kern is not None:
            for attr in ("alpha", "beta", "c", "mark_power"):
                if hasattr(kern, attr):
                    result[attr] = float(getattr(kern, attr))

        kernel_matrix = getattr(proc, "kernel_matrix", None)
        if kernel_matrix is not None:
            for m, row in enumerate(kernel_matrix):
                for k, cell_kern in enumerate(row):
                    for attr in ("alpha", "beta"):
                        if hasattr(cell_kern, attr):
                            result[f"{attr}_{m}_{k}"] = float(getattr(cell_kern, attr))

        if hasattr(proc, "mark_power"):
            result["mark_power"] = float(proc.mark_power)

        return result

    def plot_diagnostics(self):
        """2x2 diagnostic figure: intensity, QQ, kernel (if any), summary text."""
        import matplotlib.pyplot as plt

        from ...visualization.connectivity import plot_connectivity
        from ...visualization.intensity import plot_intensity
        from ...visualization.kernels import plot_kernel
        from ..diagnostics.goodness_of_fit import qq_plot
        from ..processes.hawkes import MultivariateHawkes

        if self.process is None or self.events is None:
            raise ValueError(
                "process and events must be set on FitResult for plot_diagnostics. "
                "This is usually done automatically by model.fit()."
            )

        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        T = self.T if self.T is not None else float(max(self.events))

        is_multivariate = isinstance(self.process, MultivariateHawkes)

        if not is_multivariate:
            plot_intensity(self, events=self.events, T=T, ax=axes[0, 0])
            qq_plot(self, events=self.events, T=T, ax=axes[0, 1])
        else:
            axes[0, 0].set_axis_off()
            axes[0, 0].text(
                0.5, 0.5, "Intensity plot N/A\n(multivariate)", ha="center", va="center"
            )
            axes[0, 1].set_axis_off()
            axes[0, 1].text(
                0.5, 0.5, "QQ plot N/A\n(multivariate)", ha="center", va="center"
            )

        proc = self.process
        kern = getattr(proc, "kernel", None)
        kernel_matrix = getattr(proc, "kernel_matrix", None)
        if kern is not None:
            plot_kernel(kern, ax=axes[1, 0])
        elif kernel_matrix is not None:
            plot_connectivity(self, ax=axes[1, 0])
        else:
            axes[1, 0].set_axis_off()
            axes[1, 0].text(0.5, 0.5, "No kernel", ha="center", va="center")

        axes[1, 1].set_axis_off()
        axes[1, 1].text(
            0.05, 0.95, self.summary(), va="top", family="monospace", fontsize=8
        )
        fig.tight_layout()
        return fig

    def connectivity_matrix(self) -> Any:
        """Return :math:`M\\times M` matrix of kernel L1 norms for multivariate Hawkes."""

        from ..processes.hawkes import MultivariateHawkes

        if self.process is None:
            raise ValueError(
                "No fitted process on this FitResult. "
                "Did you forget to call model.fit()?"
            )
        if not isinstance(self.process, MultivariateHawkes):
            raise TypeError(
                f"connectivity_matrix() requires MultivariateHawkes, "
                f"but process is {type(self.process).__name__}"
            )
        proc: MultivariateHawkes = self.process
        M = proc.n_dims
        W = np.zeros((M, M), dtype=float)
        for m in range(M):
            for k in range(M):
                W[m, k] = float(proc.kernel_matrix[m][k].l1_norm())
        return W

    def significant_connections(self, significance_level: float = 0.05) -> Any:
        """
        Boolean mask of ``(m,k)`` edges with evidence of non-zero connection.

        If ``std_errors`` contains ``\"alpha_{m}_{k}\"`` entries (multivariate MLE),
        uses a normal approximation for two-sided tests. Otherwise uses a coarse
        threshold ``|W_{mk}| > significance_level`` as a proxy.
        """
        import warnings

        from scipy.stats import norm  # type: ignore[import-untyped]

        W = self.connectivity_matrix()
        M = W.shape[0]
        thr = float(significance_level)
        if self.std_errors is None:
            warnings.warn(
                "std_errors missing; using |connectivity| > significance_level as proxy",
                UserWarning,
            )
            return W > thr

        sig = np.zeros((M, M), dtype=bool)
        for m in range(M):
            for k in range(M):
                key = f"alpha_{m}_{k}"
                if key not in self.std_errors:
                    sig[m, k] = abs(W[m, k]) > thr
                    continue
                se = float(self.std_errors[key])
                est = float(W[m, k])
                z = est / max(se, 1e-12)
                p = 2.0 * (1.0 - float(norm.cdf(abs(z))))
                sig[m, k] = p < thr
        return sig

    def plot_posterior(self, max_vars: int = 8) -> Any:
        """Trace and marginal histograms for variables in ``posterior_samples_``."""
        import matplotlib.pyplot as plt

        if not self.posterior_samples_:
            raise ValueError(
                "posterior_samples_ not set; use Bayesian inference first."
            )
        samples = self.posterior_samples_
        keys = list(samples.keys())[: int(max_vars)]
        n = len(keys)
        fig, axes = plt.subplots(n, 2, figsize=(8, 2.5 * max(n, 1)))
        for i, k in enumerate(keys):
            ar = np.asarray(samples[k]).ravel()
            if n == 1:
                ax0, ax1 = axes[0], axes[1]
            else:
                ax0, ax1 = axes[i, 0], axes[i, 1]
            ax0.plot(ar, lw=0.8)
            ax0.set_title(f"{k} trace")
            ax1.hist(ar, bins=30, density=True, color="steelblue")
            ax1.set_title(f"{k} marginal")
        fig.tight_layout()
        return fig


class InferenceEngine(ABC):
    """Abstract base for inference algorithms."""

    @abstractmethod
    def fit(self, process: Any, events: Any, T: float, **kwargs: Any) -> FitResult:
        pass


_ENGINES: dict[str, InferenceEngine] = {}


def register_engine(name: str, engine: InferenceEngine) -> None:
    _ENGINES[name] = engine


def get_inference_engine(name: str) -> InferenceEngine:
    if name not in _ENGINES:
        raise ValueError(
            f"Unknown inference engine '{name}'. Available: {list(_ENGINES.keys())}"
        )
    return _ENGINES[name]


from .bayesian import BayesianInference
from .mle import (
    MLEInference,
)
from .mle import (
    _general_likelihood_numpy as _general_likelihood,
)
from .mle import (
    _recursive_likelihood_numpy as _recursive_likelihood,
)
from .online import OnlineInference

try:
    from .em import EMInference
except ImportError:
    EMInference = None  # type: ignore[misc, assignment]

__all__ = [
    "InferenceEngine",
    "FitResult",
    "compute_information_criteria",
    "get_inference_engine",
    "register_engine",
    "MLEInference",
    "EMInference",
    "OnlineInference",
    "BayesianInference",
    "_recursive_likelihood",
    "_general_likelihood",
]
