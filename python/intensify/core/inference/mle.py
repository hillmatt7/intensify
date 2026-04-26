"""Maximum likelihood inference: scipy L-BFGS-B + Rust analytic gradient."""

import warnings
from typing import Any

import numpy as np

from ..._config import config_get

from . import FitResult, InferenceEngine, compute_information_criteria
from .univariate_hawkes_mle_params import (
    hawkes_mle_apply_vector,
    hawkes_mle_bounds,
    hawkes_mle_initial_vector,
    hawkes_mle_param_names,
)


class PerformanceWarning(UserWarning):
    """Emitted when an expensive code path is selected."""


def _resolve_regularization(regularization: Any | None) -> Any | None:
    """Accept a Regularizer instance, None, or a string shorthand.

    Strings accepted: "l1", "elasticnet" (case-insensitive). Strings map to
    ``L1()`` / ``ElasticNet()`` with default strength.
    """
    if regularization is None:
        return None
    if isinstance(regularization, str):
        from ..regularizers import ElasticNet, L1

        key = regularization.strip().lower().replace("-", "").replace("_", "")
        if key == "l1":
            return L1()
        if key in ("elasticnet", "en"):
            return ElasticNet()
        raise ValueError(
            f"Unknown regularization shorthand {regularization!r}; "
            "use 'l1', 'elasticnet', or pass a Regularizer instance."
        )
    return regularization


def _lock_beta_bounds_univariate(
    bounds: list, x0: np.ndarray, process,
) -> list:
    """Replace the β bound(s) in ``bounds`` with zero-width intervals at x0.

    Only supported for kernels where β is the exponential decay rate:
    ``ExponentialKernel`` and ``SumExponentialKernel``. Other kernels
    raise TypeError because their parameter layout does not expose a
    decay-rate axis (``PowerLawKernel.beta`` is a tail exponent, not
    a decay rate).
    """
    from ..kernels.exponential import ExponentialKernel
    from ..kernels.sum_exponential import SumExponentialKernel

    locked = list(bounds)
    kern = process.kernel
    if isinstance(kern, ExponentialKernel):
        locked[2] = (float(x0[2]), float(x0[2]))
    elif isinstance(kern, SumExponentialKernel):
        K = kern.n_components
        for i in range(K):
            j = 1 + K + i
            locked[j] = (float(x0[j]), float(x0[j]))
    else:
        raise TypeError(
            f"fit_decay=False not supported for {type(kern).__name__}; "
            "only ExponentialKernel and SumExponentialKernel have an "
            "identifiable decay-rate axis."
        )
    return locked


def _lock_beta_bounds_multivariate(
    bounds: list, x0: np.ndarray, M: int,
) -> list:
    """Lock every β slot in the flat multivariate vector to its initial value.

    Flat layout: ``[mu (M), (alpha, beta) interleaved (M*M cells)]``.
    β slots are at indices ``M + 2k + 1`` for ``k`` in ``[0, M*M)``.
    """
    locked = list(bounds)
    for k in range(M * M):
        j = M + 2 * k + 1
        locked[j] = (float(x0[j]), float(x0[j]))
    return locked


def _warn_if_not_converged(result_opt, model_label: str) -> None:
    """Emit a RuntimeWarning when SciPy optimizer did not converge."""
    if not result_opt.success:
        warnings.warn(
            f"{model_label} MLE did not converge: {result_opt.message}. "
            "Parameters may be unreliable; check result.convergence_info.",
            RuntimeWarning,
            stacklevel=3,
        )


def _validate_events(events: np.ndarray, T: float | None) -> None:
    """Raise on unsorted, negative, or out-of-window events."""
    if events.size == 0:
        return
    if not np.all(np.diff(events) >= 0):
        raise ValueError("events must be sorted in non-decreasing order")
    if float(events[0]) < 0:
        raise ValueError("events must be non-negative")
    if T is not None and float(events[-1]) > T:
        raise ValueError(
            f"events must be in [0, T={T}], but max event = {float(events[-1])}"
        )


def _finite_difference_std_errors(
    obj, x_opt: np.ndarray, param_names: list[str]
) -> dict[str, float] | None:
    """Compute standard errors via finite-difference Hessian with conditioning checks."""
    try:
        from scipy.optimize import approx_fprime

        eps = np.sqrt(np.finfo(float).eps) ** (1 / 3)
        n_p = len(x_opt)
        g0 = approx_fprime(x_opt, obj, epsilon=eps)
        H = np.zeros((n_p, n_p))
        for i in range(n_p):
            xp = x_opt.copy()
            xp[i] += eps
            gi = approx_fprime(xp, obj, epsilon=eps)
            H[:, i] = (gi - g0) / eps
        H = (H + H.T) / 2

        cond = np.linalg.cond(H)
        if cond > 1e12:
            warnings.warn(
                f"Hessian condition number {cond:.2e} is very large; "
                "standard errors may be unreliable.",
                RuntimeWarning,
            )

        cov = np.linalg.pinv(H + 1e-8 * np.eye(n_p))
        return {
            param_names[i]: float(np.sqrt(max(cov[i, i], 0.0)))
            for i in range(n_p)
        }
    except Exception as e:  # pragma: no cover
        warnings.warn(f"Could not approximate Hessian: {e}", RuntimeWarning)
        return None


class MLEInference(InferenceEngine):
    """
    Maximum likelihood inference for point processes using gradient-based optimization.

    Automatically selects O(N) recursive likelihood if kernel.has_recursive_form()
    is True, otherwise uses O(N²) general likelihood.

    Drives scipy.optimize.L-BFGS-B with analytic gradients from the Rust
    core (or finite-diff for kernels not yet ported).

    Parameters (optional)
    ---------------------
    optimizer : str, default "adam"
        Legacy parameter, no longer used (was for JAX adam path pre-0.3.0).
    lr : float, default 1e-3
        Legacy parameter, no longer used.
    max_iter : int, default 5000
        Maximum L-BFGS-B iterations.
    tol : float, default 1e-5
        Convergence tolerance.
    """

    def __init__(self, optimizer: str = "adam", lr: float = 1e-3, max_iter: int = 5000, tol: float = 1e-5):
        self.optimizer = optimizer
        self.lr = float(lr)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def fit(self, process, events: Any, T: float, *, fit_decay: bool = True, **kwargs: Any) -> FitResult:
        """
        Fit model parameters via maximum likelihood.

        Parameters
        ----------
        process : PointProcess
            The model to fit (will be modified in-place).
        events : array
            Event timestamps, or ``(times, marks)`` for :class:`MarkedHawkes`.
        T : float
            Observation window end.

        Returns
        -------
        result : FitResult
        """
        from ..processes.hawkes import MultivariateHawkes
        from ..processes.marked_hawkes import MarkedHawkes

        if isinstance(process, MultivariateHawkes):
            ev_list = [np.asarray(e, dtype=float).ravel() for e in events]
            for e in ev_list:
                _validate_events(e, float(T))
            n_obs = int(sum(len(e) for e in ev_list)) or 1
            try:
                import scipy.optimize as spo  # noqa: F401
            except ImportError as e:
                raise RuntimeError("SciPy is required for multivariate MLE.") from e
            return self._fit_multivariate_numpy(
                process,
                ev_list,
                float(T),
                n_obs,
                regularization=_resolve_regularization(kwargs.get("regularization")),
                fit_decay=fit_decay,
            )

        if isinstance(process, MarkedHawkes):
            ev, marks = events
            ev = np.asarray(ev, dtype=float).ravel()
            marks = np.asarray(marks, dtype=float).ravel()
            _validate_events(ev, float(T))
            n_obs = len(ev)
            try:
                import scipy.optimize as spo  # noqa: F401
            except ImportError:
                spo = None
            if spo is None:
                raise RuntimeError("SciPy is required for MarkedHawkes MLE.")
            from ..._rust import has_rust_marked_exp_path
            if has_rust_marked_exp_path(process):
                return self._fit_marked_uni_exp_rust(
                    process, ev, marks, T, n_obs, fit_decay=fit_decay,
                )
            return self._fit_marked_numpy(process, ev, marks, T, n_obs, fit_decay=fit_decay)

        events = np.asarray(events)
        _validate_events(np.asarray(events), float(T))
        n_obs = len(events)

        try:
            import scipy.optimize as spo  # noqa: F401
        except ImportError:
            spo = None

        if spo is None:
            raise RuntimeError("SciPy is required for MLE; install scipy.")

        return self._fit_numpy(process, events, T, n_obs, fit_decay=fit_decay)

    def _fit_marked_numpy(
        self, process: Any, events: np.ndarray, marks: np.ndarray, T: float, n_obs: int,
        *, fit_decay: bool = True,
    ) -> FitResult:
        """MLE for MarkedHawkes (any kernel supported by the univariate helpers)."""
        x0 = hawkes_mle_initial_vector(process)
        bounds = hawkes_mle_bounds(process)
        names = hawkes_mle_param_names(process)
        if not fit_decay:
            bounds = _lock_beta_bounds_univariate(bounds, x0, process)

        def obj(x: np.ndarray) -> float:
            hawkes_mle_apply_vector(process, x)
            return -float(process.log_likelihood(events, marks, T))

        import scipy.optimize as spo

        result_opt = spo.minimize(
            obj,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": self.tol, "maxiter": self.max_iter},
        )
        hawkes_mle_apply_vector(process, result_opt.x)
        _warn_if_not_converged(result_opt, "MarkedHawkes")
        final_params = process.get_params()
        final_ll = -result_opt.fun
        n_params = len(result_opt.x)
        aic = 2 * n_params - 2 * final_ll
        bic = n_params * float(np.log(n_obs)) - 2 * final_ll if n_obs > 0 else float("nan")
        result = FitResult(
            params=final_params,
            log_likelihood=final_ll,
            aic=aic,
            bic=bic,
            std_errors=None,
            convergence_info={
                "iterations": result_opt.nit,
                "success": result_opt.success,
                "message": result_opt.message,
                "backend": "numpy",
                "model": "marked_hawkes",
            },
        )
        result.process = process
        result.events = events
        result.marks_ = marks
        result.T = T
        br = float(process.kernel.l1_norm())
        result.branching_ratio_ = br
        if br < 1.0:
            result.endogeneity_index_ = br / (1.0 + br)
        return result

    def _fit_nonlinear_numpy(
        self, process: Any, events: Any, T: float, n_obs: int,
        *, fit_decay: bool = True,
    ) -> FitResult:
        """MLE for :class:`NonlinearHawkes` with any kernel supported by the univariate helpers."""
        from ..processes.nonlinear_hawkes import NonlinearHawkes

        if not isinstance(process, NonlinearHawkes):
            raise TypeError("expected NonlinearHawkes")
        ev = np.asarray(events, dtype=float).ravel()

        # Phase 3: route through Rust when kernel is ExponentialKernel + builtin link.
        from ..._rust import has_rust_nonlinear_exp_path
        if has_rust_nonlinear_exp_path(process):
            return self._fit_nonlinear_uni_exp_rust(process, ev, T, n_obs, fit_decay=fit_decay)

        x0 = hawkes_mle_initial_vector(process)
        bounds = hawkes_mle_bounds(process)
        names = hawkes_mle_param_names(process)
        if not fit_decay:
            bounds = _lock_beta_bounds_univariate(bounds, x0, process)

        def obj(x: np.ndarray) -> float:
            hawkes_mle_apply_vector(process, x)
            return -float(process.log_likelihood(ev, T))

        import scipy.optimize as spo

        result_opt = spo.minimize(
            obj,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": self.tol, "maxiter": self.max_iter},
        )
        hawkes_mle_apply_vector(process, result_opt.x)
        _warn_if_not_converged(result_opt, "NonlinearHawkes")
        final_params = process.get_params()
        final_ll = -result_opt.fun
        aic, bic = compute_information_criteria(final_ll, final_params, n_obs)
        std_errors = None
        if result_opt.success and len(result_opt.x) <= 12:
            std_errors = _finite_difference_std_errors(obj, result_opt.x, names)
        result = FitResult(
            params=final_params,
            log_likelihood=final_ll,
            aic=aic,
            bic=bic,
            std_errors=std_errors,
            convergence_info={
                "iterations": result_opt.nit,
                "success": result_opt.success,
                "message": result_opt.message,
                "backend": "numpy",
                "model": "nonlinear_hawkes",
            },
        )
        result.process = process
        result.events = ev
        result.T = float(T)
        br = float(process.kernel.l1_norm())
        result.branching_ratio_ = br
        if br < 1.0:
            result.endogeneity_index_ = br / (1.0 + br)
        return result

    def _fit_mv_exp_recursive_rust(
        self,
        process: Any,
        events_list: list[np.ndarray],
        T: float,
        n_obs: int,
        *,
        fit_decay: bool = True,
    ) -> FitResult:
        """MLE for :class:`MultivariateHawkes` with shared-β exponential
        kernels and β locked, via the Rust core.

        Compute path: scipy.optimize.minimize(L-BFGS-B) drives the loop;
        each value+grad call is a single Rust function (~10µs at M=5,
        N=1099) returning analytic gradient via tick's per-target weight
        precomputation. Coefficient layout is [μ, α row-major]; β is
        passed at construction time and never enters the optimizer.
        """
        import scipy.optimize as spo

        from ..._rust import (
            _ext,
            mv_apply_rust_coeffs,
            mv_initial_rust_coeffs,
            mv_shared_beta,
        )

        M = process.n_dims
        beta = mv_shared_beta(process)
        if beta is None:  # pragma: no cover  — predicate already gated on this
            raise RuntimeError("mv_exp_recursive Rust path requires shared β")

        # Sanitize event arrays for the Rust boundary.
        ev_clean = [
            np.ascontiguousarray(np.asarray(e, dtype=np.float64).ravel())
            for e in events_list
        ]

        rust_model = _ext.likelihood.MvExpRecursiveLogLik(ev_clean, float(T), float(beta))

        x0 = mv_initial_rust_coeffs(process)
        n_coeffs = M + M * M
        bounds: list[tuple[float | None, float | None]] = [(1e-8, None)] * M
        bounds.extend([(1e-8, 0.999)] * (M * M))

        # Pre-allocated grad buffer; we copy() into the closure return
        # so scipy never sees an aliased mutable.
        grad_buf = np.zeros(n_coeffs, dtype=np.float64)

        def obj_and_grad(x: np.ndarray):
            val = rust_model.loss_and_grad(x, grad_buf)
            return float(val), grad_buf.copy()

        def obj_only(x: np.ndarray) -> float:
            return rust_model.loss(x)

        result_opt = spo.minimize(
            obj_and_grad,
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"ftol": self.tol, "maxiter": self.max_iter},
        )

        # Write fitted coefficients back into the process and project to
        # a stationary parameter region.
        mv_apply_rust_coeffs(process, result_opt.x, beta)
        process.project_params()
        _warn_if_not_converged(result_opt, "MultivariateHawkes (Rust)")

        final_params = process.get_params()
        final_ll = -result_opt.fun
        aic, bic = compute_information_criteria(final_ll, final_params, n_obs)

        std_errors = None
        if result_opt.success and len(result_opt.x) <= 24:
            param_names = [f"mu_{m}" for m in range(M)] + [
                f"alpha_{i}_{j}" for i in range(M) for j in range(M)
            ]
            std_errors = _finite_difference_std_errors(
                obj_only, result_opt.x, param_names,
            )

        # Spectral radius of the L1-norm matrix W = (|alpha_{i,j}|).
        W = np.zeros((M, M))
        for m in range(M):
            for k in range(M):
                W[m, k] = float(process.kernel_matrix[m][k].l1_norm())
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(W))))
        if spectral_radius >= 1.0:
            warnings.warn(
                f"Fitted multivariate Hawkes has spectral radius "
                f"{spectral_radius:.4f} >= 1 after projection; the process is "
                "non-stationary and simulation/forecasts will diverge. "
                "Inspect connectivity_matrix() and consider regularization.",
                RuntimeWarning,
            )

        result = FitResult(
            params=final_params,
            log_likelihood=final_ll,
            aic=aic,
            bic=bic,
            std_errors=std_errors,
            convergence_info={
                "iterations": result_opt.nit,
                "success": result_opt.success,
                "message": result_opt.message,
                "backend": "rust",
                "model": "multivariate_hawkes_exp_recursive",
            },
        )
        result.process = process
        result.events = events_list
        result.T = float(T)
        result.branching_ratio_ = spectral_radius
        if spectral_radius < 1.0:
            result.endogeneity_index_ = spectral_radius / (1.0 + spectral_radius)
        return result

    def _fit_mv_exp_dense_rust(
        self,
        process: Any,
        events_list: list[np.ndarray],
        T: float,
        n_obs: int,
    ) -> FitResult:
        """MLE for :class:`MultivariateHawkes` with per-cell ExponentialKernel
        and **β fitted jointly** with α and μ, via the Rust core.

        Coefficient layout for the optimizer matches the existing intensify
        Python flat layout: ``[μ (M), (α_{m,k}, β_{m,k}) interleaved per
        cell row-major]``. The Rust core uses three flat row-major buffers
        internally; this method translates between layouts.
        """
        import scipy.optimize as spo

        from ..._rust import _ext
        from ..kernels.exponential import ExponentialKernel

        M = process.n_dims
        n_coeffs = M + 2 * M * M

        # Flatten events globally into (times, sources) sorted ascending.
        ev_clean = [
            np.ascontiguousarray(np.asarray(e, dtype=np.float64).ravel())
            for e in events_list
        ]
        times_list: list[float] = []
        sources_list: list[int] = []
        for k, ev in enumerate(ev_clean):
            for t in ev:
                times_list.append(float(t))
                sources_list.append(k)
        order = np.argsort(times_list, kind="stable")
        times_all = np.asarray(times_list, dtype=np.float64)[order]
        sources_all = np.asarray(sources_list, dtype=np.int64)[order]

        # Initial flat coefficients (intensify layout).
        from .multivariate_hawkes_mle_params import (
            multivariate_hawkes_apply_vector,
            multivariate_hawkes_bounds,
            multivariate_hawkes_initial_vector,
            multivariate_hawkes_param_names,
        )
        x0 = multivariate_hawkes_initial_vector(process)
        bounds = multivariate_hawkes_bounds(process)
        names = multivariate_hawkes_param_names(process)

        def split_layout(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            mu = np.ascontiguousarray(x[:M], dtype=np.float64)
            rest = x[M:].reshape(M * M, 2)
            alpha = np.ascontiguousarray(rest[:, 0], dtype=np.float64)
            beta = np.ascontiguousarray(rest[:, 1], dtype=np.float64)
            return mu, alpha, beta

        def merge_layout(grad_mu: np.ndarray, grad_alpha: np.ndarray, grad_beta: np.ndarray) -> np.ndarray:
            out = np.empty(n_coeffs, dtype=np.float64)
            out[:M] = grad_mu
            interleaved = out[M:].reshape(M * M, 2)
            interleaved[:, 0] = grad_alpha
            interleaved[:, 1] = grad_beta
            return out

        def obj_and_grad(x: np.ndarray):
            mu_x, a_x, b_x = split_layout(x)
            val, gm, ga, gb = _ext.likelihood.mv_exp_dense_neg_ll_with_grad(
                times_all, sources_all, float(T), int(M), mu_x, a_x, b_x,
            )
            return float(val), merge_layout(gm, ga, gb)

        def obj_only(x: np.ndarray) -> float:
            mu_x, a_x, b_x = split_layout(x)
            return _ext.likelihood.mv_exp_dense_neg_ll(
                times_all, sources_all, float(T), int(M), mu_x, a_x, b_x,
            )

        result_opt = spo.minimize(
            obj_and_grad,
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"ftol": self.tol, "maxiter": self.max_iter},
        )

        multivariate_hawkes_apply_vector(process, result_opt.x)
        process.project_params()
        _warn_if_not_converged(result_opt, "MultivariateHawkes (Rust dense)")

        final_params = process.get_params()
        final_ll = -result_opt.fun
        aic, bic = compute_information_criteria(final_ll, final_params, n_obs)

        std_errors = None
        if result_opt.success and len(result_opt.x) <= 24:
            std_errors = _finite_difference_std_errors(obj_only, result_opt.x, names)

        # Spectral radius of the L1-norm matrix W = (|alpha_{i,j}|).
        W = np.zeros((M, M))
        for m in range(M):
            for k in range(M):
                W[m, k] = float(process.kernel_matrix[m][k].l1_norm())
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(W))))
        if spectral_radius >= 1.0:
            warnings.warn(
                f"Fitted multivariate Hawkes has spectral radius "
                f"{spectral_radius:.4f} >= 1 after projection; the process is "
                "non-stationary and simulation/forecasts will diverge. "
                "Inspect connectivity_matrix() and consider regularization.",
                RuntimeWarning,
            )

        result = FitResult(
            params=final_params,
            log_likelihood=final_ll,
            aic=aic,
            bic=bic,
            std_errors=std_errors,
            convergence_info={
                "iterations": result_opt.nit,
                "success": result_opt.success,
                "message": result_opt.message,
                "backend": "rust",
                "model": "multivariate_hawkes_exp_dense",
            },
        )
        result.process = process
        result.events = events_list
        result.T = float(T)
        result.branching_ratio_ = spectral_radius
        if spectral_radius < 1.0:
            result.endogeneity_index_ = spectral_radius / (1.0 + spectral_radius)
        return result

    def _fit_multivariate_numpy(
        self,
        process: Any,
        events_list: list[np.ndarray],
        T: float,
        n_obs: int,
        *,
        regularization: Any | None = None,
        fit_decay: bool = True,
    ) -> FitResult:
        """MLE for :class:`MultivariateHawkes` with exponential kernels in each cell."""
        from ..processes.hawkes import MultivariateHawkes

        if not isinstance(process, MultivariateHawkes):
            raise TypeError("expected MultivariateHawkes")
        M = process.n_dims
        n_free_params = M + 2 * M * M
        if n_free_params > 0 and (n_obs / n_free_params) < 20:
            warnings.warn(
                f"Only {n_obs} events for {n_free_params} free parameters "
                f"(M={M}). MLE estimates may be unreliable.",
                UserWarning,
            )

        # Rust dispatch:
        #   * Decay-given (β fixed, shared) → Rust mv_exp_recursive (Phase 1b)
        #   * Joint-decay (β fitted per cell) → Rust mv_exp_dense   (Phase 2)
        # Both paths require all kernel-matrix cells to be ExponentialKernel
        # without allow_signed. Regularized fits also use Rust value+grad
        # (gradient adds the regularizer's analytic gradient on top).
        from ..._rust import (
            has_rust_mv_dense_path,
            has_rust_mv_recursive_path,
        )
        if regularization is None and has_rust_mv_recursive_path(process, fit_decay):
            return self._fit_mv_exp_recursive_rust(
                process, events_list, T, n_obs, fit_decay=fit_decay,
            )
        if regularization is None and fit_decay and has_rust_mv_dense_path(process):
            return self._fit_mv_exp_dense_rust(
                process, events_list, T, n_obs,
            )

        from .multivariate_hawkes_mle_params import (
            multivariate_hawkes_apply_vector,
            multivariate_hawkes_bounds,
            multivariate_hawkes_initial_vector,
            multivariate_hawkes_param_names,
        )

        x0 = multivariate_hawkes_initial_vector(process)
        bounds = multivariate_hawkes_bounds(process)
        names = multivariate_hawkes_param_names(process)
        if not fit_decay:
            bounds = _lock_beta_bounds_multivariate(bounds, x0, M)

        # Regularized MV exp fit. Uses Rust value+grad (recursive when β is
        # shared and fit_decay=False, else dense) and adds the regularizer's
        # penalty + analytic gradient on top.
        from ..._rust import _ext, mv_shared_beta as _shared_beta
        ev_clean = [
            np.ascontiguousarray(np.asarray(e, dtype=np.float64).ravel())
            for e in events_list
        ]
        # Flat global event array for the dense path.
        times_list: list[float] = []
        sources_list: list[int] = []
        for k, ev in enumerate(ev_clean):
            for t in ev:
                times_list.append(float(t))
                sources_list.append(k)
        order = np.argsort(times_list, kind="stable")
        times_all_np = np.asarray(times_list, dtype=np.float64)[order]
        sources_all_np = np.asarray(sources_list, dtype=np.int64)[order]

        shared_beta = _shared_beta(process)
        use_recursive = shared_beta is not None and not fit_decay

        if use_recursive:
            # Build [μ, α row-major] coeffs, length M + M².
            rust_model = _ext.likelihood.MvExpRecursiveLogLik(
                ev_clean, float(T), float(shared_beta),
            )
            grad_buf = np.zeros(M + M * M, dtype=np.float64)

            def coeffs_full_to_compact(x_full: np.ndarray) -> np.ndarray:
                # full layout [μ, (α, β) interleaved] → [μ, α row-major]
                out = np.empty(M + M * M, dtype=np.float64)
                out[:M] = x_full[:M]
                rest = x_full[M:].reshape(M * M, 2)
                out[M:] = rest[:, 0]
                return out

            def grad_compact_to_full(g_compact: np.ndarray) -> np.ndarray:
                out = np.zeros(M + 2 * M * M, dtype=np.float64)
                out[:M] = g_compact[:M]
                # ∂L/∂α_{m,k} only; ∂L/∂β slots stay 0 (β locked via bounds)
                out[M::2][:] = g_compact[M:]
                return out

            def obj_and_grad(x: np.ndarray):
                x_compact = coeffs_full_to_compact(x)
                base = float(rust_model.loss_and_grad(x_compact, grad_buf))
                grad_full = grad_compact_to_full(grad_buf)
                if regularization is not None:
                    base += float(regularization.penalty(x, M))
                    grad_full += regularization.gradient(x, M)
                return base, grad_full
        else:
            # Dense (joint-decay) path: full M + 2·M² layout.
            def split(x: np.ndarray):
                mu_x = np.ascontiguousarray(x[:M], dtype=np.float64)
                rest = x[M:].reshape(M * M, 2)
                a_x = np.ascontiguousarray(rest[:, 0], dtype=np.float64)
                b_x = np.ascontiguousarray(rest[:, 1], dtype=np.float64)
                return mu_x, a_x, b_x

            def merge(g_mu, g_a, g_b):
                out = np.empty(M + 2 * M * M, dtype=np.float64)
                out[:M] = g_mu
                interleaved = out[M:].reshape(M * M, 2)
                interleaved[:, 0] = g_a
                interleaved[:, 1] = g_b
                return out

            def obj_and_grad(x: np.ndarray):
                mu_x, a_x, b_x = split(x)
                val, gm, ga, gb = _ext.likelihood.mv_exp_dense_neg_ll_with_grad(
                    times_all_np, sources_all_np, float(T), int(M), mu_x, a_x, b_x,
                )
                grad = merge(gm, ga, gb)
                base = float(val)
                if regularization is not None:
                    base += float(regularization.penalty(x, M))
                    grad += regularization.gradient(x, M)
                return base, grad

        def obj(x_np):
            return obj_and_grad(x_np)[0]

        import scipy.optimize as spo

        result_opt = spo.minimize(
            obj_and_grad,
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"ftol": self.tol, "maxiter": self.max_iter},
        )
        multivariate_hawkes_apply_vector(process, result_opt.x)
        process.project_params()
        _warn_if_not_converged(result_opt, "MultivariateHawkes")
        final_params = process.get_params()
        final_ll = -result_opt.fun
        aic, bic = compute_information_criteria(final_ll, final_params, n_obs)
        std_errors = None
        if result_opt.success and len(result_opt.x) <= 24:
            std_errors = _finite_difference_std_errors(obj, result_opt.x, names)

        W = np.zeros((M, M))
        for m in range(M):
            for k in range(M):
                W[m, k] = float(process.kernel_matrix[m][k].l1_norm())
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(W))))
        if spectral_radius >= 1.0:
            warnings.warn(
                f"Fitted multivariate Hawkes has spectral radius "
                f"{spectral_radius:.4f} >= 1 after projection; the process is "
                "non-stationary and simulation/forecasts will diverge. "
                "Inspect connectivity_matrix() and consider regularization.",
                RuntimeWarning,
            )

        result = FitResult(
            params=final_params,
            log_likelihood=final_ll,
            aic=aic,
            bic=bic,
            std_errors=std_errors,
            convergence_info={
                "iterations": result_opt.nit,
                "success": result_opt.success,
                "message": result_opt.message,
                "backend": "numpy",
                "model": "multivariate_hawkes",
            },
        )
        result.process = process
        result.events = events_list
        result.T = float(T)
        result.branching_ratio_ = spectral_radius
        if spectral_radius < 1.0:
            result.endogeneity_index_ = spectral_radius / (1.0 + spectral_radius)
        return result

    def _fit_uni_exp_rust(
        self,
        process,
        events: Any,
        T: float,
        n_obs: int,
        *,
        fit_decay: bool = True,
    ) -> FitResult:
        """MLE for UnivariateHawkes(ExponentialKernel) via the Rust core.

        Compute path: scipy.optimize.minimize(L-BFGS-B) drives the loop;
        each value+grad call is a single Rust function (~µs at any
        realistic N) returning analytic gradient. β is locked when
        `fit_decay=False` via a zero-width L-BFGS-B bound.
        """
        import scipy.optimize as spo

        from ..._rust import _ext
        from ..kernels.exponential import ExponentialKernel

        events_np = np.ascontiguousarray(np.asarray(events, dtype=np.float64).ravel())

        # x = [μ, α, β]
        x0 = np.array(
            [float(process.mu), float(process.kernel.alpha), float(process.kernel.beta)],
            dtype=np.float64,
        )
        bounds: list[tuple[float | None, float | None]] = [
            (1e-8, None),       # μ
            (1e-8, 0.999),      # α: positive and below stationarity heuristic
            (1e-8, None),       # β
        ]
        if not fit_decay:
            beta_locked = float(process.kernel.beta)
            bounds[2] = (beta_locked, beta_locked)

        def obj_and_grad(x: np.ndarray):
            val, grad = _ext.likelihood.uni_exp_neg_ll_with_grad(
                events_np, float(T), float(x[0]), float(x[1]), float(x[2]),
            )
            return float(val), grad

        def obj_only(x: np.ndarray) -> float:
            return _ext.likelihood.uni_exp_neg_ll(
                events_np, float(T), float(x[0]), float(x[1]), float(x[2]),
            )

        result_opt = spo.minimize(
            obj_and_grad,
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"ftol": self.tol, "maxiter": self.max_iter},
        )

        process.mu = float(result_opt.x[0])
        process.kernel = ExponentialKernel(
            alpha=float(result_opt.x[1]),
            beta=float(result_opt.x[2]),
            allow_signed=False,
        )
        process.project_params()
        _warn_if_not_converged(result_opt, "UnivariateHawkes (Rust)")

        final_params = process.get_params()
        final_ll = -result_opt.fun
        aic, bic = compute_information_criteria(final_ll, final_params, n_obs)

        std_errors = None
        if result_opt.success and len(result_opt.x) <= 12:
            std_errors = _finite_difference_std_errors(
                obj_only, result_opt.x, ["mu", "alpha", "beta"],
            )

        br = float(process.kernel.l1_norm())
        result = FitResult(
            params=final_params,
            log_likelihood=final_ll,
            aic=aic,
            bic=bic,
            std_errors=std_errors,
            convergence_info={
                "iterations": result_opt.nit,
                "success": result_opt.success,
                "message": result_opt.message,
                "backend": "rust",
                "model": "univariate_hawkes_exp",
            },
        )
        result.process = process
        result.events = events
        result.T = float(T)
        result.branching_ratio_ = br
        if br < 1.0:
            result.endogeneity_index_ = br / (1.0 + br)
        return result

    def _fit_uni_approx_powerlaw_rust(
        self,
        process,
        events: Any,
        T: float,
        n_obs: int,
    ) -> FitResult:
        """MLE for UnivariateHawkes(ApproxPowerLawKernel) via the Rust core.

        Fits (μ, α, β_pow, β_min); r and n_components are fixed structural
        parameters of the kernel approximation.
        """
        import scipy.optimize as spo

        from ..._rust import _ext
        from ..kernels.approx_power_law import ApproxPowerLawKernel

        events_np = np.ascontiguousarray(np.asarray(events, dtype=np.float64).ravel())
        k_struct = process.kernel
        r = float(k_struct.r)
        n_components = int(k_struct.n_components)

        x0 = np.array(
            [float(process.mu), float(k_struct.alpha), float(k_struct.beta_pow), float(k_struct.beta_min)],
            dtype=np.float64,
        )
        bounds: list[tuple[float | None, float | None]] = [
            (1e-8, None),     # μ
            (1e-8, 0.999),    # α (L1 norm; bounded for stationarity heuristic)
            (1e-4, None),     # β_pow
            (1e-4, None),     # β_min
        ]

        def obj_and_grad(x: np.ndarray):
            val, grad = _ext.likelihood.uni_approx_powerlaw_neg_ll_with_grad(
                events_np, float(T),
                float(x[0]), float(x[1]), float(x[2]), float(x[3]),
                r, n_components,
            )
            return float(val), grad

        def obj_only(x: np.ndarray) -> float:
            return _ext.likelihood.uni_approx_powerlaw_neg_ll(
                events_np, float(T),
                float(x[0]), float(x[1]), float(x[2]), float(x[3]),
                r, n_components,
            )

        result_opt = spo.minimize(
            obj_and_grad, x0, method="L-BFGS-B", jac=True, bounds=bounds,
            options={"ftol": self.tol, "maxiter": self.max_iter},
        )

        process.mu = float(result_opt.x[0])
        process.kernel = ApproxPowerLawKernel(
            alpha=float(result_opt.x[1]),
            beta_pow=float(result_opt.x[2]),
            beta_min=float(result_opt.x[3]),
            r=r,
            n_components=n_components,
        )
        process.project_params()
        _warn_if_not_converged(result_opt, "UnivariateHawkes (Rust approx-powerlaw)")

        final_params = process.get_params()
        final_ll = -result_opt.fun
        aic, bic = compute_information_criteria(final_ll, final_params, n_obs)

        std_errors = None
        if result_opt.success and len(result_opt.x) <= 12:
            std_errors = _finite_difference_std_errors(
                obj_only, result_opt.x, ["mu", "alpha", "beta_pow", "beta_min"],
            )

        br = float(process.kernel.l1_norm())
        result = FitResult(
            params=final_params,
            log_likelihood=final_ll,
            aic=aic,
            bic=bic,
            std_errors=std_errors,
            convergence_info={
                "iterations": result_opt.nit,
                "success": result_opt.success,
                "message": result_opt.message,
                "backend": "rust",
                "model": "univariate_hawkes_approx_powerlaw",
            },
        )
        result.process = process
        result.events = events
        result.T = float(T)
        result.branching_ratio_ = br
        if br < 1.0:
            result.endogeneity_index_ = br / (1.0 + br)
        return result

    def _fit_nonlinear_uni_exp_rust(
        self,
        process,
        events: np.ndarray,
        T: float,
        n_obs: int,
        *,
        fit_decay: bool = True,
    ) -> FitResult:
        """MLE for NonlinearHawkes(ExponentialKernel) with builtin link.

        Compensator approximated by trapezoidal rule on a uniform
        n_quad-point grid (matches the existing Python reference).
        Closed-form gradient via chain rule through the link.
        """
        import scipy.optimize as spo

        from ..._rust import _ext, nonlinear_link_kind
        from ..kernels.exponential import ExponentialKernel

        events_np = np.ascontiguousarray(np.asarray(events, dtype=np.float64).ravel())
        link_kind, sigmoid_scale = nonlinear_link_kind(process)
        n_quad = 512

        x0 = np.array(
            [float(process.mu), float(process.kernel.alpha), float(process.kernel.beta)],
            dtype=np.float64,
        )
        bounds: list[tuple[float | None, float | None]] = [
            (1e-8, None),
            (1e-8, 0.999),
            (1e-8, None),
        ]
        if not fit_decay:
            beta_locked = float(process.kernel.beta)
            bounds[2] = (beta_locked, beta_locked)

        def obj_and_grad(x: np.ndarray):
            val, grad = _ext.likelihood.nonlinear_uni_exp_neg_ll_with_grad(
                events_np, float(T),
                float(x[0]), float(x[1]), float(x[2]),
                link_kind, sigmoid_scale, n_quad,
            )
            return float(val), grad

        def obj_only(x: np.ndarray) -> float:
            return _ext.likelihood.nonlinear_uni_exp_neg_ll(
                events_np, float(T),
                float(x[0]), float(x[1]), float(x[2]),
                link_kind, sigmoid_scale, n_quad,
            )

        result_opt = spo.minimize(
            obj_and_grad, x0, method="L-BFGS-B", jac=True, bounds=bounds,
            options={"ftol": self.tol, "maxiter": self.max_iter},
        )

        process.mu = float(result_opt.x[0])
        process.kernel = ExponentialKernel(
            alpha=float(result_opt.x[1]),
            beta=float(result_opt.x[2]),
            allow_signed=False,
        )
        _warn_if_not_converged(result_opt, "NonlinearHawkes (Rust)")

        final_params = process.get_params()
        final_ll = -result_opt.fun
        n_params = len(result_opt.x)
        aic = 2 * n_params - 2 * final_ll
        bic = n_params * float(np.log(n_obs)) - 2 * final_ll if n_obs > 0 else float("nan")

        std_errors = None
        if result_opt.success:
            std_errors = _finite_difference_std_errors(
                obj_only, result_opt.x, ["mu", "alpha", "beta"],
            )

        br = float(process.kernel.l1_norm())
        result = FitResult(
            params=final_params,
            log_likelihood=final_ll,
            aic=aic,
            bic=bic,
            std_errors=std_errors,
            convergence_info={
                "iterations": result_opt.nit,
                "success": result_opt.success,
                "message": result_opt.message,
                "backend": "rust",
                "model": "nonlinear_hawkes_exp",
            },
        )
        result.process = process
        result.events = events
        result.T = float(T)
        result.branching_ratio_ = br
        if br < 1.0:
            result.endogeneity_index_ = br / (1.0 + br)
        return result

    def _fit_marked_uni_exp_rust(
        self,
        process,
        events: np.ndarray,
        marks: np.ndarray,
        T: float,
        n_obs: int,
        *,
        fit_decay: bool = True,
    ) -> FitResult:
        """MLE for MarkedHawkes(ExponentialKernel) via the Rust core.

        Supports all four mark_influence kinds (linear, log, power, callable).
        The mark-influence function g(m) is evaluated once per event before
        optimization starts (it's constant w.r.t. params), and a flat
        ``g_values`` NumPy array is passed to the Rust hot loop. No
        per-pair Python callback overhead even for user callables.
        """
        import scipy.optimize as spo

        from ..._rust import _ext, evaluate_mark_influence
        from ..kernels.exponential import ExponentialKernel

        events_np = np.ascontiguousarray(np.asarray(events, dtype=np.float64).ravel())
        # Pre-evaluate the mark-influence function once. Constant w.r.t. (μ, α, β).
        g_values = np.ascontiguousarray(
            evaluate_mark_influence(process, marks), dtype=np.float64,
        )

        x0 = np.array(
            [
                float(process.mu),
                float(process.kernel.alpha),
                float(process.kernel.beta),
            ],
            dtype=np.float64,
        )
        bounds: list[tuple[float | None, float | None]] = [
            (1e-8, None),
            (1e-8, 0.999),
            (1e-8, None),
        ]
        if not fit_decay:
            beta_locked = float(process.kernel.beta)
            bounds[2] = (beta_locked, beta_locked)

        def obj_and_grad(x: np.ndarray):
            val, grad = _ext.likelihood.marked_uni_exp_neg_ll_with_grad(
                events_np, g_values, float(T),
                float(x[0]), float(x[1]), float(x[2]),
            )
            return float(val), grad

        def obj_only(x: np.ndarray) -> float:
            return _ext.likelihood.marked_uni_exp_neg_ll(
                events_np, g_values, float(T),
                float(x[0]), float(x[1]), float(x[2]),
            )

        result_opt = spo.minimize(
            obj_and_grad,
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"ftol": self.tol, "maxiter": self.max_iter},
        )

        process.mu = float(result_opt.x[0])
        process.kernel = ExponentialKernel(
            alpha=float(result_opt.x[1]),
            beta=float(result_opt.x[2]),
            allow_signed=False,
        )
        _warn_if_not_converged(result_opt, "MarkedHawkes (Rust exp)")

        final_params = process.get_params()
        final_ll = -result_opt.fun
        n_params = len(result_opt.x)
        aic = 2 * n_params - 2 * final_ll
        bic = n_params * float(np.log(n_obs)) - 2 * final_ll if n_obs > 0 else float("nan")

        std_errors = None
        if result_opt.success:
            std_errors = _finite_difference_std_errors(
                obj_only, result_opt.x, ["mu", "alpha", "beta"],
            )

        br = float(process.kernel.l1_norm())
        result = FitResult(
            params=final_params,
            log_likelihood=final_ll,
            aic=aic,
            bic=bic,
            std_errors=std_errors,
            convergence_info={
                "iterations": result_opt.nit,
                "success": result_opt.success,
                "message": result_opt.message,
                "backend": "rust",
                "model": "marked_hawkes_exp",
            },
        )
        result.process = process
        result.events = events
        result.marks_ = marks
        result.T = float(T)
        result.branching_ratio_ = br
        if br < 1.0:
            result.endogeneity_index_ = br / (1.0 + br)
        return result

    def _fit_uni_sumexp_rust(
        self,
        process,
        events: Any,
        T: float,
        n_obs: int,
        *,
        fit_decay: bool = True,
    ) -> FitResult:
        """MLE for UnivariateHawkes(SumExponentialKernel) via the Rust core.

        Coefficient layout matches existing intensify Python:
        ``[μ, α_0..α_{K-1}, β_0..β_{K-1}]`` of length 1 + 2K. β slots are
        locked via zero-width L-BFGS-B bounds when ``fit_decay=False``.
        """
        import scipy.optimize as spo

        from ..._rust import _ext
        from ..kernels.sum_exponential import SumExponentialKernel

        events_np = np.ascontiguousarray(np.asarray(events, dtype=np.float64).ravel())
        K = process.kernel.n_components
        x0 = np.empty(1 + 2 * K, dtype=np.float64)
        x0[0] = float(process.mu)
        x0[1 : 1 + K] = np.asarray(process.kernel.alphas, dtype=np.float64)
        x0[1 + K : 1 + 2 * K] = np.asarray(process.kernel.betas, dtype=np.float64)
        bounds: list[tuple[float | None, float | None]] = [(1e-8, None)]
        bounds.extend([(1e-8, 0.999)] * K)  # alphas
        bounds.extend([(1e-8, None)] * K)   # betas
        if not fit_decay:
            for k in range(K):
                idx = 1 + K + k
                bounds[idx] = (float(x0[idx]), float(x0[idx]))

        def obj_and_grad(x: np.ndarray):
            mu_x = float(x[0])
            alphas_x = np.ascontiguousarray(x[1 : 1 + K], dtype=np.float64)
            betas_x = np.ascontiguousarray(x[1 + K : 1 + 2 * K], dtype=np.float64)
            val, gmu, ga, gb = _ext.likelihood.uni_sumexp_neg_ll_with_grad(
                events_np, float(T), mu_x, alphas_x, betas_x,
            )
            grad = np.empty_like(x)
            grad[0] = gmu
            grad[1 : 1 + K] = ga
            grad[1 + K : 1 + 2 * K] = gb
            return float(val), grad

        def obj_only(x: np.ndarray) -> float:
            mu_x = float(x[0])
            alphas_x = np.ascontiguousarray(x[1 : 1 + K], dtype=np.float64)
            betas_x = np.ascontiguousarray(x[1 + K : 1 + 2 * K], dtype=np.float64)
            return _ext.likelihood.uni_sumexp_neg_ll(
                events_np, float(T), mu_x, alphas_x, betas_x,
            )

        result_opt = spo.minimize(
            obj_and_grad,
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"ftol": self.tol, "maxiter": self.max_iter},
        )

        process.mu = float(result_opt.x[0])
        process.kernel = SumExponentialKernel(
            alphas=[float(a) for a in result_opt.x[1 : 1 + K]],
            betas=[float(b) for b in result_opt.x[1 + K : 1 + 2 * K]],
        )
        process.project_params()
        _warn_if_not_converged(result_opt, "UnivariateHawkes (Rust sumexp)")

        final_params = process.get_params()
        final_ll = -result_opt.fun
        aic, bic = compute_information_criteria(final_ll, final_params, n_obs)

        std_errors = None
        if result_opt.success and len(result_opt.x) <= 24:
            param_names = (
                ["mu"]
                + [f"alpha_{k}" for k in range(K)]
                + [f"beta_{k}" for k in range(K)]
            )
            std_errors = _finite_difference_std_errors(
                obj_only, result_opt.x, param_names,
            )

        br = float(process.kernel.l1_norm())
        result = FitResult(
            params=final_params,
            log_likelihood=final_ll,
            aic=aic,
            bic=bic,
            std_errors=std_errors,
            convergence_info={
                "iterations": result_opt.nit,
                "success": result_opt.success,
                "message": result_opt.message,
                "backend": "rust",
                "model": "univariate_hawkes_sumexp",
            },
        )
        result.process = process
        result.events = events
        result.T = float(T)
        result.branching_ratio_ = br
        if br < 1.0:
            result.endogeneity_index_ = br / (1.0 + br)
        return result

    def _fit_uni_nonparametric_rust(
        self,
        process,
        events: Any,
        T: float,
        n_obs: int,
    ) -> FitResult:
        """MLE for UnivariateHawkes(NonparametricKernel) via the Rust core.

        Edges are FIXED during the fit (chosen up-front by user, e.g.
        `NonparametricKernel.select_bin_count_aic`); the optimizer fits μ
        plus the K bin heights. Closed-form analytic gradient is sparse:
        each (i, j) pair contributes to exactly one bin.

        Resolves ISSUES.md #8: the existing JAX path was unusable above
        N≈300 due to per-pair `kernel.evaluate(jnp.array([lag]))[0]`
        allocations. Rust does a binary-search bin lookup per pair with
        no allocation.
        """
        import scipy.optimize as spo

        from ..._rust import _ext
        from ..kernels.nonparametric import NonparametricKernel

        events_np = np.ascontiguousarray(np.asarray(events, dtype=np.float64).ravel())
        edges_np = np.ascontiguousarray(
            np.asarray(process.kernel.edges, dtype=np.float64).ravel()
        )
        n_bins = len(process.kernel.values)

        # x = [μ, values_0, ..., values_{K-1}]
        x0 = np.empty(1 + n_bins, dtype=np.float64)
        x0[0] = float(process.mu)
        x0[1:] = np.asarray(process.kernel.values, dtype=np.float64)
        bounds: list[tuple[float | None, float | None]] = [(1e-8, None)]
        bounds.extend([(0.0, None)] * n_bins)

        def obj_and_grad(x: np.ndarray):
            mu_x = float(x[0])
            values_x = np.ascontiguousarray(x[1:], dtype=np.float64)
            val, gmu, gv = _ext.likelihood.uni_nonparametric_neg_ll_with_grad(
                events_np, float(T), mu_x, edges_np, values_x,
            )
            grad = np.empty_like(x)
            grad[0] = gmu
            grad[1:] = gv
            return float(val), grad

        def obj_only(x: np.ndarray) -> float:
            mu_x = float(x[0])
            values_x = np.ascontiguousarray(x[1:], dtype=np.float64)
            return _ext.likelihood.uni_nonparametric_neg_ll(
                events_np, float(T), mu_x, edges_np, values_x,
            )

        result_opt = spo.minimize(
            obj_and_grad,
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"ftol": self.tol, "maxiter": self.max_iter},
        )

        process.mu = float(result_opt.x[0])
        process.kernel = NonparametricKernel(
            edges=process.kernel.edges,
            values=[float(v) for v in result_opt.x[1:]],
        )
        process.project_params()
        _warn_if_not_converged(result_opt, "UnivariateHawkes (Rust nonparametric)")

        final_params = process.get_params()
        final_ll = -result_opt.fun
        aic, bic = compute_information_criteria(final_ll, final_params, n_obs)

        std_errors = None
        if result_opt.success and len(result_opt.x) <= 24:
            param_names = ["mu"] + [f"values_{k}" for k in range(n_bins)]
            std_errors = _finite_difference_std_errors(
                obj_only, result_opt.x, param_names,
            )

        br = float(process.kernel.l1_norm())
        result = FitResult(
            params=final_params,
            log_likelihood=final_ll,
            aic=aic,
            bic=bic,
            std_errors=std_errors,
            convergence_info={
                "iterations": result_opt.nit,
                "success": result_opt.success,
                "message": result_opt.message,
                "backend": "rust",
                "model": "univariate_hawkes_nonparametric",
            },
        )
        result.process = process
        result.events = events
        result.T = float(T)
        result.branching_ratio_ = br
        if br < 1.0:
            result.endogeneity_index_ = br / (1.0 + br)
        return result

    def _fit_uni_powerlaw_rust(
        self,
        process,
        events: Any,
        T: float,
        n_obs: int,
    ) -> FitResult:
        """MLE for UnivariateHawkes(PowerLawKernel) via the Rust core.

        PowerLawKernel has no recursive form, so the likelihood is O(N²)
        in pairwise event sums. The Rust implementation provides
        closed-form analytic gradient w.r.t. (μ, α, β, c).
        """
        import scipy.optimize as spo

        from ..._rust import _ext
        from ..kernels.power_law import PowerLawKernel

        events_np = np.ascontiguousarray(np.asarray(events, dtype=np.float64).ravel())

        # x = [μ, α, β, c]
        x0 = np.array(
            [
                float(process.mu),
                float(process.kernel.alpha),
                float(process.kernel.beta),
                float(process.kernel.c),
            ],
            dtype=np.float64,
        )
        bounds: list[tuple[float | None, float | None]] = [
            (1e-8, None),  # μ
            (1e-8, None),  # α (no upper bound — heavy-tail kernels can have l1 > 1)
            (1e-4, None),  # β must be positive
            (1e-4, None),  # c must be positive (singularity at t=0 otherwise)
        ]

        def obj_and_grad(x: np.ndarray):
            val, grad = _ext.likelihood.uni_powerlaw_neg_ll_with_grad(
                events_np, float(T), float(x[0]), float(x[1]), float(x[2]), float(x[3]),
            )
            return float(val), grad

        def obj_only(x: np.ndarray) -> float:
            return _ext.likelihood.uni_powerlaw_neg_ll(
                events_np, float(T), float(x[0]), float(x[1]), float(x[2]), float(x[3]),
            )

        result_opt = spo.minimize(
            obj_and_grad,
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"ftol": self.tol, "maxiter": self.max_iter},
        )

        process.mu = float(result_opt.x[0])
        process.kernel = PowerLawKernel(
            alpha=float(result_opt.x[1]),
            beta=float(result_opt.x[2]),
            c=float(result_opt.x[3]),
        )
        process.project_params()
        _warn_if_not_converged(result_opt, "UnivariateHawkes (Rust power-law)")

        final_params = process.get_params()
        final_ll = -result_opt.fun
        aic, bic = compute_information_criteria(final_ll, final_params, n_obs)

        std_errors = None
        if result_opt.success and len(result_opt.x) <= 12:
            std_errors = _finite_difference_std_errors(
                obj_only, result_opt.x, ["mu", "alpha", "beta", "c"],
            )

        br = float(process.kernel.l1_norm())
        result = FitResult(
            params=final_params,
            log_likelihood=final_ll,
            aic=aic,
            bic=bic,
            std_errors=std_errors,
            convergence_info={
                "iterations": result_opt.nit,
                "success": result_opt.success,
                "message": result_opt.message,
                "backend": "rust",
                "model": "univariate_hawkes_powerlaw",
            },
        )
        result.process = process
        result.events = events
        result.T = float(T)
        result.branching_ratio_ = br
        if br < 1.0:
            result.endogeneity_index_ = br / (1.0 + br)
        return result

    def _fit_numpy(self, process, events: Any, T: float, n_obs: int, *, fit_decay: bool = True) -> FitResult:
        """MLE using SciPy L-BFGS-B on a flat parameter vector."""
        from ..processes.hawkes import UnivariateHawkes
        from ..processes.nonlinear_hawkes import NonlinearHawkes

        if isinstance(process, NonlinearHawkes):
            return self._fit_nonlinear_numpy(process, events, T, n_obs, fit_decay=fit_decay)

        if not isinstance(process, UnivariateHawkes):
            raise NotImplementedError(
                "MLEInference currently only supports UnivariateHawkes; "
                "use process-specific fitting for other models."
            )

        # Fast path: ExponentialKernel routes through Rust (Phase 1 port);
        # PowerLawKernel via Rust (Phase 3). Remaining kernels (Nonparametric,
        # SumExp, ApproxPowerLaw, signed exp) fall through to existing path.
        from ..._rust import (
            has_rust_uni_approx_powerlaw_path,
            has_rust_uni_exp_path,
            has_rust_uni_nonparametric_path,
            has_rust_uni_powerlaw_path,
            has_rust_uni_sumexp_path,
        )
        if has_rust_uni_exp_path(process):
            return self._fit_uni_exp_rust(
                process, events, T, n_obs, fit_decay=fit_decay,
            )
        if has_rust_uni_powerlaw_path(process):
            return self._fit_uni_powerlaw_rust(
                process, events, T, n_obs,
            )
        if has_rust_uni_nonparametric_path(process):
            return self._fit_uni_nonparametric_rust(
                process, events, T, n_obs,
            )
        if has_rust_uni_sumexp_path(process):
            return self._fit_uni_sumexp_rust(
                process, events, T, n_obs, fit_decay=fit_decay,
            )
        if has_rust_uni_approx_powerlaw_path(process):
            return self._fit_uni_approx_powerlaw_rust(
                process, events, T, n_obs,
            )

        threshold = int(config_get("recursive_warning_threshold") or 50_000)
        if process.kernel.has_recursive_form():
            loglik_fn = _recursive_likelihood_numpy
        else:
            if n_obs > threshold:
                warnings.warn(
                    f"{process.kernel.__class__.__name__} requires O(N²) computation. "
                    f"With {n_obs:,} events this may be slow. Consider "
                    f"ApproxPowerLawKernel or another recursive kernel for large N.",
                    PerformanceWarning,
                )
            loglik_fn = _general_likelihood_numpy

        x0 = hawkes_mle_initial_vector(process)
        bounds = hawkes_mle_bounds(process)
        names = hawkes_mle_param_names(process)
        if not fit_decay:
            bounds = _lock_beta_bounds_univariate(bounds, x0, process)

        def obj(x):
            hawkes_mle_apply_vector(process, x)
            return -float(loglik_fn(process, events, T))

        import scipy.optimize as spo

        result_opt = spo.minimize(
            obj,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": self.tol, "maxiter": self.max_iter},
        )

        hawkes_mle_apply_vector(process, result_opt.x)
        process.project_params()
        _warn_if_not_converged(result_opt, "UnivariateHawkes")
        final_params = process.get_params()
        final_ll = -result_opt.fun
        aic, bic = compute_information_criteria(final_ll, final_params, n_obs)

        std_errors = None
        if result_opt.success and len(result_opt.x) <= 12:
            std_errors = _finite_difference_std_errors(obj, result_opt.x, names)

        br = float(process.kernel.l1_norm())
        result = FitResult(
            params=final_params,
            log_likelihood=final_ll,
            aic=aic,
            bic=bic,
            std_errors=std_errors,
            convergence_info={
                "iterations": result_opt.nit,
                "success": result_opt.success,
                "message": result_opt.message,
                "backend": "numpy",
            },
        )
        result.process = process
        result.events = events
        result.T = T
        result.branching_ratio_ = br
        if br < 1.0:
            result.endogeneity_index_ = br / (1.0 + br)
        return result


class MLEInferenceEngine(MLEInference):
    """Alias for backward compatibility."""
    pass


# ---------------------------------------------------------------------------
# JIT-compiled neg-log-likelihood factories
# ---------------------------------------------------------------------------
# These are module-level so JAX caches the compiled traces across fits.
# Each function takes ``events_jax`` and ``T_jax`` as arguments (rather than
# closing over them) so JAX keys the cache by argument shape/dtype — same
# event-array length → cache hit, regardless of which model instance.


def _recursive_likelihood_numpy(process, events: Any, T: float) -> float:
    """NumPy version: plain Python loop for O(N) recursive likelihood."""
    if len(events) == 0:
        return 0.0

    mu = process.mu
    kernel = process.kernel

    R = kernel.recursive_init_state()
    t_prev = 0.0
    log_intensity_sum = 0.0

    for t_i in events:
        dt = float(t_i) - t_prev
        R = kernel.recursive_decay(R, dt)
        exc = kernel.recursive_intensity_excitation(R)
        intensity = mu + exc
        li = np.log(intensity)
        log_intensity_sum += float(li.item() if hasattr(li, "item") else li)
        R = kernel.recursive_absorb(R)
        t_prev = float(t_i)

    tails = T - np.asarray(events, dtype=float)
    comp_tail = float(np.sum(kernel.integrate_vec(tails)))
    compensator = mu * T + comp_tail
    return log_intensity_sum - compensator


def _general_likelihood_numpy(process, events: Any, T: float) -> float:
    """NumPy O(N^2) likelihood via vectorized lag matrix.

    Builds the full lower-triangular lag matrix and evaluates the kernel
    once over all positive lags to avoid per-element array creation.
    """
    import numpy as np

    n = len(events)
    if n == 0:
        return 0.0

    mu = process.mu
    kernel = process.kernel
    events_np = np.asarray(events, dtype=float)

    lags = events_np[:, None] - events_np[None, :]
    causal_mask = lags > 0

    positive_lags = lags[causal_mask]
    if positive_lags.size > 0:
        kernel_at_lags = np.asarray(
            kernel.evaluate(np.asarray(positive_lags)), dtype=float
        )
        kernel_matrix = np.zeros((n, n), dtype=float)
        kernel_matrix[causal_mask] = kernel_at_lags
    else:
        kernel_matrix = np.zeros((n, n), dtype=float)

    intensities = mu + kernel_matrix.sum(axis=1)
    log_intensity_sum = float(np.sum(np.log(intensities)))

    tails = T - np.asarray(events_np, dtype=float)
    comp = mu * T + float(np.sum(kernel.integrate_vec(tails)))

    return log_intensity_sum - comp


# Auto-register this engine upon module import
from . import register_engine

register_engine("mle", MLEInference())