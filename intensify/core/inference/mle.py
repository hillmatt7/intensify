"""Maximum likelihood inference via JAX autodiff or NumPy optim."""

import warnings
from typing import Any

import numpy as np

from ..._config import config_get
from ...backends import get_backend
from ...backends._backend import get_backend_name

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

    Uses JAX autodiff when available; NumPy backend uses scipy.optimize with
    finite differences.

    Parameters (optional)
    ---------------------
    optimizer : str, default "adam"
        Optimizer name for JAX backend ("adam", "sgd", etc.).
    lr : float, default 1e-3
        Learning rate for JAX optimizer.
    max_iter : int, default 5000
        Maximum iterations.
    tol : float, default 1e-5
        Convergence tolerance on gradient norm (JAX) or parameter change (NumPy).
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

        bt = get_backend()
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
            return self._fit_marked_numpy(process, ev, marks, T, n_obs, fit_decay=fit_decay)

        events = bt.asarray(events)
        _validate_events(np.asarray(events), float(T))
        n_obs = len(events)

        try:
            import scipy.optimize as spo  # noqa: F401
        except ImportError:
            spo = None

        if spo is None:
            raise RuntimeError("SciPy is required for MLE; install scipy.")

        jit_ok = (
            get_backend_name() == "jax"
            and getattr(process.kernel, "jit_compatible", False)
        )
        if jit_ok:
            return self._fit_jax(process, events, T, n_obs, fit_decay=fit_decay)
        return self._fit_numpy(process, events, T, n_obs, fit_decay=fit_decay)

    def _fit_jax(self, process, events: Any, T: float, n_obs: int, *, fit_decay: bool = True) -> FitResult:
        """MLE using SciPy L-BFGS-B with exact JAX gradients via jax.value_and_grad."""
        import jax
        import jax.numpy as jnp
        import scipy.optimize as spo

        from ..processes.hawkes import UnivariateHawkes
        from ..processes.nonlinear_hawkes import NonlinearHawkes

        if isinstance(process, NonlinearHawkes):
            return self._fit_nonlinear_numpy(process, events, T, n_obs)
        if not isinstance(process, UnivariateHawkes):
            return self._fit_numpy(process, events, T, n_obs)

        events_jax = jnp.asarray(np.asarray(events, dtype=float), dtype=jnp.float64)
        T_jax = jnp.asarray(float(T), dtype=jnp.float64)

        val_grad = _make_jit_neg_loglik(process, events_jax, T_jax)
        if val_grad is None:
            return self._fit_numpy(process, events, T, n_obs)

        x0 = hawkes_mle_initial_vector(process)
        bounds = hawkes_mle_bounds(process)
        names = hawkes_mle_param_names(process)
        if not fit_decay:
            bounds = _lock_beta_bounds_univariate(bounds, x0, process)

        def obj_and_grad(x_np):
            x_jax = jnp.asarray(x_np, dtype=jnp.float64)
            val, grad = val_grad(x_jax, events_jax, T_jax)
            return float(val), np.asarray(grad, dtype=float)

        result_opt = spo.minimize(
            obj_and_grad,
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"ftol": self.tol, "maxiter": self.max_iter},
        )

        hawkes_mle_apply_vector(process, result_opt.x)
        process.project_params()
        _warn_if_not_converged(result_opt, "UnivariateHawkes (JAX)")
        final_params = process.get_params()
        final_ll = -result_opt.fun
        aic, bic = compute_information_criteria(final_ll, final_params, n_obs)

        std_errors = None
        if result_opt.success and len(result_opt.x) <= 12:
            std_errors = _jax_hessian_std_errors(
                process.kernel, result_opt.x, events_jax, T_jax, names,
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
                "backend": "jax",
                "jit_compiled": True,
            },
        )
        result.process = process
        result.events = events
        result.T = T
        result.branching_ratio_ = br
        if br < 1.0:
            result.endogeneity_index_ = br / (1.0 + br)
        return result

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

        # Try the JAX-JIT fast path first. Requires every cell of
        # kernel_matrix to be ExponentialKernel (the flat-vector layout
        # already enforces this via multivariate_hawkes_initial_vector).
        use_jax_path = get_backend_name() == "jax"

        if use_jax_path:
            import jax.numpy as jnp

            # Flatten events into a single sorted (times, sources) pair.
            times_list = []
            sources_list = []
            for k, ev in enumerate(events_list):
                for t in np.asarray(ev, dtype=float):
                    times_list.append(float(t))
                    sources_list.append(k)
            order = np.argsort(times_list, kind="stable")
            times_all_np = np.asarray(times_list, dtype=float)[order]
            sources_all_np = np.asarray(sources_list, dtype=np.int32)[order]
            times_all = jnp.asarray(times_all_np, dtype=jnp.float64)
            sources_all = jnp.asarray(sources_all_np, dtype=jnp.int32)
            T_jax = jnp.asarray(float(T), dtype=jnp.float64)

            # Use the O(N·M) recursive path when β is shared across every
            # cell (and the decay is locked — otherwise the optimizer
            # would try to move β, invalidating the shared-β assumption).
            shared_beta = _mv_shared_beta(process)
            use_recursive = shared_beta is not None and not fit_decay

            if use_recursive:
                beta_jax = jnp.asarray(shared_beta, dtype=jnp.float64)
                val_grad_rec = _get_mv_val_grad_recursive(M)

                def obj_and_grad(x_np):
                    x_jax = jnp.asarray(x_np, dtype=jnp.float64)
                    val, grad = val_grad_rec(
                        x_jax, times_all, sources_all, T_jax, beta_jax,
                    )
                    obj_val = float(val)
                    if regularization is not None:
                        obj_val += float(regularization.penalty(x_np, M))
                    return obj_val, np.asarray(grad, dtype=float)
            else:
                val_grad = _get_mv_val_grad(M)

                def obj_and_grad(x_np):
                    x_jax = jnp.asarray(x_np, dtype=jnp.float64)
                    val, grad = val_grad(x_jax, times_all, sources_all, T_jax)
                    obj_val = float(val)
                    if regularization is not None:
                        obj_val += float(regularization.penalty(x_np, M))
                    return obj_val, np.asarray(grad, dtype=float)

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
        else:
            _bt = get_backend()
            ev_bt = [_bt.array(e) for e in events_list]

            def obj(x: np.ndarray) -> float:
                multivariate_hawkes_apply_vector(process, x)
                ll = float(process.log_likelihood(ev_bt, T))
                if regularization is not None:
                    ll -= float(regularization.penalty(x, M))
                return -ll

            import scipy.optimize as spo

            result_opt = spo.minimize(
                obj,
                x0,
                method="L-BFGS-B",
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


def _neg_ll_exp(params, events_jax, T_jax):
    import jax
    import jax.numpy as jnp

    mu = params[0]
    alpha = params[1]
    beta = params[2]
    dts = jnp.diff(events_jax, prepend=jnp.array(0.0))

    def scan_fn(R, dt):
        R_decayed = jnp.exp(-beta * dt) * R
        lam = mu + alpha * beta * R_decayed
        R_next = R_decayed + 1.0
        return R_next, jnp.log(lam)

    _, log_terms = jax.lax.scan(scan_fn, jnp.array(0.0), dts)
    log_int_sum = jnp.sum(log_terms)
    comp_tail = jnp.sum(alpha * (1.0 - jnp.exp(-beta * (T_jax - events_jax))))
    comp = mu * T_jax + comp_tail
    return -(log_int_sum - comp)


def _neg_ll_sum_exp(params, events_jax, T_jax, n_components):
    import jax
    import jax.numpy as jnp

    K = n_components
    mu = params[0]
    alphas = params[1: 1 + K]
    betas = params[1 + K: 1 + 2 * K]
    dts = jnp.diff(events_jax, prepend=jnp.array(0.0))

    def scan_fn(R, dt):
        R_decayed = jnp.exp(-betas * dt) * R
        exc = jnp.sum(alphas * betas * R_decayed)
        lam = mu + exc
        R_next = R_decayed + 1.0
        return R_next, jnp.log(lam)

    _, log_terms = jax.lax.scan(scan_fn, jnp.zeros(K), dts)
    log_int_sum = jnp.sum(log_terms)
    tails = T_jax - events_jax
    comp_kernel = jnp.sum(
        alphas[None, :] * (1.0 - jnp.exp(-betas[None, :] * tails[:, None]))
    )
    comp = mu * T_jax + comp_kernel
    return -(log_int_sum - comp)


def _neg_ll_approx_pl(params, events_jax, T_jax, r_f, n_components):
    import jax
    import jax.numpy as jnp

    K = n_components
    k_indices = jnp.arange(K, dtype=jnp.float64)
    mu = params[0]
    alpha = params[1]
    beta_pow = params[2]
    beta_min = params[3]
    betas = beta_min * jnp.power(r_f, k_indices)
    weights_raw = jnp.power(betas, beta_pow - 1.0)
    weights = weights_raw / jnp.sum(weights_raw)
    dts = jnp.diff(events_jax, prepend=jnp.array(0.0))

    def scan_fn(R, dt):
        R_decayed = jnp.exp(-betas * dt) * R
        exc = jnp.sum(alpha * weights * betas * R_decayed)
        lam = mu + exc
        R_next = R_decayed + 1.0
        return R_next, jnp.log(lam)

    _, log_terms = jax.lax.scan(scan_fn, jnp.zeros(K), dts)
    log_int_sum = jnp.sum(log_terms)
    tails = T_jax - events_jax
    comp_kernel = jnp.sum(
        alpha * weights[None, :] * (1.0 - jnp.exp(-betas[None, :] * tails[:, None]))
    )
    comp = mu * T_jax + comp_kernel
    return -(log_int_sum - comp)


def _neg_ll_mv_exp(params, times_all, sources_all, T_jax, M):
    """Vectorized multivariate exp-Hawkes neg-log-likelihood.

    Parameters layout (flat): ``[mu (M), (alpha_mk, beta_mk) interleaved per
    row-major (m, k)]`` — matches
    ``multivariate_hawkes_initial_vector``.

    Uses a dense N×N lag matrix (N = total events across dims) with a
    strict-lower-triangular causal mask. ``times_all`` and ``sources_all``
    are sorted by time. Shape ``(N,)`` each; ``sources_all`` is int.
    """
    import jax.numpy as jnp

    mu = params[:M]
    rest = params[M:].reshape(M * M, 2)
    alpha_mat = rest[:, 0].reshape(M, M)
    beta_mat = rest[:, 1].reshape(M, M)

    N = times_all.shape[0]

    # Pairwise lag matrix, strict lower triangular mask
    lags = times_all[:, None] - times_all[None, :]  # (N, N)
    causal = lags > 0

    # Gather α_{src_i, src_j}, β_{src_i, src_j} for every (i, j) pair
    src_i = sources_all[:, None]  # (N, 1)
    src_j = sources_all[None, :]  # (1, N)
    alpha_ij = alpha_mat[src_i, src_j]  # (N, N)
    beta_ij = beta_mat[src_i, src_j]   # (N, N)

    # phi_{mk}(lag) = α_mk · β_mk · exp(-β_mk · lag), zero off-causal
    safe_lags = jnp.where(causal, lags, 0.0)
    phi = jnp.where(
        causal, alpha_ij * beta_ij * jnp.exp(-beta_ij * safe_lags), 0.0,
    )
    # Intensity of event i's source dim at time t_i = μ_{src_i} + Σ_j phi[i, j]
    intensities = mu[sources_all] + jnp.sum(phi, axis=1)
    # Floor for numerical stability
    sum_log = jnp.sum(jnp.log(jnp.maximum(intensities, 1e-30)))

    # Compensator: Λ_m(T) = μ_m·T + Σ_j α_{m, src_j}·(1 - exp(-β_{m, src_j}·(T - t_j)))
    tails = T_jax - times_all  # (N,)
    # For each m, sum over j: alpha_mat[m, src_j] · (1 - exp(-beta_mat[m, src_j] · tails_j))
    # We broadcast: alpha_per_m[m, j] = alpha_mat[m, src_j]
    alpha_per_m = alpha_mat[:, sources_all]  # (M, N)
    beta_per_m = beta_mat[:, sources_all]   # (M, N)
    integ = alpha_per_m * (1.0 - jnp.exp(-beta_per_m * tails[None, :]))  # (M, N)
    compensator_per_m = mu * T_jax + jnp.sum(integ, axis=1)  # (M,)
    total_comp = jnp.sum(compensator_per_m)

    return -(sum_log - total_comp)


_MV_VAL_GRAD_CACHE: dict = {}


def _get_mv_val_grad(M: int):
    """Cached jit-compiled value+grad for multivariate exp Hawkes at dim *M*."""
    import jax

    cached = _MV_VAL_GRAD_CACHE.get(M)
    if cached is not None:
        return cached

    def fn(params, times_all, sources_all, T_jax):
        return _neg_ll_mv_exp(params, times_all, sources_all, T_jax, M)

    val_grad = jax.jit(jax.value_and_grad(fn, argnums=0))
    _MV_VAL_GRAD_CACHE[M] = val_grad
    return val_grad


def _neg_ll_mv_exp_recursive(params, times_all, sources_all, T_jax, M, beta_scalar):
    """O(N·M) multivariate exp-Hawkes neg-log-likelihood, shared β across cells.

    Uses a Hawkes recursion with an M-vector state R_k that tracks the
    exponential-decay sum of past events from source k. Correct only when
    every β_{m,k} is the same scalar; the caller is responsible for
    verifying this (via ``_mv_shared_beta``).

    Parameters layout: same as ``_neg_ll_mv_exp`` — ``[mu (M),
    (alpha_mk, beta_mk) interleaved]``. The β slots are ignored (replaced
    by ``beta_scalar``).
    """
    import jax
    import jax.numpy as jnp

    mu = params[:M]
    rest = params[M:].reshape(M * M, 2)
    alpha_mat = rest[:, 0].reshape(M, M)

    beta = beta_scalar

    N = times_all.shape[0]
    # Pre-compute dt between successive events (including t_0 from 0)
    dts = jnp.diff(times_all, prepend=jnp.array(0.0, dtype=times_all.dtype))
    # One-hot source indicators, shape (N, M)
    one_hot = jax.nn.one_hot(sources_all, M, dtype=params.dtype)

    def step(R, inputs):
        dt, src_onehot = inputs
        R_dec = R * jnp.exp(-beta * dt)  # (M,)
        # intensity of source dim i: μ_{src_i} + Σ_k α_{src_i, k} · β · R_dec_k
        # src_i = argmax(src_onehot); use one_hot to pick the right row of alpha
        alpha_row = src_onehot @ alpha_mat  # (M,)
        mu_src = jnp.dot(src_onehot, mu)
        lam = mu_src + beta * jnp.dot(alpha_row, R_dec)
        log_lam = jnp.log(jnp.maximum(lam, 1e-30))
        R_new = R_dec + src_onehot  # absorb the event into its source dim
        return R_new, log_lam

    _, log_terms = jax.lax.scan(step, jnp.zeros(M, dtype=params.dtype), (dts, one_hot))
    sum_log = jnp.sum(log_terms)

    # Compensator: Λ_m = μ_m·T + Σ_j α_{m, src_j} · (1 − exp(−β·(T − t_j)))
    tails = T_jax - times_all
    per_j = 1.0 - jnp.exp(-beta * tails)  # (N,)
    # Σ_j α_{m, src_j} · per_j  = Σ_j (alpha_mat[m] @ one_hot[j]) · per_j
    #                           = alpha_mat @ (one_hot.T · per_j)   (M,)
    per_k = one_hot.T @ per_j  # (M,) = Σ_{j: src(j)=k} per_j
    integ_per_m = alpha_mat @ per_k  # (M,)
    compensator_per_m = mu * T_jax + integ_per_m
    total_comp = jnp.sum(compensator_per_m)

    return -(sum_log - total_comp)


_MV_VAL_GRAD_REC_CACHE: dict = {}


def _get_mv_val_grad_recursive(M: int):
    """Cached jit-compiled value+grad for the shared-β recursive path."""
    import jax

    cached = _MV_VAL_GRAD_REC_CACHE.get(M)
    if cached is not None:
        return cached

    def fn(params, times_all, sources_all, T_jax, beta_scalar):
        return _neg_ll_mv_exp_recursive(
            params, times_all, sources_all, T_jax, M, beta_scalar,
        )

    val_grad = jax.jit(jax.value_and_grad(fn, argnums=0))
    _MV_VAL_GRAD_REC_CACHE[M] = val_grad
    return val_grad



def _mv_shared_beta(process) -> float | None:
    """Return the shared β if every cell in the kernel matrix has the same
    ExponentialKernel β (and is an ExponentialKernel); else None.
    """
    from ..kernels.exponential import ExponentialKernel

    betas = []
    for row in process.kernel_matrix:
        for k in row:
            if not isinstance(k, ExponentialKernel):
                return None
            betas.append(float(k.beta))
    if not betas:
        return None
    first = betas[0]
    if all(abs(b - first) < 1e-12 for b in betas):
        return first
    return None


def _neg_ll_power_law(params, events_jax, T_jax):
    import jax.numpy as jnp

    mu = params[0]
    alpha = params[1]
    beta = params[2]
    c = params[3]
    lags = events_jax[:, None] - events_jax[None, :]
    causal = lags > 0
    safe_lags = jnp.maximum(lags, 0.0)
    kernel_vals = jnp.where(
        causal, alpha * jnp.power(safe_lags + c, -(1.0 + beta)), 0.0,
    )
    intensities = mu + jnp.sum(kernel_vals, axis=1)
    log_int_sum = jnp.sum(jnp.log(intensities))
    tails = T_jax - events_jax
    term0 = jnp.power(c, -beta)
    termt = jnp.power(tails + c, -beta)
    comp_kernel = jnp.sum((alpha / beta) * (term0 - termt))
    comp = mu * T_jax + comp_kernel
    return -(log_int_sum - comp)


# Cache JIT-compiled value-and-grad factories so they are only built once
# per kernel-shape. Caches by static-arg keys (kernel kind + n_components).
_VAL_GRAD_CACHE: dict = {}


def _get_val_grad(kernel_kind: str, n_components: int = 0, r_f: float = 0.0):
    """Return a (cached) jit-compiled value+grad for the given kernel kind.

    The returned callable signature is ``f(params, events_jax, T_jax)`` and
    returns a JAX (value, grad_wrt_params) pair. JAX traces once per
    unique (events_jax shape, dtype) and reuses the trace across fits.
    """
    import jax

    key = (kernel_kind, int(n_components), float(r_f))
    cached = _VAL_GRAD_CACHE.get(key)
    if cached is not None:
        return cached

    if kernel_kind == "exp":
        fn = _neg_ll_exp
    elif kernel_kind == "sum_exp":
        K = n_components

        def fn(params, events_jax, T_jax):
            return _neg_ll_sum_exp(params, events_jax, T_jax, K)
    elif kernel_kind == "approx_pl":
        K = n_components
        rf = r_f

        def fn(params, events_jax, T_jax):
            return _neg_ll_approx_pl(params, events_jax, T_jax, rf, K)
    elif kernel_kind == "power_law":
        fn = _neg_ll_power_law
    else:
        return None

    val_grad = jax.jit(jax.value_and_grad(fn, argnums=0))
    _VAL_GRAD_CACHE[key] = val_grad
    return val_grad


# Back-compat shims. The pre-0.2.0 API exposed per-kernel factories that
# returned a closure over (events_jax, T_jax). New code should use
# ``_get_val_grad`` and call ``f(params, events_jax, T_jax)`` directly.

def _make_jit_neg_loglik_exp(events_jax, T_jax):
    return lambda params: _neg_ll_exp(params, events_jax, T_jax)


def _make_jit_neg_loglik_sum_exp(events_jax, T_jax, n_components):
    return lambda params: _neg_ll_sum_exp(params, events_jax, T_jax, n_components)


def _make_jit_neg_loglik_approx_pl(events_jax, T_jax, r, n_components):
    return lambda params: _neg_ll_approx_pl(
        params, events_jax, T_jax, float(r), int(n_components),
    )


def _make_jit_neg_loglik_power_law(events_jax, T_jax):
    return lambda params: _neg_ll_power_law(params, events_jax, T_jax)


def _make_jit_neg_loglik(process, events_jax, T_jax):
    """Return a (cached) jit'd value+grad for the given kernel kind.

    The returned callable signature is ``f(params, events_jax, T_jax)``.
    Returns ``None`` if the kernel type has no JIT implementation.
    """
    from ..kernels.approx_power_law import ApproxPowerLawKernel
    from ..kernels.exponential import ExponentialKernel
    from ..kernels.power_law import PowerLawKernel
    from ..kernels.sum_exponential import SumExponentialKernel

    kernel = process.kernel
    if not getattr(kernel, "jit_compatible", True):
        return None

    if isinstance(kernel, ExponentialKernel):
        return _get_val_grad("exp")
    if isinstance(kernel, SumExponentialKernel):
        return _get_val_grad("sum_exp", n_components=kernel.n_components)
    if isinstance(kernel, ApproxPowerLawKernel):
        return _get_val_grad(
            "approx_pl", n_components=kernel.n_components, r_f=float(kernel.r),
        )
    if isinstance(kernel, PowerLawKernel):
        return _get_val_grad("power_law")
    return None


_HESSIAN_CACHE: dict = {}


def _get_hessian_fn(kernel_kind: str, n_components: int = 0, r_f: float = 0.0):
    """Return a (cached) jit-compiled Hessian wrt params for the given kernel."""
    import jax

    key = (kernel_kind, int(n_components), float(r_f))
    cached = _HESSIAN_CACHE.get(key)
    if cached is not None:
        return cached

    if kernel_kind == "exp":
        fn = _neg_ll_exp
    elif kernel_kind == "sum_exp":
        K = n_components

        def fn(params, events_jax, T_jax):
            return _neg_ll_sum_exp(params, events_jax, T_jax, K)
    elif kernel_kind == "approx_pl":
        K = n_components
        rf = r_f

        def fn(params, events_jax, T_jax):
            return _neg_ll_approx_pl(params, events_jax, T_jax, rf, K)
    elif kernel_kind == "power_law":
        fn = _neg_ll_power_law
    else:
        return None

    h = jax.jit(jax.hessian(fn, argnums=0))
    _HESSIAN_CACHE[key] = h
    return h


def _hessian_kind_for(kernel) -> tuple[str, int, float] | None:
    from ..kernels.approx_power_law import ApproxPowerLawKernel
    from ..kernels.exponential import ExponentialKernel
    from ..kernels.power_law import PowerLawKernel
    from ..kernels.sum_exponential import SumExponentialKernel

    if isinstance(kernel, ExponentialKernel):
        return "exp", 0, 0.0
    if isinstance(kernel, SumExponentialKernel):
        return "sum_exp", kernel.n_components, 0.0
    if isinstance(kernel, ApproxPowerLawKernel):
        return "approx_pl", kernel.n_components, float(kernel.r)
    if isinstance(kernel, PowerLawKernel):
        return "power_law", 0, 0.0
    return None


def _jax_hessian_std_errors(kernel_or_fn, x_opt, *args, **kwargs):
    """Compute standard errors via the Hessian.

    New signature (preferred):
        _jax_hessian_std_errors(kernel, x_opt, events_jax, T_jax, names)
    Back-compat signature (pre-0.2.0):
        _jax_hessian_std_errors(neg_ll_fn, x_opt, names)
        where neg_ll_fn is a callable taking ``params`` only.
    """
    try:
        import jax
        import jax.numpy as jnp

        # Detect back-compat call: third positional is the names list.
        if len(args) == 1 and isinstance(args[0], (list, tuple)) and not kwargs:
            neg_ll_fn = kernel_or_fn
            names = list(args[0])
            x_jax = jnp.asarray(x_opt, dtype=jnp.float64)
            H = jax.hessian(neg_ll_fn)(x_jax)
        else:
            # New signature: (kernel, x_opt, events_jax, T_jax, names)
            kernel = kernel_or_fn
            events_jax, T_jax, names = args[0], args[1], args[2]
            kind = _hessian_kind_for(kernel)
            if kind is None:
                return None
            h_fn = _get_hessian_fn(*kind)
            if h_fn is None:
                return None
            x_jax = jnp.asarray(x_opt, dtype=jnp.float64)
            H = h_fn(x_jax, events_jax, T_jax)
        H = np.asarray(H, dtype=float)
        H = (H + H.T) / 2.0
        n_p = len(x_opt)

        cond = np.linalg.cond(H)
        if cond > 1e12:
            warnings.warn(
                f"Hessian condition number {cond:.2e} is very large; "
                "standard errors may be unreliable.",
                RuntimeWarning,
            )

        cov = np.linalg.pinv(H + 1e-8 * np.eye(n_p))
        return {
            names[i]: float(np.sqrt(max(cov[i, i], 0.0)))
            for i in range(n_p)
        }
    except Exception as e:
        warnings.warn(f"JAX Hessian computation failed: {e}", RuntimeWarning)
        return None


def _recursive_likelihood(process, events: Any, T: float) -> float:
    """O(N) recursive likelihood for exponential-family kernels."""
    bt = get_backend()
    if len(events) == 0:
        return 0.0

    dts = bt.diff(events, prepend=bt.array([0.0]))

    def scan_step(state, dt):
        decayed = process.kernel.recursive_decay(state, dt)
        exc = process.kernel.recursive_intensity_excitation(decayed)
        intensity_contrib = process.mu + exc
        new_state = process.kernel.recursive_absorb(decayed)
        return new_state, intensity_contrib

    init_state = process.kernel.recursive_init_state()
    _, intensities = bt.lax.scan(scan_step, init_state, dts)

    log_intensity_sum = bt.sum(bt.log(intensities))
    kernel_integrals = process.kernel.integrate_vec(T - events)
    compensator = process.mu * T + bt.sum(kernel_integrals)
    return float(log_intensity_sum - compensator)


def _general_likelihood(process, events: Any, T: float) -> float:
    """O(N²) general likelihood via pairwise kernel evaluation."""
    bt = get_backend()
    n = len(events)
    if n == 0:
        return 0.0

    # Build (n, n) matrix of time lags
    lags = events[:, None] - events[None, :]  # shape (n, n)
    causal_mask = lags > 0

    # Evaluate kernel at all lags, mask out non-causal
    kernel_matrix = bt.where(causal_mask, process.kernel.evaluate(lags), 0.0)
    intensities = process.mu + bt.sum(kernel_matrix, axis=1)
    log_intensity_sum = bt.sum(bt.log(intensities))

    # Compensator: μ T + Σ ∫_0^{T-t_i} φ(s) ds
    kernel_integrals = process.kernel.integrate_vec(T - events)
    compensator = process.mu * T + bt.sum(kernel_integrals)

    return float(log_intensity_sum - compensator)


def _recursive_likelihood_numpy(process, events: Any, T: float) -> float:
    """NumPy version: plain Python loop for O(N) recursive likelihood."""
    bt = get_backend()
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
        li = bt.log(intensity)
        log_intensity_sum += float(li.item() if hasattr(li, "item") else li)
        R = kernel.recursive_absorb(R)
        t_prev = float(t_i)

    tails = T - bt.asarray(events, dtype=float)
    comp_tail = float(bt.sum(kernel.integrate_vec(tails)))
    compensator = mu * T + comp_tail
    return log_intensity_sum - compensator


def _general_likelihood_numpy(process, events: Any, T: float) -> float:
    """NumPy O(N^2) likelihood via vectorized lag matrix.

    Builds the full lower-triangular lag matrix and evaluates the kernel
    once over all positive lags to avoid per-element array creation.
    """
    bt = get_backend()
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
            kernel.evaluate(bt.array(positive_lags)), dtype=float
        )
        kernel_matrix = np.zeros((n, n), dtype=float)
        kernel_matrix[causal_mask] = kernel_at_lags
    else:
        kernel_matrix = np.zeros((n, n), dtype=float)

    intensities = mu + kernel_matrix.sum(axis=1)
    log_intensity_sum = float(np.sum(np.log(intensities)))

    tails = T - bt.asarray(events_np, dtype=float)
    comp = mu * T + float(bt.sum(kernel.integrate_vec(tails)))

    return log_intensity_sum - comp


# Auto-register this engine upon module import
from . import register_engine

register_engine("mle", MLEInference())