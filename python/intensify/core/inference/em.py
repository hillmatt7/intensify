"""Expectation-Maximization inference for Hawkes processes."""

import warnings

from ...backends import get_backend
from . import FitResult, InferenceEngine, compute_information_criteria

bt = get_backend()


class EMInference(InferenceEngine):
    """
    EM algorithm for UnivariateHawkes with ExponentialKernel.

    Other kernels/processes not yet supported.
    """

    def __init__(self, max_iter: int = 100, mstep_iter: int = 30, lr: float = 0.01, tol: float = 1e-4):
        self.max_iter = int(max_iter)
        self.mstep_iter = int(mstep_iter)
        self.lr = float(lr)
        self.tol = float(tol)

    def fit(self, process, events: bt.array, T: float, **kwargs) -> FitResult:
        from ..kernels.exponential import ExponentialKernel
        from ..processes.hawkes import UnivariateHawkes

        if not isinstance(process, UnivariateHawkes):
            raise NotImplementedError("EMInference currently only supports UnivariateHawkes")
        if not isinstance(process.kernel, ExponentialKernel):
            raise NotImplementedError("EMInference currently only supports ExponentialKernel")

        # Ensure JAX backend for autodiff M-step
        from ...backends._backend import get_backend_name

        backend_name = get_backend_name()
        if backend_name != "jax":
            warnings.warn("EM inference best used with JAX; falling back to MLE", UserWarning)
            from .mle import MLEInference
            return MLEInference().fit(process, events, T, **kwargs)

        import jax
        import jax.numpy as jnp

        events = jnp.array(events)  # ensure JAX array
        n = len(events)

        # Initial parameters from current process state
        mu = float(process.mu)
        alpha = float(process.kernel.alpha)
        beta = float(process.kernel.beta)

        prev_ll = -jnp.inf
        final_iteration = self.max_iter

        for iteration in range(self.max_iter):
            # E-step: compute intensities λ_j via O(N) recursion
            dts = jnp.diff(events, prepend=jnp.array([0.0]))
            mu_iter = mu

            def make_scan_step(mu_val):
                def scan_step(state, dt):
                    decayed = process.kernel.recursive_decay(state, dt)
                    exc = process.kernel.recursive_intensity_excitation(decayed)
                    lam = mu_val + exc
                    new_state = process.kernel.recursive_absorb(decayed)
                    return new_state, lam

                return scan_step

            init_state = process.kernel.recursive_init_state()
            _, lambdas = jax.lax.scan(make_scan_step(mu_iter), init_state, dts)

            # Responsibilities
            p0 = mu / lambdas  # immigrant probability for each event
            p0_sum = jnp.sum(p0)
            pij_sum = n - p0_sum  # total expected offspring count

            # Compute sum of weighted lags: Σ_{i<j} p_ij * Δ_{ij}
            # Build full lag matrix (O(N^2))
            lags = events[:, None] - events[None, :]  # shape (n, n)
            causal_mask = lags > 0
            phi_vals = alpha * beta * jnp.exp(-beta * lags)
            # p_ij = phi_vals / λ_j (broadcast over columns)
            lambda_col = lambdas[:, None]
            p_matrix = jnp.where(causal_mask, phi_vals / lambda_col, 0.0)
            sum_w_dt = jnp.sum(p_matrix * lags)

            # Helper: d_i = T - t_i
            d = T - events

            # M-step: maximize expected complete-data log-likelihood Q using gradient ascent
            def Q_fn(
                mu_new,
                alpha_new,
                beta_new,
                *,
                _p0=p0_sum,
                _pij=pij_sum,
                _swdt=sum_w_dt,
                _d=d,
                _horizon=T,
            ):
                term_imm = _p0 * jnp.log(mu_new)
                term_parents = _pij * (
                    jnp.log(alpha_new)
                    + jnp.log(beta_new)
                    - beta_new * _swdt / _pij
                )
                term_mu_int = -mu_new * _horizon
                integral_sum = alpha_new * jnp.sum(1 - jnp.exp(-beta_new * _d))
                term_kernel_int = -integral_sum
                return term_imm + term_parents + term_mu_int + term_kernel_int

            def loss_fn(mu_p, alpha_p, beta_p):
                return -Q_fn(mu_p, alpha_p, beta_p)

            grad_fn = jax.grad(loss_fn, argnums=(0, 1, 2))

            # Initialize M-step from current params
            mub = jnp.array(mu)
            alphab = jnp.array(alpha)
            betab = jnp.array(beta)

            for _ in range(self.mstep_iter):
                g_mu, g_alpha, g_beta = grad_fn(mub, alphab, betab)
                mub = mub - self.lr * g_mu
                alphab = alphab - self.lr * g_alpha
                betab = betab - self.lr * g_beta
                # Project to feasible set
                mub = jnp.maximum(mub, 1e-8)
                alphab = jnp.clip(alphab, 1e-8, 0.99)  # ensure <1 for stationarity
                betab = jnp.maximum(betab, 1e-8)

            mu, alpha, beta = float(mub), float(alphab), float(betab)

            # Update process parameters to compute new log-likelihood for convergence check
            process.mu = mu
            process.kernel.alpha = alpha
            process.kernel.beta = beta
            ll = process.log_likelihood(events, T)

            if jnp.abs(ll - prev_ll) < self.tol:
                final_iteration = iteration + 1
                break
            prev_ll = ll
        else:
            final_iteration = self.max_iter

        # Finalize
        process.mu = mu
        process.kernel.alpha = alpha
        process.kernel.beta = beta
        aic, bic = compute_information_criteria(ll, process.get_params(), n)
        result = FitResult(
            params=process.get_params(),
            log_likelihood=ll,
            aic=aic,
            bic=bic,
            std_errors=None,
            convergence_info={
                "iterations": final_iteration,
                "final_log_likelihood": float(ll),
            },
        )
        result.process = process
        result.events = events
        result.T = T
        result.branching_ratio_ = alpha
        return result


# Auto-register EM engine
from . import register_engine

register_engine("em", EMInference())