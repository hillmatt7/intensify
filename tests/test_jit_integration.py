"""
Comprehensive integration tests for the JIT acceleration changes.

Exercises every affected module with realistic demo data:
  - Kernel integrate_vec (all 5 kernel types)
  - Kernel jit_compatible property
  - Vectorized UnivariateHawkes.intensity
  - Vectorized MultivariateHawkes._log_likelihood_dim
  - Vectorized NonlinearHawkes (pre_intensity, compensator, log_likelihood)
  - JIT-compiled neg-log-likelihood factories (all 4 kernel types)
  - JAX value_and_grad correctness (vs finite differences)
  - JAX Hessian-based standard errors
  - End-to-end MLE fit via _fit_jax path (ExponentialKernel)
  - End-to-end MLE fit via _fit_jax path (SumExponentialKernel)
  - End-to-end MLE fit via _fit_jax path (PowerLawKernel)
  - End-to-end MLE fit via _fit_jax path (ApproxPowerLawKernel)
  - MLE fit NumPy fallback for NonparametricKernel
  - JAX while-loop simulation (ogata_thinning)
  - Multivariate Hawkes log-likelihood vectorized path
  - Multivariate Hawkes MLE fit
  - NonlinearHawkes fit via MLE
  - EM inference still works after likelihood refactor
  - Recursive / general likelihood cross-check after integrate_vec refactor
"""

import numpy as np
import pytest

from intensify.backends import get_backend
from intensify.backends._backend import get_backend_name

bt = get_backend()

_HAS_JAX = get_backend_name() == "jax"

# ---- Deterministic demo events for reproducibility ----
EVENTS_SMALL = bt.array([0.2, 0.8, 1.5, 2.3, 3.1, 4.0, 5.5, 7.0, 8.2, 9.0])
EVENTS_MEDIUM = bt.array([
    0.12, 0.34, 0.65, 0.91, 1.23, 1.58, 1.87, 2.15, 2.49, 2.78,
    3.11, 3.45, 3.72, 4.09, 4.41, 4.78, 5.12, 5.49, 5.83, 6.21,
    6.55, 6.89, 7.23, 7.58, 7.91, 8.29, 8.65, 9.01, 9.38, 9.72,
])
T_SMALL = 10.0
T_MEDIUM = 10.0


# =====================================================================
# 1. integrate_vec: all 5 kernel types
# =====================================================================


class TestIntegrateVecAllKernels:
    """Verify integrate_vec matches scalar integrate for every kernel type."""

    def _check(self, kernel, t_arr):
        vec = kernel.integrate_vec(t_arr)
        ref = bt.array([kernel.integrate(float(ti)) for ti in t_arr])
        np.testing.assert_allclose(vec, ref, atol=1e-12, rtol=1e-12)

    def test_exponential(self):
        from intensify.core.kernels import ExponentialKernel
        self._check(ExponentialKernel(0.3, 1.5), bt.array([0.01, 0.1, 0.5, 2.0, 10.0, 100.0]))

    def test_sum_exponential(self):
        from intensify.core.kernels import SumExponentialKernel
        self._check(SumExponentialKernel([0.2, 0.1], [1.0, 5.0]), bt.array([0.01, 0.5, 2.0, 10.0]))

    def test_power_law(self):
        from intensify.core.kernels import PowerLawKernel
        self._check(PowerLawKernel(0.4, 0.7, 0.2), bt.array([0.01, 0.1, 1.0, 5.0, 20.0]))

    def test_approx_power_law(self):
        from intensify.core.kernels import ApproxPowerLawKernel
        self._check(ApproxPowerLawKernel(0.3, 0.5, 0.1, r=2.0, n_components=6), bt.array([0.05, 0.5, 2.0]))

    def test_nonparametric(self):
        from intensify.core.kernels import NonparametricKernel
        self._check(
            NonparametricKernel([0.0, 0.5, 1.0, 2.0, 4.0], [0.4, 0.3, 0.15, 0.05]),
            bt.array([0.3, 0.6, 1.5, 3.0, 5.0]),
        )


# =====================================================================
# 2. jit_compatible flag
# =====================================================================


class TestJitCompatibleFlag:
    def test_all_parametric_true(self):
        from intensify.core.kernels import (
            ApproxPowerLawKernel,
            ExponentialKernel,
            PowerLawKernel,
            SumExponentialKernel,
        )
        assert ExponentialKernel(0.3, 1.5).jit_compatible is True
        assert SumExponentialKernel([0.2], [1.0]).jit_compatible is True
        assert PowerLawKernel(0.3, 0.8, 0.5).jit_compatible is True
        assert ApproxPowerLawKernel(0.3, 0.5, 0.1).jit_compatible is True

    def test_nonparametric_false(self):
        from intensify.core.kernels import NonparametricKernel
        assert NonparametricKernel([0.0, 1.0], [0.1]).jit_compatible is False


# =====================================================================
# 3. Vectorized UnivariateHawkes.intensity
# =====================================================================


class TestVectorizedIntensityDemo:
    def test_array_matches_scalar_loop(self):
        from intensify.core.kernels import ExponentialKernel
        from intensify.core.processes import UnivariateHawkes

        proc = UnivariateHawkes(mu=0.5, kernel=ExponentialKernel(0.3, 1.5))
        history = bt.array([0.5, 1.2, 2.0, 3.5])
        t_arr = bt.array([4.0, 5.0, 6.0, 7.0, 8.0])

        vec_out = proc.intensity(t_arr, history)
        scalar_out = bt.array([proc.intensity(float(ti), history) for ti in t_arr])
        np.testing.assert_allclose(vec_out, scalar_out, atol=1e-10)

    def test_empty_history_array(self):
        from intensify.core.kernels import ExponentialKernel
        from intensify.core.processes import UnivariateHawkes

        proc = UnivariateHawkes(mu=0.7, kernel=ExponentialKernel(0.3, 1.5))
        result = proc.intensity(bt.array([1.0, 2.0, 3.0]), bt.array([]))
        np.testing.assert_allclose(result, [0.7, 0.7, 0.7])

    def test_power_law_kernel_vectorized(self):
        from intensify.core.kernels import PowerLawKernel
        from intensify.core.processes import UnivariateHawkes

        proc = UnivariateHawkes(mu=0.5, kernel=PowerLawKernel(0.3, 0.8, 0.5))
        history = bt.array([0.5, 1.5, 2.5])
        t_arr = bt.array([3.0, 4.0, 5.0])

        vec_out = proc.intensity(t_arr, history)
        scalar_out = bt.array([proc.intensity(float(ti), history) for ti in t_arr])
        np.testing.assert_allclose(vec_out, scalar_out, atol=1e-10)


# =====================================================================
# 4. Vectorized MultivariateHawkes._log_likelihood_dim
# =====================================================================


class TestVectorizedMultivariateLLDemo:
    def test_bivariate_ll_finite(self):
        from intensify.core.kernels import ExponentialKernel
        from intensify.core.processes.hawkes import MultivariateHawkes

        kernels = [
            [ExponentialKernel(0.12, 1.0), ExponentialKernel(0.08, 1.5)],
            [ExponentialKernel(0.10, 1.2), ExponentialKernel(0.15, 1.0)],
        ]
        mv = MultivariateHawkes(n_dims=2, mu=[0.3, 0.4], kernel=kernels)
        events = [bt.array([0.5, 1.5, 3.0, 5.0, 7.0]), bt.array([0.8, 2.0, 4.0, 6.5, 9.0])]
        ll = mv.log_likelihood(events, T=10.0)
        assert np.isfinite(ll), f"Multivariate LL not finite: {ll}"

    def test_trivariate_ll_finite(self):
        from intensify.core.kernels import ExponentialKernel
        from intensify.core.processes.hawkes import MultivariateHawkes

        M = 3
        kernels = [[ExponentialKernel(0.08, 1.0) for _ in range(M)] for _ in range(M)]
        mv = MultivariateHawkes(n_dims=M, mu=[0.3, 0.3, 0.3], kernel=kernels)
        events = [
            bt.array([0.3, 1.2, 2.5, 4.0, 6.0]),
            bt.array([0.5, 1.8, 3.5, 5.5]),
            bt.array([1.0, 2.2, 4.5, 7.0, 9.0]),
        ]
        ll = mv.log_likelihood(events, T=10.0)
        assert np.isfinite(ll), f"Trivariate LL not finite: {ll}"

    def test_empty_dim_handled(self):
        from intensify.core.kernels import ExponentialKernel
        from intensify.core.processes.hawkes import MultivariateHawkes

        kernels = [
            [ExponentialKernel(0.1, 1.0), ExponentialKernel(0.1, 1.0)],
            [ExponentialKernel(0.1, 1.0), ExponentialKernel(0.1, 1.0)],
        ]
        mv = MultivariateHawkes(n_dims=2, mu=[0.3, 0.3], kernel=kernels)
        events = [bt.array([0.5, 1.5, 3.0]), bt.zeros(0)]
        ll = mv.log_likelihood(events, T=4.0)
        assert np.isfinite(ll)


# =====================================================================
# 5. Vectorized NonlinearHawkes
# =====================================================================


class TestVectorizedNonlinearHawkesDemo:
    def test_softplus_ll_finite(self):
        from intensify.core.kernels import ExponentialKernel
        from intensify.core.processes.nonlinear_hawkes import NonlinearHawkes

        k = ExponentialKernel(0.3, 1.5, allow_signed=True)
        proc = NonlinearHawkes(mu=0.5, kernel=k, link_function="softplus")
        ll = proc.log_likelihood(EVENTS_SMALL, T_SMALL, n_quad=128)
        assert np.isfinite(ll), f"NonlinearHawkes softplus LL: {ll}"

    def test_relu_ll_finite(self):
        from intensify.core.kernels import ExponentialKernel
        from intensify.core.processes.nonlinear_hawkes import NonlinearHawkes

        k = ExponentialKernel(0.3, 1.5, allow_signed=True)
        proc = NonlinearHawkes(mu=0.5, kernel=k, link_function="relu")
        ll = proc.log_likelihood(EVENTS_SMALL, T_SMALL, n_quad=128)
        assert np.isfinite(ll), f"NonlinearHawkes relu LL: {ll}"

    def test_inhibition_ll_finite(self):
        """Inhibitory kernel (negative alpha) with softplus link."""
        from intensify.core.kernels import ExponentialKernel
        from intensify.core.processes.nonlinear_hawkes import NonlinearHawkes

        k = ExponentialKernel(-0.2, 1.5, allow_signed=True)
        proc = NonlinearHawkes(mu=1.0, kernel=k, link_function="softplus")
        ll = proc.log_likelihood(EVENTS_SMALL, T_SMALL, n_quad=128)
        assert np.isfinite(ll), f"Inhibition LL: {ll}"

    def test_pre_intensity_matches_manual(self):
        from intensify.core.kernels import ExponentialKernel
        from intensify.core.processes.nonlinear_hawkes import NonlinearHawkes

        k = ExponentialKernel(0.4, 2.0, allow_signed=True)
        proc = NonlinearHawkes(mu=0.5, kernel=k)
        history = np.array([1.0, 2.0])
        t = 3.0
        pre = proc._pre_intensity(t, history)
        expected = 0.5 + 0.4 * 2.0 * np.exp(-2.0 * 2.0) + 0.4 * 2.0 * np.exp(-2.0 * 1.0)
        np.testing.assert_allclose(pre, expected, atol=1e-10)


# =====================================================================
# 6. JIT-compiled neg-log-likelihood factories (JAX only)
# =====================================================================


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not available")
class TestJitNegLLFactories:
    """Each JIT factory must produce the same -LL as the Python/NumPy path."""

    def _numpy_neg_ll(self, proc, events, T):
        from intensify.core.inference.mle import (
            _general_likelihood_numpy,
            _recursive_likelihood_numpy,
        )
        if proc.kernel.has_recursive_form():
            return -float(_recursive_likelihood_numpy(proc, events, T))
        return -float(_general_likelihood_numpy(proc, events, T))

    def test_exponential(self):
        import jax.numpy as jnp
        from intensify.core.inference.mle import _make_jit_neg_loglik_exp
        from intensify.core.kernels import ExponentialKernel
        from intensify.core.processes import UnivariateHawkes

        mu, a, b = 0.5, 0.3, 1.5
        proc = UnivariateHawkes(mu=mu, kernel=ExponentialKernel(a, b))
        ej = jnp.asarray(np.asarray(EVENTS_MEDIUM, dtype=float), dtype=jnp.float64)
        neg_ll = _make_jit_neg_loglik_exp(ej, jnp.asarray(T_MEDIUM, dtype=jnp.float64))
        jit_val = float(neg_ll(jnp.array([mu, a, b])))
        ref_val = self._numpy_neg_ll(proc, EVENTS_MEDIUM, T_MEDIUM)
        np.testing.assert_allclose(jit_val, ref_val, atol=1e-8)

    def test_sum_exponential(self):
        import jax.numpy as jnp
        from intensify.core.inference.mle import _make_jit_neg_loglik_sum_exp
        from intensify.core.kernels import SumExponentialKernel
        from intensify.core.processes import UnivariateHawkes

        mu = 0.4
        alphas, betas = [0.15, 0.1], [1.0, 3.0]
        proc = UnivariateHawkes(mu=mu, kernel=SumExponentialKernel(alphas, betas))
        ej = jnp.asarray(np.asarray(EVENTS_MEDIUM, dtype=float), dtype=jnp.float64)
        neg_ll = _make_jit_neg_loglik_sum_exp(ej, jnp.asarray(T_MEDIUM, dtype=jnp.float64), 2)
        params = jnp.array([mu] + alphas + betas)
        jit_val = float(neg_ll(params))
        ref_val = self._numpy_neg_ll(proc, EVENTS_MEDIUM, T_MEDIUM)
        np.testing.assert_allclose(jit_val, ref_val, atol=1e-8)

    def test_power_law(self):
        import jax.numpy as jnp
        from intensify.core.inference.mle import _make_jit_neg_loglik_power_law
        from intensify.core.kernels import PowerLawKernel
        from intensify.core.processes import UnivariateHawkes

        mu, a, b, c = 0.5, 0.3, 0.8, 0.5
        proc = UnivariateHawkes(mu=mu, kernel=PowerLawKernel(a, b, c))
        ej = jnp.asarray(np.asarray(EVENTS_SMALL, dtype=float), dtype=jnp.float64)
        neg_ll = _make_jit_neg_loglik_power_law(ej, jnp.asarray(T_SMALL, dtype=jnp.float64))
        jit_val = float(neg_ll(jnp.array([mu, a, b, c])))
        ref_val = self._numpy_neg_ll(proc, EVENTS_SMALL, T_SMALL)
        np.testing.assert_allclose(jit_val, ref_val, atol=1e-8)

    def test_approx_power_law(self):
        import jax.numpy as jnp
        from intensify.core.inference.mle import _make_jit_neg_loglik_approx_pl
        from intensify.core.kernels import ApproxPowerLawKernel
        from intensify.core.processes import UnivariateHawkes

        mu, a, bp, bm = 0.4, 0.3, 0.5, 0.1
        k = ApproxPowerLawKernel(a, bp, bm, r=1.5, n_components=5)
        proc = UnivariateHawkes(mu=mu, kernel=k)
        ej = jnp.asarray(np.asarray(EVENTS_SMALL, dtype=float), dtype=jnp.float64)
        neg_ll = _make_jit_neg_loglik_approx_pl(ej, jnp.asarray(T_SMALL, dtype=jnp.float64), 1.5, 5)
        jit_val = float(neg_ll(jnp.array([mu, a, bp, bm])))
        ref_val = self._numpy_neg_ll(proc, EVENTS_SMALL, T_SMALL)
        np.testing.assert_allclose(jit_val, ref_val, atol=1e-8)


# =====================================================================
# 7. JAX gradient correctness (vs finite differences)
# =====================================================================


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not available")
class TestJaxGradVsFiniteDiff:
    def test_exp_grad_matches_fd(self):
        import jax
        import jax.numpy as jnp
        from intensify.core.inference.mle import _make_jit_neg_loglik_exp

        ej = jnp.asarray(np.asarray(EVENTS_SMALL, dtype=float), dtype=jnp.float64)
        neg_ll = _make_jit_neg_loglik_exp(ej, jnp.asarray(T_SMALL, dtype=jnp.float64))
        grad_fn = jax.grad(neg_ll)

        x0 = jnp.array([0.5, 0.3, 1.5])
        jax_grad = np.asarray(grad_fn(x0))
        eps = 1e-5
        fd_grad = np.zeros(3)
        for i in range(3):
            xp = np.asarray(x0).copy(); xp[i] += eps
            xm = np.asarray(x0).copy(); xm[i] -= eps
            fd_grad[i] = (float(neg_ll(jnp.array(xp))) - float(neg_ll(jnp.array(xm)))) / (2 * eps)
        np.testing.assert_allclose(jax_grad, fd_grad, atol=1e-4, rtol=1e-4)


# =====================================================================
# 8. JAX Hessian standard errors
# =====================================================================


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not available")
class TestJaxHessianDemo:
    def test_std_errors_at_optimum(self):
        """Compute SEs at the actual MLE optimum (via a quick fit) and check they are positive."""
        import jax.numpy as jnp
        import scipy.optimize as spo

        from intensify.core.inference.mle import (
            _jax_hessian_std_errors,
            _make_jit_neg_loglik_exp,
        )

        ej = jnp.asarray(np.asarray(EVENTS_MEDIUM, dtype=float), dtype=jnp.float64)
        T_jax = jnp.asarray(T_MEDIUM, dtype=jnp.float64)
        neg_ll = _make_jit_neg_loglik_exp(ej, T_jax)

        import jax
        val_grad = jax.value_and_grad(neg_ll)

        def obj(x):
            v, g = val_grad(jnp.array(x))
            return float(v), np.asarray(g, dtype=float)

        res = spo.minimize(obj, np.array([0.5, 0.3, 1.5]), method="L-BFGS-B",
                           jac=True, bounds=[(1e-8, None), (1e-8, 0.999), (1e-8, None)])

        names = ["mu", "alpha", "beta"]
        se = _jax_hessian_std_errors(neg_ll, res.x, names)
        assert se is not None, "Hessian SE should not be None"
        for name in names:
            assert se[name] >= 0, f"SE for {name} should be non-negative, got {se[name]}"
            assert se[name] < 10.0, f"SE for {name} unreasonably large: {se[name]}"
        assert any(se[n] > 0 for n in names), "At least one SE should be strictly positive"


# =====================================================================
# 9. End-to-end MLE fit via _fit_jax for each kernel
# =====================================================================


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not available")
class TestFitJaxEndToEnd:
    def test_exponential_recovery(self):
        from intensify.core.inference import MLEInference
        from intensify.core.kernels import ExponentialKernel
        from intensify.core.processes import UnivariateHawkes

        result = MLEInference(max_iter=500).fit(
            UnivariateHawkes(mu=0.4, kernel=ExponentialKernel(0.2, 1.0)),
            EVENTS_MEDIUM, T=T_MEDIUM,
        )
        # Phase 1 port: ExponentialKernel + UnivariateHawkes routes through
        # the Rust core, replacing the previous JAX path. We still verify a
        # fast compiled backend handled the fit + the math came out right.
        assert result.convergence_info["backend"] == "rust"
        assert result.convergence_info["model"] == "univariate_hawkes_exp"
        assert result.convergence_info["success"] is True, result.convergence_info["message"]
        assert np.isfinite(result.log_likelihood)
        assert result.std_errors is not None
        assert result.branching_ratio_ < 1.0

    def test_sum_exponential(self):
        from intensify.core.inference import MLEInference
        from intensify.core.kernels import SumExponentialKernel
        from intensify.core.processes import UnivariateHawkes

        result = MLEInference(max_iter=500).fit(
            UnivariateHawkes(mu=0.4, kernel=SumExponentialKernel([0.15, 0.1], [1.0, 3.0])),
            EVENTS_MEDIUM, T=T_MEDIUM,
        )
        assert result.convergence_info["jit_compiled"] is True
        assert result.convergence_info["success"] is True, result.convergence_info["message"]
        assert np.isfinite(result.log_likelihood)

    def test_power_law(self):
        from intensify.core.inference import MLEInference
        from intensify.core.kernels import PowerLawKernel
        from intensify.core.processes import UnivariateHawkes

        result = MLEInference(max_iter=300).fit(
            UnivariateHawkes(mu=0.5, kernel=PowerLawKernel(0.3, 0.8, 0.5)),
            EVENTS_MEDIUM, T=T_MEDIUM,
        )
        # Phase 3 port: PowerLawKernel + UnivariateHawkes routes through Rust.
        assert result.convergence_info["backend"] == "rust"
        assert result.convergence_info["model"] == "univariate_hawkes_powerlaw"
        assert np.isfinite(result.log_likelihood)

    def test_approx_power_law(self):
        from intensify.core.inference import MLEInference
        from intensify.core.kernels import ApproxPowerLawKernel
        from intensify.core.processes import UnivariateHawkes

        result = MLEInference(max_iter=500).fit(
            UnivariateHawkes(mu=0.4, kernel=ApproxPowerLawKernel(0.3, 0.5, 0.1, r=1.5, n_components=5)),
            EVENTS_MEDIUM, T=T_MEDIUM,
        )
        assert result.convergence_info["jit_compiled"] is True
        assert np.isfinite(result.log_likelihood)


# =====================================================================
# 10. NonparametricKernel falls back to NumPy path
# =====================================================================


class TestNonparametricFallback:
    def test_nonparametric_fit_uses_numpy(self):
        from intensify.core.inference import MLEInference
        from intensify.core.kernels import NonparametricKernel
        from intensify.core.processes import UnivariateHawkes

        k = NonparametricKernel(edges=[0.0, 0.5, 1.0, 2.0], values=[0.2, 0.1, 0.05])
        result = MLEInference(max_iter=200).fit(
            UnivariateHawkes(mu=0.5, kernel=k),
            EVENTS_MEDIUM, T=T_MEDIUM,
        )
        assert result.convergence_info["backend"] == "numpy"
        assert np.isfinite(result.log_likelihood)


# =====================================================================
# 11. JAX while-loop simulation
# =====================================================================


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not available")
class TestJaxWhileSimulation:
    def test_exponential_simulation(self):
        from intensify.core.kernels import ExponentialKernel
        from intensify.core.processes import UnivariateHawkes

        proc = UnivariateHawkes(mu=1.0, kernel=ExponentialKernel(0.3, 1.5))
        events = proc.simulate(T=20.0, seed=42)
        assert len(events) > 0, "Simulation should produce events"
        ev_np = np.asarray(events)
        assert np.all(np.diff(ev_np) >= 0), "Events must be non-decreasing"
        assert ev_np[-1] <= 20.0, "Events must be within [0, T]"

    def test_high_rate_simulation(self):
        from intensify.core.kernels import ExponentialKernel
        from intensify.core.processes import UnivariateHawkes

        proc = UnivariateHawkes(mu=5.0, kernel=ExponentialKernel(0.5, 2.0))
        events = proc.simulate(T=10.0, seed=99)
        assert len(events) > 10, f"High-rate process should produce many events, got {len(events)}"

    def test_low_rate_simulation(self):
        from intensify.core.kernels import ExponentialKernel
        from intensify.core.processes import UnivariateHawkes

        proc = UnivariateHawkes(mu=0.1, kernel=ExponentialKernel(0.05, 1.0))
        events = proc.simulate(T=5.0, seed=7)
        ev_np = np.asarray(events)
        if len(ev_np) > 0:
            assert ev_np[-1] <= 5.0


# =====================================================================
# 12. Multivariate Hawkes MLE fit
# =====================================================================


class TestMultivariateMLE:
    def test_bivariate_fit(self):
        from intensify.core.inference import MLEInference
        from intensify.core.kernels import ExponentialKernel
        from intensify.core.processes.hawkes import MultivariateHawkes

        kernels = [
            [ExponentialKernel(0.12, 1.0), ExponentialKernel(0.08, 1.5)],
            [ExponentialKernel(0.10, 1.2), ExponentialKernel(0.15, 1.0)],
        ]
        mv = MultivariateHawkes(n_dims=2, mu=[0.3, 0.4], kernel=kernels)
        events = [bt.array([0.5, 1.5, 3.0, 5.0, 7.0, 8.5]), bt.array([0.8, 2.0, 4.0, 6.5, 9.0])]
        result = MLEInference(max_iter=200).fit(mv, events, T=10.0)
        assert np.isfinite(result.log_likelihood), f"MV MLE LL={result.log_likelihood}"


# =====================================================================
# 13. NonlinearHawkes MLE fit
# =====================================================================


class TestNonlinearHawkesMLE:
    def test_softplus_fit(self):
        from intensify.core.inference import MLEInference
        from intensify.core.kernels import ExponentialKernel
        from intensify.core.processes.nonlinear_hawkes import NonlinearHawkes

        k = ExponentialKernel(0.3, 1.5, allow_signed=True)
        proc = NonlinearHawkes(mu=0.5, kernel=k, link_function="softplus")
        result = MLEInference(max_iter=200).fit(proc, EVENTS_SMALL, T=T_SMALL)
        assert np.isfinite(result.log_likelihood)


# =====================================================================
# 14. EM inference still works
# =====================================================================


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not available")
class TestEMStillWorks:
    def test_em_exponential(self):
        from intensify.core.inference.em import EMInference
        from intensify.core.kernels import ExponentialKernel
        from intensify.core.processes import UnivariateHawkes

        proc = UnivariateHawkes(mu=0.5, kernel=ExponentialKernel(0.3, 1.5))
        result = EMInference(max_iter=30, mstep_iter=10, lr=0.005).fit(
            proc, EVENTS_MEDIUM, T=T_MEDIUM,
        )
        assert np.isfinite(result.log_likelihood), f"EM LL={result.log_likelihood}"


# =====================================================================
# 15. Recursive vs general LL cross-check (post integrate_vec refactor)
# =====================================================================


class TestRecursiveVsGeneralAfterRefactor:
    """Ensure the integrate_vec change didn't break LL correctness."""

    def test_exponential_recursive_vs_general(self):
        from intensify.core.inference.mle import (
            _general_likelihood_numpy,
            _recursive_likelihood_numpy,
        )
        from intensify.core.kernels import ExponentialKernel
        from intensify.core.processes import UnivariateHawkes

        proc = UnivariateHawkes(mu=0.5, kernel=ExponentialKernel(0.3, 1.5))
        rec = _recursive_likelihood_numpy(proc, EVENTS_MEDIUM, T_MEDIUM)
        gen = _general_likelihood_numpy(proc, EVENTS_MEDIUM, T_MEDIUM)
        np.testing.assert_allclose(rec, gen, atol=1e-8)

    def test_sum_exp_recursive_vs_general(self):
        from intensify.core.inference.mle import (
            _general_likelihood_numpy,
            _recursive_likelihood_numpy,
        )
        from intensify.core.kernels import SumExponentialKernel
        from intensify.core.processes import UnivariateHawkes

        proc = UnivariateHawkes(mu=0.4, kernel=SumExponentialKernel([0.15, 0.1], [1.0, 3.0]))
        rec = _recursive_likelihood_numpy(proc, EVENTS_SMALL, T_SMALL)
        gen = _general_likelihood_numpy(proc, EVENTS_SMALL, T_SMALL)
        np.testing.assert_allclose(rec, gen, atol=1e-8)

    def test_approx_pl_recursive_vs_general(self):
        from intensify.core.inference.mle import (
            _general_likelihood_numpy,
            _recursive_likelihood_numpy,
        )
        from intensify.core.kernels import ApproxPowerLawKernel
        from intensify.core.processes import UnivariateHawkes

        proc = UnivariateHawkes(mu=0.4, kernel=ApproxPowerLawKernel(0.3, 0.5, 0.1, r=1.5, n_components=5))
        rec = _recursive_likelihood_numpy(proc, EVENTS_SMALL, T_SMALL)
        gen = _general_likelihood_numpy(proc, EVENTS_SMALL, T_SMALL)
        np.testing.assert_allclose(rec, gen, atol=1e-8)


# =====================================================================
# 16. Simulate → Fit round-trip
# =====================================================================


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not available")
class TestSimulateFitRoundTrip:
    """Simulate from known params, fit, check recovery is plausible."""

    def test_exponential_round_trip(self):
        from intensify.core.inference import MLEInference
        from intensify.core.kernels import ExponentialKernel
        from intensify.core.processes import UnivariateHawkes

        true_mu, true_a, true_b = 0.5, 0.3, 1.5
        proc_true = UnivariateHawkes(mu=true_mu, kernel=ExponentialKernel(true_a, true_b))
        events = proc_true.simulate(T=50.0, seed=2024)
        if len(events) < 20:
            pytest.skip("Too few simulated events for stable recovery")

        proc_fit = UnivariateHawkes(mu=0.3, kernel=ExponentialKernel(0.15, 1.0))
        result = MLEInference(max_iter=1000).fit(proc_fit, events, T=50.0)
        # ExponentialKernel routes through the Rust core post-Phase-1.
        assert result.convergence_info["backend"] == "rust"
        est_mu = result.params["mu"]
        est_a = result.params["kernel"].alpha
        est_b = result.params["kernel"].beta
        assert abs(est_mu - true_mu) / true_mu < 0.6, f"mu recovery: {est_mu} vs {true_mu}"
        assert abs(est_a - true_a) / true_a < 0.6, f"alpha recovery: {est_a} vs {true_a}"
        assert abs(est_b - true_b) / true_b < 0.6, f"beta recovery: {est_b} vs {true_b}"
