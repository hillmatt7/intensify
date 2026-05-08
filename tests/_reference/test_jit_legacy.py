"""Tests for JIT acceleration: vectorized ops, JIT likelihoods, JAX gradients."""

import numpy as np
import pytest
from intensify.backends import get_backend
from intensify.backends._backend import get_backend_name

bt = get_backend()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HAS_JAX = get_backend_name() == "jax"


def _make_events(mu=0.5, alpha=0.3, beta=1.5, T=10.0, seed=42):
    """Generate a small reproducible event sequence."""
    from intensify.core.kernels import ExponentialKernel
    from intensify.core.processes import UnivariateHawkes

    kernel = ExponentialKernel(alpha=alpha, beta=beta)
    proc = UnivariateHawkes(mu=mu, kernel=kernel)
    events = proc.simulate(T=T, seed=seed)
    if len(events) < 5:
        events = bt.array([0.2, 0.8, 1.5, 2.3, 3.1, 4.0, 5.5, 7.0, 8.2, 9.0])
    return events


# ---------------------------------------------------------------------------
# integrate_vec tests
# ---------------------------------------------------------------------------


class TestIntegrateVec:
    def test_exponential_matches_scalar(self):
        from intensify.core.kernels import ExponentialKernel

        k = ExponentialKernel(alpha=0.3, beta=1.5)
        t_arr = bt.array([0.1, 0.5, 1.0, 3.0, 10.0])
        vec_result = k.integrate_vec(t_arr)
        scalar_results = bt.array([k.integrate(float(ti)) for ti in t_arr])
        assert np.allclose(vec_result, scalar_results, atol=1e-12)

    def test_sum_exponential_matches_scalar(self):
        from intensify.core.kernels import SumExponentialKernel

        k = SumExponentialKernel(alphas=[0.2, 0.1], betas=[1.0, 5.0])
        t_arr = bt.array([0.1, 0.5, 1.0, 3.0])
        vec_result = k.integrate_vec(t_arr)
        scalar_results = bt.array([k.integrate(float(ti)) for ti in t_arr])
        assert np.allclose(vec_result, scalar_results, atol=1e-12)

    def test_power_law_matches_scalar(self):
        from intensify.core.kernels import PowerLawKernel

        k = PowerLawKernel(alpha=0.5, beta=0.8, c=0.1)
        t_arr = bt.array([0.1, 0.5, 1.0, 5.0])
        vec_result = k.integrate_vec(t_arr)
        scalar_results = bt.array([k.integrate(float(ti)) for ti in t_arr])
        assert np.allclose(vec_result, scalar_results, atol=1e-10)

    def test_approx_power_law_matches_scalar(self):
        from intensify.core.kernels import ApproxPowerLawKernel

        k = ApproxPowerLawKernel(alpha=0.3, beta_pow=0.5, beta_min=0.1)
        t_arr = bt.array([0.1, 0.5, 1.0, 3.0])
        vec_result = k.integrate_vec(t_arr)
        scalar_results = bt.array([k.integrate(float(ti)) for ti in t_arr])
        assert np.allclose(vec_result, scalar_results, atol=1e-10)

    def test_nonparametric_matches_scalar(self):
        from intensify.core.kernels import NonparametricKernel

        k = NonparametricKernel(edges=[0.0, 0.5, 1.0, 2.0], values=[0.3, 0.2, 0.1])
        t_arr = bt.array([0.3, 0.7, 1.5, 3.0])
        vec_result = k.integrate_vec(t_arr)
        scalar_results = bt.array([k.integrate(float(ti)) for ti in t_arr])
        assert np.allclose(vec_result, scalar_results, atol=1e-12)


# ---------------------------------------------------------------------------
# jit_compatible property
# ---------------------------------------------------------------------------


class TestJitCompatible:
    def test_exponential_is_jit_compatible(self):
        from intensify.core.kernels import ExponentialKernel

        k = ExponentialKernel(alpha=0.3, beta=1.5)
        assert k.jit_compatible is True

    def test_sum_exponential_is_jit_compatible(self):
        from intensify.core.kernels import SumExponentialKernel

        k = SumExponentialKernel(alphas=[0.2], betas=[1.0])
        assert k.jit_compatible is True

    def test_nonparametric_is_not_jit_compatible(self):
        from intensify.core.kernels import NonparametricKernel

        k = NonparametricKernel(edges=[0.0, 1.0, 2.0], values=[0.1, 0.05])
        assert k.jit_compatible is False


# ---------------------------------------------------------------------------
# Vectorized intensity
# ---------------------------------------------------------------------------


class TestVectorizedIntensity:
    def test_univariate_intensity_array(self):
        from intensify.core.kernels import ExponentialKernel
        from intensify.core.processes import UnivariateHawkes

        k = ExponentialKernel(alpha=0.3, beta=1.5)
        proc = UnivariateHawkes(mu=0.5, kernel=k)
        history = bt.array([0.5, 1.2, 2.0])
        t_arr = bt.array([2.5, 3.0, 4.0])

        vec_result = proc.intensity(t_arr, history)
        scalar_results = bt.array([proc.intensity(float(ti), history) for ti in t_arr])
        assert np.allclose(vec_result, scalar_results, atol=1e-10)

    def test_univariate_intensity_empty_history(self):
        from intensify.core.kernels import ExponentialKernel
        from intensify.core.processes import UnivariateHawkes

        k = ExponentialKernel(alpha=0.3, beta=1.5)
        proc = UnivariateHawkes(mu=0.5, kernel=k)
        t_arr = bt.array([1.0, 2.0])
        result = proc.intensity(t_arr, bt.array([]))
        assert np.allclose(result, [0.5, 0.5])


# ---------------------------------------------------------------------------
# Vectorized multivariate log-likelihood
# ---------------------------------------------------------------------------


class TestVectorizedMultivarLL:
    def test_mv_ll_runs_without_error(self):
        from intensify.core.kernels import ExponentialKernel
        from intensify.core.processes.hawkes import MultivariateHawkes

        kernels = [
            [
                ExponentialKernel(alpha=0.15, beta=1.0),
                ExponentialKernel(alpha=0.1, beta=1.0),
            ],
            [
                ExponentialKernel(alpha=0.1, beta=1.0),
                ExponentialKernel(alpha=0.2, beta=1.5),
            ],
        ]
        mv = MultivariateHawkes(n_dims=2, mu=[0.3, 0.4], kernel=kernels)
        events = [bt.array([0.5, 1.5, 3.0]), bt.array([0.8, 2.0, 4.0])]
        ll = mv.log_likelihood(events, T=5.0)
        assert np.isfinite(ll)
        assert ll < 0


# ---------------------------------------------------------------------------
# JIT-compiled likelihood functions (JAX-only)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not available")
class TestJitLikelihood:
    def test_jit_exp_matches_numpy(self):
        import jax.numpy as jnp
        from intensify.core.inference.mle import (
            _make_jit_neg_loglik_exp,
            _recursive_likelihood_numpy,
        )
        from intensify.core.kernels import ExponentialKernel
        from intensify.core.processes import UnivariateHawkes

        events = _make_events()
        T = 10.0
        mu, alpha, beta = 0.5, 0.3, 1.5
        proc = UnivariateHawkes(mu=mu, kernel=ExponentialKernel(alpha=alpha, beta=beta))

        events_jax = jnp.asarray(np.asarray(events, dtype=float), dtype=jnp.float64)
        T_jax = jnp.asarray(T, dtype=jnp.float64)
        neg_ll = _make_jit_neg_loglik_exp(events_jax, T_jax)

        params = jnp.array([mu, alpha, beta])
        jit_val = float(neg_ll(params))
        numpy_val = -float(_recursive_likelihood_numpy(proc, events, T))
        assert np.isclose(jit_val, numpy_val, atol=1e-8), (
            f"JIT={jit_val}, NumPy={numpy_val}"
        )

    def test_jit_sum_exp_matches_numpy(self):
        import jax.numpy as jnp
        from intensify.core.inference.mle import (
            _make_jit_neg_loglik_sum_exp,
            _recursive_likelihood_numpy,
        )
        from intensify.core.kernels import SumExponentialKernel
        from intensify.core.processes import UnivariateHawkes

        alphas, betas = [0.15, 0.1], [1.0, 3.0]
        mu = 0.4
        k = SumExponentialKernel(alphas=alphas, betas=betas)
        proc = UnivariateHawkes(mu=mu, kernel=k)
        events = bt.array([0.2, 0.8, 1.5, 2.3, 3.1, 4.0, 5.5])
        T = 6.0

        events_jax = jnp.asarray(np.asarray(events, dtype=float), dtype=jnp.float64)
        T_jax = jnp.asarray(T, dtype=jnp.float64)
        neg_ll = _make_jit_neg_loglik_sum_exp(events_jax, T_jax, 2)

        params = jnp.array([mu] + alphas + betas)
        jit_val = float(neg_ll(params))
        numpy_val = -float(_recursive_likelihood_numpy(proc, events, T))
        assert np.isclose(jit_val, numpy_val, atol=1e-8), (
            f"JIT={jit_val}, NumPy={numpy_val}"
        )

    def test_jit_power_law_matches_numpy(self):
        import jax.numpy as jnp
        from intensify.core.inference.mle import (
            _general_likelihood_numpy,
            _make_jit_neg_loglik_power_law,
        )
        from intensify.core.kernels import PowerLawKernel
        from intensify.core.processes import UnivariateHawkes

        mu, alpha, beta_p, c = 0.5, 0.3, 0.8, 0.5
        k = PowerLawKernel(alpha=alpha, beta=beta_p, c=c)
        proc = UnivariateHawkes(mu=mu, kernel=k)
        events = bt.array([0.3, 0.9, 1.8, 2.5, 3.5])
        T = 4.0

        events_jax = jnp.asarray(np.asarray(events, dtype=float), dtype=jnp.float64)
        T_jax = jnp.asarray(T, dtype=jnp.float64)
        neg_ll = _make_jit_neg_loglik_power_law(events_jax, T_jax)

        params = jnp.array([mu, alpha, beta_p, c])
        jit_val = float(neg_ll(params))
        numpy_val = -float(_general_likelihood_numpy(proc, events, T))
        assert np.isclose(jit_val, numpy_val, atol=1e-8), (
            f"JIT={jit_val}, NumPy={numpy_val}"
        )

    def test_jit_approx_pl_matches_numpy(self):
        import jax.numpy as jnp
        from intensify.core.inference.mle import (
            _make_jit_neg_loglik_approx_pl,
            _recursive_likelihood_numpy,
        )
        from intensify.core.kernels import ApproxPowerLawKernel
        from intensify.core.processes import UnivariateHawkes

        mu, alpha, bp, bm = 0.4, 0.3, 0.5, 0.1
        k = ApproxPowerLawKernel(
            alpha=alpha, beta_pow=bp, beta_min=bm, r=1.5, n_components=5
        )
        proc = UnivariateHawkes(mu=mu, kernel=k)
        events = bt.array([0.3, 0.9, 1.8, 2.5, 3.5])
        T = 4.0

        events_jax = jnp.asarray(np.asarray(events, dtype=float), dtype=jnp.float64)
        T_jax = jnp.asarray(T, dtype=jnp.float64)
        neg_ll = _make_jit_neg_loglik_approx_pl(events_jax, T_jax, 1.5, 5)

        params = jnp.array([mu, alpha, bp, bm])
        jit_val = float(neg_ll(params))
        numpy_val = -float(_recursive_likelihood_numpy(proc, events, T))
        assert np.isclose(jit_val, numpy_val, atol=1e-8), (
            f"JIT={jit_val}, NumPy={numpy_val}"
        )


# ---------------------------------------------------------------------------
# JAX value_and_grad sanity
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not available")
class TestJaxGrad:
    def test_grad_exp_finite(self):
        import jax
        import jax.numpy as jnp
        from intensify.core.inference.mle import _make_jit_neg_loglik_exp

        events = bt.array([0.2, 0.8, 1.5, 2.3, 3.1])
        events_jax = jnp.asarray(np.asarray(events, dtype=float), dtype=jnp.float64)
        T_jax = jnp.asarray(4.0, dtype=jnp.float64)
        neg_ll = _make_jit_neg_loglik_exp(events_jax, T_jax)
        val_grad = jax.value_and_grad(neg_ll)

        params = jnp.array([0.5, 0.3, 1.5])
        val, grad = val_grad(params)
        assert np.isfinite(float(val))
        assert np.all(np.isfinite(np.asarray(grad)))

    def test_grad_sum_exp_finite(self):
        import jax
        import jax.numpy as jnp
        from intensify.core.inference.mle import _make_jit_neg_loglik_sum_exp

        events_jax = jnp.array([0.2, 0.8, 1.5, 2.3, 3.1])
        T_jax = jnp.asarray(4.0, dtype=jnp.float64)
        neg_ll = _make_jit_neg_loglik_sum_exp(events_jax, T_jax, 2)
        val_grad = jax.value_and_grad(neg_ll)

        params = jnp.array([0.4, 0.15, 0.1, 1.0, 3.0])
        val, grad = val_grad(params)
        assert np.isfinite(float(val))
        assert np.all(np.isfinite(np.asarray(grad)))


# ---------------------------------------------------------------------------
# JAX Hessian-based std errors
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not available")
class TestJaxHessian:
    def test_hessian_std_errors(self):
        import jax.numpy as jnp
        from intensify.core.inference.mle import (
            _jax_hessian_std_errors,
            _make_jit_neg_loglik_exp,
        )

        events_jax = jnp.array([0.2, 0.8, 1.5, 2.3, 3.1, 4.0, 5.5, 7.0])
        T_jax = jnp.asarray(8.0, dtype=jnp.float64)
        neg_ll = _make_jit_neg_loglik_exp(events_jax, T_jax)

        x_opt = np.array([0.5, 0.3, 1.5])
        names = ["mu", "alpha", "beta"]
        se = _jax_hessian_std_errors(neg_ll, x_opt, names)
        assert se is not None
        for name in names:
            assert name in se
            assert se[name] >= 0


# ---------------------------------------------------------------------------
# End-to-end MLE fit with JAX backend
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not available")
class TestFitJax:
    def test_fit_jax_exponential(self):
        from intensify.core.inference import MLEInference
        from intensify.core.kernels import ExponentialKernel
        from intensify.core.processes import UnivariateHawkes

        events = bt.array(
            [
                0.2,
                0.5,
                0.9,
                1.3,
                1.8,
                2.1,
                2.7,
                3.2,
                3.9,
                4.5,
                5.0,
                5.6,
                6.1,
                6.8,
                7.3,
                8.0,
                8.5,
                9.2,
                9.8,
                10.5,
            ]
        )
        proc = UnivariateHawkes(mu=0.4, kernel=ExponentialKernel(alpha=0.2, beta=1.0))
        mle = MLEInference(max_iter=500)
        result = mle.fit(proc, events, T=11.0)
        # Phase 1 port: ExponentialKernel + UnivariateHawkes routes through
        # the Rust core. The JAX path is reserved for kernels not yet ported
        # (sum_exp, power_law, approx_power_law, nonparametric).
        assert result.convergence_info.get("backend") == "rust"
        assert result.convergence_info.get("model") == "univariate_hawkes_exp"
        assert np.isfinite(result.log_likelihood)
        assert result.branching_ratio_ < 1.0

    def test_fit_jax_power_law(self):
        from intensify.core.inference import MLEInference
        from intensify.core.kernels import PowerLawKernel
        from intensify.core.processes import UnivariateHawkes

        events = bt.array([0.3, 0.9, 1.8, 2.5, 3.5, 4.2, 5.0, 6.1, 7.3, 8.8])
        proc = UnivariateHawkes(
            mu=0.5, kernel=PowerLawKernel(alpha=0.3, beta=0.8, c=0.5)
        )
        mle = MLEInference(max_iter=300)
        result = mle.fit(proc, events, T=10.0)
        # Phase 3 port: PowerLawKernel routes through the Rust core.
        assert result.convergence_info.get("backend") == "rust"
        assert result.convergence_info.get("model") == "univariate_hawkes_powerlaw"
        assert np.isfinite(result.log_likelihood)


# ---------------------------------------------------------------------------
# Vectorized NonlinearHawkes
# ---------------------------------------------------------------------------


class TestVectorizedNonlinear:
    def test_nonlinear_ll_runs(self):
        from intensify.core.kernels import ExponentialKernel
        from intensify.core.processes.nonlinear_hawkes import NonlinearHawkes

        k = ExponentialKernel(alpha=0.3, beta=1.5, allow_signed=True)
        proc = NonlinearHawkes(mu=0.5, kernel=k, link_function="softplus")
        events = bt.array([0.5, 1.2, 2.0, 3.5, 5.0])
        ll = proc.log_likelihood(events, T=6.0, n_quad=64)
        assert np.isfinite(ll)

    def test_pre_intensity_vectorized(self):
        from intensify.core.kernels import ExponentialKernel
        from intensify.core.processes.nonlinear_hawkes import NonlinearHawkes

        k = ExponentialKernel(alpha=0.3, beta=1.5, allow_signed=True)
        proc = NonlinearHawkes(mu=0.5, kernel=k, link_function="softplus")
        events = np.array([0.5, 1.2, 2.0])
        pre = proc._pre_intensity(2.5, events)
        assert np.isfinite(pre)
        assert pre > 0


# ---------------------------------------------------------------------------
# Simulation with JIT while_loop
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not available")
class TestJitSimulation:
    def test_jax_while_simulation_produces_events(self):
        from intensify.core.kernels import ExponentialKernel
        from intensify.core.processes import UnivariateHawkes

        k = ExponentialKernel(alpha=0.3, beta=1.5)
        proc = UnivariateHawkes(mu=1.0, kernel=k)
        events = proc.simulate(T=10.0, seed=42)
        assert len(events) > 0
        if len(events) > 1:
            diffs = np.diff(np.asarray(events))
            assert np.all(diffs >= 0), "events must be sorted"

    def test_jax_while_simulation_respects_T(self):
        from intensify.core.kernels import ExponentialKernel
        from intensify.core.processes import UnivariateHawkes

        k = ExponentialKernel(alpha=0.3, beta=1.5)
        proc = UnivariateHawkes(mu=1.0, kernel=k)
        T = 5.0
        events = proc.simulate(T=T, seed=123)
        if len(events) > 0:
            assert float(np.max(np.asarray(events))) <= T
