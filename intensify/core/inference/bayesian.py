"""Bayesian inference via NumPyro NUTS (optional dependency)."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import FitResult, InferenceEngine, compute_information_criteria


def _hawkes_exp_loglik_jax(mu, alpha, beta, events, T):
    import jax
    import jax.numpy as jnp

    events = jnp.sort(jnp.asarray(events, dtype=jnp.float64))
    T = jnp.asarray(T, dtype=jnp.float64)
    dts = jnp.diff(events, prepend=jnp.array(0.0, dtype=jnp.float64))

    def scan_fn(R, dt):
        R_decayed = jnp.exp(-beta * dt) * R
        lam = mu + alpha * beta * R_decayed
        R_next = R_decayed + 1.0
        return R_next, jnp.log(lam)

    _, log_terms = jax.lax.scan(scan_fn, jnp.array(0.0), dts)
    log_int_sum = jnp.sum(log_terms)
    comp_tail = jnp.sum(alpha * (1.0 - jnp.exp(-beta * (T - events))))
    comp = mu * T + comp_tail
    return log_int_sum - comp


class BayesianInference(InferenceEngine):
    """
    NUTS sampling for a univariate exponential-kernel Hawkes (NumPyro required).

    Parameters
    ----------
    num_warmup : int
    num_samples : int
    num_chains : int
    sparse_prior : None or \"horseshoe\"
        If \"horseshoe\", uses a hierarchical Half-Cauchy / Half-Normal prior on `alpha`.
    """

    def __init__(
        self,
        num_warmup: int = 300,
        num_samples: int = 600,
        num_chains: int = 1,
        *,
        sparse_prior: str | None = None,
    ):
        self.num_warmup = int(num_warmup)
        self.num_samples = int(num_samples)
        self.num_chains = int(num_chains)
        self.sparse_prior = sparse_prior

    def fit(self, process: Any, events: Any, T: float, **kwargs: Any) -> FitResult:
        try:
            import jax.numpy as jnp
            import numpyro
            import numpyro.distributions as dist
            from numpyro.infer import MCMC, NUTS
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "BayesianInference requires numpyro and jax; install with "
                "`pip install intensify[bayesian]`."
            ) from e

        from ..kernels.exponential import ExponentialKernel
        from ..processes.hawkes import UnivariateHawkes

        if not isinstance(process, UnivariateHawkes) or not isinstance(
            process.kernel, ExponentialKernel
        ):
            raise NotImplementedError(
                "BayesianInference currently supports UnivariateHawkes + ExponentialKernel."
            )

        ev = jnp.asarray(np.asarray(events, dtype=float).ravel(), dtype=jnp.float64)
        T = float(T)
        if len(ev) == 0:
            raise ValueError("Need at least one event for Hawkes likelihood.")
        sparse = kwargs.get("sparse_prior", self.sparse_prior)

        def model():
            import jax.numpy as jnp

            mu = numpyro.sample("mu", dist.HalfNormal(1.0))
            beta = numpyro.sample("beta", dist.HalfNormal(2.0))
            if sparse == "horseshoe":
                tau = numpyro.sample("tau", dist.HalfCauchy(1.0))
                alpha = numpyro.sample("alpha", dist.HalfNormal(tau))
                alpha = jnp.minimum(alpha, 0.99)
            else:
                alpha = numpyro.sample("alpha", dist.Beta(2.0, 2.0))
            ll = _hawkes_exp_loglik_jax(mu, alpha, beta, ev, jnp.array(T))
            numpyro.factor("obs", ll)

        nuts = NUTS(model)
        mcmc = MCMC(
            nuts,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
        )
        rng_key = __import__("jax").random.PRNGKey(
            int(kwargs.get("seed", 0))
        )
        mcmc.run(rng_key)
        samples = mcmc.get_samples()
        posterior = {k: np.asarray(v) for k, v in samples.items()}
        # point estimate: posterior mean
        mu_hat = float(np.mean(posterior["mu"]))
        alpha_hat = float(np.mean(posterior["alpha"]))
        beta_hat = float(np.mean(posterior["beta"]))
        process.mu = mu_hat
        process.kernel = ExponentialKernel(alpha=alpha_hat, beta=beta_hat)
        ll = float(process.log_likelihood(np.asarray(ev), T))
        n_obs = len(ev)
        params = process.get_params()
        aic, bic = compute_information_criteria(ll, params, n_obs)
        ess: dict[str, float] = {}
        rhat: dict[str, float] = {}
        try:
            import numpyro.diagnostics as n_diag

            chain_samples = mcmc.get_samples(group_by_chain=True)
            for k, v in chain_samples.items():
                arr = np.asarray(v)
                if arr.ndim >= 2 and arr.shape[0] > 1:
                    rhat[k] = float(n_diag.gelman_rubin(arr))
                ess[k] = float(np.min(n_diag.effective_sample_size(arr)))
        except Exception:  # pragma: no cover
            pass
        q = {}
        for k, v in posterior.items():
            lo, hi = np.percentile(v, [2.5, 97.5])
            q[k] = (float(lo), float(hi))
        result = FitResult(
            params=params,
            log_likelihood=ll,
            aic=aic,
            bic=bic,
            std_errors=None,
            convergence_info={"method": "numpyro_nuts", "chains": self.num_chains},
            posterior_samples_=posterior,
            credible_intervals_=q,
            effective_sample_size_=ess or None,
            r_hat_=rhat or None,
        )
        result.process = process
        result.events = np.asarray(ev)
        result.T = T
        result.branching_ratio_ = process.kernel.l1_norm()
        return result


from . import register_engine

register_engine("bayesian", BayesianInference())
