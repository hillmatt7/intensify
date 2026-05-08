"""Tests for Phases 2–3 plan features (marked/nonlinear/online/multivariate/Bayesian helpers)."""

import numpy as np
import pytest
from intensify.core.diagnostics.metrics import branching_ratio, endogeneity_index
from intensify.core.inference import FitResult, MLEInference, get_inference_engine
from intensify.core.inference.multivariate_hawkes_mle_params import (
    multivariate_hawkes_apply_vector,
    multivariate_hawkes_initial_vector,
)
from intensify.core.kernels import ExponentialKernel
from intensify.core.processes import (
    MarkedHawkes,
    MultivariateHawkes,
    MultivariateNonlinearHawkes,
    NonlinearHawkes,
    UnivariateHawkes,
)
from intensify.core.regularizers import L1
from intensify.visualization.connectivity import plot_connectivity
from intensify.visualization.event_histograms import (
    plot_event_aligned_histogram,
    plot_inter_event_intervals,
)


def test_marked_hawkes_likelihood_and_mle():
    kern = ExponentialKernel(alpha=0.25, beta=1.2)
    model = MarkedHawkes(0.5, kern, mark_influence="linear")
    ev = np.array([0.2, 0.7, 1.4], dtype=float)
    marks = np.array([0.5, 1.0, 0.8], dtype=float)
    ll = model.log_likelihood(ev, marks, T=2.0)
    assert np.isfinite(ll)
    result = model.fit(ev, marks, T=2.0, method="mle")
    assert result.log_likelihood > -np.inf
    assert result.branching_ratio_ is not None


def test_nonlinear_hawkes_softplus_and_mle():
    kern = ExponentialKernel(alpha=-0.2, beta=1.5, allow_signed=True)
    model = NonlinearHawkes(0.6, kern, link_function="softplus")
    ev = np.array([0.15, 0.9], dtype=float)
    ll = model.log_likelihood(ev, T=1.5)
    assert np.isfinite(ll)
    result = get_inference_engine("mle").fit(model, ev, 1.5)
    assert np.isfinite(result.log_likelihood)


def test_online_engine_runs():
    model = UnivariateHawkes(0.4, ExponentialKernel(0.35, 1.1))
    ev = np.sort(np.random.default_rng(2).uniform(0, 2, size=80))
    eng = get_inference_engine("online")
    res = eng.fit(model, ev, float(ev.max()))
    assert res.log_likelihood > -np.inf


def test_multivariate_mle_and_connectivity():
    M = 2
    kern = ExponentialKernel(0.2, 1.0)
    mat = [[kern for _ in range(M)] for _ in range(M)]
    proc = MultivariateHawkes(M, 0.3, mat)
    ev_lists = [
        np.array([0.1, 0.6]),
        np.array([0.2, 0.8]),
    ]
    res = MLEInference().fit(proc, ev_lists, 1.0)
    W = res.connectivity_matrix()
    assert W.shape == (2, 2)
    sig = res.significant_connections(significance_level=0.5)
    assert sig.shape == (2, 2)


def test_multivariate_mle_with_l1():
    M = 2
    kern = ExponentialKernel(0.15, 1.0)
    mat = [[kern for _ in range(M)] for _ in range(M)]
    proc = MultivariateHawkes(M, 0.25, mat)
    ev_lists = [np.array([0.1, 0.5]), np.array([0.2, 0.6])]
    reg = L1(strength=0.01, off_diagonal_only=True)
    res = MLEInference().fit(proc, ev_lists, 1.0, regularization=reg)
    assert np.isfinite(res.log_likelihood)


def test_multivariate_nonlinear_likelihood_smoke():
    M = 2
    kern = ExponentialKernel(0.15, 1.2, allow_signed=True)
    mat = [[kern for _ in range(M)] for _ in range(M)]
    proc = MultivariateNonlinearHawkes(M, 0.3, mat, link_function="softplus")
    ev = [np.array([0.1, 0.4]), np.array([0.2])]
    ll = proc.log_likelihood(ev, 0.8, n_quad=32)
    assert np.isfinite(ll)


def test_endogeneity_index_critical_branching():
    fr = FitResult(params={}, log_likelihood=0.0, branching_ratio_=1.0)
    assert endogeneity_index(fr) == 1.0


def test_marked_fit_warns_large_marks():
    m = MarkedHawkes(0.2, ExponentialKernel(0.18, 1.0))
    ev = np.array([0.1, 0.3])
    mk = np.array([2000.0, 5000.0])
    with pytest.warns(UserWarning):
        m.fit(ev, mk, T=1.0)


def test_marked_fit_raises_mark_length():
    m = MarkedHawkes(0.2, ExponentialKernel(0.18, 1.0))
    with pytest.raises(ValueError):
        m.fit(np.array([0.1]), np.array([0.2, 0.3]), T=1.0)


def test_metrics_branching_ratio_errors():
    with pytest.raises(TypeError):
        branching_ratio(object())


def test_plot_connectivity_grid_layout():
    W = np.array([[0.0, 0.2], [0.3, 0.0]])
    fig = plot_connectivity(W, layout="grid", threshold=0.01)
    assert fig is not None


def test_metrics_helpers():
    m = UnivariateHawkes(0.5, ExponentialKernel(0.4, 1.0))
    assert abs(branching_ratio(m) - 0.4) < 1e-6
    ei = endogeneity_index(m)
    assert ei > 0
    fr = FitResult(params={}, log_likelihood=0.0, branching_ratio_=0.5)
    assert branching_ratio(fr) == 0.5


def test_plotting_smoke(tmp_path):
    ev = np.cumsum(np.random.default_rng(0).exponential(0.3, size=40))
    fig1 = plot_inter_event_intervals(ev)
    fig1.savefig(tmp_path / "isi.png", dpi=40)
    fig2 = plot_event_aligned_histogram(
        ev, reference_times=np.array([0.5, 1.2]), window=(-0.2, 0.5)
    )
    fig2.savefig(tmp_path / "align.png", dpi=40)
    W = np.array([[0.0, 0.3], [0.1, 0.0]])
    fig3 = plot_connectivity(W, labels=["a", "b"], threshold=0.05)
    fig3.savefig(tmp_path / "conn.png", dpi=40)


def test_multivariate_param_pack_roundtrip():
    M = 2
    kern = ExponentialKernel(0.2, 1.3)
    proc = MultivariateHawkes(
        M, [0.2, 0.25], [[kern for _ in range(M)] for _ in range(M)]
    )
    x0 = multivariate_hawkes_initial_vector(proc)
    multivariate_hawkes_apply_vector(proc, x0 * 0.95 + 0.01)
    assert proc.kernel_matrix[0][0].alpha > 0


def test_bayesian_engine_smoke():
    pytest.importorskip("numpyro")
    from intensify.core.inference.bayesian import BayesianInference

    true = UnivariateHawkes(0.4, ExponentialKernel(0.35, 1.1))
    ev = np.asarray(true.simulate(1.5, seed=3))
    model = UnivariateHawkes(0.3, ExponentialKernel(0.25, 1.0))
    res = BayesianInference(num_warmup=120, num_samples=120, num_chains=1).fit(
        model, ev, 1.5, seed=1
    )
    assert "mu" in res.posterior_samples_


def test_piecewise_poisson_more():
    from intensify.core.processes import InhomogeneousPoisson

    p = InhomogeneousPoisson(rates={0.0: 1.0, 0.5: 2.0, 1.0: 0.5})
    ev = p.simulate(2.0, seed=1)
    ll = p.log_likelihood(ev, 2.0)
    assert np.isfinite(ll)


def test_cox_set_last_window():
    from intensify.core.processes import LogGaussianCoxProcess

    lgcp = LogGaussianCoxProcess(n_bins=10)
    lgcp.simulate(1.0, seed=4)
    lgcp.set_last_window(1.0)
    _ = lgcp.intensity(0.33, np.array([]))


def test_exponential_signed_path():
    k = ExponentialKernel(-0.3, 1.0, allow_signed=True)
    assert not k.has_recursive_form()


def test_marked_simulate_and_power_influence():
    m = MarkedHawkes(
        0.3, ExponentialKernel(0.2, 1.0), mark_influence="power", mark_power=1.2
    )
    ev, mk = m.simulate(0.8, seed=11)
    assert len(ev) == len(mk)
    assert m.log_likelihood(ev, mk, 0.8) > -np.inf


def test_nonlinear_links_smoke():
    k = ExponentialKernel(-0.15, 1.2, allow_signed=True)
    ev = np.array([0.1, 0.4, 0.9])
    for link in ("relu", "sigmoid", "identity"):
        model = NonlinearHawkes(0.5, k, link_function=link)
        assert np.isfinite(model.log_likelihood(ev, 1.0, n_quad=64))


def test_regularizer_elastic_net():
    from intensify.core.regularizers import ElasticNet

    x = np.array([0.1, 0.2, 0.3, 0.25, 0.4, 0.35, 0.5, 0.45, 0.6, 0.55, 0.7])
    reg = ElasticNet(strength=0.01, l1_ratio=0.4, off_diagonal_only=False)
    assert reg.penalty(x, M=2) >= 0


def test_fitresult_connectivity_errors():
    fr = FitResult(
        params={},
        log_likelihood=0.0,
        process=UnivariateHawkes(0.5, ExponentialKernel(0.2, 1.0)),
    )
    with pytest.raises(TypeError):
        fr.connectivity_matrix()


def test_fitresult_plot_posterior_smoke():
    fr = FitResult(
        params={}, log_likelihood=0.0, posterior_samples_={"mu": np.random.randn(200)}
    )
    fig = fr.plot_posterior(max_vars=1)
    assert fig is not None


def test_univariate_mle_param_helpers_cover_types():
    from intensify.core.inference.univariate_hawkes_mle_params import (
        hawkes_mle_apply_vector,
        hawkes_mle_bounds,
        hawkes_mle_initial_vector,
        hawkes_mle_param_names,
    )
    from intensify.core.kernels import (
        ApproxPowerLawKernel,
        NonparametricKernel,
        PowerLawKernel,
        SumExponentialKernel,
    )

    edges = [0.0, 0.5, 1.0, 1.5]
    vals = [0.05, 0.05, 0.05]
    proc = UnivariateHawkes(0.2, NonparametricKernel(edges=edges, values=vals))
    x0 = hawkes_mle_initial_vector(proc)
    b = hawkes_mle_bounds(proc)
    n = hawkes_mle_param_names(proc)
    assert len(x0) == len(b) == len(n)
    hawkes_mle_apply_vector(proc, x0)

    proc2 = UnivariateHawkes(
        0.3, SumExponentialKernel(alphas=[0.1, 0.1], betas=[1.0, 2.0])
    )
    x1 = hawkes_mle_initial_vector(proc2)
    hawkes_mle_apply_vector(proc2, x1)

    proc3 = UnivariateHawkes(0.4, PowerLawKernel(alpha=0.1, beta=1.5, c=0.5))
    hawkes_mle_apply_vector(proc3, hawkes_mle_initial_vector(proc3))

    proc4 = UnivariateHawkes(
        0.35,
        ApproxPowerLawKernel(alpha=0.2, beta_pow=1.4, beta_min=0.8, n_components=3),
    )
    hawkes_mle_apply_vector(proc4, hawkes_mle_initial_vector(proc4))


def test_multivariate_nonlinear_simulate_smoke():
    M = 2
    k = ExponentialKernel(0.12, 1.0)
    mat = [[k for _ in range(M)] for _ in range(M)]
    p = MultivariateNonlinearHawkes(M, 0.25, mat, link_function="relu")
    out = p.simulate(0.4, seed=7)
    assert len(out) == 2


def test_univariate_hawkes_intensity_vector_time():
    m = UnivariateHawkes(0.5, ExponentialKernel(0.3, 1.0))
    hist = np.array([0.1, 0.2])
    out = m.intensity(np.array([0.25, 0.4, 0.41]), hist)
    assert len(out) == 3


def test_marked_general_kernel_likelihood():
    from intensify.core.kernels import PowerLawKernel

    m = MarkedHawkes(
        0.15,
        PowerLawKernel(alpha=0.12, beta=1.5, c=0.15),
        mark_influence="linear",
    )
    ev = np.array([0.2, 0.6, 0.9])
    mk = np.array([0.5, 0.6, 0.4])
    assert np.isfinite(m.log_likelihood(ev, mk, 1.2))


def test_nonlinear_empty_events_compensator():
    m = NonlinearHawkes(0.5, ExponentialKernel(0.2, 1.0))
    assert m.log_likelihood(np.array([]), 0.5) <= 0.0


def test_marked_log_and_callable_influence():
    m1 = MarkedHawkes(0.25, ExponentialKernel(0.2, 1.1), mark_influence="log")
    ev = np.array([0.1, 0.5])
    mk = np.array([0.2, 0.4])
    assert np.isfinite(m1.log_likelihood(ev, mk, 1.0))
    m2 = MarkedHawkes(
        0.25, ExponentialKernel(0.2, 1.1), mark_influence=lambda x: float(np.sqrt(x))
    )
    assert np.isfinite(m2.log_likelihood(ev, mk, 1.0))
    m2.project_params()


def test_marked_mle_accepts_power_law_kernel():
    """Since 0.1.1 MarkedHawkes MLE accepts any kernel the univariate helpers
    understand — PowerLawKernel included. Previously raised NotImplementedError."""
    from intensify.core.kernels import PowerLawKernel

    m = MarkedHawkes(0.2, PowerLawKernel(alpha=0.1, beta=1.5, c=0.2))
    ev = np.array([0.1, 0.4])
    mk = np.array([0.5, 0.5])
    result = MLEInference().fit(m, (ev, mk), 1.0)
    assert np.isfinite(result.log_likelihood)
    assert isinstance(result.process.kernel, PowerLawKernel)


def test_nonlinear_custom_link():
    k = ExponentialKernel(-0.1, 1.0, allow_signed=True)
    model = NonlinearHawkes(0.5, k, link_function=lambda z: float(np.maximum(z, 0.0)))
    assert np.isfinite(model.log_likelihood(np.array([0.2, 0.6]), 1.0))


def test_significant_connections_with_std_errors():
    M = 2
    kern = ExponentialKernel(0.2, 1.0)
    mat = [[kern for _ in range(M)] for _ in range(M)]
    proc = MultivariateHawkes(M, 0.3, mat)
    ev_lists = [np.array([0.1, 0.6]), np.array([0.2, 0.7])]
    res = MLEInference().fit(proc, ev_lists, 1.0)
    assert res.std_errors is not None
    _ = res.significant_connections(significance_level=0.9)


def test_bayesian_horseshoe_smoke():
    pytest.importorskip("numpyro")
    from intensify.core.inference.bayesian import BayesianInference

    ev = np.asarray([0.2, 0.7])
    model = UnivariateHawkes(0.3, ExponentialKernel(0.25, 1.0))
    res = BayesianInference(
        num_warmup=80, num_samples=80, sparse_prior="horseshoe"
    ).fit(model, ev, 1.0, seed=3)
    assert "alpha" in res.posterior_samples_
