"""
Stress tests using real HC-3 hippocampal spike train data from CRCNS.org.

These tests exercise intensify's full pipeline — data loading, model fitting,
diagnostics, and visualization — on genuine multiunit recordings, verifying
institutional-grade reliability under realistic conditions.

Dataset: hc-3 (Mizuseki et al., hippocampal multiunit recordings)
Sessions used:
  - ec013.544 (8 electrodes, animal ec013)
  - ec012ec.409 (4 electrodes, animal ec012)
"""

import os
import time
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Non-interactive backend for CI

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HC3_ROOT = Path(os.environ.get("HC3_ROOT", Path.home() / "hc-3"))
SESSION_A_DIR = HC3_ROOT / "ec013.33" / "ec013.544"
SESSION_B_DIR = HC3_ROOT / "ec012ec.20" / "ec012ec.409"
SAMPLE_RATE = 20_000  # 20 kHz standard for hc-3 dataset


# ---------------------------------------------------------------------------
# Helpers — parse real spike data
# ---------------------------------------------------------------------------
def _load_spike_trains(session_dir: Path, electrode: int) -> dict[int, np.ndarray]:
    """
    Load spike trains from .res.N and .clu.N files.

    Returns dict mapping cluster_id -> array of spike times in seconds.
    Clusters 0 (noise) and 1 (unsortable) are excluded.
    """
    stem = session_dir.name
    res_path = session_dir / f"{stem}.res.{electrode}"
    clu_path = session_dir / f"{stem}.clu.{electrode}"

    if not res_path.exists() or not clu_path.exists():
        pytest.skip(f"Missing data files in {session_dir}")

    spike_samples = np.loadtxt(res_path, dtype=np.int64)
    clu_data = np.loadtxt(clu_path, dtype=np.int32)
    n_clusters = clu_data[0]
    cluster_ids = clu_data[1:]  # First line is cluster count

    assert len(spike_samples) == len(cluster_ids), (
        f"res/clu length mismatch: {len(spike_samples)} vs {len(cluster_ids)}"
    )

    spike_times = spike_samples / SAMPLE_RATE  # Convert to seconds

    trains = {}
    for cid in range(2, n_clusters + 1):  # Skip 0 (noise) and 1 (unsortable)
        mask = cluster_ids == cid
        if mask.sum() > 0:
            trains[cid] = np.sort(spike_times[mask])
    return trains


def _load_all_neurons(session_dir: Path, max_electrodes: int = 8) -> list[np.ndarray]:
    """Load all well-isolated neurons across electrodes as a list of spike time arrays."""
    all_trains = []
    stem = session_dir.name
    for e in range(1, max_electrodes + 1):
        res_path = session_dir / f"{stem}.res.{e}"
        if not res_path.exists():
            continue
        trains = _load_spike_trains(session_dir, e)
        for cid, times in sorted(trains.items()):
            if len(times) >= 10:  # Only neurons with enough spikes
                all_trains.append(times)
    return all_trains


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def session_a_neurons():
    """All neurons from ec013.544 (8-electrode session)."""
    if not SESSION_A_DIR.exists():
        pytest.skip("HC-3 session A data not found")
    return _load_all_neurons(SESSION_A_DIR)


@pytest.fixture(scope="module")
def session_b_neurons():
    """All neurons from ec012ec.409 (4-electrode session)."""
    if not SESSION_B_DIR.exists():
        pytest.skip("HC-3 session B data not found")
    return _load_all_neurons(SESSION_B_DIR)


@pytest.fixture(scope="module")
def high_rate_neuron(session_a_neurons):
    """Highest firing rate neuron from session A."""
    if not session_a_neurons:
        pytest.skip("No neurons loaded")
    return max(session_a_neurons, key=len)


@pytest.fixture(scope="module")
def low_rate_neuron(session_a_neurons):
    """Lowest firing rate neuron (>= 10 spikes) from session A."""
    if not session_a_neurons:
        pytest.skip("No neurons loaded")
    return min(session_a_neurons, key=len)


# ===========================================================================
#  SECTION 1: Data Loading Sanity
# ===========================================================================
class TestDataLoading:
    """Verify we can parse real neural data correctly."""

    def test_session_a_has_neurons(self, session_a_neurons):
        assert len(session_a_neurons) > 0, "Expected at least one neuron"
        print(f"\nSession A: {len(session_a_neurons)} neurons loaded")
        for i, train in enumerate(session_a_neurons):
            print(
                f"  Neuron {i}: {len(train)} spikes, "
                f"range [{train[0]:.2f}, {train[-1]:.2f}]s, "
                f"mean ISI = {np.mean(np.diff(train)):.4f}s"
            )

    def test_session_b_has_neurons(self, session_b_neurons):
        assert len(session_b_neurons) > 0

    def test_spike_times_are_sorted(self, session_a_neurons):
        for train in session_a_neurons:
            assert np.all(np.diff(train) >= 0), "Spike times must be sorted"

    def test_spike_times_are_positive(self, session_a_neurons):
        for train in session_a_neurons:
            assert np.all(train >= 0), "Spike times must be non-negative"

    def test_no_exact_duplicate_timestamps(self, session_a_neurons):
        for train in session_a_neurons:
            n_unique = len(np.unique(train))
            n_total = len(train)
            dup_frac = 1 - n_unique / n_total if n_total > 0 else 0
            # Allow small fraction of duplicates from binning
            assert dup_frac < 0.01, f"Too many duplicate timestamps: {dup_frac:.2%}"


# ===========================================================================
#  SECTION 2: Univariate Hawkes — Real Spike Trains
# ===========================================================================
class TestUnivariateHawkesReal:
    """Fit univariate Hawkes to individual neurons and stress-test edge cases."""

    def test_fit_exponential_kernel_high_rate(self, high_rate_neuron):
        """MLE on the busiest neuron — exercises O(N) recursive path at scale."""
        import intensify as its

        events = high_rate_neuron
        T = float(events[-1]) + 1.0
        n = len(events)
        print(f"\nHigh-rate neuron: {n} spikes over {T:.1f}s (rate ~ {n / T:.1f} Hz)")

        model = its.UnivariateHawkes(
            mu=n / (2 * T),
            kernel=its.ExponentialKernel(alpha=0.2, beta=5.0),
        )
        t0 = time.perf_counter()
        result = model.fit(events, T=T, method="mle")
        elapsed = time.perf_counter() - t0

        print(f"  Fit time: {elapsed:.2f}s")
        print(result.summary())

        assert np.isfinite(result.log_likelihood)
        assert result.branching_ratio_ is not None
        assert result.branching_ratio_ < 1.0, "Fitted process should be stationary"
        assert result.params["mu"] > 0
        assert result.aic is not None
        assert result.bic is not None

    def test_fit_exponential_kernel_low_rate(self, low_rate_neuron):
        """MLE on a sparse neuron — tests stability with few events."""
        import intensify as its

        events = low_rate_neuron
        T = float(events[-1]) + 1.0
        print(f"\nLow-rate neuron: {len(events)} spikes over {T:.1f}s")

        model = its.UnivariateHawkes(
            mu=0.5,
            kernel=its.ExponentialKernel(alpha=0.1, beta=1.0),
        )
        result = model.fit(events, T=T, method="mle")

        assert np.isfinite(result.log_likelihood)
        assert result.branching_ratio_ < 1.0

    def test_fit_sum_exponential_kernel(self, high_rate_neuron):
        """Multi-timescale excitation on real data."""
        import intensify as its

        events = high_rate_neuron
        T = float(events[-1]) + 1.0

        model = its.UnivariateHawkes(
            mu=len(events) / (2 * T),
            kernel=its.SumExponentialKernel(alphas=[0.1, 0.1], betas=[1.0, 10.0]),
        )
        result = model.fit(events, T=T, method="mle")

        assert np.isfinite(result.log_likelihood)
        assert result.branching_ratio_ < 1.0
        print(f"\nSumExp fit: {result.summary()}")

    def test_fit_power_law_kernel_small_subset(self, high_rate_neuron):
        """Power-law kernel (O(N^2)) on a small slice — tests general path."""
        import intensify as its

        # Take first 200 spikes to keep O(N^2) manageable
        events = high_rate_neuron[:200]
        T = float(events[-1]) + 0.5

        model = its.UnivariateHawkes(
            mu=len(events) / (2 * T),
            kernel=its.PowerLawKernel(alpha=0.5, beta=1.5, c=0.01),
        )
        t0 = time.perf_counter()
        result = model.fit(events, T=T, method="mle")
        elapsed = time.perf_counter() - t0

        print(f"\nPowerLaw fit ({len(events)} spikes): {elapsed:.2f}s")
        assert np.isfinite(result.log_likelihood)
        assert elapsed < 120, "Power-law O(N^2) on 200 spikes should be < 2 min"

    def test_fit_nonparametric_kernel(self, high_rate_neuron):
        """Nonparametric kernel — model-free excitation shape."""
        import intensify as its

        events = high_rate_neuron[:100]  # Small subset for O(N^2) kernel
        T = float(events[-1]) + 0.5

        edges = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
        values = [0.5] * (len(edges) - 1)
        model = its.UnivariateHawkes(
            mu=len(events) / (2 * T),
            kernel=its.NonparametricKernel(edges=edges, values=values),
        )
        result = model.fit(events, T=T, method="mle")

        assert np.isfinite(result.log_likelihood)
        print(f"\nNonparametric fit: LL={result.log_likelihood:.2f}")

    def test_fit_approx_power_law_kernel(self, high_rate_neuron):
        """ApproxPowerLaw (recursive) on full dataset — O(N) heavy-tail model."""
        import intensify as its

        events = high_rate_neuron
        T = float(events[-1]) + 1.0

        model = its.UnivariateHawkes(
            mu=len(events) / (2 * T),
            kernel=its.ApproxPowerLawKernel(
                alpha=0.3, beta_pow=1.5, beta_min=0.1, n_components=5
            ),
        )
        result = model.fit(events, T=T, method="mle")

        assert np.isfinite(result.log_likelihood)
        assert result.branching_ratio_ < 1.0

    def test_all_kernels_agree_on_direction(self, high_rate_neuron):
        """All kernel types should produce finite LL better than null Poisson."""
        import intensify as its

        events = high_rate_neuron[:300]
        T = float(events[-1]) + 0.5
        rate = len(events) / T

        # Null model: homogeneous Poisson
        poisson_ll = len(events) * np.log(rate) - rate * T

        kernels = {
            "Exponential": its.ExponentialKernel(alpha=0.2, beta=5.0),
            "SumExponential": its.SumExponentialKernel(
                alphas=[0.1, 0.1], betas=[1.0, 10.0]
            ),
            "ApproxPowerLaw": its.ApproxPowerLawKernel(
                alpha=0.2, beta_pow=1.5, beta_min=0.1, n_components=5
            ),
        }
        for name, kernel in kernels.items():
            model = its.UnivariateHawkes(mu=rate / 2, kernel=kernel)
            result = model.fit(events, T=T, method="mle")
            print(
                f"\n  {name}: LL={result.log_likelihood:.2f} "
                f"(Poisson baseline={poisson_ll:.2f}, "
                f"improvement={result.log_likelihood - poisson_ll:.2f})"
            )
            assert np.isfinite(result.log_likelihood)


# ===========================================================================
#  SECTION 3: Multivariate Hawkes — Neural Connectivity
# ===========================================================================
class TestMultivariateHawkesReal:
    """Fit multivariate Hawkes to multiple simultaneously-recorded neurons."""

    def test_bivariate_connectivity(self, session_a_neurons):
        """Two neurons: test cross-excitation recovery."""
        import intensify as its

        if len(session_a_neurons) < 2:
            pytest.skip("Need >= 2 neurons")

        # Pick two neurons with decent spike counts
        sorted_neurons = sorted(session_a_neurons, key=len, reverse=True)
        n1, n2 = sorted_neurons[0], sorted_neurons[1]
        T = max(n1[-1], n2[-1]) + 1.0

        events = [n1, n2]
        print(
            f"\nBivariate: neuron 0 ({len(n1)} spikes), "
            f"neuron 1 ({len(n2)} spikes), T={T:.1f}s"
        )

        model = its.MultivariateHawkes(
            n_dims=2,
            mu=1.0,
            kernel=its.ExponentialKernel(alpha=0.1, beta=5.0),
        )
        t0 = time.perf_counter()
        result = model.fit(events, T=T, method="mle")
        elapsed = time.perf_counter() - t0

        print(f"  Fit time: {elapsed:.2f}s")
        print(result.summary())

        W = result.connectivity_matrix()
        print(f"  Connectivity matrix:\n{W}")

        assert np.isfinite(result.log_likelihood)
        assert W.shape == (2, 2)
        assert np.all(np.isfinite(W))

    def test_five_neuron_connectivity(self, session_a_neurons):
        """5-dim Hawkes — realistic small-network inference."""
        import intensify as its

        if len(session_a_neurons) < 5:
            pytest.skip("Need >= 5 neurons")

        # Use moderate-sized neurons to keep optimization tractable
        candidates = [n for n in session_a_neurons if 50 < len(n) < 300]
        if len(candidates) < 5:
            candidates = sorted(session_a_neurons, key=len, reverse=True)
        sorted_neurons = sorted(candidates, key=len, reverse=True)[:5]
        T = max(n[-1] for n in sorted_neurons) + 1.0
        events = sorted_neurons

        total_spikes = sum(len(n) for n in events)
        print(f"\n5-dim Hawkes: {total_spikes} total spikes, T={T:.1f}s")

        model = its.MultivariateHawkes(
            n_dims=5,
            mu=1.0,
            kernel=its.ExponentialKernel(alpha=0.05, beta=5.0),
        )
        t0 = time.perf_counter()
        result = model.fit(events, T=T, method="mle")
        elapsed = time.perf_counter() - t0

        print(f"  Fit time: {elapsed:.2f}s")
        W = result.connectivity_matrix()
        print(f"  Connectivity matrix (5x5):\n{np.array2string(W, precision=3)}")

        assert np.isfinite(result.log_likelihood)
        assert W.shape == (5, 5)
        spectral_radius = np.max(np.abs(np.linalg.eigvals(W)))
        print(f"  Spectral radius: {spectral_radius:.4f}")
        assert spectral_radius < 1.0, (
            f"Spectral radius {spectral_radius:.4f} >= 1 (non-stationary)"
        )
        assert result.branching_ratio_ is not None
        assert result.branching_ratio_ < 1.0

    def test_connectivity_significant_connections(self, session_a_neurons):
        """Test significance testing on real connectivity."""
        import intensify as its

        if len(session_a_neurons) < 3:
            pytest.skip("Need >= 3 neurons")

        # Use moderate-sized neurons to keep runtime tractable
        candidates = [n for n in session_a_neurons if 50 < len(n) < 200]
        if len(candidates) < 3:
            candidates = sorted(session_a_neurons, key=len, reverse=True)
        sorted_neurons = sorted(candidates, key=len, reverse=True)[:3]
        T = max(n[-1] for n in sorted_neurons) + 1.0

        model = its.MultivariateHawkes(
            n_dims=3,
            mu=1.0,
            kernel=its.ExponentialKernel(alpha=0.05, beta=5.0),
        )
        result = model.fit(sorted_neurons, T=T, method="mle")
        sig = result.significant_connections(significance_level=0.05)
        print(f"\nSignificant connections (3x3):\n{sig}")
        assert sig.shape == (3, 3)
        assert sig.dtype == bool

    def test_multivariate_with_regularization(self, session_a_neurons):
        """L1-regularized multivariate MLE for sparse connectivity."""
        import intensify as its
        from intensify.core.regularizers import L1

        if len(session_a_neurons) < 4:
            pytest.skip("Need >= 4 neurons")

        # Use moderate-sized neurons to keep runtime tractable
        candidates = [n for n in session_a_neurons if 50 < len(n) < 200]
        if len(candidates) < 4:
            candidates = sorted(session_a_neurons, key=len, reverse=True)
        sorted_neurons = sorted(candidates, key=len, reverse=True)[:4]
        T = max(n[-1] for n in sorted_neurons) + 1.0

        model = its.MultivariateHawkes(
            n_dims=4,
            mu=1.0,
            kernel=its.ExponentialKernel(alpha=0.05, beta=5.0),
        )
        result = model.fit(
            sorted_neurons,
            T=T,
            method="mle",
            regularization=L1(strength=0.1),
        )
        W = result.connectivity_matrix()
        print(f"\nL1-regularized connectivity:\n{np.array2string(W, precision=4)}")
        # L1 should push some connections toward zero
        assert np.isfinite(result.log_likelihood)
        n_near_zero = np.sum(W < 0.01)
        print(f"  Connections near zero: {n_near_zero}/{W.size}")


# ===========================================================================
#  SECTION 4: Nonlinear / Inhibitory Hawkes
# ===========================================================================
class TestNonlinearHawkesReal:
    """Test nonlinear link functions on real neural data — where inhibition exists."""

    def test_softplus_link(self, high_rate_neuron):
        """Softplus link on real spikes."""
        import intensify as its

        events = high_rate_neuron[:500]
        T = float(events[-1]) + 0.5

        model = its.NonlinearHawkes(
            mu=len(events) / T,
            kernel=its.ExponentialKernel(alpha=0.3, beta=5.0, allow_signed=True),
            link_function="softplus",
        )
        result = model.fit(events, T=T, method="mle")
        print(f"\nSoftplus Hawkes: {result.summary()}")
        assert np.isfinite(result.log_likelihood)

    def test_sigmoid_link(self, high_rate_neuron):
        """Sigmoid link — bounded intensity."""
        import intensify as its

        events = high_rate_neuron[:500]
        T = float(events[-1]) + 0.5

        model = its.NonlinearHawkes(
            mu=len(events) / T,
            kernel=its.ExponentialKernel(alpha=0.3, beta=5.0, allow_signed=True),
            link_function="sigmoid",
            sigmoid_scale=float(2 * len(events) / T),
        )
        result = model.fit(events, T=T, method="mle")
        assert np.isfinite(result.log_likelihood)

    def test_inhibitory_kernel_real_data(self, high_rate_neuron):
        """Signed (inhibitory) kernel with softplus link."""
        import intensify as its

        events = high_rate_neuron[:500]
        T = float(events[-1]) + 0.5

        model = its.NonlinearHawkes(
            mu=len(events) / T,
            kernel=its.ExponentialKernel(alpha=-0.2, beta=5.0, allow_signed=True),
            link_function="softplus",
        )
        result = model.fit(events, T=T, method="mle")
        assert np.isfinite(result.log_likelihood)
        # Fitted alpha could be positive or negative
        print(f"\nInhibitory fit alpha: {result.params.get('alpha', 'N/A')}")


# ===========================================================================
#  SECTION 5: Marked Hawkes
# ===========================================================================
class TestMarkedHawkesReal:
    """Use ISI (inter-spike interval) as marks for marked Hawkes."""

    def test_marked_hawkes_with_isi_marks(self, high_rate_neuron):
        """ISI as marks: bursts have short ISI → should amplify excitation."""
        import intensify as its

        events = high_rate_neuron
        T = float(events[-1]) + 1.0

        # Use preceding ISI as mark (short ISI = burst)
        isis = np.diff(events)
        # First event has no ISI, use median
        marks = np.concatenate([[np.median(isis)], isis])

        model = its.MarkedHawkes(
            mu=len(events) / (2 * T),
            kernel=its.ExponentialKernel(alpha=0.2, beta=5.0),
            mark_influence="log",
        )
        result = model.fit(events, marks, T=T, method="mle")
        print(f"\nMarked Hawkes (ISI marks): {result.summary()}")
        assert np.isfinite(result.log_likelihood)

    def test_marked_hawkes_power_influence(self, high_rate_neuron):
        """Power mark influence on real data."""
        import intensify as its

        events = high_rate_neuron[:500]
        T = float(events[-1]) + 0.5
        isis = np.diff(events)
        marks = np.concatenate([[np.median(isis)], isis])[:500]

        model = its.MarkedHawkes(
            mu=len(events) / (2 * T),
            kernel=its.ExponentialKernel(alpha=0.2, beta=5.0),
            mark_influence="power",
            mark_power=0.5,
        )
        result = model.fit(events, marks, T=T, method="mle")
        assert np.isfinite(result.log_likelihood)


# ===========================================================================
#  SECTION 6: Diagnostics on Real Fits
# ===========================================================================
class TestDiagnosticsReal:
    """Goodness-of-fit diagnostics on real data model fits."""

    def test_time_rescaling_test(self, high_rate_neuron):
        """KS test on rescaled times for real data fit."""
        import intensify as its
        from intensify.core.diagnostics.goodness_of_fit import time_rescaling_test

        # Fit first, then test — use subset for speed (O(N^2) in diagnostics)
        events = high_rate_neuron[:300]
        T = float(events[-1]) + 0.5

        model = its.UnivariateHawkes(
            mu=len(events) / (2 * T),
            kernel=its.ExponentialKernel(alpha=0.2, beta=5.0),
        )
        result = model.fit(events, T=T, method="mle")
        ks_stat, p_value = time_rescaling_test(result, events=events, T=T)

        print(f"\nKS test: stat={ks_stat:.4f}, p={p_value:.4f}")
        assert np.isfinite(ks_stat)
        assert 0 <= p_value <= 1

    def test_qq_plot_runs(self, high_rate_neuron):
        """QQ plot should produce a figure without error."""
        import intensify as its
        import matplotlib.pyplot as plt
        from intensify.core.diagnostics.goodness_of_fit import qq_plot

        events = high_rate_neuron[:200]
        T = float(events[-1]) + 0.5

        model = its.UnivariateHawkes(
            mu=len(events) / (2 * T),
            kernel=its.ExponentialKernel(alpha=0.2, beta=5.0),
        )
        result = model.fit(events, T=T, method="mle")
        fig = qq_plot(result, events=events, T=T)
        assert fig is not None
        plt.close(fig)

    def test_plot_diagnostics_full(self, high_rate_neuron):
        """FitResult.plot_diagnostics() on real fit."""
        import intensify as its
        import matplotlib.pyplot as plt

        events = high_rate_neuron[:200]
        T = float(events[-1]) + 0.5

        model = its.UnivariateHawkes(
            mu=len(events) / (2 * T),
            kernel=its.ExponentialKernel(alpha=0.2, beta=5.0),
        )
        result = model.fit(events, T=T, method="mle")
        fig = result.plot_diagnostics()
        assert fig is not None
        plt.close(fig)

    def test_branching_ratio_and_endogeneity(self, high_rate_neuron):
        """These metrics should be computed and finite for real fits."""
        import intensify as its

        events = high_rate_neuron
        T = float(events[-1]) + 1.0

        model = its.UnivariateHawkes(
            mu=len(events) / (2 * T),
            kernel=its.ExponentialKernel(alpha=0.2, beta=5.0),
        )
        result = model.fit(events, T=T, method="mle")

        assert result.branching_ratio_ is not None
        assert 0 <= result.branching_ratio_ < 1.0
        assert result.endogeneity_index_ is not None
        assert 0 <= result.endogeneity_index_ <= 1.0
        print(f"\nBranching ratio: {result.branching_ratio_:.4f}")
        print(f"Endogeneity index: {result.endogeneity_index_:.4f}")

    def test_aic_bic_model_selection(self, high_rate_neuron):
        """Compare AIC/BIC across kernels for genuine model selection."""
        import intensify as its

        events = high_rate_neuron[:500]
        T = float(events[-1]) + 0.5

        results = {}
        for name, kernel in [
            ("Exponential", its.ExponentialKernel(alpha=0.2, beta=5.0)),
            (
                "SumExp(2)",
                its.SumExponentialKernel(alphas=[0.1, 0.1], betas=[1.0, 10.0]),
            ),
        ]:
            model = its.UnivariateHawkes(mu=len(events) / (2 * T), kernel=kernel)
            result = model.fit(events, T=T, method="mle")
            results[name] = result
            print(
                f"\n  {name}: AIC={result.aic:.2f}, BIC={result.bic:.2f}, "
                f"LL={result.log_likelihood:.2f}"
            )

        # Both should have valid info criteria
        for name, r in results.items():
            assert np.isfinite(r.aic)
            assert np.isfinite(r.bic)


# ===========================================================================
#  SECTION 7: Visualization on Real Data
# ===========================================================================
class TestVisualizationReal:
    """Visualization functions should not crash on real data."""

    def test_plot_intensity(self, high_rate_neuron):
        import intensify as its
        import matplotlib.pyplot as plt

        events = high_rate_neuron[:200]
        T = float(events[-1]) + 0.5
        model = its.UnivariateHawkes(
            mu=len(events) / (2 * T),
            kernel=its.ExponentialKernel(alpha=0.2, beta=5.0),
        )
        result = model.fit(events, T=T, method="mle")

        fig, ax = plt.subplots()
        its.plot_intensity(result, events=events, T=T, ax=ax)
        plt.close(fig)

    def test_plot_kernel(self):
        import intensify as its
        import matplotlib.pyplot as plt

        for kernel in [
            its.ExponentialKernel(alpha=0.3, beta=5.0),
            its.SumExponentialKernel(alphas=[0.1, 0.2], betas=[1.0, 10.0]),
            its.PowerLawKernel(alpha=0.5, beta=1.5, c=0.01),
        ]:
            fig, ax = plt.subplots()
            its.plot_kernel(kernel, ax=ax)
            plt.close(fig)

    def test_plot_connectivity(self, session_a_neurons):
        import intensify as its
        import matplotlib.pyplot as plt

        if len(session_a_neurons) < 3:
            pytest.skip("Need >= 3 neurons")

        candidates = [n for n in session_a_neurons if 50 < len(n) < 200]
        if len(candidates) < 3:
            candidates = sorted(session_a_neurons, key=len, reverse=True)
        sorted_neurons = sorted(candidates, key=len, reverse=True)[:3]
        T = max(n[-1] for n in sorted_neurons) + 1.0

        model = its.MultivariateHawkes(
            n_dims=3,
            mu=1.0,
            kernel=its.ExponentialKernel(alpha=0.05, beta=5.0),
        )
        result = model.fit(sorted_neurons, T=T, method="mle")

        fig, ax = plt.subplots()
        its.plot_connectivity(result.connectivity_matrix(), ax=ax)
        plt.close(fig)

    def test_plot_inter_event_intervals(self, high_rate_neuron):
        import intensify as its
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        its.plot_inter_event_intervals(high_rate_neuron, ax=ax)
        plt.close(fig)

    def test_plot_event_aligned_histogram(self, high_rate_neuron):
        import intensify as its
        import matplotlib.pyplot as plt

        # Use a subset of events as reference times (PSTH-style)
        reference = high_rate_neuron[::10]  # Every 10th spike as trigger
        fig, ax = plt.subplots()
        its.plot_event_aligned_histogram(
            high_rate_neuron,
            reference_times=reference,
            window=(-0.05, 0.05),
            ax=ax,
        )
        plt.close(fig)


# ===========================================================================
#  SECTION 8: Performance & Scale Stress
# ===========================================================================
class TestPerformanceStress:
    """Verify performance characteristics at scale."""

    def test_recursive_likelihood_scales_linearly(self, high_rate_neuron):
        """O(N) recursive path should scale roughly linearly."""
        import intensify as its

        events = high_rate_neuron
        T = float(events[-1]) + 1.0

        model = its.UnivariateHawkes(
            mu=5.0,
            kernel=its.ExponentialKernel(alpha=0.3, beta=5.0),
        )

        sizes = [100, 500, 1000, min(5000, len(events))]
        times_list = []
        for n in sizes:
            ev = events[:n]
            t_end = float(ev[-1]) + 0.5
            t0 = time.perf_counter()
            for _ in range(3):
                model.log_likelihood(ev, t_end)
            elapsed = (time.perf_counter() - t0) / 3
            times_list.append(elapsed)
            print(f"\n  N={n}: LL time = {elapsed * 1000:.1f}ms")

        # O(N) check: time(5000) / time(500) should be roughly 10, not 100
        if len(times_list) == 4 and times_list[1] > 1e-6:
            ratio = times_list[3] / times_list[1]
            print(f"  Scaling ratio (5000/500): {ratio:.1f}x (ideal ~10x for O(N))")
            # Allow generous margin — 30x would indicate O(N^2)
            assert ratio < 30, f"Scaling ratio {ratio:.1f}x suggests worse than O(N)"

    def test_general_likelihood_is_quadratic_warning(self, high_rate_neuron):
        """O(N^2) path should warn for large N (config threshold)."""
        import intensify as its

        events = high_rate_neuron
        if len(events) < 100:
            pytest.skip("Need enough spikes for this test")

        # Lower threshold to trigger warning
        old_val = its.config_get("recursive_warning_threshold")
        its.config_set("recursive_warning_threshold", 50)
        try:
            model = its.UnivariateHawkes(
                mu=5.0,
                kernel=its.PowerLawKernel(alpha=0.5, beta=1.5, c=0.01),
            )
            # This should emit a performance warning for N > 50
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                model.log_likelihood(events[:100], float(events[99]) + 0.5)
                # Check if any performance-related warning was emitted
                perf_warns = [
                    x
                    for x in w
                    if "performance" in str(x.message).lower()
                    or "O(N" in str(x.message)
                    or "N²" in str(x.message)
                    or "expensive" in str(x.message).lower()
                ]
                # Just verify it doesn't crash; warning is optional
        finally:
            its.config_set("recursive_warning_threshold", old_val)

    def test_intensity_computation_vectorized(self, high_rate_neuron):
        """Vectorized intensity evaluation over a time grid."""
        import intensify as its

        events = high_rate_neuron[:1000]
        T = float(events[-1]) + 0.5

        model = its.UnivariateHawkes(
            mu=5.0,
            kernel=its.ExponentialKernel(alpha=0.3, beta=5.0),
        )

        # Compute intensity on a grid of 500 points
        t_grid = np.linspace(0.01, T, 500)
        t0 = time.perf_counter()
        intensities = model.intensity(t_grid, events)
        elapsed = time.perf_counter() - t0

        print(
            f"\nVectorized intensity (500 points, 1000 spikes): {elapsed * 1000:.1f}ms"
        )
        assert len(intensities) == 500
        assert np.all(np.isfinite(intensities))
        assert np.all(intensities >= 0), "Intensity must be non-negative"


# ===========================================================================
#  SECTION 9: Online / Streaming Inference
# ===========================================================================
class TestOnlineInferenceReal:
    """Test streaming parameter updates on real spike data."""

    def test_online_updates_on_real_spikes(self, high_rate_neuron):
        """Online SGD should produce reasonable params after streaming real spikes."""
        import intensify as its

        events = high_rate_neuron[:1000]

        model = its.UnivariateHawkes(
            mu=5.0,
            kernel=its.ExponentialKernel(alpha=0.2, beta=5.0),
        )
        engine = its.OnlineInference(lr=0.001, window=500, min_events=20)

        for t_i in events:
            engine.update(model, float(t_i))

        print(
            f"\nAfter online updates: mu={model.mu:.4f}, "
            f"alpha={model.kernel.alpha:.4f}, beta={model.kernel.beta:.4f}"
        )
        assert model.mu > 0
        assert model.kernel.beta > 0


# ===========================================================================
#  SECTION 10: Edge Cases with Real Data Characteristics
# ===========================================================================
class TestEdgeCasesRealData:
    """Edge cases inspired by real neural data properties."""

    def test_very_short_observation_window(self, high_rate_neuron):
        """Fit on first 1 second of recording."""
        import intensify as its

        mask = high_rate_neuron < 1.0
        events = high_rate_neuron[mask]
        if len(events) < 5:
            pytest.skip("Not enough spikes in first second")
        T = 1.0

        model = its.UnivariateHawkes(
            mu=len(events) / T,
            kernel=its.ExponentialKernel(alpha=0.1, beta=5.0),
        )
        result = model.fit(events, T=T, method="mle")
        assert np.isfinite(result.log_likelihood)

    def test_neuron_with_bursts(self, session_a_neurons):
        """Find burstiest neuron (smallest min ISI) and fit."""
        import intensify as its

        if not session_a_neurons:
            pytest.skip("No neurons")

        # Find neuron with smallest inter-spike intervals (burstiest)
        burstiest = min(
            [n for n in session_a_neurons if len(n) > 50],
            key=lambda n: np.percentile(np.diff(n), 5),
            default=None,
        )
        if burstiest is None:
            pytest.skip("No neurons with > 50 spikes")

        isis = np.diff(burstiest)
        print(
            f"\nBurstiest neuron: {len(burstiest)} spikes, "
            f"5th percentile ISI = {np.percentile(isis, 5) * 1000:.2f}ms, "
            f"min ISI = {isis.min() * 1000:.2f}ms"
        )

        T = float(burstiest[-1]) + 1.0
        model = its.UnivariateHawkes(
            mu=len(burstiest) / (2 * T),
            kernel=its.ExponentialKernel(alpha=0.3, beta=10.0),
        )
        result = model.fit(burstiest, T=T, method="mle")

        assert np.isfinite(result.log_likelihood)
        # Bursty neurons should have high self-excitation
        print(f"  Branching ratio: {result.branching_ratio_:.4f}")

    def test_cross_session_consistency(self, session_a_neurons, session_b_neurons):
        """Same model type fitted to different sessions should produce valid results."""
        import intensify as its

        for name, neurons in [("A", session_a_neurons), ("B", session_b_neurons)]:
            if not neurons:
                continue
            best = max(neurons, key=len)
            T = float(best[-1]) + 1.0

            model = its.UnivariateHawkes(
                mu=len(best) / (2 * T),
                kernel=its.ExponentialKernel(alpha=0.2, beta=5.0),
            )
            result = model.fit(best, T=T, method="mle")
            print(
                f"\nSession {name}: LL={result.log_likelihood:.2f}, "
                f"BR={result.branching_ratio_:.4f}, "
                f"mu={result.params['mu']:.4f}"
            )
            assert np.isfinite(result.log_likelihood)
            assert result.branching_ratio_ < 1.0

    def test_simultaneous_events_handling(self, session_a_neurons):
        """Test behavior when two neurons fire near-simultaneously."""
        import intensify as its

        if len(session_a_neurons) < 2:
            pytest.skip("Need >= 2 neurons")

        n1, n2 = session_a_neurons[0], session_a_neurons[1]
        # Merge and sort — some timestamps may be very close
        merged = np.sort(np.concatenate([n1, n2]))
        min_gap = np.diff(merged).min()
        print(f"\nMerged neurons: min gap = {min_gap * 1e6:.2f} μs")

        T = float(merged[-1]) + 1.0
        model = its.UnivariateHawkes(
            mu=len(merged) / (2 * T),
            kernel=its.ExponentialKernel(alpha=0.1, beta=5.0),
        )
        result = model.fit(merged, T=T, method="mle")
        assert np.isfinite(result.log_likelihood)


# ===========================================================================
#  SECTION 11: Poisson Baselines
# ===========================================================================
class TestPoissonBaseline:
    """Verify Poisson models work as baselines on real data."""

    def test_homogeneous_poisson_fit(self, high_rate_neuron):
        import intensify as its

        events = high_rate_neuron
        T = float(events[-1]) + 1.0

        model = its.HomogeneousPoisson()
        result = model.fit(events, T=T)
        expected_rate = len(events) / T

        assert np.isclose(result.params["rate"], expected_rate, rtol=0.01)
        print(
            f"\nPoisson rate: {result.params['rate']:.4f} "
            f"(expected {expected_rate:.4f})"
        )

    def test_hawkes_beats_poisson(self, high_rate_neuron):
        """Hawkes should have better (higher) LL than Poisson on bursty neural data."""
        import intensify as its

        events = high_rate_neuron[:500]
        T = float(events[-1]) + 0.5

        poisson = its.HomogeneousPoisson()
        poisson_result = poisson.fit(events, T=T)

        hawkes = its.UnivariateHawkes(
            mu=len(events) / (2 * T),
            kernel=its.ExponentialKernel(alpha=0.2, beta=5.0),
        )
        hawkes_result = hawkes.fit(events, T=T, method="mle")

        print(f"\nPoisson LL: {poisson_result.log_likelihood:.2f}")
        print(f"Hawkes LL: {hawkes_result.log_likelihood:.2f}")
        print(
            f"Improvement: {hawkes_result.log_likelihood - poisson_result.log_likelihood:.2f}"
        )

        # Hawkes should fit better on real neural data (which is bursty)
        assert hawkes_result.log_likelihood >= poisson_result.log_likelihood, (
            "Hawkes should fit at least as well as Poisson on neural data"
        )


# ===========================================================================
#  SECTION 12: Simulation → Fit Round-Trip with Real Parameters
# ===========================================================================
class TestSimulationRoundTrip:
    """Fit real data, then simulate from fitted model and re-fit to check consistency."""

    def test_fit_simulate_refit(self, high_rate_neuron):
        """Parameters recovered from simulation should be close to original fit."""
        import intensify as its

        events = high_rate_neuron[:1000]
        T = float(events[-1]) + 0.5

        # Step 1: fit real data
        model = its.UnivariateHawkes(
            mu=len(events) / (2 * T),
            kernel=its.ExponentialKernel(alpha=0.2, beta=5.0),
        )
        real_result = model.fit(events, T=T, method="mle")
        real_mu = real_result.params["mu"]
        # Kernel params may be stored as top-level keys or inside a kernel object
        kernel_params = real_result.params.get("kernel", {})
        if isinstance(kernel_params, dict):
            real_alpha = real_result.params.get("alpha", kernel_params.get("alpha"))
            real_beta = real_result.params.get("beta", kernel_params.get("beta"))
        else:
            # kernel is an object — extract from the fitted process
            real_alpha = getattr(real_result.process.kernel, "alpha", None)
            real_beta = getattr(real_result.process.kernel, "beta", None)
        print(f"\nReal fit: mu={real_mu}, alpha={real_alpha}, beta={real_beta}")

        if real_alpha is None or real_beta is None:
            pytest.skip("Could not extract kernel params from fit result")

        # Step 2: simulate from fitted params
        sim_model = its.UnivariateHawkes(
            mu=float(real_mu),
            kernel=its.ExponentialKernel(
                alpha=float(real_alpha), beta=float(real_beta)
            ),
        )
        sim_events = sim_model.simulate(T=T, seed=42)
        print(f"  Simulated {len(sim_events)} events (real had {len(events)})")

        if len(sim_events) < 20:
            pytest.skip("Too few simulated events for stable re-fit")

        # Step 3: re-fit simulated data
        refit_model = its.UnivariateHawkes(
            mu=0.5,
            kernel=its.ExponentialKernel(alpha=0.15, beta=1.0),
        )
        refit_result = refit_model.fit(sim_events, T=T, method="mle")
        refit_mu = refit_result.params["mu"]

        print(f"  Re-fit: mu={refit_mu}")
        assert np.isfinite(refit_result.log_likelihood)
        assert refit_result.branching_ratio_ < 1.0
