"""HC-3 head-to-head: intensify vs tick on real spike-train data.

Covers the apples-to-apples Hawkes subset (univariate exp + multivariate
exp, decay-given) plus a Poisson baseline + Cox process — to make the
point that intensify is broad-spectrum point processes, not just
Hawkes. tick has no MLE for power-law / nonparametric / marked /
nonlinear kernels, no LGCP, and no shot-noise Cox.

Usage:
    python benchmarks/run_hc3_comparison.py --lib intensify
    python benchmarks/run_hc3_comparison.py --lib tick   # needs separate env

Loads the same HC-3 fixtures used by tests/test_real_data_stress.py
(env var HC3_ROOT, default ~/hc-3).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

HC3_ROOT = Path(os.environ.get("HC3_ROOT", Path.home() / "hc-3"))
SESSION_A_DIR = HC3_ROOT / "ec013.33" / "ec013.544"
SAMPLE_RATE = 20_000

DECAY = 1.0  # β fixed across both libs (decay-given mode)
N_NEURONS_MV = 5  # 5-neuron multivariate problem, matches run_scaling settings
N_REPEATS = 3


def _load_spike_trains(session_dir: Path, electrode: int) -> dict[int, np.ndarray]:
    stem = session_dir.name
    res_path = session_dir / f"{stem}.res.{electrode}"
    clu_path = session_dir / f"{stem}.clu.{electrode}"
    if not res_path.exists() or not clu_path.exists():
        return {}
    spike_samples = np.loadtxt(res_path, dtype=np.int64)
    clu_data = np.loadtxt(clu_path, dtype=np.int32)
    n_clusters = clu_data[0]
    cluster_ids = clu_data[1:]
    spike_times = spike_samples / SAMPLE_RATE
    trains = {}
    for cid in range(2, n_clusters + 1):
        mask = cluster_ids == cid
        if mask.sum() > 0:
            trains[cid] = np.sort(spike_times[mask])
    return trains


def _load_all_neurons(session_dir: Path) -> list[np.ndarray]:
    out = []
    for e in range(1, 9):
        for _, t in sorted(_load_spike_trains(session_dir, e).items()):
            if len(t) >= 10:
                out.append(t)
    return out


def _setup() -> tuple[np.ndarray, list[np.ndarray], float]:
    if not SESSION_A_DIR.exists():
        raise SystemExit(f"HC-3 data not found at {SESSION_A_DIR}")
    neurons = _load_all_neurons(SESSION_A_DIR)
    if not neurons:
        raise SystemExit("No neurons loaded")
    high_rate = max(neurons, key=len)
    # Pick the top N_NEURONS_MV by spike count for the multivariate problem.
    mv_neurons = sorted(neurons, key=len, reverse=True)[:N_NEURONS_MV]
    T = float(max(t[-1] for t in mv_neurons)) + 0.1
    return high_rate, mv_neurons, T


def _time(fn, repeats: int = N_REPEATS) -> tuple[float, list[float]]:
    """Median wall time over `repeats` runs (after one warmup)."""
    fn()  # warmup
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)  # ms
    return statistics.median(times), times


def _run_intensify(high_rate: np.ndarray, mv_neurons: list[np.ndarray], T: float) -> dict:
    import intensify as its

    print(f"\n[intensify] uni_exp on N={len(high_rate)} spikes, T={T:.1f} s, β={DECAY}")
    def fit_uni():
        m = its.Hawkes(mu=0.5, kernel=its.ExponentialKernel(alpha=0.1, beta=DECAY))
        m.fit(high_rate, T=T, fit_decay=False)
    uni_ms, uni_runs = _time(fit_uni)
    print(f"  median: {uni_ms:.2f} ms (runs: {[f'{r:.2f}' for r in uni_runs]})")

    n_mv = len(mv_neurons)
    n_total = sum(len(t) for t in mv_neurons)
    print(f"\n[intensify] mv_exp on M={n_mv} neurons, N={n_total} total spikes, T={T:.1f} s")
    def fit_mv():
        m = its.MultivariateHawkes(
            n_dims=n_mv,
            mu=[0.5] * n_mv,
            kernel=its.ExponentialKernel(alpha=0.05, beta=DECAY),
        )
        m.fit(mv_neurons, T=T, fit_decay=False)
    mv_ms, mv_runs = _time(fit_mv)
    print(f"  median: {mv_ms:.2f} ms (runs: {[f'{r:.2f}' for r in mv_runs]})")

    # Recover branching ratio for sanity
    m = its.MultivariateHawkes(
        n_dims=n_mv, mu=[0.5] * n_mv,
        kernel=its.ExponentialKernel(alpha=0.05, beta=DECAY),
    )
    r = m.fit(mv_neurons, T=T, fit_decay=False)
    print(f"  fitted branching ratio: {r.branching_ratio_:.3f}")

    # Poisson baseline — same data, simpler model.
    print(f"\n[intensify] HomogeneousPoisson on N={len(high_rate)} spikes")
    def fit_poisson():
        p = its.HomogeneousPoisson()
        p.fit(high_rate, T=T)
    poisson_ms, poisson_runs = _time(fit_poisson)
    print(f"  median: {poisson_ms:.3f} ms (runs: {[f'{r:.3f}' for r in poisson_runs]})")

    # Cox: LGCP (intensify-exclusive — tick has no LGCP). Simulate then
    # evaluate log-likelihood; tick has no equivalent capability.
    print("\n[intensify] LogGaussianCoxProcess: simulate + log_likelihood (T=29.2 s, n_bins=50)")
    cox = its.LogGaussianCoxProcess(n_bins=50)
    def cox_simlik():
        e = cox.simulate(T=T, seed=0)
        cox.log_likelihood(e, T=T)
    lgcp_ms, lgcp_runs = _time(cox_simlik)
    print(f"  median: {lgcp_ms:.3f} ms (runs: {[f'{r:.3f}' for r in lgcp_runs]})")

    return {
        "library": "intensify",
        "uni_exp_ms": uni_ms,
        "uni_exp_runs": uni_runs,
        "uni_exp_n_events": int(len(high_rate)),
        "mv_exp_ms": mv_ms,
        "mv_exp_runs": mv_runs,
        "mv_exp_n_dims": n_mv,
        "mv_exp_n_total_events": int(n_total),
        "T": T,
        "decay": DECAY,
        "branching_ratio": float(r.branching_ratio_),
        "homogeneous_poisson_fit_ms": poisson_ms,
        "homogeneous_poisson_runs": poisson_runs,
        "lgcp_log_likelihood_ms": lgcp_ms,
        "lgcp_log_likelihood_runs": lgcp_runs,
    }


def _patch_tick_setattr():
    """Work around tick 0.8.0.1's broken `__setattr__`.

    BaseMeta installs a per-class `__setattr__` on every subclass that
    rejects attributes not pre-declared in `_attrinfos`. On recent Python
    + numpy combos the registry is incomplete; tick can't even construct
    its own learners. Patch every subclass of `tick.base.Base` to fall
    through to `object.__setattr__` when the registry rejects.
    """
    import tick.base as _tb
    # Force-import the learner + model trees so all the subclasses we need
    # are present in Base.__subclasses__ before we walk it.
    import tick.hawkes  # noqa: F401
    import tick.base_model  # noqa: F401

    def make_patched_setter(orig):
        def patched(self, key, value):
            try:
                orig(self, key, value)
            except AttributeError:
                object.__setattr__(self, key, value)
        return patched

    seen = set()

    def walk(cls):
        for sub in cls.__subclasses__():
            if sub in seen:
                continue
            seen.add(sub)
            for name in ("__setattr__", "_set"):
                if name in vars(sub):
                    try:
                        setattr(sub, name, make_patched_setter(vars(sub)[name]))
                    except TypeError:
                        pass
            walk(sub)

    for name in ("__setattr__", "_set"):
        if name in vars(_tb.Base):
            setattr(_tb.Base, name, make_patched_setter(vars(_tb.Base)[name]))
    walk(_tb.Base)


def _run_tick(high_rate: np.ndarray, mv_neurons: list[np.ndarray], T: float) -> dict:
    _patch_tick_setattr()
    from tick.hawkes import HawkesExpKern

    print(f"\n[tick] uni_exp (likelihood) on N={len(high_rate)} spikes, T={T:.1f} s, β={DECAY}")
    def fit_uni():
        m = HawkesExpKern(decays=DECAY, gofit="likelihood")
        m.fit([high_rate])
    uni_ms, uni_runs = _time(fit_uni)
    print(f"  median: {uni_ms:.2f} ms (runs: {[f'{r:.2f}' for r in uni_runs]})")

    n_mv = len(mv_neurons)
    n_total = sum(len(t) for t in mv_neurons)

    # MV likelihood — tick's positivity prox doesn't constrain the search
    # path, so AGD diverges into negative-α territory and the C++ loss
    # raises. Record the failure.
    print(f"\n[tick] mv_exp (likelihood) on M={n_mv} neurons, N={n_total} total spikes")
    mv_lik_status = "ok"
    mv_lik_ms = None
    mv_lik_runs = None
    try:
        def fit_mv_lik():
            m = HawkesExpKern(decays=DECAY, gofit="likelihood")
            m.fit(mv_neurons)
        mv_lik_ms, mv_lik_runs = _time(fit_mv_lik)
        print(f"  median: {mv_lik_ms:.2f} ms (runs: {[f'{r:.2f}' for r in mv_lik_runs]})")
    except Exception as e:
        mv_lik_status = f"FAIL: {type(e).__name__}: {e}"
        print(f"  FAILED: {mv_lik_status[:120]}")

    # tick HomogeneousPoisson: not directly a fit API; tick has no
    # `HomogeneousPoissonFit` — only `SimuPoissonProcess` for simulation.
    # Record this gap honestly.
    print("\n[tick] HomogeneousPoisson.fit:  not provided by tick (only SimuPoissonProcess for sim)")

    # tick LGCP / Shot-Noise Cox: also unsupported.
    print("[tick] LogGaussianCoxProcess:    not provided")
    print("[tick] ShotNoiseCoxProcess:      not provided")

    # MV least-squares — different objective, but it's the MV path tick
    # actually finishes on real spike data. Record both for honesty.
    print(f"\n[tick] mv_exp (least-squares) on M={n_mv} neurons, N={n_total}")
    mv_ls_status = "ok"
    mv_ls_ms = None
    mv_ls_runs = None
    try:
        def fit_mv_ls():
            m = HawkesExpKern(decays=DECAY, gofit="least-squares")
            m.fit(mv_neurons)
        mv_ls_ms, mv_ls_runs = _time(fit_mv_ls)
        print(f"  median: {mv_ls_ms:.2f} ms (runs: {[f'{r:.2f}' for r in mv_ls_runs]})")
    except Exception as e:
        mv_ls_status = f"FAIL: {type(e).__name__}: {e}"
        print(f"  FAILED: {mv_ls_status[:120]}")

    return {
        "library": "tick",
        "uni_exp_ms": uni_ms,
        "uni_exp_runs": uni_runs,
        "uni_exp_n_events": int(len(high_rate)),
        "mv_exp_likelihood_ms": mv_lik_ms,
        "mv_exp_likelihood_runs": mv_lik_runs,
        "mv_exp_likelihood_status": mv_lik_status,
        "mv_exp_least_squares_ms": mv_ls_ms,
        "mv_exp_least_squares_runs": mv_ls_runs,
        "mv_exp_least_squares_status": mv_ls_status,
        "mv_exp_n_dims": n_mv,
        "mv_exp_n_total_events": int(n_total),
        "T": T,
        "decay": DECAY,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", choices=["intensify", "tick"], required=True)
    parser.add_argument(
        "--out",
        default=None,
        help="Optional results JSON path (default: benchmarks/results/hc3_<lib>.json)",
    )
    args = parser.parse_args()

    high_rate, mv_neurons, T = _setup()
    print(f"Loaded HC-3 session A: {len(mv_neurons)} top neurons, T={T:.1f} s")

    if args.lib == "intensify":
        results = _run_intensify(high_rate, mv_neurons, T)
    else:
        results = _run_tick(high_rate, mv_neurons, T)

    out = Path(args.out) if args.out else Path(__file__).parent / "results" / f"hc3_{args.lib}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\n→ Results written to {out}")


if __name__ == "__main__":
    main()
