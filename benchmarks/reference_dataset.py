"""Generate reproducible reference datasets for benchmarking.

Each scenario writes a single ``.npz`` file under ``benchmarks/data/`` with:

- ``events``: event array (univariate) or list of arrays (multivariate)
- ``T``: observation window
- ``ground_truth``: flat dict of the parameters used to simulate
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

import intensify as its

DATA_DIR = pathlib.Path(__file__).parent / "data"


def gen_uni_exp(n: int, seed: int) -> dict:
    np.random.seed(seed)
    mu, alpha, beta = 0.2, 0.5, 1.5
    model = its.Hawkes(mu=mu, kernel=its.ExponentialKernel(alpha=alpha, beta=beta))
    T_est = n / (mu / (1.0 - alpha))  # expected mean event count ≈ n
    events = np.asarray(model.simulate(T=T_est, seed=seed))
    return {
        "events": events,
        "T": float(T_est),
        "ground_truth": {"mu": mu, "alpha": alpha, "beta": beta},
    }


def gen_uni_power_law(n: int, seed: int) -> dict:
    np.random.seed(seed)
    mu, alpha, beta, c = 0.3, 0.4, 1.2, 1.0
    model = its.Hawkes(
        mu=mu, kernel=its.PowerLawKernel(alpha=alpha, beta=beta, c=c),
    )
    T_est = 1000.0
    events = np.asarray(model.simulate(T=T_est, seed=seed))[:n]
    T = float(events[-1]) + 0.1 if events.size else T_est
    return {
        "events": events,
        "T": T,
        "ground_truth": {"mu": mu, "alpha": alpha, "beta": beta, "c": c},
    }


def gen_mv_exp(M: int, T: float, seed: int, *, method: str = "thinning") -> dict:
    """Simulate a 5-d multivariate Hawkes over fixed window T.

    Two methods: ``"thinning"`` (Ogata, default, slower at large T) and
    ``"branching"`` (cluster / Galton-Watson, O(expected-events), used
    for scaling-study datasets where T is large).
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.1, 0.4, size=M).tolist()
    # Diagonally dominant alpha matrix (row-norms ~0.5 < 1 → stationary)
    alpha = np.eye(M) * 0.3 + rng.uniform(0, 0.05, size=(M, M)) * (1 - np.eye(M))
    beta = 1.5
    kernels = [
        [its.ExponentialKernel(alpha=float(alpha[m, k]), beta=beta) for k in range(M)]
        for m in range(M)
    ]
    model = its.MultivariateHawkes(n_dims=M, mu=mu, kernel=kernels)
    if method == "branching":
        from intensify.core.simulation.cluster import branching_simulation_multivariate
        events = branching_simulation_multivariate(model, T=T, seed=seed)
    else:
        events = model.simulate(T=T, seed=seed)
    return {
        "events": [np.asarray(e) for e in events],
        "T": float(T),
        "ground_truth": {"mu": mu, "alpha": alpha.tolist(), "beta": beta},
    }


SCENARIOS = {
    "uni_exp_small": lambda: gen_uni_exp(n=500, seed=42),
    "uni_exp_large": lambda: gen_uni_exp(n=50_000, seed=42),
    "uni_power_law": lambda: gen_uni_power_law(n=5_000, seed=42),
    "mv_exp_5d": lambda: gen_mv_exp(M=5, T=500.0, seed=42),
    # Scaling-study scenarios: fixed M, fixed process parameters, growing T
    "mv_exp_5d_scale_small": lambda: gen_mv_exp(M=5, T=200.0, seed=42, method="branching"),
    "mv_exp_5d_scale_medium": lambda: gen_mv_exp(M=5, T=1_000.0, seed=42, method="branching"),
    "mv_exp_5d_scale_large": lambda: gen_mv_exp(M=5, T=4_000.0, seed=42, method="branching"),
    "mv_exp_5d_scale_xl": lambda: gen_mv_exp(M=5, T=12_000.0, seed=42, method="branching"),
    "mv_exp_5d_scale_xxl": lambda: gen_mv_exp(M=5, T=40_000.0, seed=42, method="branching"),
}


def _save(path: pathlib.Path, payload: dict) -> None:
    """Write events as raw .npy arrays (numpy-version-portable) plus a JSON
    sidecar for T + ground-truth. Avoids pickle cross-version issues.
    """
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.with_suffix("")
    ev = payload["events"]
    if isinstance(ev, list):
        for i, arr in enumerate(ev):
            np.save(f"{stem}.dim{i}.npy", np.asarray(arr, dtype=float))
        meta = {"kind": "multivariate", "M": len(ev),
                "T": float(payload["T"]),
                "ground_truth": _json_safe(payload["ground_truth"])}
    else:
        np.save(f"{stem}.npy", np.asarray(ev, dtype=float))
        meta = {"kind": "univariate",
                "T": float(payload["T"]),
                "ground_truth": _json_safe(payload["ground_truth"])}
    with open(f"{stem}.json", "w") as f:
        json.dump(meta, f, indent=2)


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "scenarios", nargs="*", default=list(SCENARIOS),
        choices=list(SCENARIOS),
    )
    args = ap.parse_args()

    for name in args.scenarios:
        print(f"Generating {name}...", flush=True)
        data = SCENARIOS[name]()
        stem = DATA_DIR / f"reference_{name}"
        _save(stem.with_suffix(".npz"), data)
        print(f"  -> {stem}.json + .npy  (T={data['T']:.1f})")


if __name__ == "__main__":
    main()
