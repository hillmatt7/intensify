"""Scaling study: fit time vs N (total events) for intensify and tick.

Runs the same scenarios across both libraries (different Python envs),
writes results to ``benchmarks/results/scaling_<lib>.json``.

Usage:
    .venv/bin/python benchmarks/run_scaling.py --lib intensify
    micromamba run -n tickbench38 python benchmarks/run_scaling.py --lib tick
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time

import numpy as np

DATA_DIR = pathlib.Path(__file__).parent / "data"
RESULTS_DIR = pathlib.Path(__file__).parent / "results"

SCENARIOS = [
    "mv_exp_5d_scale_small",
    "mv_exp_5d_scale_medium",
    "mv_exp_5d_scale_large",
    "mv_exp_5d_scale_xl",
    "mv_exp_5d_scale_xxl",
]


def _load(name: str) -> dict:
    stem = DATA_DIR / f"reference_{name}"
    meta_path = pathlib.Path(f"{stem}.json")
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    if meta["kind"] == "univariate":
        events = np.load(f"{stem}.npy")
    else:
        events = [np.load(f"{stem}.dim{i}.npy") for i in range(meta["M"])]
    return {"events": events, "T": meta["T"], "ground_truth": meta["ground_truth"]}


def run_intensify(name: str, d: dict, n_runs: int = 3) -> dict:
    import intensify as its

    events = [np.asarray(e) for e in d["events"]]
    M = len(events)
    T = d["T"]
    beta_true = float(d["ground_truth"]["beta"])
    gt_alpha = np.asarray(d["ground_truth"]["alpha"])
    n = int(sum(len(e) for e in events))

    # Warm up JIT with a throwaway fit
    m = its.MultivariateHawkes(
        M, [0.3] * M, its.ExponentialKernel(0.1, beta_true),
    )
    m.fit(events, T=T, fit_decay=False)

    # Timed runs
    times = []
    rmse = None
    for _ in range(n_runs):
        m = its.MultivariateHawkes(
            M, [0.3] * M, its.ExponentialKernel(0.1, beta_true),
        )
        t = time.perf_counter()
        r = m.fit(events, T=T, fit_decay=False)
        times.append(time.perf_counter() - t)
        # RMSE on last run
        fitted = r.flat_params()
        alpha = np.zeros_like(gt_alpha)
        for mm in range(M):
            for kk in range(M):
                alpha[mm, kk] = fitted.get(f"alpha_{mm}_{kk}", float("nan"))
        rmse = float(np.sqrt(np.nanmean((alpha - gt_alpha) ** 2)))

    return {"n": n, "fit_time_s": statistics.median(times), "rmse": rmse}


def run_tick(name: str, d: dict, n_runs: int = 3) -> dict:
    from tick.hawkes import HawkesExpKern

    events = [np.asarray(e, dtype=float) for e in d["events"]]
    M = len(events)
    T = d["T"]
    beta_true = float(d["ground_truth"]["beta"])
    gt_alpha = np.asarray(d["ground_truth"]["alpha"])
    n = int(sum(len(e) for e in events))

    # Warm up
    model = HawkesExpKern(decays=beta_true)
    model.fit([events])

    times = []
    rmse = None
    for _ in range(n_runs):
        model = HawkesExpKern(decays=beta_true)
        t = time.perf_counter()
        model.fit([events])
        times.append(time.perf_counter() - t)
        alpha = np.asarray(model.adjacency)
        rmse = float(np.sqrt(np.nanmean((alpha - gt_alpha) ** 2)))

    return {"n": n, "fit_time_s": statistics.median(times), "rmse": rmse}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", choices=["intensify", "tick"], required=True)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    runner = run_intensify if args.lib == "intensify" else run_tick
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for name in SCENARIOS:
        d = _load(name)
        if d is None:
            print(f"[skip] {name}: dataset missing")
            continue
        try:
            rec = runner(name, d, n_runs=args.runs)
        except Exception as e:
            print(f"[error] {name}: {e}")
            continue
        rec["scenario"] = name
        rec["library"] = args.lib
        results.append(rec)
        print(
            f"{name}: N={rec['n']}  fit_time={rec['fit_time_s']:.3f}s  RMSE={rec['rmse']:.4f}"
        )

    out = RESULTS_DIR / f"scaling_{args.lib}.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
