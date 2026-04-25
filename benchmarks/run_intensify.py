"""Run intensify MLE on each reference scenario and emit JSON results."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time

import numpy as np

import intensify as its

DATA_DIR = pathlib.Path(__file__).parent / "data"
RESULTS_DIR = pathlib.Path(__file__).parent / "results"


def _load(name: str) -> dict:
    import json
    stem = DATA_DIR / f"reference_{name}"
    with open(f"{stem}.json") as f:
        meta = json.load(f)
    if meta["kind"] == "univariate":
        events = np.load(f"{stem}.npy")
    else:
        events = [np.load(f"{stem}.dim{i}.npy") for i in range(meta["M"])]
    return {"events": events, "T": meta["T"], "ground_truth": meta["ground_truth"]}


def _fit_and_time(build_model, events, T, n_runs=3):
    times = []
    result = None
    for _ in range(n_runs):
        model = build_model()
        t0 = time.perf_counter()
        result = model.fit(events, T=T)
        times.append(time.perf_counter() - t0)
    return result, statistics.median(times)


def _fit_and_time_with_kw(build_model, events, T, n_runs=3, **fit_kwargs):
    times = []
    result = None
    for _ in range(n_runs):
        model = build_model()
        t0 = time.perf_counter()
        result = model.fit(events, T=T, **fit_kwargs)
        times.append(time.perf_counter() - t0)
    return result, statistics.median(times)


def bench_uni_exp(name: str) -> dict:
    d = _load(name)
    events = np.asarray(d["events"])
    gt = d["ground_truth"]

    # Joint fit: all 3 params (μ, α, β)
    result, dt = _fit_and_time(
        lambda: its.Hawkes(mu=0.5, kernel=its.ExponentialKernel(alpha=0.5, beta=1.0)),
        events, d["T"],
    )
    fitted = result.flat_params()
    rmse = float(np.sqrt(np.mean([
        (fitted.get(k, float("nan")) - v) ** 2 for k, v in gt.items()
    ])))

    # Decay-given fit (apples-to-apples with tick): β fixed at ground truth
    beta_true = float(gt["beta"])
    result_dg, dt_dg = _fit_and_time_with_kw(
        lambda: its.Hawkes(
            mu=0.5, kernel=its.ExponentialKernel(alpha=0.5, beta=beta_true),
        ),
        events, d["T"], fit_decay=False,
    )
    fitted_dg = result_dg.flat_params()
    rmse_dg = float(np.sqrt(np.mean([
        (fitted_dg.get(k, float("nan")) - v) ** 2 for k, v in gt.items()
    ])))

    return {
        "scenario": name, "library": "intensify", "n": int(events.size),
        "fit_time_s": dt, "rmse": rmse, "fitted": fitted, "ground_truth": gt,
        "log_likelihood": result.log_likelihood,
        "decay_given": {
            "fit_time_s": dt_dg, "rmse": rmse_dg, "fitted": fitted_dg,
        },
    }


def bench_uni_power_law(name: str) -> dict:
    d = _load(name)
    events = np.asarray(d["events"])
    result, dt = _fit_and_time(
        lambda: its.Hawkes(
            mu=0.5, kernel=its.PowerLawKernel(alpha=0.3, beta=1.0, c=1.0),
        ),
        events, d["T"],
    )
    gt = d["ground_truth"]
    fitted = result.flat_params()
    rmse = float(np.sqrt(np.mean([
        (fitted.get(k, float("nan")) - v) ** 2 for k, v in gt.items()
    ])))
    return {
        "scenario": name, "library": "intensify", "n": int(events.size),
        "fit_time_s": dt, "rmse": rmse, "fitted": fitted, "ground_truth": gt,
        "log_likelihood": result.log_likelihood,
    }


def bench_mv_exp(name: str) -> dict:
    d = _load(name)
    events = [np.asarray(e) for e in d["events"]]
    M = len(events)
    gt_alpha = np.asarray(d["ground_truth"]["alpha"])
    beta_true = float(d["ground_truth"]["beta"])

    def _alpha_rmse(fitted_dict):
        fitted_alpha = np.zeros_like(gt_alpha)
        for m in range(M):
            for k in range(M):
                fitted_alpha[m, k] = fitted_dict.get(f"alpha_{m}_{k}", float("nan"))
        return float(np.sqrt(np.nanmean((fitted_alpha - gt_alpha) ** 2)))

    # Joint fit
    result, dt = _fit_and_time(
        lambda: its.MultivariateHawkes(
            n_dims=M, mu=[0.3] * M,
            kernel=its.ExponentialKernel(alpha=0.1, beta=1.0),
        ),
        events, d["T"],
    )
    rmse = _alpha_rmse(result.flat_params())

    # Decay-given fit: every β cell locked to ground-truth β
    result_dg, dt_dg = _fit_and_time_with_kw(
        lambda: its.MultivariateHawkes(
            n_dims=M, mu=[0.3] * M,
            kernel=its.ExponentialKernel(alpha=0.1, beta=beta_true),
        ),
        events, d["T"], fit_decay=False,
    )
    rmse_dg = _alpha_rmse(result_dg.flat_params())

    return {
        "scenario": name, "library": "intensify", "n": int(sum(len(e) for e in events)),
        "fit_time_s": dt, "rmse": rmse,
        "log_likelihood": result.log_likelihood,
        "spectral_radius": float(result.branching_ratio_),
        "decay_given": {
            "fit_time_s": dt_dg, "rmse": rmse_dg,
            "spectral_radius": float(result_dg.branching_ratio_),
        },
    }


RUNNERS = {
    "uni_exp_small": bench_uni_exp,
    "uni_exp_large": bench_uni_exp,
    "uni_power_law": bench_uni_power_law,
    "mv_exp_5d": bench_mv_exp,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="*", default=list(RUNNERS), choices=list(RUNNERS))
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.scenarios:
        try:
            record = RUNNERS[name](name)
        except FileNotFoundError:
            print(
                f"[skip] {name}: dataset missing — run `python benchmarks/reference_dataset.py {name}`"
            )
            continue
        out = RESULTS_DIR / f"intensify_{name}.json"
        out.write_text(json.dumps(record, indent=2, default=float))
        print(f"{name}: fit_time={record['fit_time_s']:.3f}s rmse={record['rmse']:.4f}  -> {out}")


if __name__ == "__main__":
    main()
