"""Run the `tick` library's Hawkes MLE on each reference scenario.

Run from a py3.8 env with tick installed. Writes results JSON matching the
schema used by ``run_intensify.py`` for apples-to-apples comparison.

Example:

    micromamba run -n tickbench38 python benchmarks/run_tick.py uni_exp_small mv_exp_5d
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


def bench_uni_exp(name: str) -> dict:
    from tick.hawkes import HawkesExpKern

    d = _load(name)
    events = np.asarray(d["events"], dtype=float)
    gt = d["ground_truth"]
    # tick's HawkesExpKern fixes decay; provide ground-truth beta so it's a
    # fair recovery-of-(mu,alpha) comparison.
    beta = float(gt["beta"])

    times = []
    model = None
    for _ in range(3):
        model = HawkesExpKern(decays=beta)
        t0 = time.perf_counter()
        model.fit([[events]])
        times.append(time.perf_counter() - t0)

    mu_hat = float(model.baseline[0])
    alpha_hat = float(model.adjacency[0][0])  # (1,1) matrix for univariate
    fitted = {"mu": mu_hat, "alpha": alpha_hat, "beta": beta}
    rmse = float(np.sqrt(np.mean([
        (fitted.get(k, float("nan")) - v) ** 2 for k, v in gt.items()
    ])))
    return {
        "scenario": name, "library": "tick", "n": int(events.size),
        "fit_time_s": statistics.median(times),
        "rmse": rmse, "fitted": fitted, "ground_truth": gt,
    }


def bench_mv_exp(name: str) -> dict:
    from tick.hawkes import HawkesExpKern

    d = _load(name)
    events_list = [np.asarray(e, dtype=float) for e in d["events"]]
    M = len(events_list)
    gt = d["ground_truth"]
    beta = float(gt["beta"])

    times = []
    model = None
    for _ in range(3):
        model = HawkesExpKern(decays=beta)
        t0 = time.perf_counter()
        model.fit([events_list])
        times.append(time.perf_counter() - t0)

    gt_alpha = np.asarray(gt["alpha"])
    alpha_hat = np.asarray(model.adjacency)
    rmse = float(np.sqrt(np.nanmean((alpha_hat - gt_alpha) ** 2)))
    return {
        "scenario": name, "library": "tick", "n": int(sum(len(e) for e in events_list)),
        "fit_time_s": statistics.median(times),
        "rmse": rmse,
    }


RUNNERS = {
    "uni_exp_small": bench_uni_exp,
    "uni_exp_large": bench_uni_exp,
    "mv_exp_5d": bench_mv_exp,
    # uni_power_law: tick only ships sum-of-exponentials HawkesSumExpKern, so
    # the power-law scenario is out of scope for a head-to-head.
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
            print(f"[skip] {name}: dataset missing")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {name}: {exc}")
            continue
        out = RESULTS_DIR / f"tick_{name}.json"
        out.write_text(json.dumps(record, indent=2, default=float))
        print(
            f"{name}: fit_time={record['fit_time_s']:.3f}s "
            f"rmse={record['rmse']:.4f}  -> {out}"
        )


if __name__ == "__main__":
    main()
