"""Top-of-book Hawkes comparison: intensify vs tick on real Kalshi L2 data.

Typical TOB point-process strategies model bid/ask quote arrivals as a
mutually-exciting Hawkes process (Bacry et al. 2013, "Modelling
microstructure noise with mutually exciting point processes"). The
bivariate process (bid → bid/ask, ask → bid/ask) captures cross-side
clustering and self-excitation.

This script fits both univariate (bid-only) and bivariate (bid + ask)
Hawkes on increasing windows of real Kalshi KXBTC15M L2 book data.

Data: /home/etrigan/Downloads/KXBTC15M_L2.csv (28.3M events, 2 tickers)
Schema: ts,ticker,side,price,qty

Usage:
    python benchmarks/run_kalshi_tob.py --lib intensify
    python benchmarks/run_kalshi_tob.py --lib tick   # needs separate env
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/home/etrigan/Downloads/KXBTC15M_L2.csv")
TICKER = "KXBTC15M-26MAR190045-45"  # most active ticker
DECAY = 100.0  # 1/β = 10 ms — typical sub-second self-excitation timescale
N_REPEATS = 3

# Window sizes (in seconds) to fit on. Each window slices a contiguous
# sub-period of the L2 stream starting from the first event for TICKER.
WINDOWS_S = [0.5, 2.0, 10.0]


def _load_window(window_s: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Load (bid_times, ask_times, T) for the active ticker over `window_s` seconds."""
    bids: list[float] = []
    asks: list[float] = []
    t0 = None
    target_end = None

    for chunk in pd.read_csv(DATA, chunksize=500_000, usecols=["ts", "ticker", "side"]):
        chunk = chunk[chunk["ticker"] == TICKER]
        if chunk.empty:
            continue
        ts = pd.to_datetime(chunk["ts"]).astype("int64") / 1e9  # seconds
        if t0 is None:
            t0 = float(ts.iloc[0])
            target_end = t0 + window_s
        rel = ts - t0
        keep = rel < window_s
        if not keep.any():
            break
        sub = chunk[keep]
        sub_rel = rel[keep].to_numpy()
        is_bid = (sub["side"] == "bid").to_numpy()
        bids.extend(sub_rel[is_bid].tolist())
        asks.extend(sub_rel[~is_bid].tolist())
        if (rel >= window_s).any():
            break

    return (
        np.sort(np.asarray(bids, dtype=np.float64)),
        np.sort(np.asarray(asks, dtype=np.float64)),
        float(window_s),
    )


def _time_fit(fn, repeats: int = N_REPEATS) -> tuple[float, list[float]]:
    fn()  # warmup
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times), times


def _run_intensify(slices: dict) -> dict:
    import intensify as its

    out = {}
    for window_s, (bids, asks, T) in slices.items():
        n_b, n_a = len(bids), len(asks)
        print(f"\n=== window={window_s}s — {n_b} bid + {n_a} ask events (T={T:.2f}s) ===")
        result = {"n_bid": n_b, "n_ask": n_a, "T": T}

        # Univariate Hawkes on bids only
        print(f"[intensify] uni Hawkes(exp) on {n_b} bid events…")
        def fit_uni():
            m = its.UnivariateHawkes(
                mu=10.0,
                kernel=its.ExponentialKernel(alpha=0.5, beta=DECAY),
            )
            m.fit(bids, T=T, fit_decay=False)
        try:
            ms, runs = _time_fit(fit_uni)
            print(f"  median {ms:.2f} ms  (runs: {[f'{r:.2f}' for r in runs]})")
            result["uni_intensify_ms"] = ms
            result["uni_intensify_runs"] = runs
        except Exception as e:
            print(f"  FAIL: {e}")
            result["uni_intensify_status"] = f"FAIL: {type(e).__name__}: {e}"

        # Bivariate Hawkes (bid + ask)
        print(f"[intensify] mv Hawkes(exp 2x2) on {n_b}+{n_a} bid/ask events…")
        def fit_mv():
            m = its.MultivariateHawkes(
                n_dims=2,
                mu=[10.0, 10.0],
                kernel=its.ExponentialKernel(alpha=0.3, beta=DECAY),
            )
            m.fit([bids, asks], T=T, fit_decay=False)
        try:
            ms, runs = _time_fit(fit_mv)
            # Recover branching ratio for sanity
            m = its.MultivariateHawkes(
                n_dims=2, mu=[10.0, 10.0],
                kernel=its.ExponentialKernel(alpha=0.3, beta=DECAY),
            )
            r = m.fit([bids, asks], T=T, fit_decay=False)
            print(f"  median {ms:.2f} ms  (runs: {[f'{r:.2f}' for r in runs]})")
            print(f"  branching ratio: {r.branching_ratio_:.3f}")
            result["mv_intensify_ms"] = ms
            result["mv_intensify_runs"] = runs
            result["mv_intensify_branching"] = float(r.branching_ratio_)
        except Exception as e:
            print(f"  FAIL: {e}")
            result["mv_intensify_status"] = f"FAIL: {type(e).__name__}: {e}"

        out[str(window_s)] = result
    return out


def _patch_tick():
    """Work around tick 0.8.0.1's broken `__setattr__`/`_set` registry."""
    import tick.base as _tb
    import tick.hawkes  # noqa: F401
    import tick.base_model  # noqa: F401

    def make_patched(orig):
        def patched(self, key, value):
            try:
                orig(self, key, value)
            except AttributeError:
                object.__setattr__(self, key, value)
        return patched

    seen = set()

    def walk(cls):
        for sub in cls.__subclasses__():
            if sub in seen or sub is object:
                continue
            seen.add(sub)
            for name in ("__setattr__", "_set"):
                if name in vars(sub):
                    try:
                        setattr(sub, name, make_patched(vars(sub)[name]))
                    except TypeError:
                        pass
            walk(sub)

    for name in ("__setattr__", "_set"):
        if name in vars(_tb.Base):
            setattr(_tb.Base, name, make_patched(vars(_tb.Base)[name]))
    walk(_tb.Base)


def _run_tick(slices: dict) -> dict:
    _patch_tick()
    from tick.hawkes import HawkesExpKern

    out = {}
    for window_s, (bids, asks, T) in slices.items():
        n_b, n_a = len(bids), len(asks)
        print(f"\n=== window={window_s}s — {n_b} bid + {n_a} ask events ===")
        result = {"n_bid": n_b, "n_ask": n_a, "T": T}

        # Univariate
        print(f"[tick] uni Hawkes(exp,β={DECAY}) on {n_b} bid events…")
        def fit_uni():
            m = HawkesExpKern(decays=DECAY, gofit="likelihood")
            m.fit([bids])
        try:
            ms, runs = _time_fit(fit_uni)
            print(f"  median {ms:.2f} ms  (runs: {[f'{r:.2f}' for r in runs]})")
            result["uni_tick_lik_ms"] = ms
        except Exception as e:
            print(f"  FAIL (likelihood): {str(e)[:80]}")
            result["uni_tick_lik_status"] = f"FAIL: {type(e).__name__}"

        # Bivariate likelihood
        print(f"[tick] mv Hawkes(exp 2x2,β={DECAY}) likelihood on bid+ask…")
        def fit_mv_lik():
            m = HawkesExpKern(decays=DECAY, gofit="likelihood")
            m.fit([bids, asks])
        try:
            ms, runs = _time_fit(fit_mv_lik)
            adj = HawkesExpKern(decays=DECAY, gofit="likelihood")
            adj.fit([bids, asks])
            print(f"  median {ms:.2f} ms  (runs: {[f'{r:.2f}' for r in runs]})")
            print(f"  spectral radius: {np.linalg.eigvals(adj.adjacency).max().real:.3f}")
            result["mv_tick_lik_ms"] = ms
            result["mv_tick_lik_branching"] = float(np.linalg.eigvals(adj.adjacency).max().real)
        except Exception as e:
            print(f"  FAIL (likelihood): {str(e)[:80]}")
            result["mv_tick_lik_status"] = f"FAIL: {type(e).__name__}: {str(e)[:60]}"

        # Bivariate least-squares (different objective; tick's actual working MV path)
        print(f"[tick] mv Hawkes(exp 2x2,β={DECAY}) least-squares on bid+ask…")
        def fit_mv_ls():
            m = HawkesExpKern(decays=DECAY, gofit="least-squares")
            m.fit([bids, asks])
        try:
            ms, runs = _time_fit(fit_mv_ls)
            adj = HawkesExpKern(decays=DECAY, gofit="least-squares")
            adj.fit([bids, asks])
            print(f"  median {ms:.2f} ms  (runs: {[f'{r:.2f}' for r in runs]})")
            print(f"  spectral radius: {np.linalg.eigvals(adj.adjacency).max().real:.3f}")
            result["mv_tick_ls_ms"] = ms
            result["mv_tick_ls_branching"] = float(np.linalg.eigvals(adj.adjacency).max().real)
        except Exception as e:
            print(f"  FAIL (LS): {str(e)[:80]}")
            result["mv_tick_ls_status"] = f"FAIL: {type(e).__name__}"

        out[str(window_s)] = result
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", choices=["intensify", "tick"], required=True)
    args = parser.parse_args()

    print(f"Loading Kalshi KXBTC15M L2 windows for ticker {TICKER}…")
    slices = {}
    for w in WINDOWS_S:
        bids, asks, T = _load_window(w)
        print(f"  window={w}s: {len(bids)} bid + {len(asks)} ask events (T={T:.2f}s)")
        slices[w] = (bids, asks, T)

    if args.lib == "intensify":
        out = _run_intensify(slices)
    else:
        out = _run_tick(slices)

    out_path = Path(__file__).parent / "results" / f"kalshi_tob_{args.lib}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n→ Results written to {out_path}")


if __name__ == "__main__":
    main()
