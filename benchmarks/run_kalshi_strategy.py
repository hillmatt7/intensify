"""Hawkes Imbalance Momentum (HIM): a basic TOB strategy backtest.

Strategy spec.
  At each test-window L2 quote event t_i, compute the bivariate exp
  Hawkes intensity ratio
        p̂_bid(t_i⁻) = λ̂_bid(t_i⁻) / (λ̂_bid(t_i⁻) + λ̂_ask(t_i⁻))
  using parameters fit on the train window. Generate a signal:
        signal_i = +1   if p̂_bid > 0.5 + θ        (bet on bid)
                 = −1   if p̂_bid < 0.5 − θ        (bet on ask)
                 =  0   otherwise                  (stand aside)
  Define realised direction:
        actual_i = +1   if event i is bid
                 = −1   if event i is ask
  Per-event P&L (no transaction cost — we are scoring a *signal*, not
  net of latency or fees):
        pnl_i = signal_i * actual_i ∈ {−1, 0, +1}

Latency accounting. The fit happens BEFORE the strategy goes live;
its wall-clock time is the "warmup" the strategy pays. Any event
that arrives while we are still fitting is missed. We explicitly
report: fit-time, equivalent missed-event count at this stream's
event rate, and the resulting P&L when those events are dropped.

Two libraries are compared with identical pre-/post-processing,
identical signal threshold, identical evaluation loop. The only
differences are:
  1. The optimizer (intensify uses scipy L-BFGS-B + closed-form Rust
     gradient with branching-ratio projection; tick uses AGD on a C++
     loss with no stationarity projection).
  2. Wall-clock fit time.

Usage:
  python benchmarks/run_kalshi_strategy.py --lib intensify
  python benchmarks/run_kalshi_strategy.py --lib tick
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/home/etrigan/Downloads/KXBTC15M_L2.csv")
TICKER = "KXBTC15M-26MAR190030-30"
DECAY = 100.0
TRAIN_FRAC = 0.5    # 1s train, 1s evaluate over a 2s window
WINDOW_S = 2.0
THRESHOLD = 0.05  # signal generated only when |p_bid - 0.5| > θ


def _load_window(window_s: float):
    bids: list[float] = []
    asks: list[float] = []
    t0 = None
    for chunk in pd.read_csv(DATA, chunksize=500_000, usecols=["ts", "ticker", "side"]):
        chunk = chunk[chunk["ticker"] == TICKER]
        if chunk.empty:
            continue
        ts = pd.to_datetime(chunk["ts"]).astype("int64") / 1e9
        if t0 is None:
            t0 = float(ts.iloc[0])
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


def _split(bids, asks, T, frac):
    all_t = np.sort(np.concatenate([bids, asks]))
    t_split = float(all_t[int(frac * len(all_t))])
    return (
        bids[bids < t_split], asks[asks < t_split],
        bids[bids >= t_split], asks[asks >= t_split],
        t_split,
    )


def _build_test_stream(bids_te, asks_te):
    times = np.concatenate([bids_te, asks_te])
    sides = np.concatenate([np.zeros(len(bids_te), dtype=np.int64),
                             np.ones(len(asks_te), dtype=np.int64)])
    order = np.argsort(times, kind="stable")
    return times[order], sides[order]


def _evaluate_intensities(times, sides, hist_bids, hist_asks, mu, alpha, beta):
    n = len(times)
    p_bid = np.zeros(n, dtype=np.float64)
    R = np.zeros(2, dtype=np.float64)
    t0 = times[0] if n > 0 else 0.0
    for t_j in hist_bids:
        R[0] += np.exp(-beta * max(t0 - t_j, 0.0))
    for t_j in hist_asks:
        R[1] += np.exp(-beta * max(t0 - t_j, 0.0))
    last_t = t0
    for i in range(n):
        t = times[i]
        R *= np.exp(-beta * (t - last_t))
        lam_bid = mu[0] + alpha[0, 0] * R[0] + alpha[0, 1] * R[1]
        lam_ask = mu[1] + alpha[1, 0] * R[0] + alpha[1, 1] * R[1]
        denom = lam_bid + lam_ask
        p_bid[i] = lam_bid / denom if denom > 0 else 0.5
        R[sides[i]] += 1.0
        last_t = t
    return p_bid


def _intensify_fit(bids_train, asks_train, T_train):
    import intensify as its
    t0 = time.perf_counter()
    m = its.MultivariateHawkes(
        n_dims=2, mu=[10.0, 10.0],
        kernel=its.ExponentialKernel(alpha=0.3, beta=DECAY),
    )
    r = m.fit([bids_train, asks_train], T=T_train, fit_decay=False)
    fit_time_ms = (time.perf_counter() - t0) * 1000.0
    mu_hat = np.asarray(r.process.mu, dtype=np.float64).reshape(-1)
    alpha_hat = np.zeros((2, 2), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            alpha_hat[i, j] = float(r.process.kernel_matrix[i][j].alpha)
    return mu_hat, alpha_hat, DECAY, fit_time_ms, float(r.branching_ratio_)


def _patch_tick():
    import tick.base as _tb, tick.hawkes, tick.base_model  # noqa
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
                    try: setattr(sub, name, make_patched(vars(sub)[name]))
                    except TypeError: pass
            walk(sub)
    for name in ("__setattr__", "_set"):
        if name in vars(_tb.Base):
            setattr(_tb.Base, name, make_patched(vars(_tb.Base)[name]))
    walk(_tb.Base)


def _tick_fit(bids_train, asks_train):
    _patch_tick()
    from tick.hawkes import HawkesExpKern
    t0 = time.perf_counter()
    m = HawkesExpKern(decays=DECAY, gofit="likelihood")
    m.fit([bids_train, asks_train])
    fit_time_ms = (time.perf_counter() - t0) * 1000.0
    mu_hat = np.asarray(m.baseline, dtype=np.float64)
    alpha_hat = np.asarray(m.adjacency, dtype=np.float64)
    branching = float(np.linalg.eigvals(alpha_hat).max().real)
    return mu_hat, alpha_hat, DECAY, fit_time_ms, branching


def _backtest_with_latency(times_te, sides_te, p_bid, fit_time_ms, threshold):
    """Apply the warmup-latency penalty, then score the strategy.

    Events that arrive in the first `fit_time_ms` of the test window are
    missed (the model isn't live yet). We score only events arriving
    after the warmup completes.
    """
    fit_time_s = fit_time_ms / 1000.0
    # times_te[0] is t at which the test window begins (= t_split).
    # We measure latency relative to t_split.
    t_warmup_done = times_te[0] + fit_time_s
    live_mask = times_te >= t_warmup_done
    n_total = len(times_te)
    n_live = int(live_mask.sum())
    n_missed = n_total - n_live

    p_bid_live = p_bid[live_mask]
    sides_live = sides_te[live_mask]
    actual = np.where(sides_live == 0, +1, -1)  # bid → +1, ask → −1

    signal = np.zeros_like(actual)
    signal[p_bid_live > 0.5 + threshold] = +1
    signal[p_bid_live < 0.5 - threshold] = -1

    pnl = signal * actual  # ∈ {−1, 0, +1}
    n_signals = int((signal != 0).sum())
    n_correct = int((pnl > 0).sum())
    n_wrong = int((pnl < 0).sum())
    hit_rate = n_correct / n_signals if n_signals else float("nan")
    total_pnl = int(pnl.sum())
    pnl_per_event = pnl.mean()
    sharpe = (pnl.mean() / pnl.std()) if pnl.std() > 0 else float("nan")

    # Latency-adjusted P&L: P&L AS-IF the live mask doesn't apply (perfect
    # warmup, for context), and the actual realised P&L (with warmup loss).
    actual_full = np.where(sides_te == 0, +1, -1)
    signal_full = np.zeros_like(actual_full)
    signal_full[p_bid > 0.5 + threshold] = +1
    signal_full[p_bid < 0.5 - threshold] = -1
    pnl_full = (signal_full * actual_full).sum()
    pnl_lost_to_latency = int(pnl_full) - total_pnl

    return {
        "n_test_events_total": n_total,
        "n_test_events_missed_during_warmup": n_missed,
        "n_test_events_scored": n_live,
        "warmup_fraction_of_test": n_missed / n_total if n_total else float("nan"),
        "n_signals": n_signals,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "hit_rate": hit_rate,
        "total_pnl": total_pnl,
        "pnl_per_event": float(pnl_per_event),
        "sharpe_per_event": float(sharpe) if not np.isnan(sharpe) else None,
        "pnl_if_zero_warmup": int(pnl_full),
        "pnl_lost_to_warmup_latency": pnl_lost_to_latency,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", choices=["intensify", "tick"], required=True)
    args = parser.parse_args()

    print(f"Loading {WINDOW_S}s window of {TICKER}…")
    bids, asks, T = _load_window(WINDOW_S)
    print(f"  total: {len(bids)} bid + {len(asks)} ask events (T={T:.2f}s)")

    bids_tr, asks_tr, bids_te, asks_te, t_split = _split(bids, asks, T, TRAIN_FRAC)
    print(f"  train: {len(bids_tr)} bid + {len(asks_tr)} ask  (t<{t_split:.4f}s)")
    print(f"  test : {len(bids_te)} bid + {len(asks_te)} ask  (t≥{t_split:.4f}s)")

    times_te, sides_te = _build_test_stream(bids_te, asks_te)
    test_duration_s = float(times_te[-1] - times_te[0])
    test_event_rate = len(times_te) / test_duration_s
    print(f"  test stream: {len(times_te)} events, duration={test_duration_s*1000:.1f} ms, rate={test_event_rate:.0f} events/s")

    print(f"\n[fit] {args.lib}…")
    if args.lib == "intensify":
        mu, alpha, beta, fit_time_ms, branching = _intensify_fit(bids_tr, asks_tr, t_split)
    else:
        mu, alpha, beta, fit_time_ms, branching = _tick_fit(bids_tr, asks_tr)
    print(f"  fit time: {fit_time_ms:.1f} ms")
    print(f"  fitted spectral radius: {branching:.4f}  ({'OK' if branching < 1 else 'NON-STATIONARY'})")
    print(f"  μ̂ = {mu}")
    print(f"  α̂ = {alpha.flatten()}")

    print("\n[predict] computing per-event p_bid on test stream…")
    p_bid = _evaluate_intensities(
        times_te - t_split, sides_te,
        bids_tr - t_split, asks_tr - t_split,
        mu, alpha, beta,
    )

    print(f"\n[strategy] HIM with threshold θ={THRESHOLD}")
    res = _backtest_with_latency(times_te - t_split, sides_te, p_bid, fit_time_ms, THRESHOLD)
    test_duration_ms = test_duration_s * 1000.0
    res["test_duration_ms"] = test_duration_ms
    res["fit_overshoot_ms"] = max(0.0, fit_time_ms - test_duration_ms)
    res["realtime_refit_viable"] = fit_time_ms < test_duration_ms

    # Per-event evaluation cost (the hot path for online deployment).
    eval_t0 = time.perf_counter()
    _ = _evaluate_intensities(
        times_te - t_split, sides_te,
        bids_tr - t_split, asks_tr - t_split,
        mu, alpha, beta,
    )
    eval_time_ms = (time.perf_counter() - eval_t0) * 1000.0
    res["eval_time_ms"] = eval_time_ms
    res["eval_per_event_us"] = eval_time_ms * 1000.0 / max(len(times_te), 1)
    res["fit_time_ms"] = fit_time_ms
    res["fitted_spectral_radius"] = branching
    res["fitted_mu"] = mu.tolist()
    res["fitted_alpha"] = alpha.tolist()
    res["test_event_rate_per_s"] = test_event_rate
    res["test_duration_s"] = test_duration_s

    # =========================================================================
    # Scenario A: Pre-trained / offline fit. Fit before market open, deploy.
    # Warmup is not on the live clock; per-event evaluation is the only
    # online cost. This is how you'd actually run this strategy.
    # =========================================================================
    print("\n  --- Scenario A: pre-trained (fit offline, deploy) ---")
    print(f"    P&L over entire test:     {res['pnl_if_zero_warmup']:+,d}")
    actual_full = np.where(sides_te == 0, +1, -1)
    signal_full = np.zeros_like(actual_full)
    signal_full[p_bid > 0.5 + THRESHOLD] = +1
    signal_full[p_bid < 0.5 - THRESHOLD] = -1
    n_sig_full = int((signal_full != 0).sum())
    n_corr_full = int(((signal_full * actual_full) > 0).sum())
    print(f"    signals fired:            {n_sig_full:,} / {len(sides_te):,} events ({n_sig_full/max(len(sides_te),1)*100:.1f}%)")
    print(f"    hit rate:                 {n_corr_full/max(n_sig_full,1)*100:.2f}%")
    print(f"    P&L per signal:           {res['pnl_if_zero_warmup']/max(n_sig_full,1):+.4f}")
    print(f"    online eval cost:         {res['eval_time_ms']:.1f} ms total / {res['eval_per_event_us']:.2f} µs per event")
    res["scenario_a_offline_fit"] = {
        "pnl": res["pnl_if_zero_warmup"],
        "n_signals": n_sig_full,
        "n_correct": n_corr_full,
        "hit_rate": n_corr_full / max(n_sig_full, 1),
        "pnl_per_signal": res["pnl_if_zero_warmup"] / max(n_sig_full, 1),
    }

    # =========================================================================
    # Scenario B: Realtime refit each cycle. Fit must complete inside the
    # data window, otherwise the strategy missed every event.
    # =========================================================================
    print("\n  --- Scenario B: realtime refit each cycle ---")
    print(f"    fit time:                 {fit_time_ms:.1f} ms")
    print(f"    test window duration:     {test_duration_ms:.1f} ms")
    if res["realtime_refit_viable"]:
        print(f"    realtime refit VIABLE: model goes live with {test_duration_ms - fit_time_ms:.1f} ms remaining")
    else:
        print(f"    realtime refit NOT VIABLE: fit overshoots window by {res['fit_overshoot_ms']:.1f} ms")
    print(f"    warmup latency cost:  {fit_time_ms:.1f} ms = {res['n_test_events_missed_during_warmup']} missed events ({res['warmup_fraction_of_test']*100:.1f}% of test window)")
    print(f"  signals fired:        {res['n_signals']:,} / {res['n_test_events_scored']:,} live events ({res['n_signals']/max(res['n_test_events_scored'],1)*100:.1f}%)")
    print(f"  hit rate:             {res['hit_rate']*100:.2f}%  ({res['n_correct']:,} correct, {res['n_wrong']:,} wrong)")
    print(f"  total P&L (live):     {res['total_pnl']:+,d}")
    print(f"  P&L per event:        {res['pnl_per_event']:+.6f}")
    sharpe = res["sharpe_per_event"]
    print(f"  Sharpe per event:     {sharpe:+.4f}" if sharpe is not None else "  Sharpe per event:     n/a (no live events)")
    print(f"  P&L if zero warmup:   {res['pnl_if_zero_warmup']:+,d}  (lost to latency: {res['pnl_lost_to_warmup_latency']:+,d})")

    out = Path(__file__).parent / "results" / f"kalshi_strategy_{args.lib}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
