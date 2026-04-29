"""Out-of-sample midprice prediction accuracy: intensify vs tick.

Setup. Given bid+ask quote arrivals on real Kalshi L2 data, fit a
bivariate Hawkes process on a training window and use the fitted
parameters to predict, at each event time in a held-out test window,
whether the next event will be a bid or an ask.

Why this measures midprice accuracy. The bivariate Hawkes intensity
ratio  ρ̂(t) = λ_bid(t) / (λ_bid(t) + λ_ask(t))  is the model's
forward-looking estimate of book imbalance. In microstructure
literature (Stoikov 2018 "microprice", Bacry et al. 2013) this ratio
is the canonical signal for short-horizon mid-price drift: ρ̂ > 0.5
implies upward pressure on mid (bids arriving faster than asks),
ρ̂ < 0.5 implies downward pressure. So the per-event classification
accuracy of a bivariate Hawkes is a direct readout of how well that
fitted model captures mid-drift direction.

Baselines:
  - majority-class predictor (constant "bid" or "ask"): floor.
  - empirical-rate predictor (use unconditional bid:ask ratio
    observed in the training set): tighter floor; doesn't use
    temporal structure.

Both libraries are fit on identical data, then we evaluate prediction
accuracy with the same scoring function. The test asks: does Hawkes
add directional skill beyond the marginal rate?

Usage:
  python benchmarks/run_kalshi_midprice_accuracy.py --lib intensify
  python benchmarks/run_kalshi_midprice_accuracy.py --lib tick
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/home/etrigan/Downloads/KXBTC15M_L2.csv")
TICKER = "KXBTC15M-26MAR190030-30"  # active early in file
DECAY = 100.0  # 1/β = 10 ms timescale
TRAIN_FRAC = 0.7
WINDOW_S = 0.2  # tight window — keeps train/test distributions similar
              # (longer windows on this stream contain regime shifts that
              # neither baseline nor Hawkes can predict)


def _load_window(window_s: float) -> tuple[np.ndarray, np.ndarray, float]:
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


def _split_train_test(bids: np.ndarray, asks: np.ndarray, T: float, train_frac: float):
    """Split at the event-count quantile so train/test both contain both sides
    even when one side is bursty/front-loaded in time."""
    all_times = np.sort(np.concatenate([bids, asks]))
    t_split = float(all_times[int(train_frac * len(all_times))])
    return (
        bids[bids < t_split],
        asks[asks < t_split],
        bids[bids >= t_split],
        asks[asks >= t_split],
        t_split,
    )


def _build_test_stream(bids_test: np.ndarray, asks_test: np.ndarray):
    """Combine test bid/ask events into a single time-sorted stream with side labels."""
    times = np.concatenate([bids_test, asks_test])
    sides = np.concatenate([np.zeros(len(bids_test), dtype=np.int64),
                             np.ones(len(asks_test), dtype=np.int64)])  # 0=bid, 1=ask
    order = np.argsort(times, kind="stable")
    return times[order], sides[order]


def _evaluate_intensities(
    times: np.ndarray, sides: np.ndarray,
    history_bids: np.ndarray, history_asks: np.ndarray,
    mu: np.ndarray, alpha: np.ndarray, beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict next-side at each test event using the bivariate exp Hawkes intensity.

    At each test event t_i we compute λ_bid(t_i⁻) and λ_ask(t_i⁻) from
    the event history (training events plus prior test events) using
    the recursive exp form:
        λ_m(t) = μ_m + Σ_k Σ_{t_j^k < t} α_{m,k} * exp(-β (t - t_j^k))
    Returns (preds, p_bid) where p_bid[i] = λ_bid/(λ_bid+λ_ask) at t_i⁻.
    """
    # State carried as exponentially-weighted "influence" per pair.
    # R[m,k] holds Σ_{t_j^k < t_i} exp(-β (t_i - t_j^k)) weighted, and
    # is updated O(1) per event. Track per-source histories.
    n = len(times)
    preds = np.zeros(n, dtype=np.int64)
    p_bid = np.zeros(n, dtype=np.float64)

    # Initialize R from training history at t_split (= times[0]).
    t0 = times[0] if n > 0 else 0.0

    def init_R():
        R = np.zeros(2, dtype=np.float64)  # R[k] = Σ_{train events of side k} exp(-β (t0 - t_j))
        for t_j in history_bids:
            R[0] += np.exp(-beta * max(t0 - t_j, 0.0))
        for t_j in history_asks:
            R[1] += np.exp(-beta * max(t0 - t_j, 0.0))
        return R
    R = init_R()

    last_t = t0
    for i in range(n):
        t = times[i]
        # Decay R from last_t to t
        decay = np.exp(-beta * (t - last_t))
        R *= decay
        # Intensities just before t
        lam_bid = mu[0] + alpha[0, 0] * R[0] + alpha[0, 1] * R[1]
        lam_ask = mu[1] + alpha[1, 0] * R[0] + alpha[1, 1] * R[1]
        denom = lam_bid + lam_ask
        p_bid[i] = lam_bid / denom if denom > 0 else 0.5
        preds[i] = 0 if lam_bid >= lam_ask else 1
        # Update R with this event (which contributes to itself for next event)
        R[sides[i]] += 1.0
        last_t = t
    return preds, p_bid


def _summarize(name: str, preds: np.ndarray, sides: np.ndarray) -> dict:
    n = len(sides)
    n_bid = int((sides == 0).sum())
    n_ask = int((sides == 1).sum())
    correct = int((preds == sides).sum())
    acc = correct / n if n else float("nan")
    # Per-side
    bid_mask = sides == 0
    ask_mask = sides == 1
    bid_recall = float((preds[bid_mask] == 0).mean()) if bid_mask.any() else float("nan")
    ask_recall = float((preds[ask_mask] == 1).mean()) if ask_mask.any() else float("nan")
    pred_bid = preds == 0
    pred_ask = preds == 1
    bid_prec = float((sides[pred_bid] == 0).mean()) if pred_bid.any() else float("nan")
    ask_prec = float((sides[pred_ask] == 1).mean()) if pred_ask.any() else float("nan")
    print(f"  {name}:")
    print(f"    accuracy: {acc:.4f}  ({correct}/{n})")
    print(f"    bid recall: {bid_recall:.4f}   ask recall: {ask_recall:.4f}")
    print(f"    bid precision: {bid_prec:.4f}   ask precision: {ask_prec:.4f}")
    return {
        "accuracy": acc,
        "n": n,
        "n_bid": n_bid,
        "n_ask": n_ask,
        "correct": correct,
        "bid_recall": bid_recall,
        "ask_recall": ask_recall,
        "bid_precision": bid_prec,
        "ask_precision": ask_prec,
    }


def _intensify_fit(bids_train, asks_train, T_train):
    import intensify as its
    m = its.MultivariateHawkes(
        n_dims=2,
        mu=[10.0, 10.0],
        kernel=its.ExponentialKernel(alpha=0.3, beta=DECAY),
    )
    r = m.fit([bids_train, asks_train], T=T_train, fit_decay=False)
    mu_hat = np.asarray(r.process.mu, dtype=np.float64).reshape(-1)
    alpha_hat = np.zeros((2, 2), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            alpha_hat[i, j] = float(r.process.kernel_matrix[i][j].alpha)
    print(f"  intensify fit: μ̂={mu_hat}  α̂={alpha_hat.flatten()}  branching={r.branching_ratio_:.3f}")
    return mu_hat, alpha_hat, DECAY


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
    m = HawkesExpKern(decays=DECAY, gofit="likelihood")
    m.fit([bids_train, asks_train])
    mu_hat = np.asarray(m.baseline, dtype=np.float64)
    alpha_hat = np.asarray(m.adjacency, dtype=np.float64)
    branching = float(np.linalg.eigvals(alpha_hat).max().real)
    print(f"  tick fit: μ̂={mu_hat}  α̂={alpha_hat.flatten()}  spectral={branching:.3f}")
    return mu_hat, alpha_hat, DECAY


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", choices=["intensify", "tick"], required=True)
    args = parser.parse_args()

    print(f"Loading {WINDOW_S}s window of {TICKER}…")
    bids, asks, T = _load_window(WINDOW_S)
    print(f"  total: {len(bids)} bid + {len(asks)} ask events (T={T:.2f}s)")

    bids_tr, asks_tr, bids_te, asks_te, t_split = _split_train_test(bids, asks, T, TRAIN_FRAC)
    print(f"  train (t<{t_split:.3f}s): {len(bids_tr)} bid + {len(asks_tr)} ask")
    print(f"  test  (t≥{t_split:.3f}s): {len(bids_te)} bid + {len(asks_te)} ask")

    times_te, sides_te = _build_test_stream(bids_te, asks_te)
    print(f"  test stream: {len(times_te)} events  ({(sides_te==0).sum()} bid, {(sides_te==1).sum()} ask)")

    print(f"\n[fit] {args.lib}…")
    if args.lib == "intensify":
        mu, alpha, beta = _intensify_fit(bids_tr, asks_tr, t_split)
    else:
        mu, alpha, beta = _tick_fit(bids_tr, asks_tr)

    print(f"\n[predict] evaluating on {len(times_te)} test events…")
    preds, p_bid = _evaluate_intensities(
        times_te - t_split, sides_te,
        bids_tr - t_split, asks_tr - t_split,
        mu, alpha, beta,
    )

    # Log-loss: probabilistic accuracy. A Hawkes that doesn't add
    # info beyond the marginal rate will match the rate-based baseline.
    eps = 1e-12
    p_bid_clipped = np.clip(p_bid, eps, 1.0 - eps)
    p_actual_bid = (sides_te == 0).astype(np.float64)
    hawkes_log_loss = float(
        -(p_actual_bid * np.log(p_bid_clipped)
          + (1 - p_actual_bid) * np.log(1 - p_bid_clipped)).mean()
    )
    n_bid_tr2 = len(bids_tr)
    n_ask_tr2 = len(asks_tr)
    p_bid_train = n_bid_tr2 / (n_bid_tr2 + n_ask_tr2)
    p_bid_train = np.clip(p_bid_train, eps, 1.0 - eps)
    rate_log_loss = float(
        -(p_actual_bid * np.log(p_bid_train)
          + (1 - p_actual_bid) * np.log(1 - p_bid_train)).mean()
    )
    print(f"\n  log-loss (Hawkes):           {hawkes_log_loss:.6f}")
    print(f"  log-loss (train marginal):   {rate_log_loss:.6f}")
    print(f"  Δ (lower = Hawkes better):   {hawkes_log_loss - rate_log_loss:+.6f}")

    print("\n[results]")
    res_hawkes = _summarize(f"{args.lib} bivariate Hawkes", preds, sides_te)

    # Baselines
    n_bid_tr = len(bids_tr)
    n_ask_tr = len(asks_tr)
    p_bid_train = n_bid_tr / (n_bid_tr + n_ask_tr)
    # (a) train-majority — fair: doesn't peek at test
    train_majority = 0 if p_bid_train >= 0.5 else 1
    res_train_majority = _summarize(
        f"baseline (train-majority = {'bid' if train_majority==0 else 'ask'})",
        np.full(len(sides_te), train_majority, dtype=np.int64), sides_te,
    )
    # (b) test-majority — unfair (peeks): the ceiling that any
    # constant-prediction strategy could achieve if it knew the test
    # distribution. Useful as a sanity check on whether Hawkes adds
    # anything beyond "predict whichever side is more common."
    n_bid_te = int((sides_te == 0).sum())
    n_ask_te = int((sides_te == 1).sum())
    test_majority = 0 if n_bid_te >= n_ask_te else 1
    res_test_majority = _summarize(
        f"oracle baseline (test-majority = {'bid' if test_majority==0 else 'ask'})",
        np.full(len(sides_te), test_majority, dtype=np.int64), sides_te,
    )

    out = {
        "library": args.lib,
        "window_s": WINDOW_S,
        "train_frac": TRAIN_FRAC,
        "n_train_bid": n_bid_tr,
        "n_train_ask": n_ask_tr,
        "n_test_bid": int((sides_te == 0).sum()),
        "n_test_ask": int((sides_te == 1).sum()),
        "fitted_mu": mu.tolist(),
        "fitted_alpha": alpha.tolist(),
        "fitted_beta": beta,
        "hawkes": res_hawkes,
        "train_majority_baseline": res_train_majority,
        "test_majority_oracle": res_test_majority,
        "lift_over_train_majority": res_hawkes["accuracy"] - res_train_majority["accuracy"],
        "lift_over_test_majority_oracle": res_hawkes["accuracy"] - res_test_majority["accuracy"],
        "hawkes_log_loss": hawkes_log_loss,
        "marginal_rate_log_loss": rate_log_loss,
        "log_loss_delta": hawkes_log_loss - rate_log_loss,
    }
    out_path = Path(__file__).parent / "results" / f"kalshi_midprice_accuracy_{args.lib}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n→ Results: {out_path}")
    print(f"\nLift vs train-majority (fair):     {out['lift_over_train_majority']*100:+.2f} pp")
    print(f"Lift vs test-majority (oracle):    {out['lift_over_test_majority_oracle']*100:+.2f} pp")


if __name__ == "__main__":
    main()
