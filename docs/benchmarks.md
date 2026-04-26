# Benchmarks

Head-to-head comparison with [tick][] on reproducible seeded datasets.
[pyhawkes][] was evaluated and dropped — upstream has been unmaintained
since 2018 and its transitive dependency `pybasicbayes` still uses
`scipy.misc.logsumexp` (removed in SciPy 1.0, 2017).

All numbers from a single machine, median of 3 runs, Python 3.12 for
`intensify`, Python 3.8 for `tick` (its last supported version). Reproduce:

```bash
pip install -e ".[benchmark]"
maturin develop --release
python benchmarks/reference_dataset.py
python benchmarks/run_intensify.py
# tick needs its own py3.8 env:
python benchmarks/run_tick.py         # run inside a tick-compatible env
```

[tick]: https://github.com/X-DataInitiative/tick
[pyhawkes]: https://github.com/slinderman/pyhawkes

## Headline results — `intensify 0.3.0` vs `tick 0.7.0.1`

intensify can be run two ways. The **joint** mode fits the full kernel
(`μ`, `α`, **and** `β`) — what most users want. The **decay-given** mode
locks `β` to its initial value (set via `model.fit(..., fit_decay=False)`)
which is the same problem tick's `HawkesExpKern` solves.

### Multivariate exponential, decay-given (apples-to-apples vs tick)

| N | tick (ms) | intensify 0.2.0 (ms) | **intensify 0.3.0** (ms) | vs tick |
|---:|---:|---:|---:|---:|
| 501 | 1.0 | 8 | **0.5** | **2.0× faster** |
| 2,249 | 2.0 | 21 | **0.8** | **2.5× faster** |
| 9,271 | 6.0 | 38 | **2.4** | **2.5× faster** |
| 27,519 | 15.0 | 189 | **6.9** | **2.2× faster** |
| 91,249 | 48.0 | 549 | **22.2** | **2.2× faster** |

Parameter-recovery RMSE is preserved across the port (matches the 0.2.0
numbers identically — the optimizer is unchanged; only the hot inner
loop went from XLA-on-CPU to native Rust).

### Multivariate exponential, joint-decay (β fit per cell)

tick **cannot fit β**, so this comparison is intensify-vs-itself.

| N | intensify 0.2.0 (ms) | **intensify 0.3.0** (ms) | speedup |
|---:|---:|---:|---:|
| 1,099 | 1100 | **14** | **~80×** |

The joint-decay path keeps a per-cell recursive state and accumulates
∂R/∂β analytically — closed-form gradient through the M² recursive
states, no autodiff in the hot path.

### Other kernels — tick has no MLE for these

| Scenario | intensify 0.2.0 | **intensify 0.3.0** |
|---|---:|---:|
| `uni_power_law` (N=451) | 56 ms | **35 ms** |
| `uni_nonparametric` (N=500) | killed (>7 min) | **<1 s** ⭐ |

The nonparametric speedup resolves [ISSUES.md][] #8 — the dense lag
expansion is gone in favor of a binary-search bin lookup over the
piecewise-constant kernel.

[ISSUES.md]: ../ISSUES.md

### Scenario summary table

| Scenario | Mode | **intensify 0.3.0** | tick `0.7.0.1` |
|---|---|---|---|
| `uni_exp_small` (516 events) | joint, time | 1.5 ms | n/a |
| `uni_exp_small` | joint, RMSE | 0.188 | n/a |
| `uni_exp_small` | **decay-given, time** | **1.5 ms** | **1.0 ms** |
| `uni_exp_small` | **decay-given, RMSE** | **0.042** | **0.029** |
| `uni_power_law` (451 events) | joint, time | 35 ms | not supported |
| `uni_power_law` | joint, RMSE | 0.094 | — |
| `mv_exp_5d` (1099 events, 5d) | joint, time | **14 ms** | n/a |
| `mv_exp_5d` | joint, RMSE | 0.075 | n/a |
| `mv_exp_5d` | **decay-given, time** | **1.4 ms** | **2.0 ms** |
| `mv_exp_5d` | **decay-given, RMSE** | **0.041** | **0.052** |

In **decay-given** multivariate mode, intensify is now both **faster
and more accurate** than tick on the same data. The univariate
sub-millisecond gap is the scipy.optimize.minimize Python wrapper —
each L-BFGS-B iteration pays a ~10 µs round-trip into Python that
tick's pure-C++ inner loop avoids. Closing that fully would mean
porting L-BFGS-B itself; out of scope until a user reports needing
sub-tick wall-clock on N<1000.

In **joint** mode intensify is doing strictly more work (fitting one
or more decay rates that tick can't fit at all), and most of the
residual RMSE is β-error. The full kernel fit is what most lab users
want; tick either requires a separate cross-validation loop over `β`
or acceptance of a possibly-wrong decay.

## How to read these numbers — the comparison is not apples-to-apples

- **`tick.hawkes.HawkesExpKern` requires the user to supply the decay
  rate `β` as a constructor argument.** Its MLE then fits only `μ` and
  the amplitude(s) `α`. That is a 2-parameter convex problem for
  univariate (or M²-parameter for multivariate), which C++ can close in
  microseconds.
- **`intensify` fits the whole kernel jointly** — baseline, amplitude,
  and decay — as a 3-parameter (univariate) or 2M²+M-parameter
  (multivariate) non-convex problem, via `scipy.optimize.L-BFGS-B`
  driving a hand-derived closed-form gradient computed in Rust.

## Where each library earns its place

**Prefer `tick` when** you have a well-studied process and already
know the decay rate (e.g. from literature or a separate estimator),
the kernel is pure exponential or sum-of-exponentials, and you want
sub-millisecond fits at N<1000 on Linux/macOS with Python 3.8.

**Prefer `intensify` when** any of the following apply:

1. You don't know the decay a priori and want to fit it jointly.
2. Your kernel is power-law (`PowerLawKernel`, `ApproxPowerLawKernel`)
   or piecewise-constant (`NonparametricKernel`) — tick has no MLE
   for these.
3. You need marked Hawkes or nonlinear (inhibitory, softplus, sigmoid)
   Hawkes — tick's fit API covers only linear exp/sum-exp kernels.
4. You need the time-rescaling theorem test on inter-compensator
   increments (the mathematically correct form). Several alternative
   implementations — including older intensify code — use cumulative
   compensators and produce wrong p-values.
5. You want projected-gradient stationarity enforcement with per-fit
   `FitResult.branching_ratio_` (spectral radius for multivariate).
6. You need a modern Python 3.10+ runtime with prebuilt wheels.
7. You need fits that scale: at N=91,249 intensify is 2.2× faster
   than tick on the multivariate decay-given problem.

## Scaling behavior

See [scaling.md](scaling.md) for the full curve across dataset sizes
from 500 to 91,000 events. TL;DR: both libraries scale linearly in N,
intensify is **2–2.5× faster than tick at every benchmarked size**,
and parameter-recovery RMSE is preserved versus the 0.2.0 baseline.

## Reproducibility

All datasets are seeded and written as portable `.npy` + `.json`
pairs under `benchmarks/data/`. Results JSON is under
`benchmarks/results/`.

## What changed in 0.3.0

The full backend (every kernel evaluator, every likelihood, every
gradient, both simulators, every compensator) was ported from JAX +
NumPy to Rust + PyO3. Highlights:

1. **Closed-form analytic gradients replace JAX autodiff** in the hot
   path. Hand-derived from Ozaki 1979 (univariate) and Bacry et al.
   2015 (multivariate), then cross-validated to 1e-10 against the
   frozen JAX oracle in `tests/_reference/` across many seeds.

2. **`MultivariateHawkes` joint-decay** uses a per-cell recursive
   state with closed-form ∂R/∂β. `mv_exp_5d` joint went from 1100 ms
   to 14 ms (~80×). tick can't do this case at all.

3. **Nonparametric kernel** uses a binary-search bin lookup
   (`partition_point`) instead of an O(N²) lag-matrix expansion.
   Resolves [ISSUES.md][] #8: N=500 went from killed-after-7-min to
   <1 s.

4. **Marked Hawkes covers all four mark-influence kinds** (linear,
   log, power, callable) via a precomputed g_values pattern that
   keeps the inner loop branch-free.

5. **NonlinearHawkes** (softplus / sigmoid / relu / identity links)
   uses a numerical compensator on a quadrature grid with closed-form
   chain-rule through the link.

6. **Simulators** (Ogata thinning + Galton–Watson branching, both
   univariate and multivariate) are Rust-backed.

7. **JAX is excised from the runtime.** Every user-facing inference
   path now hits Rust exclusively. JAX is retained only as a
   cross-validation oracle in `tests/_reference/` (dev-only, never
   imported at runtime). Bayesian inference still uses numpyro/JAX
   internally and is gated behind the optional `[bayesian]` extra.

## Known limitations

- The univariate small-N gap to tick (~0.5 ms vs tick's ~0.001 s) is
  scipy.optimize.minimize Python wrapper overhead. A Rust-native
  L-BFGS-B would close it but isn't shipped: tick is already
  sub-millisecond at this N and the benchmark suite isn't latency-
  bound there.
- `ApproxPowerLawKernel` Rust-side gradient propagates through the
  weight-normalization chain rule for `β_pow`; ∂w/∂β_min cancels
  exactly so β_min uses the same gradient code path. Verified at
  1e-10 against the JAX reference.
- GPU is out of scope (the recursive likelihood is sequential; CPU
  cache locality dominates and tick is CPU-only).
