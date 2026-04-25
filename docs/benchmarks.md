# Benchmarks

Head-to-head comparison with [tick][] on reproducible seeded datasets.
[pyhawkes][] was evaluated and dropped — upstream has been unmaintained
since 2018 and its transitive dependency `pybasicbayes` still uses
`scipy.misc.logsumexp` (removed in SciPy 1.0, 2017).

All numbers from a single machine, median of 3 runs, Python 3.14 for
`intensify`, Python 3.8 for `tick` (its last supported version). Reproduce:

```bash
pip install -e ".[benchmark]"
python benchmarks/reference_dataset.py
python benchmarks/run_intensify.py
# tick needs its own py3.8 env:
python benchmarks/run_tick.py         # run inside a tick-compatible env
```

[tick]: https://github.com/X-DataInitiative/tick
[pyhawkes]: https://github.com/slinderman/pyhawkes

## Results

intensify can be run two ways. The **joint** mode fits the full kernel
(`μ`, `α`, **and** `β`) — what most users want. The **decay-given** mode
locks `β` to its initial value (set via `model.fit(..., fit_decay=False)`)
which is the same problem tick's `HawkesExpKern` solves.

| Scenario | Mode | intensify `0.2.0` | tick `0.7.0.1` |
|---|---|---|---|
| `uni_exp_small` (516 events) | joint, time | 0.004 s | n/a |
| `uni_exp_small` | joint, RMSE | 0.188 | n/a |
| `uni_exp_small` | **decay-given, time** | **0.0026 s** | **0.001 s** |
| `uni_exp_small` | **decay-given, RMSE** | **0.042** | **0.029** |
| `uni_power_law` (451 events) | joint, time | 0.056 s | not supported |
| `uni_power_law` | joint, RMSE | 0.094 | — |
| `mv_exp_5d` (1099 events, 5 dims) | joint, time | 1.10 s | n/a |
| `mv_exp_5d` | joint, RMSE | 0.109 | n/a |
| `mv_exp_5d` | **decay-given, time** | **0.017 s** | **0.002 s** |
| `mv_exp_5d` | **decay-given, RMSE** | **0.041** | **0.052** |

In **decay-given** mode (the apples-to-apples problem tick solves):

- Univariate: intensify 2.6 ms vs tick 1 ms — within 2.6×.
- **Multivariate: intensify 17 ms vs tick 2 ms — within ~8×, and RMSE
  0.041 vs 0.052, *intensify is more accurate* on the same data.** The
  O(N·M) recursive likelihood (enabled when every β_{m,k} is the same
  scalar) gives us the speed, and scipy L-BFGS-B finds a meaningfully
  better optimum than tick's fast inner loop.

In **joint** mode intensify is doing strictly more work (fitting one or
more decay rates that tick can't fit at all), and most of the residual
RMSE is β-error. The full kernel fit is what most lab users want; tick
either requires a separate cross-validation loop over `β` or acceptance
of a possibly-wrong decay.

## How to read these numbers — the comparison is not apples-to-apples

- **`tick.hawkes.HawkesExpKern` requires the user to supply the decay rate
  `β` as a constructor argument.** Its MLE then fits only the baseline
  `μ` and the amplitude(s) `α`. That is a 2-parameter convex problem for
  univariate (or M²-parameter for multivariate), which C++ can close in
  microseconds.
- **`intensify` fits the whole kernel jointly** — baseline, amplitude,
  and decay — as a 3-parameter (univariate) or 2M²+M-parameter
  (multivariate) non-convex problem, via `scipy.optimize.L-BFGS-B` with
  JAX autodiff on the loss. Python overhead per objective evaluation
  dominates at these sample sizes.
- The RMSE gap on `uni_exp_small` is therefore mostly β-error
  (`intensify` recovered β=1.19 vs true 1.5; μ and α are both close).

## Where each library earns its place

**Prefer `tick` when** you have a well-studied process and already know
the decay rate (e.g. from literature or a separate estimator), the kernel
is pure exponential or sum-of-exponentials, and you want microsecond fits
on Linux/macOS with Python 3.8.

**Prefer `intensify` when** any of the following apply:

1. You don't know the decay a priori and want to fit it jointly.
2. Your kernel is power-law (`PowerLawKernel`, `ApproxPowerLawKernel`) or
   piecewise-constant (`NonparametricKernel`) — tick has no MLE for these.
3. You need marked Hawkes or nonlinear (inhibitory, softplus, sigmoid)
   Hawkes — tick's fit API covers only linear exp/sum-exp kernels.
4. You need the time-rescaling theorem test on inter-compensator
   increments (the mathematically correct form). Several alternative
   implementations — including older intensify code — use cumulative
   compensators and produce wrong p-values.
5. You want projected-gradient stationarity enforcement with
   per-fit `FitResult.branching_ratio_` (spectral radius for
   multivariate) and `endogeneity_index_`.
6. You need a modern Python 3.10+ runtime.
7. You want Python 3.13/3.14 support. tick's last PyPI release was 2020;
   wheels ship up to Python 3.8 only.

## Scaling behavior

See [scaling.md](scaling.md) for the full curve across dataset sizes
from 500 to 91,000 events. TL;DR: both libraries scale linearly in N,
the ~10× speed gap is **stable across all sizes** (does not grow with
data), and parameter-recovery RMSE becomes indistinguishable by ~10k
events. Neither library's asymptotic behavior is concerning — the
choice between them is about capabilities, not scaling.

## Reproducibility

All datasets are seeded (`jax.random.PRNGKey(42)` + `np.random.seed(42)`)
and written as portable `.npy` + `.json` pairs under `benchmarks/data/`.
Results JSON is under `benchmarks/results/`.

## What changed since the first run of this comparison

Two issues were found and fixed in the same session as this benchmark.
Both were latent before any user reported them; the only reason we
caught them was running ground-truth-vs-recovered numbers against
`tick` for the first time.

1. **JAX neg-log-likelihood was being recompiled every fit.** The
   univariate path's `_make_jit_neg_loglik_*` factories returned a
   fresh `@jax.jit`-decorated closure per call, baking
   `events_jax`/`T_jax` into the closure. JAX caches by function
   identity, so each fit triggered a full retrace. Lifting the
   compiled functions to module scope and caching the
   `value+grad` / Hessian wrappers by `(kernel kind, n_components, r)`
   dropped `uni_exp_small` from 376 ms → 5 ms steady-state.

2. **Multivariate log-likelihood was mathematically wrong.**
   `_log_likelihood_dim` summed `log λ_m(t)` over **every** event in
   the system, then summed across `m` — an `M`-fold overcounting of the
   log-intensity term. The textbook multivariate Hawkes likelihood is
   `Σ_n log λ_{k_n}(t_n) − Σ_m Λ_m(T)`, where each event contributes
   `log` of its own *source* dim's intensity exactly once. Verified
   against a hand-computed reference on a 2-d sim. After the fix the
   `mv_exp_5d` parameter RMSE dropped from 0.167 → 0.075 (and the
   subsequent JIT path bumped it to 0.109 due to a different optimum
   neighborhood — still markedly better than the buggy version).

3. **Multivariate now has a JAX-JIT path.** A vectorized neg-log-
   likelihood for the all-`ExponentialKernel` multivariate matrix
   (cached per `M`) replaces the Python loop. Drop: 77 s → 1.1 s.

4. **Decay-given + shared-β recursion.** When every β_{m,k} is the same
   scalar and `fit_decay=False` locks β, intensify uses an O(N·M)
   Hawkes recursion (`jax.lax.scan` with an M-vector state per source
   dim) instead of the dense O(N²) lag matrix. For `mv_exp_5d` (N=1099,
   M=5) that's 1099 × 5 = 5,495 compute steps vs 1,207,801 — an almost
   220× reduction in compute. `mv_exp_5d` decay-given drops
   **200 ms → 17 ms** (~12× speedup) and now lands within 8× of tick
   while still recovering parameters **more accurately** than tick on
   the same data.

## Known limitations we plan to close

- ~8× speed gap to tick on multivariate decay-given is now mostly
  Python→XLA dispatch overhead (~1 ms per `value+grad` call; each L-BFGS
  iteration pays one round-trip). A pure-JAX optimizer (`optax.lbfgs` or
  `jaxopt.ScipyBoundedMinimize`) would run the L-BFGS loop inside JIT
  and remove the per-iteration round-trip. Tracked for 0.3.0.
- The recursive multivariate likelihood currently requires a shared `β`
  across the kernel matrix. Per-cell `β_{m,k}` falls back to the
  O(N²) dense path. A per-source-dim recursion would let joint-fit
  scenarios (β varying) also use the recursive form. Tracked for 0.3.0.
- The fast multivariate path is `ExponentialKernel`-only; mixed-kernel
  multivariate matrices (e.g. exp+power-law per cell) still take the
  numpy fallback. Same fix as above plus a per-cell dispatch table.
- A "fixed-decay" MLE mode (tick-style, 2-param per kernel) would make
  a like-for-like comparison possible. Also 0.3.0.
