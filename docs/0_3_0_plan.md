# intensify 0.3.0 plan

Started 2026-04-25. Successor to the 0.2.0 launch (see `SESSION_HANDOFF.md`).

## Strategic frame

The 0.2.0 benchmarks confirmed a stable ~10× constant-factor wall-clock
gap vs `tick` on multivariate decay-given fits, with parameter-recovery
RMSE ~20% better than tick. The gap is dominated by Python↔XLA dispatch,
not compute (compute is sub-ms; dispatch is ~12 ms × ~16 L-BFGS iters).

Labs aren't transitioning off tick because the kernel-variety value-add
doesn't outweigh the speed delta. **0.3.0 is the decision point on
whether intensify stays pure-Python or gets a compiled backend / port.**
If WS-1 + WS-2 don't move the adoption needle, the next move is a
Rust/C++ backend or a full port.

## Workstreams

### WS-1 — close the dispatch gap (perf)

Replace the SciPy L-BFGS-B driver in the multivariate JAX path with
`jaxopt.LBFGSB` (bounded variant — keeps β positivity and stationarity
constraints). Run the optimization loop inside a single `jax.jit` so each
fit pays one XLA dispatch instead of one per iteration.

**Target:** 2–3× on `mv_exp_5d` decay-given (200 ms → ~70–100 ms),
shrinking the 10× constant factor toward 3–5×.

**Touch points:**
- `intensify/core/inference/mle.py` — `_fit_multivariate_jax` (or whichever
  branch dispatches to `spo.minimize` for the JAX-JIT MV path).
- New optional dep on `jaxopt` (or upstream into `jax.scipy.optimize`).

### WS-2 — capabilities and correctness tick can't match

#### 2a — Multivariate stationarity actually enforced

ISSUES.md #1: `MultivariateHawkes.project_params()` currently warns when
the per-dimension branching ratio ≥ 1 but does nothing. HC-3 stress
testing produced spectral_radius=3.98 silently. `project_params()` is
also never called during MLE optimization.

**Plan:**
1. Implement real projection — when row L1 norm ≥ 1, scale the row down
   so its norm is `(1 - eps)` (eps small, configurable, default 1e-3).
2. Call `project_params()` after the optimizer returns (a final
   projected-gradient step) and emit a `RuntimeWarning` describing the
   adjustment. Unconditional in-loop projection is too costly; one-shot
   post-fix matches user expectation that fits are stationary.
3. Compute and store `branching_ratio_` = spectral radius of the kernel
   norm matrix on the multivariate `FitResult`.
4. Tests: synthetic MV setup with overshooting initial guess; verify
   final spectral radius < 1 and a warning is emitted.

#### 2b — `FitResult.flat_params()` + top-level regularizer exports

ISSUES.md #6, #7. Currently:
- `result.params["alpha"]` → `KeyError`; the user has to know
  `result.process.kernel.alpha`.
- `from intensify.core.regularizers import L1` is the only path; not
  exported from `intensify.__init__`.

**Plan:**
1. Add `FitResult.flat_params() -> dict[str, float]` that walks the
   process (univariate / multivariate / marked / nonlinear) and returns
   a dict of named scalar values.
2. Export `L1`, `ElasticNet` from `intensify/__init__.py`.
3. Tests: parameter recovery test asserts `flat_params()["alpha"]`
   matches the fitted `result.process.kernel.alpha`.

## Out of scope for 0.3.0

- ISSUES.md #4 (MarkedHawkes.fit signature) — breaking; defer to 0.4.0
  with a deprecation cycle.
- Rust/C++ backend — only triggered if WS-1+WS-2 fall short.
- ISSUES.md #2 endogeneity_index — minor; pick up if there's slack.

## 2026-04-25 update: WS-2 was already shipped; WS-1 hypothesis was wrong

Audit of the post-handoff repo state showed both WS-2 deliverables had
already landed in the uncommitted 0.2.0 work:

- **WS-2a (multivariate stationarity)** — `MultivariateHawkes.project_params`
  rescales row L1 norms to 0.99 when ≥1, both numpy and JAX MV fit
  paths call it post-fit, and `branching_ratio_` stores the true
  spectral radius (eigvals of the L1-norm matrix). Dedicated tests in
  `tests/test_multivariate_hawkes.py` and `tests/test_real_data_stress.py:381`
  cover the HC-3 stress scenario.
- **WS-2b (flat_params + regularizer exports)** — `FitResult.flat_params()`
  exists at `intensify/core/inference/__init__.py:97`, `L1` and
  `ElasticNet` are exported from `intensify/__init__.py`.

So the full session collapsed to **WS-1**, the language-decision lever.

### WS-1 result: not viable as designed

Implemented `jax.scipy.optimize.minimize(method='BFGS')` inside
`jax.jit` with log-reparameterization for positivity. Tested against
the 0.2.0 scipy L-BFGS-B baseline across N ∈ {501, 2249, 9271, 27519,
91249} with `mv_exp_5d_scale_*`.

**Two failure modes emerged.**

1. **The dispatch hypothesis was wrong.** SESSION_HANDOFF claimed the
   200 ms multivariate decay-given gap was "Python→XLA dispatch
   overhead (~12 ms × 16 iters)". Direct measurement of the cached
   `value_and_grad` shows actual dispatch+compute is ~0.5 ms per
   iter at N=501 and ~45 ms per iter at N=91k. **At small N dispatch
   is small in absolute terms; at large N "dispatch" *is* compute.**
   The 10× gap to tick is dominated by per-iteration XLA-on-CPU
   compute losing to tick's hand-tuned C++ recursive likelihood, not
   by Python loop overhead. JIT'ing the L-BFGS loop saves at most
   ~6 ms total at small N and nothing at large N.

2. **`jax.scipy` BFGS is materially worse than scipy's L-BFGS-B.**
   It hits status=3 (line-search failed) before reaching the true
   minimum — at N=91k it stops at fun=150438 vs scipy's 149849. RMSE
   at every N tested was 1.5–5× worse than the scipy baseline. A
   "polish with scipy from x_jit" hybrid recovers RMSE but the polish
   redoes most of the work, so total wall-clock regressed below
   baseline (e.g., 549 ms → 1250 ms at N=91k).

Net: BFGS-reparam alone ships worse RMSE; BFGS-reparam + scipy polish
ships worse wall-clock. No regime where it wins.

### Where this leaves the language decision

The 10× constant factor is not closable in pure Python without a
compiled inner loop. Options ranked by ROI:

1. **Numba JIT for the recursive likelihood (~days).** Likely closes
   the gap to ~2–3× by generating tight C-like code. Loses JAX
   autodiff for that path; either hand-code gradients (closed-form
   for exp recursion) or use finite differences for std errors.
2. **Rust backend with PyO3 for the recursive likelihood (~weeks).**
   Probably matches tick. Keeps Python ergonomics; one compiled crate
   in `intensify_core/`. Stays as a `jax`-optional dep.
3. **Stay pure Python, lean on capabilities.** Update README to be
   honest about the 10× constant factor and frame intensify as
   "kernel variety + correct diagnostics + modern Python", not
   "tick but faster". Existing capability lead is the actual moat.

Recommend a fresh user check-in before picking; the language decision
is the real strategic choice, not WS-1's implementation details.

## Definition of done

- `mv_exp_5d` decay-given benchmark drops to ≤100 ms median (currently
  200 ms).
- Synthetic non-stationary MV fit produces a final result with spectral
  radius < 1, with a warning explaining the projection.
- `result.flat_params()` and top-level `its.L1` work as documented in
  `getting_started.md`.
- All existing tests still green; new tests cover the three changes.
- README "Why intensify?" table refreshed with the new MV decay-given
  number.
