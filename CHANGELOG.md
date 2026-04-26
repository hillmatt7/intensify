# Changelog

All notable changes to `intensify` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added (0.3.0 — Rust port, infrastructure complete; not yet published to PyPI)
- **Rust core** (`intensify._libintensify`): every kernel evaluator,
  every likelihood, every analytic gradient, both simulators (Ogata
  thinning + Galton–Watson branching), and every compensator now
  runs in Rust. Cargo workspace of six crates (`core`, `kernels`,
  `likelihood`, `simulation`, `diagnostics`, `pyo3` aggregator)
  modeled on the Nautilus Trader pattern. Built by maturin; one
  Python extension module produced from one workspace.
- **Closed-form analytic gradients** for every likelihood (Ozaki 1979
  for univariate, Bacry et al. 2015 for multivariate, hand-derived
  for the rest). Cross-validated to 1e-10 against the frozen JAX
  reference oracle in `tests/_reference/` across many seeds.
- **Joint-decay multivariate exp** with per-cell β: closed-form
  ∂R/∂β through the M² recursive states. `mv_exp_5d` joint went
  1100 ms → 14 ms (~80×).
- **Marked Hawkes — all four mark-influence kinds in Rust**
  (linear, log, power, callable) via a precomputed g_values pattern.
  Previously `callable` fell through to JAX; that fallback is gone.
- **NonlinearHawkes (Rust)**: softplus / sigmoid / relu / identity
  links over the linear pre-intensity, with numerical compensator on
  a quadrature grid and closed-form chain rule through the link.
- **Nonparametric kernel binary-search bin lookup** replaces the
  O(N²) lag-matrix expansion. Resolves ISSUES.md #8: N=500 went
  from killed-after-7-min to <1 s.
- **Headline performance**: intensify now beats tick at every
  benchmarked N on the multivariate decay-given problem (2.0–2.5×
  faster across N ∈ [501, 91249]) while preserving accuracy. See
  `docs/benchmarks.md` and `docs/scaling.md` for the full curves.
- **HC-3 stress test** (42 tests against real CRCNS hc-3 spike-train
  recordings) dropped from 8m 13s on the 0.2.0 JAX baseline to ~1.3 s
  on the Rust core — ~380× faster end-to-end. All 42 tests still pass.
- **`[fast]` extra**: documents that source builds need a Rust
  toolchain (`pip install 'intensify[fast]'`). Binary wheels for
  Linux x86_64/aarch64, macOS Intel/Apple-Silicon, and Windows
  x86_64 ship via cibuildwheel — workflow file present; no v* tag
  is created yet so PyPI publish has not been triggered.

### Removed (0.3.0)
- **JAX excised from runtime.** Every user-facing inference path now
  hits Rust exclusively. JAX is retained only as a cross-validation
  oracle in `tests/_reference/` (dev-only, never imported by
  anything in `python/intensify/`). `intensify.set_backend("jax")`
  raises `ValueError`; the JAX/numpy backend swap is gone.
- `jax`, `jaxlib`, `optax` removed from runtime dependencies. Moved
  to the `[dev]` extra (cross-val oracle) and the `[bayesian]` extra
  (numpyro is JAX-based by design and stays gated).
- `intensify/intensify/` source layout replaced by `python/intensify/`
  per the maturin convention. `pip install intensify` is unaffected.
- ~1700 lines of obsolete JAX hot-path code (`_neg_ll_*` factories,
  JIT-cached `value_and_grad` providers, `_jax_hessian_std_errors`,
  the JAX backend module) deleted from `mle.py` and the kernel
  modules.

### Changed (0.3.0)
- `MLEInference` now dispatches every supported (kernel, process)
  pair through Rust via the `python/intensify/_rust.py` shim. A
  loud `ImportError` is raised at import time if the compiled
  extension is missing — there is no JAX runtime fallback.
- `EMInference` and `OnlineInference` route through the same shim.
- Build backend changed from `hatchling` to `maturin`. CI workflow
  rewritten around `cargo nextest` + `maturin develop` + `pytest`.

### Added (0.2.0 carry-forward)
- `MLEInference.fit(..., fit_decay=False)` — locks every β slot in the
  flat parameter vector to its initial value via a zero-width L-BFGS-B
  bound. Reduces the active parameter count, speeds up the fit, and
  matches the problem `tick.HawkesExpKern` solves (β supplied as
  input). Supported for `ExponentialKernel` and `SumExponentialKernel`
  in univariate, marked, nonlinear, and multivariate paths. Raises
  `TypeError` for kernels without an identifiable decay-rate axis
  (`PowerLawKernel.beta` is a tail exponent, not a decay rate).
- **Scaling-study benchmarks** in `docs/scaling.md` showing
  `mv_exp_5d` fit time vs N from 501 to 91,249 total events. Both
  intensify and tick scale linearly; the ~10× wall-clock ratio is
  stable across the whole range (it does not grow at scale);
  parameter-recovery RMSE is statistically indistinguishable by
  N≈10k. Reproducible via `benchmarks/run_scaling.py`.
- **O(N·M) recursive multivariate neg-log-likelihood** for the
  shared-β exp-Hawkes case (every β_{m,k} the same scalar), cached
  per `M`. Auto-selected by `_fit_multivariate_numpy` when
  `fit_decay=False` and the kernel matrix has a shared β. Replaces the
  dense O(N²) lag matrix with a `jax.lax.scan` carrying an M-vector
  state per source dim. For `mv_exp_5d` (N=1099, M=5): ~220× fewer
  compute steps, **17 ms fits vs 200 ms** on the same problem.
  Per-cell β still falls back to the dense path.

### Fixed
- **Correctness (critical, multivariate Hawkes likelihood)**:
  `MultivariateHawkes._log_likelihood_dim` summed `log λ_m(t)` over
  every event in the system, then summed across `m` — an `M`-fold
  overcounting of the log-intensity term. The textbook likelihood is
  `Σ_n log λ_{k_n}(t_n) − Σ_m Λ_m(T)`, i.e. each event contributes
  `log` of its own *source* dim's intensity exactly once. Verified
  against a hand-computed reference on a 2-d sim (current −63.0 vs
  correct −44.4 on an identical data set). Fix causes multivariate MLE
  parameter RMSE on the `mv_exp_5d` benchmark to drop from 0.167 → ~0.1,
  now within 2× of tick on the same data. This bug was silently
  affecting every multivariate Hawkes fit since the library's
  inception.
- **Performance (univariate JAX MLE)**: each fit was paying ~370 ms on
  fresh JIT compilation because `_make_jit_neg_loglik_*` returned a new
  `@jax.jit`-decorated closure per call (closing over `events_jax` /
  `T_jax`). JAX caches by function identity, so the trace was
  recompiled every fit. The neg-log-likelihood functions are now lifted
  to module scope and the `value+grad` / Hessian closures are cached by
  `(kernel kind, n_components, r)`. Result: `uni_exp_small` fit dropped
  from **376 ms → 5 ms** steady-state (~75× faster), bringing univariate
  intensify within ~5× of `tick.HawkesExpKern` on the same data while
  still fitting the decay rate jointly.
- **Performance (multivariate JAX MLE)**: `_fit_multivariate_numpy` now
  has a JAX-JIT fast path for the all-`ExponentialKernel` case
  (indirectly required by the flat-vector parameter layout already).
  Uses a dense N×N causal lag matrix with gathered per-pair
  `(α_mk, β_mk)` from `(source_i, source_j)` indexing, and caches the
  compiled `value+grad` per `M`. Multivariate `mv_exp_5d` fit dropped
  **77 s → 1.1 s** (~70× faster). Mixed-kernel multivariate matrices
  still take the numpy fallback path (unchanged).
- Pre-0.2.0 callers of `_make_jit_neg_loglik_exp` /
  `_make_jit_neg_loglik_sum_exp` / `_make_jit_neg_loglik_power_law` /
  `_make_jit_neg_loglik_approx_pl` and the old
  `_jax_hessian_std_errors(neg_ll_fn, x_opt, names)` signature still
  work — they're shimmed to the cached implementation.

## [0.2.0] - 2026-04-20

Public launch release. Consolidates correctness, API polish, and the
full documentation, CI, and scientific-credibility scaffolding needed
for lab adoption. Includes verified head-to-head benchmarks vs
[tick](https://github.com/X-DataInitiative/tick).

### Fixed
- **Correctness (critical)**: `_recursive_compensators` in the time-rescaling
  diagnostic had incorrect math for `ExponentialKernel` /
  `SumExponentialKernel` — the computed cumulative intensities were off by a
  factor of β, causing KS p-values to be catastrophically small under
  well-specified models. Replaced with the correct recursion (equivalent
  to the verified general pairwise-integral path, but O(N) instead of
  O(N²)). Unknown recursive kernels now fall back to the general path
  rather than applying the buggy formula. Before this fix the fast path of
  `time_rescaling_test` and `qq_plot` produced wrong p-values for every
  exponential-family Hawkes fit.

### Added
- `benchmarks/` with README, `reference_dataset.py`, `run_intensify.py`,
  `run_tick.py`, and verified head-to-head results in
  `docs/benchmarks.md`. Datasets use portable `.npy`+`.json` pairs so the
  same files work across NumPy 1.x and 2.x environments. pyhawkes was
  evaluated and dropped (transitive deps reference
  `scipy.misc.logsumexp`, removed in SciPy 1.0).
- `tests/test_textbook_cases.py`: parameter-recovery on seeded simulations,
  log-likelihood scaling, well-specified-vs-misspecified KS sanity check.
- Community infrastructure: `CITATION.cff`, `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `SECURITY.md`, GitHub issue + PR templates.
- Documentation: expanded `docs/getting_started.md` with regularization
  shorthand, backend-switching, and input-validation notes; new
  `docs/user_guide/diagnostics.md` covering the (corrected) time-rescaling
  theorem; new `docs/user_guide/simulation.md`; kernel-coverage table in
  `docs/user_guide/inference.md`.
- README rewrite: PyPI / Python / license / CI badges, "Why intensify?"
  comparison vs `tick` and `pyhawkes`, citation block, doc links.
- CI: multi-OS matrix (ubuntu / macOS / windows), mypy job, Sphinx
  docs-build job with `-W` treat-warnings-as-errors, README-quickstart
  doctest job (`MPLBACKEND=Agg`), PyPI publish workflow on tags using
  OIDC trusted publishing.
- Pre-commit: `codespell`, `check-yaml`, `check-toml`,
  `check-added-large-files`, `end-of-file-fixer`, `trailing-whitespace`,
  `mixed-line-ending`.

## [0.1.1] - 2026-04-19

API-consistency release. No correctness changes; surface-area polish.

### Added
- `MarkedHawkes` and `NonlinearHawkes` MLE now support every kernel type
  the univariate helpers understand: `ExponentialKernel` (signed and
  unsigned), `SumExponentialKernel`, `PowerLawKernel`, `ApproxPowerLawKernel`,
  `NonparametricKernel`. Previously both processes raised
  `NotImplementedError` for anything other than unsigned `ExponentialKernel`.
- `regularization="l1"` / `"elasticnet"` string shorthand accepted by
  `MLEInference.fit`, resolving to default `L1()` / `ElasticNet()`.
  Instances still pass through unchanged.
- Backend proxy: `get_backend()` returns a lazy proxy that delegates every
  attribute to the currently active backend. Module-level
  `bt = get_backend()` captures now pick up runtime `set_backend()` switches.
- `tests/test_mle_kernel_expansion.py`: parametric regression suite for the
  expanded MLE kernel surface + regularization shorthand + backend proxy.

### Changed
- `hawkes_mle_bounds` and `hawkes_mle_apply_vector` now preserve
  `ExponentialKernel.allow_signed` so signed kernels round-trip through MLE
  correctly when reused by `NonlinearHawkes`.

### Fixed
- Pre-existing test/spec mismatches aligned with documented behavior:
  `endogeneity_index` saturates at 1.0 at critical branching (per docstring);
  `FitResult.connectivity_matrix()` raises `TypeError` (not `ValueError`)
  when called on a non-multivariate process — test expectations updated.

## [0.1.0] - 2026-04-19

First versioned release. Focus: correctness on real data, stable packaging.

### Added
- `__version__` attribute on the top-level package.
- `py.typed` marker for PEP 561 type-checker support.
- `Kernel.scale(factor)` method on all built-in kernels so
  `project_params` can enforce stationarity uniformly across kernel types
  (`ExponentialKernel`, `SumExponentialKernel`, `PowerLawKernel`,
  `ApproxPowerLawKernel`, `NonparametricKernel`).
- `RuntimeWarning` emitted from every MLE fit path when the SciPy optimizer
  does not converge — previously only the multivariate path warned.
- `RuntimeWarning` when the fitted multivariate Hawkes spectral radius
  remains ≥ 1 after projection (non-stationary fit).
- Minimum-version pins on all runtime dependencies (`numpy>=1.24`,
  `jax>=0.4.20`, `scipy>=1.11`, `optax>=0.1.7`, `matplotlib>=3.7`).
- Project URLs: Documentation, Issues, Changelog.
- Classifiers: `Operating System :: OS Independent`,
  `Topic :: Scientific/Engineering :: Mathematics`, `Typing :: Typed`.

### Changed
- `UnivariateHawkes.project_params` and `MultivariateHawkes.project_params`
  now use `Kernel.scale()`, so non-`alpha` kernels (e.g. `SumExponentialKernel`,
  `NonparametricKernel`) are actually projected instead of silently skipped.
- Version bumped from `0.1.0-alpha` to `0.1.0`.

### Fixed
- Replaced runtime-validation `assert` statements in
  `intensify/core/kernels/nonparametric.py` and
  `intensify/core/processes/marked_hawkes.py` with explicit `ValueError` /
  `RuntimeError` raises. Asserts are stripped under `python -O` and previously
  hid real errors.
