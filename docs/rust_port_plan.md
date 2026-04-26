# intensify Rust port plan

Started 2026-04-25. Supersedes `docs/0_3_0_plan.md` after WS-1 (pure-Python
JIT'd L-BFGS) failed to close the gap and confirmed the bottleneck is
per-iteration XLA-on-CPU compute vs tick's hand-tuned C++. Modeled on
Nautilus Trader's Rust+PyO3 architecture.

---

## STATUS — last updated 2026-04-25, mid-Phase-1c

### Branches + commits

- `main` at `v0.2.0` (clean release, tagged locally; not on PyPI per plan)
- `rust-port` ahead by 4 commits:
  - `4c5b37b` planning docs
  - `6fe8343` Phase 0 — workspace bootstrap
  - `3bbf734` Phase 1a — ExponentialKernel + uni_exp Rust + 92 cross-val tests
  - `4e7e800` Phase 1b — mv_exp_recursive Rust (tick-modeled) + 47 cross-val tests
- Phase 1c (live MLE wire-up) **uncommitted in working tree**

### Done ✅

| Phase | Scope | Commit |
|---|---|---|
| Pre-port | v0.2.0 squashed + tagged, planning docs separated, rust-port branch | bfaddc5 + 4c5b37b |
| Phase 0 | Cargo workspace (6 crates), maturin pyproject, JAX → [dev]/[bayesian] extras, intensify/ → python/intensify/, dispatch shim with loud-fail import | 6fe8343 |
| Phase 1a | ExponentialKernel + uni_exp_neg_ll_with_grad with closed-form Ozaki gradient + PyO3 + 92 cross-val tests at 1e-10 vs JAX | 3bbf734 |
| Phase 1b | MvExpRecursiveLogLik modeled on tick's C++ (per-target weight precomputation, separable per-row loss). 47 cross-val tests at 1e-10. **Beats tick at every scale.** | 4e7e800 |
| Phase 1c (live wire-up, in progress) | `_fit_uni_exp_rust` + `_fit_mv_exp_recursive_rust` methods, dispatch helpers in `_rust.py`, JAX-specific assertions updated. Uncommitted. | (pending) |

### Headline numbers (decay-given fit, scipy L-BFGS-B + Rust value+grad)

| N | tick (ms) | intensify 0.2.0 (ms) | **intensify Rust** (ms) | vs tick |
|---:|---:|---:|---:|---:|
| 501 | 1.0 | 8 | **0.5** | **2.0×** |
| 2,249 | 2.0 | 21 | **0.8** | **2.5×** |
| 9,271 | 6.0 | 38 | **2.4** | **2.5×** |
| 27,519 | 15.0 | 189 | **6.9** | **2.2×** |
| 91,249 | 48.0 | 549 | **22.2** | **2.2×** |

RMSE preserved (matches 0.2.0 numbers identically — deterministic on same data).

### Phase 1c — to finish

1. Run full test suite (interrupted; the 3 failures we saw mid-run were JAX-specific `jit_compiled is True` assertions on now-Rust ExponentialKernel paths; 2 already updated, suite needs rerun to confirm zero failures)
2. Commit Phase 1c with the live wire-up
3. Refresh `benchmarks/run_intensify.py` numbers running through the **public API** (currently they hit the JAX path; post-1c they hit Rust)

### Still to do

#### Phase 2 — mv_exp dense (joint-decay β fit) + general likelihood (~1 week)
- `crates/likelihood/src/mv_exp_dense.rs` with **β gradient** (the deferred bit from Phase 1b)
- `crates/likelihood/src/general.rs` for non-recursive kernels (lag-matrix O(N²))
- Cross-val tests at 1e-10
- Wire dispatch to Rust for joint-decay MV
- Exit: joint-decay `mv_exp_5d` ≥ 5× faster than 0.2.0 baseline

#### Phase 3 — remaining kernels + simulation + EM/online + diagnostics (~2 weeks)
- PowerLawKernel, ApproxPowerLawKernel, SumExponentialKernel, NonparametricKernel, signed variants
- Compensators paired with each kernel (not deferred)
- Marked + Nonlinear likelihoods (shared ParamLayout via crates/core)
- Thinning + branching simulators (Ogata's algorithm)
- EMInference + OnlineInference using same Rust dispatch shim
- HC-3 stress test as regression check
- **Excise JAX entirely from `python/intensify/`** — only `tests/_reference/` retains JAX

#### Phase 4 — cibuildwheel, .pyi shipping, release (~1 week)
- pyo3-stub-gen pre-build hook
- cibuildwheel matrix: `{linux-x86_64, macos-arm64, macos-x86_64, windows-x86_64} × {py3.10, 3.11, 3.12}`
- OIDC PyPI publish on `v*` tags
- Documentation refresh + tag v0.3.0

### Strategic verdict so far

The question "can intensify match or beat tick on speed while keeping its capability lead?" is **answered: yes, by 2–3× on tick's home turf**. Phase 1b confirmed this end-to-end. Phases 2–4 are now execution work, not research.

---

## Goal

Match or beat tick on speed for every case tick supports, retain the
full intensify capability lead (kernel variety, marked/nonlinear/signed,
diagnostics, modern Python), and ship one `pip install intensify` that
does both. The Python API does not change for users.

Decision criteria for "done":

- `mv_exp_5d` decay-given at N=91k: ≤ tick (currently 549 ms vs tick 48 ms).
- Univariate exp recursive at any N: ≤ tick (currently 8 ms vs tick 1 ms at N=516).
- All 0.2.0 capabilities still work: PowerLawKernel, NonparametricKernel,
  MarkedHawkes, NonlinearHawkes, signed kernels, all diagnostics.
- 224+ tests still green; new Rust unit tests + Rust↔Python cross-validation tests.

## Reference architecture: Nautilus Trader

Studied at `/home/etrigan/SoftwareDev/Projects/nautilus_trader`. Patterns we adopt:

1. **Cargo workspace of domain crates** — `crates/<domain>/` each pure-Rust by default.
2. **Single aggregator `crates/pyo3/`** — `cdylib` + `rlib`, one `.so` output (`intensify._libintensify`). Uses `wrap_pymodule!` to register each domain crate's `python::<domain>` module.
3. **Inline conditional bindings** — `#[cfg_attr(feature = "python", pyclass)]` on the same struct used in pure-Rust builds. Zero PyO3 overhead when feature off; supports downstream Rust consumers.
4. **`src/python/` submodule per crate** — `#[pymethods]` and `#[pymodule]` live here, separate from pure-Rust impl.
5. **Parallel `python/intensify/` tree** — each `__init__.py` does `from intensify._libintensify.<domain> import *`. Hand-maintained, ships the wheel.
6. **Maturin** with `manifest-path = "../crates/pyo3/Cargo.toml"`, `module-name = "intensify._libintensify"`, `python-source = "."`.
7. **`rust-toolchain.toml`** pins stable Rust (1.92).
8. **`pyo3-stub-gen`** for auto-generated `.pyi` files via a `pre-build` script, packaged via `include = ["**/*.pyi"]`.

**Departure from Nautilus:** Nautilus is scalar-per-tick — zero NumPy interop. Intensify is array-heavy. We add `numpy = "0.22"` (pyo3-numpy) to receive event arrays as `PyReadonlyArray1<f64>` zero-copy views and return results as `PyArray1<f64>`. The standard pattern looks like:

```rust
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};

#[pyfunction]
fn uni_exp_neg_loglik<'py>(
    py: Python<'py>,
    times: PyReadonlyArray1<'py, f64>,
    t_horizon: f64,
    mu: f64,
    alpha: f64,
    beta: f64,
) -> PyResult<(f64, Bound<'py, PyArray1<f64>>)> {
    let t = times.as_slice()?;          // zero-copy &[f64]
    let (val, grad) = uni_exp_neg_ll_with_grad(t, t_horizon, mu, alpha, beta);
    Ok((val, grad.to_pyarray(py).into()))
}
```

## Target repo layout

```
intensify/                                      # repo root
├── Cargo.toml                                  # [workspace] root
├── Cargo.lock
├── rust-toolchain.toml                         # stable 1.92
├── rustfmt.toml, clippy.toml, deny.toml
├── pyproject.toml                              # maturin build-backend
├── README.md, CHANGELOG.md, etc.
│
├── crates/
│   ├── core/                                   # types: Event, Sample, ParamView
│   │   ├── Cargo.toml
│   │   ├── src/lib.rs                          # pure Rust types
│   │   └── src/python/{mod.rs}                 # PyO3 wrappers, gated by feature="python"
│   ├── kernels/                                # all kernel evaluators
│   │   ├── src/{exponential,power_law,approx_power_law,sum_exp,nonparametric}.rs
│   │   └── src/python/                         # #[pyclass] mirrors of each kernel
│   ├── likelihood/                             # the perf-critical surface
│   │   ├── src/uni_exp.rs                      # univariate recursive + grad
│   │   ├── src/mv_exp.rs                       # multivariate recursive + grad
│   │   ├── src/general.rs                      # O(N²) for non-recursive kernels
│   │   ├── src/marked.rs, nonlinear.rs
│   │   └── src/python/                         # #[pyfunction] entry points
│   ├── simulation/                             # thinning, branching
│   │   └── src/{thinning,cluster}.rs
│   ├── diagnostics/                            # compensators (recursive + general)
│   │   └── src/{compensators,time_rescaling}.rs
│   └── pyo3/                                   # AGGREGATOR
│       ├── Cargo.toml                          # cdylib + rlib, depends on all sibs with features=["python"]
│       └── src/lib.rs                          # #[pymodule] _libintensify { wrap_pymodule!(...) × N }
│
├── python/
│   └── intensify/
│       ├── __init__.py                         # user-facing API; imports from _libintensify
│       ├── _libintensify.so                    # (compiled, gitignored)
│       ├── kernels/__init__.py + .pyi          # re-export wrapper (auto-stub)
│       ├── core/, inference/, diagnostics/, simulation/, visualization/
│       └── ...
│
├── tests/                                      # Python integration tests (existing 224)
└── benchmarks/                                 # existing scaling + tick comparison
```

The existing `intensify/intensify/` tree becomes `python/intensify/` (preserves the Python public API exactly). Hot-path implementations move to Rust crates; thin Python wrappers stay so user-facing classes (`Hawkes`, `MultivariateHawkes`, etc.) continue to work as today.

## What goes to Rust, what stays Python

**Rust (compiled, the hot loop):**

- All kernel evaluators: `evaluate(t)`, `integrate(t)`, `integrate_vec(t)`, `l1_norm()`, `scale(factor)`.
- All likelihood functions: `_recursive_likelihood_numpy`, `_recursive_likelihood_jax`, `_neg_ll_mv_exp`, `_neg_ll_mv_exp_recursive`, `_general_likelihood_*`, marked, nonlinear variants.
- Closed-form gradients for all of the above (eliminates dependency on JAX autodiff for the hot path).
- Compensators: `_recursive_compensators` and `_general_compensators` from `diagnostics/goodness_of_fit.py`.
- Simulators: `simulate_thinning`, `simulate_cluster`.
- Time-rescaling residual computation.

**Python (thin interface layer):**

- All process classes (`Hawkes`, `MultivariateHawkes`, `MarkedHawkes`, `NonlinearHawkes`).
- `FitResult` dataclass + `flat_params`, `connectivity_matrix`, `significant_connections`.
- All inference engines (`MLEInference`, EM, Bayesian, online) — they call into Rust for value+grad and let scipy drive L-BFGS-B (we keep scipy because it's already faster than JAX at this size and the polish issue we hit on WS-1 is a non-issue when value+grad is a fast Rust call).
- `MultivariateHawkes.project_params()` (cheap, called once post-fit).
- All visualization (matplotlib).
- Bayesian inference (numpyro-based, untouched).
- Backend abstraction layer (potentially simplified — JAX backend stays as fallback for autodiff cases not yet ported, e.g., user-defined kernels).

## Phase plan

Each phase is independently shippable — the Python API works after every phase, with the Rust crate as an *optional accelerator*. If `intensify._libintensify` is missing at import time (source install without Rust toolchain, unsupported platform), Python falls back to the existing JAX/numpy paths. This is a hard requirement to keep contributor friction low and `pip install` fast.

### Phase 0 — Bootstrap workspace (~1 day)

- Add `Cargo.toml`, `rust-toolchain.toml`, `clippy.toml`, `rustfmt.toml`.
- Move `intensify/` → `python/intensify/`.
- Create empty `crates/{core,kernels,likelihood,simulation,diagnostics,pyo3}/` with stub `Cargo.toml`s and `lib.rs`.
- Update `pyproject.toml`: maturin build-backend, `manifest-path`, `python-source = "."`.
- Verify `pip install -e .` still works — at this point the wheel still contains only Python; Rust crates are stub libraries. All 224 existing tests must still pass.

**Exit criterion:** `cd /home/etrigan/SoftwareDev/Libraries/intensify && pip install -e . && pytest` is green. No behavior changes.

### Phase 1 — First slice end-to-end (~3 days)

Smallest possible PyO3-bound slice that proves the pipeline. Pick **univariate exponential recursive log-likelihood** because:
- The math is one page of textbook (Ozaki 1979).
- Closed-form gradients are trivial.
- It maps to a `tests/test_textbook_cases.py::test_uni_exp_recovery` we already have.
- The rest of the architecture (workspace, pyo3 aggregator, stub gen, NumPy interop) gets exercised on the smallest possible math surface.

Deliverables:
- `crates/kernels/`: `ExponentialKernel { alpha: f64, beta: f64, allow_signed: bool }` with `evaluate`, `integrate`, `integrate_vec`, `l1_norm`. Pure Rust + `#[cfg_attr(feature="python", pyclass)]`.
- `crates/likelihood/src/uni_exp.rs`: `uni_exp_neg_ll_with_grad(times: &[f64], T: f64, mu: f64, alpha: f64, beta: f64) -> (f64, [f64; 3])`.
- `crates/pyo3/src/lib.rs`: aggregator wraps `kernels` and `likelihood` modules.
- `python/intensify/_rust.py`: feature-detected import — `try: from intensify._libintensify import likelihood as _rust_lik; except ImportError: _rust_lik = None`.
- `python/intensify/core/inference/mle.py`: when `_rust_lik` is available and kernel is ExponentialKernel + UnivariateHawkes, route value+grad through Rust.
- New tests: `tests/test_rust_uni_exp.py` asserts `rust_loglik(...) == jax_loglik(...)` to 1e-10 across 50 seeds.

**Exit criterion:** uni_exp_small benchmark drops from 8 ms to ≤ 2 ms (matching tick at 1 ms within a small constant). `tests/test_rust_uni_exp.py` green.

### Phase 2 — Multivariate exp recursive (the headline) (~5 days)

This is the case tick "wins" today and where the language decision pays off.

- `crates/likelihood/src/mv_exp.rs`: `mv_exp_recursive_neg_ll_with_grad(times: &[f64], sources: &[i32], T, mu: &[f64], alpha: ArrayView2<f64>, beta: f64) -> (f64, MvGrad)`. O(N·M) state vector, O(N·M²) for the alpha block of the gradient.
- Hand-coded analytic gradient (vs autodiff). Reference: Bacry et al. 2015 — closed form is straightforward for the recursive case.
- `crates/likelihood/src/mv_exp_dense.rs`: O(N²) dense path for the joint-decay case (β fit too) using the same lag-matrix structure as the current numpy version, but with SIMD via packed_simd or std::simd.
- Wire into `_fit_multivariate_numpy` in Python.
- Re-run scaling benchmarks at N=501, 2249, 9271, 27519, 91249.

**Exit criterion:** at every N, `mv_exp_5d` decay-given is within 1.5× of tick. Joint-decay at every N is faster than today by ≥ 5×.

### Phase 3 — Remaining kernels, general likelihood, simulation (~5 days)

- `crates/kernels/`: `PowerLawKernel`, `ApproxPowerLawKernel`, `SumExponentialKernel`, `NonparametricKernel`, signed variants.
- `crates/likelihood/src/general.rs`: O(N²) lag-matrix likelihood, vectorized; analytic-gradient-via-ndarray over the same lag matrix.
- `crates/likelihood/src/marked.rs`, `nonlinear.rs`.
- `crates/simulation/src/thinning.rs`, `cluster.rs`. The thinning bound check is a tight inner loop (Ogata's thinning algorithm); ideal for Rust.
- Update inference dispatch to route every (kernel × process) pair through Rust.

**Exit criterion:** All capability paths (power-law, nonparametric, marked, nonlinear, signed, simulation) are Rust-backed. Existing tests green. Benchmarks: power-law ≥ 5× faster than today; nonparametric usable at N=10,000 (currently dies at N=300).

### Phase 4 — Diagnostics, CI wheels, release (~3 days)

- `crates/diagnostics/`: recursive + general compensators. Time-rescaling residuals (the math we already fixed in 0.2.0).
- `pyo3-stub-gen` integration; `.pyi` files generated at build time and shipped.
- GitHub Actions: `cibuildwheel` matrix for Linux (manylinux2014) / macOS (universal2) / Windows × Python 3.10/3.11/3.12. Wheels published to PyPI on `v*` tags via OIDC trusted publisher.
- Source builds remain possible for unsupported platforms (require Rust toolchain — gated as a noisy ImportError that points at `pip install intensify[fast]`).
- Update `README.md`, `docs/benchmarks.md`, `docs/scaling.md` with final numbers.
- Tag `v0.3.0`.

**Exit criterion:** `pip install intensify` on a clean Linux machine pulls a binary wheel and runs the full scaling benchmark in under 1 minute. Numbers in README match what users experience.

## Crate-by-crate dependency graph

```
core ─────────────────────────────────────────────┐
  ↑                                               │
kernels ◄── likelihood ◄── simulation ◄──┐        │
                  ↑                      │        │
                  └────── diagnostics ◄──┤        │
                                         │        │
                                       pyo3 ◄─────┘  (one .so, depends on all with feature="python")
```

`core` exposes shared types (`Sample = (events: Vec<f64>, T: f64)`, `Param`, error types).
`kernels` is leaf-most among the math crates — pure functions on `f64` slices.
`likelihood`, `simulation`, `diagnostics` all use kernels via the `Kernel` trait defined in `kernels`.
`pyo3` depends on every domain crate with `features = ["python"]` enabled and aggregates their `python::<domain>` modules.

## Build + test commands

```bash
# Dev install (Python venv with Rust extension built in debug)
cd /home/etrigan/SoftwareDev/Libraries/intensify
maturin develop --release        # release for benchmarking, dev for fast iteration

# Pure Rust tests (no Python)
cargo test --workspace --all-features

# Python tests (driving Rust + Python interface)
.venv/bin/pytest tests/

# Benchmarks
.venv/bin/python benchmarks/run_intensify.py mv_exp_5d
.venv/bin/python benchmarks/run_scaling.py --lib intensify

# Type stub generation (also runs at build time via pre-build hook)
maturin develop --release && python python/intensify/_stubgen.py
```

## Test strategy

Three layers, each catching a different bug class:

1. **Pure Rust unit tests** in each crate (`crates/<x>/tests/`) — math correctness. Fast (<1 s/crate); run on every save via `cargo nextest`.
2. **Cross-validation tests** in Python (`tests/test_rust_*.py`) — assert `rust_path(args) == jax_path(args)` to 1e-10 across many seeds. Catches FFI bugs, numerical drift between Rust and the existing reference implementation.
3. **Existing 224 integration tests** — unchanged. Drive the public API; transparently use the Rust path when available. The fall-back path stays so they also work without the Rust extension.

The HC-3 stress test (8m 13s, 42 tests) becomes a useful end-of-phase regression check.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| **Rust ↔ JAX numerical drift on the order of 1e-12** breaks existing tests' tight tolerances | Cross-val tests at 1e-10; relax tolerance only when measured drift < 1e-12 in 1000 seeds |
| **Wheel coverage gaps** force users to compile from source with a slow `pip install` | cibuildwheel publishes manylinux2014 + macOS universal2 + Windows wheels for py3.10/3.11/3.12; document the `[fast]` extra clearly |
| **PyO3 GIL contention** if Python calls Rust in tight loops | Likelihood is one Rust call per L-BFGS iteration (≈12 calls), GIL contention is irrelevant at this granularity |
| **NumPy↔Rust copying overhead** if we get the interop wrong | Use `PyReadonlyArray1::as_slice()` for zero-copy views; assert in tests that hot-path receives `&[f64]` not `Vec<f64>` |
| **Two-language maintenance burden** | Domain split mirrors the existing Python module split; PRs touching one domain rarely cross language boundaries. Rust API stays small (~10 public functions per crate). |
| **Incremental migration breaks tests mid-phase** | Each phase is feature-flagged: Python keeps the old path and only routes to Rust when `intensify._libintensify` import succeeds AND the kernel/process matches. Old tests stay green throughout. |
| **Closed-form gradient bugs** silently produce wrong fits | Cross-validation against JAX `jax.grad` to 1e-10 in test suite. JAX path stays in the codebase as the reference oracle. |

## What we are explicitly *not* doing

- **No GPU port.** CPU-Rust suffices to match tick. GPU is a future phase if any user requests it.
- **No removal of JAX dependency.** JAX path stays as fallback + reference oracle. Could be removed in a future major version after Rust paths are battle-tested.
- **No removal of the backend abstraction layer.** It's already minimal.
- **No port of the Bayesian / numpyro path.** That's a different perf regime.
- **No async.** Hawkes fitting is synchronous; Nautilus's Tokio runtime is not relevant here.
- **No port of the Cython/build.py legacy bits Nautilus has.** We start clean with maturin only.

## Open questions for the user

1. **Where does this work happen?** This repo (`SoftwareDev/Libraries/intensify`) currently has only `Initial commit` tracked — everything from 0.2.0 is uncommitted. Two options: (a) commit the 0.2.0 work first as one big squashed commit, then start the Rust restructure on a `rust-port` branch; (b) skip 0.2.0 commits and bundle them with the Rust port into 0.3.0. **Recommend (a)** — the 0.2.0 work is a coherent unit and worth its own version-tagged release.
2. **Are you OK with shipping the wheel-only path as the primary install?** Source installs without Rust toolchain will print a useful error and fall back to JAX. Or do you want a fully pure-Python fallback that's silently slow?
3. **Do you want to ship 0.2.0 to PyPI now** as a pre-Rust release, or wait until 0.3.0 lands the Rust core?

Once those are answered I can start Phase 0.
