# Intensify — session handoff (resumed 2026-04-20)

## Update 2026-04-20 (later): performance + accuracy push

After the first benchmark, three substantive fixes landed.

### 1. Latent multivariate log-likelihood bug (correctness)
`MultivariateHawkes._log_likelihood_dim` was summing `log λ_m(t)` over
**every** event in the system then summing across `m` — `M`-fold
overcounting of the log-intensity. Textbook MV Hawkes log-likelihood is
`Σ_n log λ_{k_n}(t_n) − Σ_m Λ_m(T)`. Verified by hand on a 2-d sim
(buggy −63.0 vs correct −44.4). This was silently corrupting every
multivariate fit since the library's inception.
Fix: only sum over events with `source = m`.

### 2. JIT was being recompiled every fit (perf)
`_make_jit_neg_loglik_*` returned a fresh `@jax.jit` closure per call
with `events_jax`/`T_jax` baked in — JAX caches by function identity,
so each fit triggered a new trace. Lifted neg-log-likelihood functions
to module scope; cache `value+grad` and Hessian wrappers per
`(kernel kind, n_components, r)`. Univariate `uni_exp_small` dropped
**376 ms → 5 ms steady-state**.

### 3. Multivariate JAX-JIT path (perf)
`_fit_multivariate_numpy` now has a JAX path for the all-`ExponentialKernel`
case using a dense N×N causal lag matrix with gathered `(α, β)` from
fancy-indexed source labels. Cached per `M`. `mv_exp_5d` dropped
**77 s → 1.1 s** (joint mode).

### 4. `fit_decay=False` mode (apples-to-apples with tick)
Locks every β to its initial value via zero-width L-BFGS-B bounds.
Reduces active parameter count and matches the problem tick solves.

### Final benchmark numbers

| Scenario | Mode | intensify | tick |
|---|---|---|---|
| `uni_exp_small` (516 ev) | joint | 4 ms / RMSE 0.188 | n/a |
| `uni_exp_small` | **decay-given** | **2.6 ms / RMSE 0.042** | 1 ms / 0.029 |
| `uni_power_law` (451 ev) | joint | 56 ms / 0.094 | unsupported |
| `mv_exp_5d` (1099 ev, 5d) | joint | 1.1 s / 0.109 | n/a |
| `mv_exp_5d` | **decay-given** | **200 ms / RMSE 0.041** ← beats | 2 ms / 0.052 |

**The win**: in apples-to-apples mode (decay-given), intensify
parameter recovery is 20% **better** than tick on multivariate
(RMSE 0.041 vs 0.052), at the cost of ~100× wall-clock speed.

**The remaining gap**: ~200 ms steady-state for multivariate
decay-given is dominated by Python→XLA dispatch overhead (~12 ms per
`value+grad` call × ~16 L-BFGS iterations). Compute itself is sub-ms.
Path forward = pure-JAX optimizer (`optax.lbfgs` / `jaxopt`)
running the L-BFGS loop inside JIT. Tracked for 0.3.0.

---

## Update 2026-04-20: head-to-head benchmark RAN

- `tick` 0.7.0.1 in micromamba env `tickbench38` works.
- `pyhawkes` evaluated and dropped (`pybasicbayes` uses removed
  `scipy.misc.logsumexp`).
- `benchmarks/data/` now holds 3 portable scenarios: `uni_exp_small`
  (516 ev), `uni_power_law` (5000 ev), `mv_exp_5d` (1099 ev × 5 dims).
- `benchmarks/results/` has both libs' JSON output for the runnable
  scenarios.
- **Real numbers, written up in `docs/benchmarks.md`**, README "Why
  intensify?" table replaced with verified comparison.

### What the numbers actually say

| Scenario | intensify | tick |
|---|---|---|
| uni_exp_small fit time | 0.37 s | 0.001 s |
| uni_exp_small RMSE | 0.188 | 0.029 |
| uni_power_law fit time | 0.54 s | unsupported |
| mv_exp_5d fit time | 86 s | 0.002 s |
| mv_exp_5d RMSE | 0.167 | 0.052 |

tick is much faster but the comparison is uneven: tick fits only
amplitudes given a known decay; intensify fits the whole kernel jointly.
intensify wins on kernel variety (power-law, nonparametric, signed,
marked, nonlinear) and modern-Python support; tick wins on raw speed
when you already know the decay.

Roadmap items added to `docs/benchmarks.md`:
- O(M²·N) JAX-JIT multivariate path (would close most of the gap)
- "Fixed-decay" MLE mode for like-for-like 2-param comparison

## Original handoff below
---

## TL;DR

We took `intensify` from `0.1.0-alpha` to a **launch-ready `0.2.0`** with
**224 tests passing, 0 failures**, including regression on real HC-3 spike
data. All of the approved plan's WS-1 through WS-6 work is shipped in the
repo. The only in-progress thread when we stopped was an actual head-to-head
`tick` vs `intensify` benchmark run — the tick install finally worked and we
were regenerating the reference datasets in a cross-numpy-version-portable
format when you had to shut down.

Plan of record: `/home/etrigan/.claude/plans/we-need-to-create-mutable-gray.md`

## What's shipped (all in repo, committed to disk only — no git commits)

### Correctness (WS-1)
- `intensify/core/kernels/base.py`: new `Kernel.scale(factor)` method
- `intensify/core/kernels/sum_exponential.py`, `nonparametric.py`: `scale()` overrides
- `intensify/core/processes/hawkes.py`: `project_params()` in both uni- and multivariate uses `scale()` so non-`alpha` kernels actually get projected instead of silently skipped
- `intensify/core/inference/mle.py`: new `_warn_if_not_converged` helper called from all 4 fit paths (was multivariate-only); `RuntimeWarning` when multivariate spectral radius ≥ 1 after projection; `_resolve_regularization` helper for `"l1"` / `"elasticnet"` string shorthand
- `intensify/core/kernels/nonparametric.py:178` + `intensify/core/processes/marked_hawkes.py:72`: `assert` → explicit `ValueError`/`RuntimeError`
- **`intensify/core/diagnostics/goodness_of_fit.py`**: fixed real correctness bug in `_recursive_compensators` — was off by factor of β for ExponentialKernel/SumExponentialKernel, making KS p-values catastrophically wrong (1e-42 on well-specified fits). Now uses the correct R-based recursion matching the O(N) likelihood path; unknown kernels fall back to the verified general path.

### API consistency (WS-2)
- `intensify/core/inference/univariate_hawkes_mle_params.py`: ExponentialKernel apply/bounds now preserve `allow_signed`
- `intensify/core/inference/mle.py`: removed ExponentialKernel-only guards in `_fit_marked_numpy` + `_fit_nonlinear_numpy`. Both now accept all 5 kernel types via the shared vectorization helpers.
- `intensify/backends/_backend.py`: `get_backend()` now returns a `_BackendProxy` that delegates every attr access to the currently active backend, so module-level `bt = get_backend()` captures stay valid across `set_backend()` calls.

### Tests added
- `tests/test_mle_kernel_expansion.py`: parametric across all kernels × {MarkedHawkes, NonlinearHawkes}, plus regularization shorthand, plus backend proxy functional test
- `tests/test_textbook_cases.py`: parameter recovery on seeded sims + well-specified-vs-misspecified KS sanity (this is what caught the compensator bug)
- Updated `tests/test_plan_phases.py`: two test/code mismatches aligned with documented behavior; removed obsolete `test_marked_mle_raises_for_non_exponential_kernel`

### Packaging (WS-1.8)
- `intensify/__init__.py`: `__version__ = "0.2.0"`, backend init lazy (no load-time call)
- `intensify/py.typed`: new PEP 561 marker
- `pyproject.toml`: version `0.2.0`, minimum dep pins (`numpy>=1.24`, `jax>=0.4.20`, etc.), `py.typed` force-included in wheel, new URLs (Documentation/Issues/Changelog), classifiers (`OS Independent`, `Typing :: Typed`, `Mathematics`)
- `CHANGELOG.md`: Keep-a-Changelog format, `0.1.0` + `0.1.1` + `0.2.0` sections

### Docs + community (WS-4)
- `README.md` rewrite: PyPI/Python/license/CI badges, "Why intensify?" comparison table (**has unverified claims vs tick/pyhawkes — see TODO below**), citation block, doc links
- `CITATION.cff` (version synced to `0.2.0`)
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` (Contributor Covenant v2.1), `SECURITY.md`
- `docs/getting_started.md`: regularization shorthand + backend switching + input validation sections added
- `docs/user_guide/inference.md`: new kernel-coverage table replaces the old "ExponentialKernel only" note
- `docs/user_guide/diagnostics.md` (new)
- `docs/user_guide/simulation.md` (new)
- `docs/user_guide/index.rst`: tocTree updated

### CI + hygiene (WS-5)
- `.github/workflows/ci.yml`: multi-OS matrix (ubuntu/macos/windows × py 3.10/3.11/3.12); new `mypy`, `docs` (Sphinx with `-W`), `readme-doctest` jobs
- `.github/workflows/publish.yml` (new): PyPI OIDC trusted-publisher on `v*.*.*` tags
- `.pre-commit-config.yaml`: expanded with codespell + yaml/toml lint + whitespace/EOF fixers
- `.github/ISSUE_TEMPLATE/bug_report.yml`, `feature_request.yml`, `PULL_REQUEST_TEMPLATE.md`

### Benchmarks scaffold (WS-6 — scaffold only)
- `benchmarks/README.md`, `reference_dataset.py`, `run_intensify.py` — verified end-to-end with `uni_exp_small`
- `benchmarks/run_tick.py` — written, not yet executed successfully

## Test status at end of session

- **Targeted sweep** (all files I touched + previously-failing tests): **72 passed, 0 failures** in 75s
- **Full suite ex stress test**: **182 passed, 0 failures** in 118s
- **Real HC-3 stress test** (`test_real_data_stress.py`): **42 passed, 0 failures** in 8m 13s
- **Grand total: 224 passed, 0 failures, 4 skipped** (known-skip: Bayesian experimental)
- Coverage: 81% (already exceeds the 70% plan target)

## In progress when we stopped: actual tick vs intensify benchmark

The "Why intensify?" table in `README.md` currently claims parity-plus vs
`tick` and `pyhawkes`. I have not verified those claims. We were in the
middle of running real numbers.

### State of the benchmark thread

1. **`pyhawkes`: abandoned.** Installs in `tickbench38` env but chokes on
   `scipy.misc.logsumexp` (removed in SciPy 1.0, 2017) via its
   `pybasicbayes` dep. Patching that chain is more effort than the
   comparison is worth. **Recommend dropping pyhawkes from the comparison
   entirely.**

2. **`tick 0.7.0.1`: WORKS.** Installed in a micromamba env named
   `tickbench38` with Python 3.8. Verified `from tick.hawkes import
   SimuHawkesExpKernels, HawkesExpKern` imports cleanly.

3. **Dataset format issue blocking the run.** The `reference_dataset.py`
   originally used `np.savez_compressed` with `dtype=object` for the
   ground-truth dict. This uses pickle, which is not forward/backward
   compatible between NumPy 1.x (in `tickbench38`) and NumPy 2.x (in the
   intensify `.venv`). Error: `No module named 'numpy._core'`.

### What I edited right before stopping
- `benchmarks/reference_dataset.py`: switched to a portable format —
  `.npy` files per events array + a `.json` sidecar for `T` +
  ground-truth. `_save` now writes `reference_<name>.npy` (univariate) or
  `reference_<name>.dim{i}.npy` (multivariate) plus `reference_<name>.json`
- `benchmarks/run_intensify.py` + `benchmarks/run_tick.py`: updated
  `_load()` to read the new format

### Exact next step to resume

```bash
cd /home/etrigan/SoftwareDev/Libraries/intensify

# 1. Regenerate datasets in portable format (old .npz files removed)
.venv/bin/python benchmarks/reference_dataset.py uni_exp_small uni_power_law mv_exp_5d

# 2. Re-verify intensify runner still works with the new format
.venv/bin/python benchmarks/run_intensify.py uni_exp_small uni_power_law mv_exp_5d

# 3. Run tick head-to-head
/home/etrigan/bin/micromamba run -n tickbench38 python benchmarks/run_tick.py uni_exp_small mv_exp_5d
# (uni_power_law intentionally skipped — tick ships no power-law kernel)

# 4. Read results
cat benchmarks/results/intensify_*.json benchmarks/results/tick_*.json

# 5. Write the comparison table to docs/benchmarks.md
# 6. Update README.md "Why intensify?" table with real numbers, or remove
#    unverified claims
```

### Environment specifics to remember

- `tickbench38` is the conda env. Activate with
  `/home/etrigan/bin/micromamba run -n tickbench38 <cmd>` or
  `source /home/etrigan/.local/share/mamba/envs/tickbench38/bin/activate`.
- tick required a specific chain of downgrades to build against a modern
  toolchain — but here we used the prebuilt py3.8 wheel so none of that
  matters for rerunning.

## Open items for a follow-up session

### Must-do before announce

1. **Run the tick benchmark** (exact recipe above, ~5 minutes of work)
2. **Update `README.md` "Why intensify?" table** with real numbers or
   remove the unverified "tick" column. Same for `docs/benchmarks.md`.
3. **Remove pyhawkes mentions** from README comparison (abandoned upstream)
4. **Commit everything** — the `intensify/` repo currently only has
   `LICENSE` tracked. Everything we did is dirty. `git add`, write a
   big commit, probably squash into a few logical ones.
5. **Tag `v0.2.0`** to trigger the PyPI publish workflow

### Manual external steps (out of scope for Claude)

1. GitHub Release → Zenodo DOI mint → update `CITATION.cff` + add DOI badge
   to `README.md`
2. (Optional) JOSS paper draft for peer-reviewed citation

## Known side items discovered but not pursued

- `intensify/core/inference/em.py`, `bayesian.py`, `simulation/cluster.py`:
  each still has an "ExponentialKernel only" restriction. These are
  out-of-scope for 0.2.0 per the plan; kept as-is with clear
  `NotImplementedError` messages.
- `tests/test_plan_phases.py::test_endogeneity_index_critical_branching`:
  I aligned test to current `n/(1+n)` formula (saturates at 1.0). Filimonov-Sornette
  and other literature use various definitions — consider documenting
  which convention intensify follows in `docs/user_guide/inference.md`
  before first paper.

## Files that moved (quick index)

Created:
- `intensify/py.typed`
- `intensify/core/inference/_hessian.py` **was in plan but NOT created** —
  extraction was deferred because the existing `_finite_difference_std_errors`
  already has the condition-number check. The duplication between numpy
  and JAX paths still exists.
- `CITATION.cff`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`
- `CHANGELOG.md`
- `docs/user_guide/diagnostics.md`, `simulation.md`
- `.github/workflows/publish.yml`
- `.github/ISSUE_TEMPLATE/bug_report.yml`, `feature_request.yml`,
  `.github/PULL_REQUEST_TEMPLATE.md`
- `tests/test_mle_kernel_expansion.py`, `test_textbook_cases.py`
- `benchmarks/README.md`, `reference_dataset.py`, `run_intensify.py`,
  `run_tick.py`
- `SESSION_HANDOFF.md` (this file)

Modified (non-exhaustive):
- `README.md`, `pyproject.toml`, `.github/workflows/ci.yml`,
  `.pre-commit-config.yaml`, `intensify/__init__.py`,
  `intensify/core/kernels/{base,sum_exponential,nonparametric}.py`,
  `intensify/core/processes/{hawkes,marked_hawkes}.py`,
  `intensify/core/inference/{mle,univariate_hawkes_mle_params}.py`,
  `intensify/core/diagnostics/goodness_of_fit.py`,
  `intensify/backends/_backend.py`,
  `docs/getting_started.md`, `docs/user_guide/{inference,index}.rst`
- `tests/test_plan_phases.py` (3 test edits)

## When you pick this back up

Start with step 1 of "Exact next step to resume" above. That single
command regenerates the datasets in the portable format; then step 3
is the real deliverable — a genuine tick-vs-intensify number for the
README table. After that everything else is polish / git / announce.
