# Intensify: Comprehensive Issue Audit

Results from stress-testing the library against real HC-3 hippocampal spike recording data (37 neurons, 11–1171 spikes each, multielectrode array recordings at 20 kHz). Every issue below was discovered through hands-on use, not code review.

---

## 1. CRITICAL: Multivariate Hawkes Does Not Enforce Stationarity

**File:** `intensify/core/processes/hawkes.py:346-358`  
**File:** `intensify/core/inference/mle.py:340-433`

`MultivariateHawkes.project_params()` detects when the per-dimension branching ratio >= 1.0 but **only emits a warning** — it does not actually project the parameters back into the stationary regime. Compare with `UnivariateHawkes.project_params()` (line 141) which at least sets `alpha = 0.99`.

```python
# MultivariateHawkes.project_params() — warns but does nothing:
def project_params(self) -> None:
    for m in range(self.n_dims):
        total_norm = 0.0
        for k in range(self.n_dims):
            total_norm += self.kernel_matrix[m][k].l1_norm()
        if total_norm >= 1.0:
            warnings.warn(...)  # <-- that's it. no projection.
```

Worse, `project_params()` is **never called** during MLE optimization. The optimizer runs unconstrained, and L-BFGS-B bounds alone don't enforce the spectral radius constraint. In testing, a 5-dimensional fit on real neural data produced **spectral_radius = 3.98** — a wildly non-stationary result that a user would silently trust.

**Impact:** Users fitting multivariate Hawkes models on real data will routinely get non-stationary parameter estimates with no error or warning. This is the single biggest correctness issue in the library.

**Fix options:**
- Actually project alpha values in `project_params()` (scale the kernel matrix row so its L1 norm < 1)
- Call `project_params()` after each optimization step (projected gradient)
- Add a spectral radius constraint to the optimizer bounds
- At minimum, compute and store spectral radius on `FitResult` so users can check it
- `FitResult` for multivariate models never sets `branching_ratio_` — it should set it to the spectral radius of the kernel norm matrix

---

## 2. CRITICAL: `endogeneity_index_` Is Never Computed

**File:** `intensify/core/inference/__init__.py:46`  
**File:** `intensify/core/inference/mle.py` (all fit paths)

`FitResult` declares `endogeneity_index_: float | None = None` and `summary()` will display it if set, but **no inference path ever computes it**. The endogeneity index (fraction of events attributable to self-excitation vs. exogenous baseline) is one of the most important diagnostics for Hawkes models. It's advertised in the dataclass but always None.

**Fix:** Compute `endogeneity_index_ = branching_ratio / (1 + branching_ratio)` or the more precise estimator `1 - (mu * T / N)` and set it in every MLE fit path.

---

## 3. CRITICAL: Multivariate MLE Never Sets `branching_ratio_`

**File:** `intensify/core/inference/mle.py:416-433`

The `_fit_multivariate_numpy` method constructs a `FitResult` but never sets `branching_ratio_`. Compare with univariate (line 525), marked (line 238), and nonlinear (line 337) — all of which set it. For the multivariate case, the natural quantity is the spectral radius of the M×M kernel norm matrix.

---

## 4. API Inconsistency: MarkedHawkes.fit() Signature

**File:** `intensify/core/processes/marked_hawkes.py:199-219`

Every other process uses `model.fit(events, T=T)`. MarkedHawkes uses `model.fit(events, marks, T=T)` — marks as a separate positional argument. This is the single most common source of user error.

The natural first guess is `model.fit((events, marks), T=T)` because:
- The base class signature is `fit(self, events, T, method, **kwargs)`
- You think of (events, marks) as "the data"
- Internally, `MarkedHawkes.fit()` bundles them into a tuple anyway for the engine: `engine.fit(self, (events_np, marks_np), T)`

**Fix options (pick one):**
- Accept a tuple: `def fit(self, events_or_tuple, marks=None, ...)` — detect tuple input
- Keep separate args but make the base class signature clearly document that subclasses may extend it
- At minimum, add a clear error message when events is a tuple: `"Did you mean model.fit(events, marks, T=T)?"`

---

## 5. API Inconsistency: `plot_connectivity()` Requires Pre-extracted Matrix

**File:** `intensify/visualization/connectivity.py:11-17`

`plot_connectivity()` takes a raw `np.ndarray` matrix, not a `FitResult`. Every other visualization function (`plot_intensity`, `qq_plot`) takes a `FitResult`. So the user has to know to write:

```python
its.plot_connectivity(result.connectivity_matrix(), ax=ax)  # correct
its.plot_connectivity(result, ax=ax)  # wrong — no error, just garbage
```

The inconsistency is compounded because `result.connectivity_matrix()` itself is a method on `FitResult`, so clearly the two are meant to work together. But the user has to know the intermediate step.

**Fix:** Accept either `FitResult` or `np.ndarray` in `plot_connectivity()`:
```python
def plot_connectivity(matrix_or_result, ...):
    if isinstance(matrix_or_result, FitResult):
        matrix = matrix_or_result.connectivity_matrix()
    else:
        matrix = matrix_or_result
```

---

## 6. API Inconsistency: Regularization Requires Importing Internal Class

**File:** `intensify/core/regularizers.py`  
**File:** `intensify/__init__.py` (not exported)

To use regularization you must:
```python
from intensify.core.regularizers import L1
result = model.fit(events, T=T, method='mle', regularization=L1(strength=0.1))
```

`L1` and `ElasticNet` are not exported from the top-level `intensify` namespace. A user scanning the API or using autocomplete will never discover them. The natural expectation would be either:
- `regularization="l1"` (string shorthand)
- `its.L1(strength=0.1)` (top-level export)

**Fix:** Export `L1` and `ElasticNet` from `intensify/__init__.py`. Optionally support string shorthand `regularization="l1"` that creates a default `L1()`.

---

## 7. `FitResult.params` Stores Opaque Objects, Not Values

**File:** `intensify/core/processes/hawkes.py:132-133`

`get_params()` returns `{"mu": self.mu, "kernel": self.kernel}` — the kernel is an **object**, not a dict of numeric values. After fitting:

```python
result.params["mu"]        # works, returns float
result.params["kernel"]    # returns ExponentialKernel object
result.params["alpha"]     # KeyError
```

To get the fitted alpha, you need `result.process.kernel.alpha`. This is unintuitive and undocumented. The same applies to multivariate (`params["kernel_matrix"]` returns a list of lists of Kernel objects) and marked (`params["kernel"]` is an object, `params["mark_power"]` is a float).

**Impact:** Every user's first instinct after fitting is `result.params["alpha"]` or iterating `result.params.items()` to see numeric values. They get objects instead.

**Fix options:**
- Add a `flat_params()` method on `FitResult` that extracts all scalar values
- Change `get_params()` to return `{"mu": 0.5, "alpha": 0.35, "beta": 1.2}` (flat scalars)
- At minimum, document the current behavior prominently

---

## 8. Performance: Nonparametric Kernel Is Unusable at Moderate Scale

**File:** `intensify/core/inference/mle.py:788-813` (O(N²) path)

`NonparametricKernel` triggers the O(N²) general likelihood path. In testing:
- 100 spikes: ~420 seconds to fit
- 300 spikes: did not complete in 7 minutes (killed)

This makes `NonparametricKernel` effectively unusable for anything beyond toy examples. The `PerformanceWarning` threshold is set at 50,000 events (line 449-453), but the real pain starts at ~100 events for nonparametric due to the per-iteration cost.

**The O(N²) general likelihood itself** (`_general_likelihood_numpy`, line 788) uses **nested Python loops** — a double for-loop over all events. This is orders of magnitude slower than a vectorized implementation.

```python
# Current: nested Python loops
for i in range(n):
    for j in range(i):
        lag = t_i - events[j]
        intensity += kernel.evaluate(bt.array([lag]))[0]  # creates array per call!
```

Each inner iteration creates a new array `bt.array([lag])` just to evaluate the kernel at a single point. This is an enormous overhead.

**Fix:**
- Vectorize the O(N²) path: build the full lag matrix and evaluate the kernel once
- For nonparametric specifically, consider binned approximations
- Lower the performance warning threshold for non-recursive kernels
- The `_general_likelihood` function (JAX path, line 736) is already vectorized but only used when JAX is the backend

---

## 9. Performance: Hessian Standard Errors Are O(p²) Extra Evaluations

**File:** `intensify/core/inference/mle.py:486-507` (and duplicated at 393-414, 299-319)

Standard error computation uses finite-difference Hessian approximation: for p parameters, it evaluates the objective p additional times (each with an `approx_fprime` call that itself does p evaluations). For the multivariate case with M dimensions, p = M + M² + M² = M(1+2M) parameters. At M=5, that's 55 parameters and ~3000 extra likelihood evaluations.

The Hessian code is also **copy-pasted three times** across `_fit_numpy`, `_fit_multivariate_numpy`, and `_fit_nonlinear_numpy`. Any fix has to be applied in three places.

**Fix:**
- Extract Hessian computation to a shared helper
- For the JAX path, use `jax.hessian()` (already done in `_jax_hessian_std_errors`, but only for the univariate JAX path)
- Gate multivariate Hessian computation behind `n_params <= 24` (already done) but consider making this configurable

---

## 10. Hessian Regularization Is Hardcoded and Fragile

**File:** `intensify/core/inference/mle.py:501` (and duplicated lines)

```python
cov = np.linalg.inv(H + 1e-8 * np.eye(n_p))
```

The `1e-8` Tikhonov regularization is hardcoded. For ill-conditioned problems (common in multivariate Hawkes), this may not be enough and `np.linalg.inv` will produce garbage standard errors with no warning. There's no condition number check.

**Fix:** Check condition number of H, warn if ill-conditioned, use `np.linalg.solve` or pseudoinverse instead of `inv`.

---

## 11. Documentation: Getting Started Is a Stub

**File:** `docs/getting_started.md`

The entire getting started guide is 20 lines showing a single univariate simulation + fit. It doesn't cover:
- How to use real data (loading, formatting)
- How to access fitted parameters (the `result.process.kernel.alpha` pattern)
- Multivariate models
- Marked models
- Diagnostics (time-rescaling test, QQ plots)
- Visualization
- Choosing kernels
- What `FitResult` contains and how to use it

A user trying to use the library for the first time will hit every API inconsistency in this document within their first 30 minutes.

---

## 12. Documentation: Inference Guide Is 15 Lines

**File:** `docs/user_guide/inference.md`

The entire inference documentation is:
- One-line descriptions of mle/em/online/bayesian
- One code example showing regularization

Missing:
- How to interpret `FitResult` (params, log_likelihood, std_errors, branching_ratio_)
- How to access fitted parameter values (the kernel object issue)
- What `convergence_info` contains and when to worry
- When to use each method (MLE vs EM vs online)
- Performance characteristics (O(N) vs O(N²), when each path triggers)
- Multivariate workflow (passing list of event arrays, regularization, connectivity matrix)
- Marked Hawkes workflow (separate args for events and marks)
- How stationarity is (not) enforced
- Standard error interpretation

---

## 13. Documentation: Kernels Guide Is 6 Lines

**File:** `docs/user_guide/kernels.md`

The entire kernel documentation is a bullet list. Missing:
- Mathematical form of each kernel
- Parameter meanings and typical ranges
- How to choose between exponential and power-law
- Performance implications (recursive vs general path)
- `allow_signed=True` for inhibitory processes — what it does, when to use it
- `NonparametricKernel` — bin count selection, severe performance limitations
- `ApproxPowerLawKernel` — what `r` and `n_components` mean

---

## 14. No Examples or Tutorials Exist

**File:** `docs/getting_started.md:21` ("See the tutorials in the `tutorials/` directory")

The getting started guide references a `tutorials/` directory that **does not exist**. There are no Jupyter notebooks, no example scripts, no worked examples anywhere in the repository.

---

## 15. `plot_event_aligned_histogram` Required Args Not Obvious

**File:** `intensify/visualization/event_histograms.py:36-40`

```python
def plot_event_aligned_histogram(
    events: np.ndarray,
    reference_times: np.ndarray,  # required
    window: tuple[float, float],  # required
    ...
)
```

Unlike `plot_intensity(result)` which can derive everything from the FitResult, `plot_event_aligned_histogram` requires the user to separately supply `reference_times` and `window`. These are positional arguments with no defaults. A user calling `plot_event_aligned_histogram(events)` gets a confusing TypeError about missing positional args.

This is a documentation issue more than an API issue — the function signature is reasonable, but the user needs to know what reference_times means in context.

---

## 16. README Quickstart Has Errors

**File:** `README.md:21`

```python
model = its.Hawkes(kernel=its.ExponentialKernel(alpha=0.3, beta=1.5))
```

`Hawkes()` constructor requires `mu` as a positional argument, but the README omits it. This code will raise `TypeError: __init__() missing 1 required positional argument: 'mu'`.

Line 27: `result.plot_intensity()` — `FitResult` has `plot_diagnostics()` but no `plot_intensity()` method. The standalone function is `its.plot_intensity(result)`.

---

## 17. `FitResult.plot_diagnostics()` Is Fragile

**File:** `intensify/core/inference/__init__.py:98-123`

`plot_diagnostics()` requires both `process` and `events` to be set on the FitResult. While MLE sets these, the method gives a generic ValueError with no guidance if they're missing. It also unconditionally tries to plot a kernel subplot, which fails for Poisson processes (no kernel attribute).

The `qq_plot` call inside uses `time_rescaling_test` internally, which is O(N²) — this can silently hang on large datasets.

---

## 18. `connectivity_matrix()` and `significant_connections()` Error Messages Are Unhelpful

**File:** `intensify/core/inference/__init__.py:125-176`

Both methods check `isinstance(self.process, MultivariateHawkes)` but the error message says "requires a fitted MultivariateHawkes process" — it doesn't tell the user they called it on the wrong model type. If `process` is None (e.g., manually constructed FitResult), the error is even more confusing.

---

## 19. Goodness-of-Fit Diagnostics Are O(N²)

**File:** `intensify/core/diagnostics/goodness_of_fit.py:60-66`

`time_rescaling_test()` and `qq_plot()` both compute compensators using nested Python loops:

```python
for i in range(n):
    for j in range(i):
        lag = t_i - t_arr[j]
        comp += kernel.integrate(lag)
```

For exponential kernels this could be O(N) using the recursive form, but the diagnostic code doesn't use it. On 1000 events, this is ~500K kernel integrations via Python loops.

---

## 20. Time-Rescaling Test Uses Cumulative Intensity, Not Intervals

**File:** `intensify/core/diagnostics/goodness_of_fit.py:68-78`

The code computes τ_i = Λ(t_i) (cumulative intensity at each event time) and tests these against Exp(1). But the time-rescaling theorem says the **inter-compensator intervals** Λ(t_i) - Λ(t_{i-1}) should be Exp(1), not the cumulative values themselves. The cumulative values are an increasing sequence and can't be i.i.d. Exponential.

The comment on line 72 says "the τ_i should be i.i.d. Exponential(1)" which is incorrect as stated — it should say the **differences** of τ_i are Exp(1).

**Impact:** The KS test p-values from `time_rescaling_test()` are mathematically wrong.

---

## 21. Duplicate Code Between `time_rescaling_test()` and `qq_plot()`

**File:** `intensify/core/diagnostics/goodness_of_fit.py:54-66` vs `96-108`

The compensator computation (the slow O(N²) loop) is copy-pasted between `time_rescaling_test()` and `qq_plot()`. Same loop, same variables, same logic. Should be extracted to a shared `_compute_compensators(process, events)` helper.

---

## 22. Multivariate MLE Low-Data Warning Threshold Is Too Generous

**File:** `intensify/core/inference/mle.py:355-359`

```python
if M > 0 and (n_obs / (M * M)) < 100:
    warnings.warn("Event count may be low relative to M^2 ...")
```

For M=5, this requires 2500 total events before warning. But with M²=25 alpha parameters, M=5 mu values, and M²=25 beta values, that's 55 free parameters. Standard MLE guidance suggests 10-20 observations per parameter minimum, so the threshold should be closer to `n_obs / n_params < 20` rather than `n_obs / M² < 100`.

---

## 23. No Input Validation on Event Arrays

Event arrays are assumed sorted and non-negative throughout the library. No function validates this. Passing unsorted events silently produces wrong likelihoods (the recursive computation depends on sorted order). Passing events outside [0, T] silently produces wrong compensators.

**Fix:** Add a single validation check in the MLE `fit()` entry point:
```python
if not np.all(np.diff(events) >= 0):
    raise ValueError("events must be sorted in non-decreasing order")
if events[0] < 0 or events[-1] > T:
    raise ValueError("events must be in [0, T]")
```

---

## 24. `_general_likelihood_numpy` Creates Arrays in Inner Loop

**File:** `intensify/core/inference/mle.py:806-808`

```python
for j in range(i):
    lag = t_i - events[j]
    intensity += kernel.evaluate(bt.array([lag]))[0]  # array creation per iteration
```

Every single inner-loop iteration creates a new backend array `bt.array([lag])` to evaluate the kernel at one point. For N=1000, that's ~500K array allocations. The vectorized path (`_general_likelihood` on line 736) does this correctly with a single matrix operation.

---

## 25. No Convergence Diagnostics for Multivariate MLE

**File:** `intensify/core/inference/mle.py:416-433`

When `result_opt.success` is False (optimizer didn't converge), the multivariate fit silently returns whatever parameters the optimizer stopped at. There's no warning, no special flag. The user has to manually check `result.convergence_info["success"]`.

For univariate fits, this is less dangerous because the parameter space is small. For multivariate with 55+ parameters, non-convergence is common and the resulting parameters can be meaningless.

---

## 26. Backend Module-Level Initialization Is Fragile

**File:** `intensify/core/diagnostics/goodness_of_fit.py:10`  
**File:** `intensify/visualization/intensity.py:9`

Several modules call `bt = get_backend()` at module level. If the user changes the backend after import (`its.set_backend("numpy")`), these modules still hold the old backend reference. This is a latent bug that manifests as confusing type mismatches.

---

## 27. `MarkedHawkes` MLE Only Supports ExponentialKernel

**File:** `intensify/core/inference/mle.py:194-197`

```python
if not isinstance(process.kernel, ExponentialKernel) or process.kernel.allow_signed:
    raise NotImplementedError(
        "MarkedHawkes MLE currently supports ExponentialKernel without allow_signed."
    )
```

MarkedHawkes with PowerLawKernel, SumExponentialKernel, or NonparametricKernel cannot be fitted. The `log_likelihood()` method on MarkedHawkes handles general kernels (line 145), so the limitation is purely in the MLE parameter vectorization, not the math.

---

## 28. `NonlinearHawkes` MLE Only Supports ExponentialKernel

**File:** `intensify/core/inference/mle.py:251`

Same issue as above. `NonlinearHawkes` has a general `log_likelihood()` but MLE fitting only works with ExponentialKernel.

---

## Summary by Priority

### Must Fix (Correctness)
1. **Multivariate stationarity not enforced** — produces silently wrong results
2. **Time-rescaling test uses wrong quantity** — p-values are mathematically incorrect  
3. **README quickstart has errors** — first impression is broken code

### Should Fix (Reliability)
4. Multivariate `branching_ratio_` never set
5. `endogeneity_index_` never computed
6. No input validation on event arrays
7. No convergence warning for multivariate MLE
8. `project_params()` never called during optimization

### Should Fix (Usability)
9. `MarkedHawkes.fit()` signature differs from all other processes
10. `plot_connectivity()` doesn't accept `FitResult`
11. Regularizers not exported from top-level namespace
12. `FitResult.params` stores objects instead of scalar values
13. Tutorials directory doesn't exist despite being referenced

### Should Fix (Performance)
14. O(N²) numpy path uses Python loops + per-element array creation
15. Diagnostics are O(N²) even for recursive kernels
16. Hessian code is copy-pasted three times

### Should Fix (Documentation)
17. Getting started guide is a stub
18. Inference guide is 15 lines
19. Kernels guide is 6 lines
20. No worked examples for multivariate, marked, nonlinear, or diagnostic workflows
21. `FitResult` structure and parameter access pattern undocumented
22. Performance characteristics and scaling undocumented
