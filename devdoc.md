# Point Process Library — Design Document & Execution Plan

**Status:** Pre-development  
**Author:** Matt [LastName]  
**Version:** 0.2 — updated kernel dispatch architecture, Section 3.1 / 5 / 6.1 / 14  

---

## 1. Vision & Positioning

A modern, actively maintained Python point process library with deep Hawkes specialization and the only clean Python implementation of marked and inhibitory Hawkes variants.

### What This Is Not
- A general stochastic process library (that's `diffrax`, `stochastic`)
- A statistics kitchen-sink library (that's `statsmodels`)
- A research artifact or notebook (that's `pyhawkes`)

### What This Is
A production-grade, pip-installable, modern Python library for modeling event arrival times. Hawkes is the flagship. Poisson and Cox are included because they share the mathematical framework and serve as baselines. Everything beyond point processes is out of scope.

### Target Users
Researchers and practitioners working with self-exciting event data — anyone fitting, simulating, or diagnosing point process models.

---

## 2. Architecture

### 2.1 Layer Diagram

```
┌─────────────────────────────────────────────────────────┐
│                       Public API                         │
│                                                          │
│   Process classes · ingestion helpers · marked/multi    │
│   variants · connectivity · streaming/online estimator  │
│   · diagnostics · I/O                                   │
│                            │                             │
│   ┌────────────────────────▼──────────────────────────┐ │
│   │                  Point Process Core                 │ │
│   │                                                     │ │
│   │  ┌────────────┐  ┌──────────┐  ┌────────────────┐  │ │
│   │  │  Poisson   │  │   Cox    │  │    Hawkes      │  │ │
│   │  │            │  │          │  │                │  │ │
│   │  │ homogeneous│  │ log-Gauss│  │ - univariate   │  │ │
│   │  │ inhomogen. │  │ shot noise│  │ - multivariate │  │ │
│   │  └────────────┘  └──────────┘  │ - marked       │  │ │
│   │                                │ - nonlinear    │  │ │
│   │                                └────────────────┘  │ │
│   │                                                     │ │
│   │  Shared Infrastructure:                             │ │
│   │  kernels · inference · simulation · likelihood      │ │
│   │  diagnostics · JAX backend · visualization         │ │
│   └─────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────┘
```

### 2.2 Package Structure

```
pointprocess/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── base.py              # Abstract base classes
│   ├── kernels/
│   │   ├── __init__.py
│   │   ├── base.py          # Kernel ABC
│   │   ├── exponential.py
│   │   ├── sum_exponential.py
│   │   ├── power_law.py
│   │   ├── approx_power_law.py
│   │   └── nonparametric.py
│   ├── processes/
│   │   ├── __init__.py
│   │   ├── poisson.py
│   │   ├── cox.py
│   │   └── hawkes.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── mle.py           # JAX-based gradient MLE
│   │   ├── em.py            # EM algorithm
│   │   └── bayesian.py      # MCMC (Phase 2)
│   ├── simulation/
│   │   ├── __init__.py
│   │   ├── thinning.py      # Ogata thinning
│   │   └── cluster.py       # Cluster representation (branching)
│   └── diagnostics/
│       ├── __init__.py
│       ├── goodness_of_fit.py
│       └── residuals.py
├── ingestion/
│   ├── __init__.py
│   ├── dataframe.py         # pandas/polars I/O
│   └── streams.py           # Streaming / online ingestion
├── extensions/
│   ├── __init__.py
│   ├── marked.py            # Marked Hawkes
│   ├── online.py            # Streaming / online inference
│   ├── connectivity.py      # Adjacency matrix → connectivity graph
│   ├── inhibitory.py        # Nonlinear / inhibitory Hawkes
│   └── metrics.py           # Branching ratio, endogeneity index
├── visualization/
│   ├── __init__.py
│   ├── intensity.py         # plot_intensity()
│   ├── kernels.py           # plot_kernel()
│   └── connectivity.py      # plot_connectivity()
└── backends/
    ├── __init__.py
    ├── jax_backend.py
    └── numpy_backend.py     # Fallback for environments without JAX
```

---

## 3. Core Abstractions

### 3.1 The Kernel ABC

Every kernel implements this interface. JAX-compatible by design — `evaluate()` and `integrate()` must be differentiable via `jax.grad`.

```python
from abc import ABC, abstractmethod
import jax.numpy as jnp

class Kernel(ABC):
    """
    Abstract base class for Hawkes excitation kernels.
    
    All kernels must be:
    - Non-negative: phi(t) >= 0 for all t >= 0
    - Causal: phi(t) = 0 for t < 0
    - JAX-compatible for autodiff-based inference

    Computation path is selected automatically based on has_recursive_form():
    - True  → O(N) recursive likelihood via jax.lax.scan
    - False → O(N²) general likelihood via JAX autodiff

    Kernels with recursive form (ExponentialKernel, SumExponentialKernel,
    ApproxPowerLawKernel) override has_recursive_form() and
    recursive_state_update(). All other kernels use the general path
    automatically — no changes required when adding new kernels.
    """

    @abstractmethod
    def evaluate(self, t: jnp.ndarray) -> jnp.ndarray:
        """Evaluate kernel at time lags t. Shape: (n,) -> (n,)"""
        pass

    @abstractmethod
    def integrate(self, t: float) -> float:
        """Compute integral of kernel from 0 to t (compensator term)."""
        pass

    @abstractmethod
    def l1_norm(self) -> float:
        """Integral from 0 to infinity. Must be < 1 for stationarity."""
        pass

    def is_stationary(self) -> bool:
        return self.l1_norm() < 1.0

    # --- Recursive dispatch interface ---
    # Default: general O(N²) path. Fast kernels override both methods below.

    def has_recursive_form(self) -> bool:
        """
        Return True if this kernel admits O(N) recursive likelihood computation.
        Only exponential-family kernels can return True.
        Default is False — safe for any new kernel added to the library.
        """
        return False

    def recursive_state_update(self, state: jnp.ndarray, dt: float) -> jnp.ndarray:
        """
        Update the recursive sufficient statistic R given time elapsed dt
        since the last event. Used by _recursive_likelihood() in the
        inference engine when has_recursive_form() is True.

        For ExponentialKernel:
            R_i = exp(-β · dt) · (1 + R_{i-1})

        For SumExponentialKernel:
            R_i^k = exp(-β_k · dt) · (1 + R_{i-1}^k)  for each component k

        Raises NotImplementedError by default — only called when
        has_recursive_form() returns True.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} declared has_recursive_form()=True "
            f"but did not implement recursive_state_update()."
        )
```

### 3.2 The Process ABC

```python
class PointProcess(ABC):
    """Abstract base for all point processes."""

    @abstractmethod
    def simulate(self, T: float, seed: int = None) -> jnp.ndarray:
        """Generate event times on [0, T]."""
        pass

    @abstractmethod
    def intensity(self, t: float, history: jnp.ndarray) -> float:
        """Evaluate conditional intensity at time t given history."""
        pass

    @abstractmethod
    def log_likelihood(self, events: jnp.ndarray, T: float) -> float:
        """Compute log-likelihood of observed event sequence."""
        pass

    def fit(self, events, T: float = None, method: str = "mle"):
        """
        Fit process parameters to observed event data.
        
        Parameters
        ----------
        events : array-like or domain-specific data object
            Event timestamps. Accepts raw arrays, pandas Series,
            SpikeTrainData objects, or OrderBookStream objects.
        T : float, optional
            Observation window end time. Inferred if not provided.
        method : str
            Inference method: 'mle', 'em', 'bayesian'
        """
        from .inference import get_inference_engine
        engine = get_inference_engine(method)
        return engine.fit(self, events, T)
```

### 3.3 The Inference Engine Interface

Decouples inference method from process type. Any inference method works with any process.

```python
class InferenceEngine(ABC):
    
    @abstractmethod
    def fit(self, process: PointProcess, 
            events: jnp.ndarray, T: float) -> FitResult:
        pass

class FitResult:
    """Standardized container for all inference results."""
    params: dict
    log_likelihood: float
    aic: float
    bic: float
    std_errors: dict        # From Hessian at MLE
    convergence_info: dict
    
    def summary(self) -> str: ...
    def plot_diagnostics(self): ...
```

---

## 4. Process Specifications

### 4.1 Poisson Processes

**Homogeneous Poisson**
- Constant rate λ
- MLE: λ̂ = N(T)/T
- Serves as the null model for Hawkes — branching ratio test against Poisson

**Inhomogeneous Poisson**  
- Time-varying intensity λ(t)
- Parametric: log-linear in covariates
- Nonparametric: piecewise constant or spline-based

**Why include it:** Every Hawkes goodness-of-fit test involves comparing against a Poisson baseline. Having it in the same library makes this seamless.

### 4.2 Cox Processes

A Cox process (doubly stochastic Poisson) is driven by a latent random intensity process Λ(t). Two variants worth implementing:

**Log-Gaussian Cox Process (LGCP)**
- Λ(t) = exp(GP(t)) where GP is a Gaussian process
- Useful when event rates are driven by an unobserved smoothly-varying latent process
- Inference via MCMC or Laplace approximation

**Shot-Noise Cox Process**
- Intensity driven by a sum of exponentially decaying shots
- Closely related to Hawkes — useful for comparison

**Why include it:** Many practitioners need to distinguish between self-exciting dynamics (Hawkes) and externally driven dynamics (Cox). Having both in one library with shared diagnostics makes this comparison trivial.

### 4.3 Hawkes Processes

The flagship. Full specification:

**Univariate**
```
λ*(t) = μ + Σ_{t_i < t} φ(t - t_i)
```

**Multivariate (M dimensions)**
```
λ*_m(t) = μ_m + Σ_k Σ_{t_i^k < t} φ_mk(t - t_i^k)
```
Where φ_mk is the kernel from dimension k to dimension m. The M×M kernel matrix encodes the full excitation structure across dimensions.

**Marked**
```
λ*(t) = μ + Σ_{t_i < t} g(m_i) · φ(t - t_i)
```
Where m_i is the per-event mark and g(·) is a user-supplied mark-influence function.

**Nonlinear / Inhibitory**
```
λ*(t) = f(μ + Σ_{t_i < t} φ(t - t_i))
```
Where f is a nonlinear link function (sigmoid, softplus, ReLU). Allows negative contributions from inhibitory kernels. Standard Hawkes is the special case f = identity with non-negative kernels.

---

## 5. Kernel Specifications

### 5.0 Kernel Selection Guide

All kernels are first-class options. The user selects based on the mathematical properties of their data — not performance. The library handles computation path selection transparently via the dispatch pattern in Section 3.1.

**Two computation paths exist:**

| Path | Complexity | Kernels | When |
|---|---|---|---|
| Recursive | O(N) | Exponential, SumExponential, ApproxPowerLaw | Large N, real-time inference |
| General (JAX autodiff) | O(N²) | PowerLaw, Nonparametric, any custom kernel | Smaller N, maximum flexibility |

**Kernel selection by data characteristics:**

| Data characteristic | Recommended Kernel | Reason |
|---|---|---|
| Single-timescale, large N, real-time | `ExponentialKernel` | Markov, O(N) recursive |
| Multi-timescale, large N | `SumExponentialKernel` | Multiple decay rates, still O(N) |
| Long-memory decay, large N | `ApproxPowerLawKernel` | Long memory, O(N) approximation |
| Long-memory decay, small N | `PowerLawKernel` | Exact long memory, O(N²) acceptable |
| Unknown / non-parametric shape | `NonparametricKernel` | Data-driven shape, EM inference |
| Fast within-component decay | `ExponentialKernel` | Single sharp timescale |
| Multiple coupled timescales | `SumExponentialKernel` | Mixture of decay rates |
| Heavy-tailed coupling | `PowerLawKernel` | Heavy-tailed influence over long lags |
| Exploratory / unknown structure | `NonparametricKernel` | No assumption on kernel shape |

**The bridge kernel:** `ApproxPowerLawKernel` (Bacry-Muzy) approximates power-law decay using geometrically spaced exponential components. It gets the O(N) recursive path while capturing long-memory behavior — the right choice when dataset size makes `PowerLawKernel` intractable.

The library will emit a performance warning when `PowerLawKernel` or `NonparametricKernel` is used with N > 50,000 events, suggesting `ApproxPowerLawKernel` as an alternative. See Section 14 for edge case details.

---

### Exponential
```
φ(t) = α · β · exp(-β·t)
```
Parameters: α (jump size), β (decay rate)  
L1 norm: α (must be < 1 for stationarity)  
Advantage: Admits recursive computation — O(N) likelihood instead of O(N²)

### Sum-of-Exponentials
```
φ(t) = Σ_k α_k · β_k · exp(-β_k · t)
```
More flexible decay shape. Same recursive trick applies per component.

### Power-Law
```
φ(t) = α · (t + c)^(-(1+β))
```
Parameters: α (amplitude), c (offset, prevents singularity at t=0), β (tail exponent)  
Captures long-memory decay where event influence falls off slowly.  
**Edge case:** Power-law kernels may be non-stationary for heavy tails. Must check L1 norm numerically and warn user.

### Approximate Power-Law (Bacry-Muzy)
```
φ(t) = α · Σ_k w_k · exp(-β_k · t)
```
Approximates power-law with a sum of exponentials using geometric spacing of decay rates. Recovers recursive computation while approximating long-memory behavior.

### Nonparametric (Piecewise Constant)
```
φ(t) = Σ_k a_k · 1[t ∈ [τ_k, τ_{k+1})]
```
No parametric assumption on kernel shape. Estimated via EM.  
**Edge case:** Bin width selection significantly affects results. Must implement cross-validation or AIC-based bin selection.

---

## 6. Inference Specifications

### 6.1 MLE via JAX Autodiff

The technical centerpiece. The log-likelihood for a Hawkes process on [0,T]:

```
log L = Σ_i log λ*(t_i) - ∫_0^T λ*(t) dt
```

The compensator integral ∫λ*(t)dt has closed form for exponential kernels (via recursion) and requires numerical integration for others.

#### Dispatch Architecture

The inference engine selects the computation path based on the kernel's `has_recursive_form()` declaration. This is the central architectural decision that makes the library both fast for exponential-family kernels and general for everything else.

```python
class MLEInference(InferenceEngine):

    def fit(self, process, events, T,
            optimizer="adam", lr=1e-3, max_iter=5000):

        # Select computation path based on kernel capability
        if process.kernel.has_recursive_form():
            log_likelihood_fn = self._recursive_likelihood
        else:
            # Warn if N is large — O(N²) will be slow
            if len(events) > 50_000:
                warnings.warn(
                    f"{process.kernel.__class__.__name__} requires O(N²) "
                    f"computation. With {len(events):,} events this may be slow. "
                    f"Consider ApproxPowerLawKernel for long-memory behavior "
                    f"with O(N) performance.",
                    PerformanceWarning
                )
            log_likelihood_fn = self._general_likelihood

        @jit
        def neg_log_likelihood(params):
            process.set_params(params)
            return -log_likelihood_fn(process, events, T)

        # Gradient via autodiff — correct for ANY kernel automatically
        grad_fn = jit(grad(neg_log_likelihood))

        opt = optax.adam(lr)
        opt_state = opt.init(process.get_params())

        for i in range(max_iter):
            grads = grad_fn(process.get_params())
            updates, opt_state = opt.update(grads, opt_state)
            process.update_params(updates)
            # Project back to stationary manifold after every step
            process.project_params()

        return FitResult(...)

    def _recursive_likelihood(self, process, events, T):
        """
        O(N) likelihood using jax.lax.scan over the recursive state.
        Only called when kernel.has_recursive_form() is True.

        For ExponentialKernel, the sufficient statistic R_i satisfies:
            R_i = exp(-β · (t_i - t_{i-1})) · (1 + R_{i-1})

        This is scanned over events — JIT-compiled, as fast as C.
        The kernel provides its own state update via recursive_state_update().
        """
        dts = jnp.diff(events, prepend=0.0)

        def scan_step(state, dt):
            new_state = process.kernel.recursive_state_update(state, dt)
            intensity_contrib = process.mu + new_state
            return new_state, intensity_contrib

        _, intensities = jax.lax.scan(scan_step, 0.0, dts)
        log_intensity_sum = jnp.sum(jnp.log(intensities))
        compensator = process.mu * T + process.kernel.l1_norm() * len(events)
        return log_intensity_sum - compensator

    def _general_likelihood(self, process, events, T):
        """
        O(N²) likelihood via full pairwise kernel evaluation.
        Works for any kernel — JAX autodiff differentiates through it.
        This is the fallback path for non-recursive kernels.
        """
        n = len(events)
        # Build (n, n) matrix of time lags — upper triangle is causal
        lags = events[:, None] - events[None, :]  # (n, n)
        causal_mask = lags > 0
        kernel_matrix = jnp.where(causal_mask,
                                   process.kernel.evaluate(lags),
                                   0.0)
        intensities = process.mu + kernel_matrix.sum(axis=1)
        log_intensity_sum = jnp.sum(jnp.log(intensities))
        compensator = process.mu * T + jnp.sum(
            process.kernel.integrate(T - events)
        )
        return log_intensity_sum - compensator
```

**Key advantage over tick:** tick required kernel-specific C++ implementations for gradient computation. JAX autodiff gives correct gradients for any kernel automatically — adding a new kernel requires zero changes to the inference engine. The dispatch pattern means new kernels automatically get the general path, and can opt into the fast path by implementing `has_recursive_form()` and `recursive_state_update()`.

### 6.2 EM Algorithm

Exploits the branching structure of Hawkes processes. Each event is either:
- An immigrant (from background rate μ)
- An offspring (triggered by a prior event via kernel φ)

The E-step computes the probability p_ij that event j was triggered by event i. The M-step updates parameters given these soft assignments. More stable than gradient MLE for nonparametric kernels.

**Edge cases:**
- EM convergence is slow for high-dimensional multivariate case — implement SQUAREM acceleration
- Degenerate assignments (p_ij → 0 or 1) cause numerical issues — clip probabilities
- Empty bins in nonparametric kernel: regularize with small prior

### 6.3 Recursive Computation for Exponential Kernels

For exponential kernels only, the likelihood can be computed in O(N) instead of O(N²) via the recursion:

```
R_i = Σ_{j<i} β · exp(-β(t_i - t_j)) = exp(-β(t_i - t_{i-1})) · (1 + R_{i-1})
```

This must be implemented as a special case for exponential and sum-of-exponential kernels. The O(N²) general computation becomes prohibitive above ~10^5 events — which is well within normal for high-frequency financial data.

**Implementation note:** Use `jax.lax.scan` for the recursion — this JIT-compiles the loop and is as fast as C for this pattern.

---

## 7. Simulation

### 7.1 Ogata Thinning Algorithm

The standard simulation method. Works for any bounded intensity.

```
Algorithm:
1. Generate candidate event from Poisson process with rate λ_upper bound
2. Accept with probability λ*(t_candidate) / λ_upper
3. If accepted: add to event sequence, update intensity upper bound
4. Repeat until T is reached
```

**Edge cases:**
- Upper bound selection: too tight causes many proposals (slow), too loose causes many rejections (also slow). Use adaptive upper bound that tracks recent intensity.
- Multivariate: must attribute each accepted event to a dimension based on relative intensities
- Power-law kernels: intensity can spike very high immediately after an event — upper bound must account for this

### 7.2 Cluster/Branching Simulation

Alternative to thinning, exploits the Galton-Watson tree structure of Hawkes. More efficient for high branching ratio (near-critical) processes.

```
Algorithm:
1. Generate immigrants from Poisson(μ)
2. For each immigrant, generate offspring from Poisson(||φ||_1)
3. Recursively generate offspring of offspring
4. Collect all events, sort by time
```

**Advantage:** Exact, no rejection needed, gives you the branching structure for free (useful for diagnostics)  
**Disadvantage:** Memory intensive for near-critical processes (branching ratio → 1)

---

## 8. Extension Module Specifications

### 8.1 Ingestion and Marked / Multivariate Extensions

**Data ingestion**
```python
from pointprocess.ingestion import from_dataframe, EventStream

# From pandas
model.fit(from_dataframe(df, time_col="timestamp", mark_col="size"))

# From raw arrays
model.fit(event_times, marks=event_marks)

# Streaming
stream = EventStream(buffer_size=10000)
stream.push(new_events)
online_estimator.update(stream)
```

**Marked Hawkes**
The mark influence function g(m) is user-specifiable:
```python
# Linear mark influence
model = MarkedHawkes(kernel=Exponential(), mark_influence="linear")

# Custom mark influence
model = MarkedHawkes(kernel=Exponential(),
                     mark_influence=lambda m: jnp.log1p(m))
```

**Edge case:** Marks must be normalized or the kernel amplitude becomes unidentifiable. Provide automatic normalization with a warning.

**Branching ratio and endogeneity**
```python
result = model.fit(events, T)
result.branching_ratio_      # n = ||φ||_1, fraction of endogenous events
result.endogeneity_index_    # Hawkes (2014) endogeneity measure
result.plot_intensity()      # Visualize fitted intensity over time
```

**Online/Streaming Estimator**
Recursive parameter updates as new events arrive. Based on stochastic gradient descent on the streaming log-likelihood. Approximate — trades statistical efficiency for real-time capability.

```python
online = OnlineEstimator(kernel=Exponential(),
                          lr=0.01,
                          window=10000)  # events in memory

for event in live_feed:
    online.update(event)
    params = online.current_params()
```

**Edge cases:**
- Concept drift: old events should decay in influence. Implement exponential forgetting factor.
- Cold start: online estimator needs warm-up period. Return uncertainty estimates during warm-up.
- Clock synchronization: high-resolution event timestamps need float64 precision throughout.

### 8.2 Connectivity Inference and Inhibitory Variants

**Event-sequence data object**
```python
from pointprocess.extensions import EventSequenceData

# From raw arrays
data = EventSequenceData(
    event_times=[array_dim1, array_dim2, ...],
    dim_ids=["dim_001", "dim_002", ...],
    observation_window=300.0,  # seconds
)

# From NWB file
data = EventSequenceData.from_nwb("recording.nwb")
```

**Connectivity inference**
```python
model = MultivariateHawkes(kernel=Exponential(), n_dims=len(data.dim_ids))
result = model.fit(data)

# Connectivity matrix — this IS the adjacency matrix
conn = result.connectivity_matrix_   # shape: (n_dims, n_dims)
conn_significant = result.significant_connections(alpha=0.05)

# Visualize
result.plot_connectivity(
    dim_ids=data.dim_ids,
    threshold=0.05,
    layout="circular"
)
```

**Inhibitory extension**
```python
from pointprocess.extensions import InhibitoryHawkes

model = InhibitoryHawkes(
    excitatory_kernel=Exponential(),
    inhibitory_kernel=Exponential(),
    link_function="softplus"  # or "sigmoid", "relu"
)
model.fit(data)
```

**Edge cases:**
- Refractory period: when modeling sources with hard refractoriness, intensity must go to zero within the refractory window. Add a refractory parameter with sensible default.
- Contamination: misattribution between dimensions inflates cross-correlations. Consider a contamination-robust fitting option.
- Non-stationarity across the observation window: baseline rates often drift. Add a stationarity test as a pre-fit diagnostic.
- High-dimensional: 100+ simultaneous dimensions makes full multivariate Hawkes intractable. Need a sparse connectivity prior (L1 regularization on adjacency matrix).

**Diagnostics**
```python
# Time-rescaling theorem — standard GoF for point processes
result.time_rescaling_test()    # Returns KS statistic and p-value

# Inter-event interval analysis
result.plot_iei_distribution()  # Observed vs. model-predicted IEI

# Stimulus-locked event histogram
result.plot_event_histogram(reference_times=ref_array, window=(-0.1, 0.5))
```

---

## 9. Goodness-of-Fit Framework

### Time-Rescaling Theorem
If the model is correct, the rescaled times τ_i = ∫_0^{t_i} λ*(s)ds should be i.i.d. Exponential(1). Test via:
- KS test against Exponential(1)
- QQ plot of rescaled inter-event times against Uniform(0,1)

This is the standard goodness-of-fit diagnostic for point process models.

### AIC / BIC
Standard model comparison. Automatically computed for all fitted models.

### Residual Intensity Plot
Plot λ*(t) over the observation window with actual events marked. Should look like events cluster where intensity is high.

---

## 10. Visualization Standards

Every plot function:
- Returns a `matplotlib` Figure object (not plt.show() — don't hijack the user's environment)
- Accepts optional `ax` argument to plot into existing axes
- Has sensible defaults but is fully customizable
- Documented with example output in the docs

```python
# All of these should just work
result.plot_intensity()
result.plot_intensity(ax=my_ax, color="steelblue", figsize=(12,4))
result.plot_kernel()
result.plot_kernel(log_scale=True)  # Essential for power-law
result.plot_connectivity()
result.plot_diagnostics()           # Grid of all relevant plots
```

---

## 11. Backend System

### JAX vs NumPy Fallback

JAX is the primary backend. NumPy fallback exists for environments where JAX is unavailable (certain HPC clusters, Windows without GPU).

```python
import pointprocess as pp

# Default: JAX if available, numpy otherwise
pp.set_backend("jax")     # Force JAX
pp.set_backend("numpy")   # Force numpy (slower, no GPU)
pp.get_backend()          # Query current backend
```

**Edge cases:**
- JAX default uses 32-bit floats. Financial timestamps need 64-bit. Must set `jax.config.update("jax_enable_x64", True)` automatically when float64 data is detected.
- JAX traces through Python control flow — branching logic in kernels must use `jax.lax.cond` not Python `if`. Abstract this behind kernel interface so users never see it.
- GPU memory: very long event sequences may OOM on GPU. Implement automatic CPU fallback with a warning above a configurable threshold.

---

## 12. API Design Principles

### Principle 1: Composability
Kernel choice and inference method are always independent:
```python
# All of these are valid
MultivariateHawkes(kernel=Exponential()).fit(data, method="mle")
MultivariateHawkes(kernel=PowerLaw()).fit(data, method="em")
MultivariateHawkes(kernel=Nonparametric(n_bins=20)).fit(data, method="em")
```

### Principle 2: Progressive Disclosure
Simple things should be one line. Complex things should be possible.
```python
# Simple — works immediately
import pointprocess as pp
model = pp.Hawkes()
model.fit(event_times)
model.plot_intensity()

# Advanced — full control
model = pp.MultivariateHawkes(
    n_dims=10,
    kernel=pp.kernels.SumExponential(n_components=3),
    baseline="inhomogeneous",
    regularization=pp.regularizers.L1(alpha=0.01)
)
result = model.fit(
    events, T=3600.0,
    method="mle",
    optimizer=optax.adam(1e-3),
    max_iter=10000,
    init_params="random",
    n_restarts=5
)
```

### Principle 3: Scikit-learn Compatible Where It Makes Sense
`.fit()`, `.predict()`, parameter naming conventions. Not forced — point processes don't map cleanly to supervised learning — but where the analogy holds, follow sklearn conventions.

### Principle 4: Fail Loudly on Physical Constraints
Don't silently return non-stationary fits. Warn loudly when:
- Branching ratio ≥ 1 (non-stationary process)
- Negative intensity values (nonlinear models)
- Refractory period violations in fitted model
- Poor convergence (gradient norm above threshold)

---

## 13. Execution Plan

### Phase 0 — Foundation (Weeks 1–2)
**Goal:** Skeleton that compiles, installs, and runs a basic exponential Hawkes.

- [ ] Repository setup: pyproject.toml, CI/CD (GitHub Actions), pre-commit hooks
- [ ] Abstract base classes: `Kernel`, `PointProcess`, `InferenceEngine`, `FitResult`
- [ ] `ExponentialKernel` — first concrete kernel
- [ ] `UnivariateHawkes` — univariate process
- [ ] `HomogeneousPoisson` — baseline process
- [ ] `MLEInference` — JAX-based MLE with optax
- [ ] `OgataThinning` — simulation
- [ ] Basic `plot_intensity()` and `plot_kernel()`
- [ ] Minimal test suite
- [ ] README with installation instructions and one working example

**Release:** v0.1.0-alpha — don't publicize, share with 1-2 people for feedback

---

### Phase 1 — Core Complete (Weeks 3–6)
**Goal:** Feature-complete core that surpasses all existing libraries.

- [ ] `SumExponentialKernel`
- [ ] `PowerLawKernel` with stationarity warning
- [ ] `ApproxPowerLawKernel` (Bacry-Muzy)
- [ ] `NonparametricKernel` with bin selection
- [ ] `MultivariateHawkes`
- [ ] `EMInference` algorithm
- [ ] Recursive O(N) computation for exponential kernels
- [ ] `CoxProcess` (log-Gaussian, shot-noise)
- [ ] Full goodness-of-fit suite: time-rescaling, KS test, QQ plots
- [ ] AIC/BIC on all models
- [ ] NumPy fallback backend
- [ ] 80%+ test coverage
- [ ] Full API documentation (Sphinx + autodoc)
- [ ] Jupyter notebook tutorials: one per process type

**Release:** v0.1.0 — announce on Reddit (r/MachineLearning, r/algotrading), HackerNews

---

### Phase 2 — Data ingestion, marked/multivariate features, diagnostics (Weeks 7–12)
**Goal:** Round out ingestion, multivariate inference, and diagnostic surface.

**Ingestion + marked/multivariate:**
- [ ] `from_dataframe()` and `from_polars()` ingestion
- [ ] `MarkedHawkes` with configurable mark influence
- [ ] `BranchingRatio` and `EndogeneityIndex` metrics
- [ ] `OnlineEstimator` streaming inference
- [ ] `EventStream` data object
- [ ] Tutorial: fitting on a real event-stream dataset

**Connectivity + diagnostics:**
- [ ] `EventSequenceData` object
- [ ] NWB file reader
- [ ] `ConnectivityInference` with significance testing
- [ ] `plot_connectivity()` graph visualization
- [ ] `InhibitoryHawkes` (sigmoid link function)
- [ ] Time-rescaling diagnostic
- [ ] ISI distribution analysis
- [ ] PSTH overlay
- [ ] Tutorial: fitting on a public event-sequence dataset

**Release:** v0.2.0

---

### Phase 3 — Novel Contributions (Weeks 13–20)
**Goal:** Features that appear nowhere else. Citation potential.

- [ ] `BayesianInference` via MCMC (NumPyro backend)
- [ ] Sparse multivariate Hawkes (L1 on adjacency, scales to 100+ dimensions)
- [ ] Inhibitory extension with ReLU and softplus link functions (full paper writeup)
- [ ] GPU acceleration benchmarks vs. tick (where tick installs)
- [ ] arXiv preprint: "A Modern Python Library for Point Process Modeling"

**Release:** v0.3.0

---

## 14. Nuances & Edge Cases Master List

### Mathematical
- **Stationarity:** Branching ratio must be < 1. Enforce via parameter projection during optimization. Never return a non-stationary fit silently.
- **Identifiability:** In multivariate models, baseline rate μ and kernel amplitudes are not separately identified without sufficient data. Warn when N/M² < 100 (rule of thumb).
- **Power-law singularity:** φ(0) = ∞ for power-law kernels. The offset parameter c prevents this. Never allow c = 0.
- **Near-critical regime:** Branching ratio → 1 causes extremely long simulation times (supercritical cascade). Cap simulation depth and warn.
- **Zero events:** Some dimensions in a multivariate model may observe no events. Handle gracefully — don't divide by zero in MLE.

### Numerical
- **O(N²) performance warning:** When a non-recursive kernel (PowerLawKernel, NonparametricKernel, or any custom kernel where `has_recursive_form()` returns False) is used with N > 50,000 events, emit a `PerformanceWarning` at fit time. The warning message must name the specific kernel, state the event count, explain the O(N²) complexity, and suggest `ApproxPowerLawKernel` by name as the bridge option for long-memory behavior with O(N) performance. The threshold of 50,000 is configurable via `pp.config.set("recursive_warning_threshold", n)`. Never silently degrade — always warn explicitly.
- **Float64 precision:** Financial timestamps in nanoseconds × large event counts can overflow float32. Enforce float64 throughout financial pipelines.
- **Log-sum-exp trick:** Log-likelihood computation involves log(Σ exp(...)). Must use numerically stable implementation.
- **Compensator integral:** For non-exponential kernels, numerical integration of ∫λ*(t)dt accumulates error. Use adaptive quadrature (scipy.integrate.quad) as reference, fast approximation for optimization.
- **Hessian computation:** Standard errors require the Hessian of the log-likelihood. Use JAX's `jax.hessian` — exact, not finite differences.

### Software Engineering
- **JAX random state:** JAX uses explicit PRNG keys, not global state. All stochastic functions must accept an explicit `seed` or `key` parameter.
- **JIT compilation:** First call to a JIT-compiled function is slow (compilation). Warn users about this or pre-compile common configurations.
- **Memory:** Multivariate Hawkes with M dimensions and N events stores an M×N influence matrix in the EM algorithm. Memory is O(M×N). Warn when this exceeds 1GB.
- **Thread safety:** Multiple simultaneous fits should be safe. Don't use global mutable state.
- **Serialization:** Models must be serializable to disk (`pickle`, or preferably a custom JSON-compatible format for portability).

### Data
- **Timestamp formats:** Accept Python datetime, numpy datetime64, pandas Timestamp, and raw float seconds. Convert internally to float64 seconds from epoch.
- **Duplicate timestamps:** Real data occasionally has duplicate event times (same millisecond). Don't crash — add small jitter or handle as simultaneous events.
- **Unsorted events:** Always sort input events. Warn if input is unsorted (suggests data pipeline issue).
- **Empty input:** Graceful failure with informative error for zero-length event arrays.
- **Observation window T:** If not specified, infer as max(event_times). This is subtly wrong if the process was observed beyond the last event — offer a warning.

---

## 15. Testing Strategy

### Unit Tests
- Every kernel: evaluate, integrate, l1_norm, gradient via JAX autodiff
- Every process: simulate (distributional tests), log_likelihood (against known analytical results)
- Every inference method: recover known parameters from simulated data within confidence interval

### Integration Tests
- Full fit-simulate-refit cycle: simulate from known params, fit, check recovery
- Extension round-trips: ingest data object, fit, extract derived metrics

### Benchmarks (not unit tests, run separately)
- Simulation speed vs. event count
- Inference speed vs. event count and dimension
- Memory usage for multivariate models
- Comparison against tick (exponential kernel, where tick installs)

### Statistical Tests
- Branching ratio recovery: simulate at known branching ratio, fit, check
- Goodness-of-fit calibration: time-rescaling test should have uniform p-values under correct model
- Multivariate connectivity recovery: simulate with known adjacency, fit, check adjacency recovery

---

## 16. Documentation Plan

### Docs Structure
```
docs/
├── getting_started.md        # Install + 5-minute example
├── user_guide/
│   ├── point_processes.md    # Mathematical background
│   ├── kernels.md
│   ├── inference.md
│   ├── simulation.md
│   └── diagnostics.md
├── api_reference/            # Auto-generated from docstrings
└── tutorials/                # Jupyter notebooks
    ├── 01_basic_hawkes.ipynb
    └── 02_multivariate.ipynb
```

### Mathematical Exposition Standard
Every process and kernel page includes:
1. Formal definition (LaTeX rendered)
2. Physical interpretation
3. When to use this vs. alternatives
4. Parameter guidance (typical ranges, interpretation)
5. Working code example

This level of mathematical documentation does not exist in any current Python point process library. It is a significant differentiator.

---

## 17. Naming

Library name requirements:
- Short, pip-installable (no conflicts on PyPI)
- Not domain-specific (not "spikehawkes", not "orderflow")
- Mathematically evocative or neutral
- Professional

Candidates to evaluate:
- `pprocesses` — clean, descriptive
- `hawkespp` — Hawkes-forward but general
- `intensit` — references intensity functions
- `ppoint` — minimal
- `eventflow` — accessible, domain-neutral

**Decision pending.**

---

*Document version 0.2 — living document, update as design evolves.*