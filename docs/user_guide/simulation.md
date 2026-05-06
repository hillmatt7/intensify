# Simulation

intensify supports two simulation algorithms, each with different trade-offs.

## Ogata thinning (default)

The thinning algorithm tracks an adaptive upper bound $\bar\lambda$ on the
intensity and accepts candidate times with probability
$\lambda^*(t)/\bar\lambda$. It works for any conditional intensity as long
as an upper bound can be computed.

```python
import intensify as its

model = its.Hawkes(mu=0.5, kernel=its.ExponentialKernel(alpha=0.3, beta=1.5))
events = model.simulate(T=100.0, seed=42)
```

- Universal: works for any kernel and any nonlinear link function.
- Slightly conservative — spends time on rejected candidates near bursts.

## Cluster (branching) representation

Hawkes processes admit a Galton–Watson branching interpretation: immigrant
events from the baseline rate $\mu$ each spawn offspring whose arrival
times follow the kernel density $\phi(t)/\|\phi\|_1$.

```python
from intensify.core.simulation.cluster import cluster_simulate

events = cluster_simulate(model, T=100.0, seed=42)
```

- Faster when the branching ratio is well below 1.
- Exact finite-sample algorithm (no rejection overhead).
- Restricted to stationary processes ($\|\phi\|_1 < 1$).

## Reproducibility

All simulation entry points accept a `seed` argument. For nested
simulations (e.g., parametric bootstrap), pass different integer seeds or
derive them from a NumPy random generator.

```python
# Bootstrap: 200 independent realizations
from joblib import Parallel, delayed

runs = Parallel(n_jobs=-1)(
    delayed(model.simulate)(100.0, seed=i) for i in range(200)
)
```

## Stationarity and divergence

If $\|\phi\|_1 \ge 1$ the process is non-stationary and the expected event
count diverges in $T$. Thinning will simulate it regardless, but you'll
see event counts explode. The `MultivariateHawkes.project_params()` method
enforces stationarity after fitting; for manual construction you can check:

```python
model.kernel.is_stationary()     # univariate
result.branching_ratio_ < 1.0    # multivariate (spectral radius)
```
