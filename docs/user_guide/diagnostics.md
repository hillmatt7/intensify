# Diagnostics

After fitting a point process model you want to answer one question:
*did I pick the right model?* intensify ships three complementary tools.

## Time-rescaling theorem

If events are drawn from a point process with conditional intensity
$\lambda^*(t)$, the transformed times

$$\tau_i = \Lambda(t_i) - \Lambda(t_{i-1}) \quad \text{where} \quad \Lambda(t) = \int_0^t \lambda^*(s)\,ds$$

are independent $\mathrm{Exp}(1)$ variables. A Kolmogorov–Smirnov test
against $\mathrm{Exp}(1)$ gives a p-value for model adequacy.

```python
from intensify.core.diagnostics.goodness_of_fit import time_rescaling_test

ks_stat, p_value = time_rescaling_test(result)
print(f"KS stat={ks_stat:.4f}  p={p_value:.4f}")
```

**Important**: the test uses the *increments* $\Delta\tau_i$, not the
cumulative compensators — cumulative values are monotone-increasing and
cannot be i.i.d. This is a common bug in third-party implementations.
intensify fixes it.

The p-value is **anti-conservative** when parameters were estimated on the
same data — treat it as a model-selection tool, not a strict acceptance
test.

## QQ plot of residuals

Visual complement to the KS test:

```python
import matplotlib.pyplot as plt
from intensify.core.diagnostics.goodness_of_fit import qq_plot

qq_plot(result)
plt.show()
```

Systematic deviation from the diagonal identifies the misspecification:

- Points above the diagonal in the right tail → your kernel decays faster
  than the data; try a heavier tail (power-law).
- Points below the diagonal in the left tail → you over-predict short
  inter-arrival intervals; try a longer decay rate.

## Information criteria

Lower is better for both AIC and BIC:

```python
print(f"AIC={result.aic:.2f}  BIC={result.bic:.2f}")
```

Both are set automatically in `FitResult.__post_init__`. When comparing two
fits on the *same data*, BIC penalizes extra parameters more aggressively —
use it for nested-model selection. Use AIC for out-of-sample prediction.

## All-in-one

```python
result.plot_diagnostics()
```

Produces a 2×2 figure: intensity, QQ, kernel (or connectivity for
multivariate), and a text summary.

## Complexity

| Kernel family | Compensator cost | Notes |
|---|---|---|
| Exponential / SumExp / ApproxPowerLaw | `O(N)` | Uses recursive form |
| PowerLaw / Nonparametric / custom | `O(N^2)` | Pairwise integral |

For large datasets with non-recursive kernels, consider subsampling event
windows before running the diagnostics.
