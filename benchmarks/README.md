# Benchmarks

Reproducible comparisons between `intensify` and adjacent libraries.

## Goals

1. **Correctness parity** — recover known-true parameters on simulated data;
   compare to results from [tick][] and [pyhawkes][].
2. **Speed** — fit time on a fixed reference dataset, same kernel, same data.
3. **Scalability** — fit time vs `N` (events) and `M` (dimensions).

[tick]: https://github.com/X-DataInitiative/tick
[pyhawkes]: https://github.com/slinderman/pyhawkes

## Running

```bash
pip install -e ".[benchmark]"
python benchmarks/reference_dataset.py      # generates data/reference_*.npz
python benchmarks/run_intensify.py
python benchmarks/run_tick.py               # optional: requires tick
python benchmarks/run_pyhawkes.py           # optional: requires pyhawkes
python benchmarks/summarize.py              # prints a comparison table
```

Each runner writes results to `benchmarks/results/<lib>_<scenario>.json` with
wall-clock time, recovered parameters, and RMSE vs ground truth.

## Scenarios

| ID | Process | Kernel | N (typical) | Use case |
|---|---|---|---|---|
| `uni_exp_small` | Univariate Hawkes | Exponential | ~500 | Fast smoke check |
| `uni_exp_large` | Univariate Hawkes | Exponential | ~50,000 | Scalability |
| `uni_power_law` | Univariate Hawkes | Power-law | ~5,000 | Heavy-tailed memory |
| `mv_exp_5d` | 5-d Multivariate Hawkes | Exponential | ~2,000/dim | Connectivity |

All scenarios use seeded RNG (`jax.random.PRNGKey(42)`) so results are
byte-reproducible across runs.

## Reporting

Timings are the median of 5 runs on the same machine. Report both absolute
time and parameter-recovery RMSE — a faster fit that produces worse
parameters is not a win.

Publish updated numbers in `docs/benchmarks.md` when a release changes
either column.
