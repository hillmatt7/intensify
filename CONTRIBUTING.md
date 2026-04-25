# Contributing to intensify

Thank you for considering a contribution. This project aims to be a reliable,
well-tested building block for point-process research in labs and industry.

## Ground rules

- **Correctness first.** Every numerical change should ship with a test that
  asserts the quantity it affects, not just that the code runs.
- **No silent fallbacks.** If a code path can produce a wrong answer under
  certain conditions (non-stationary fit, degenerate data, optimizer stall),
  surface it as a `warning` or `raise`, never as a silent sentinel.
- **Backend-agnostic.** Anything under `intensify/core/` must work with
  both the JAX and NumPy backends. Use `from ...backends import get_backend`
  and call backend methods through the returned proxy.

## Development setup

```bash
git clone https://github.com/hillmatt7/intensify
cd intensify

# Create a virtualenv and install with dev extras
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs]"

pre-commit install
```

## Running checks

```bash
# Lint + format
hatch run lint:check

# Tests (with coverage)
pytest tests/

# Docs
hatch run docs:build
```

## Pull request checklist

- [ ] Tests cover the new or changed behavior.
- [ ] `pytest tests/` passes locally (no new warnings unless intentional).
- [ ] `ruff check intensify tests` is clean.
- [ ] Public API changes are documented in `CHANGELOG.md` under `## [Unreleased]`.
- [ ] If you changed math, a citation (paper or textbook chapter) appears in
      the docstring or in-line comment.

## Reporting bugs

Open an issue at
<https://github.com/hillmatt7/intensify/issues/new/choose>.
Please include a minimal reproducer, the observed vs expected behavior,
your Python version, `intensify.__version__`, and the active backend
(`intensify.get_backend_name()`).

## Security

See [`SECURITY.md`](SECURITY.md) for reporting security issues privately.

## Community

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By
participating you agree to uphold it.
