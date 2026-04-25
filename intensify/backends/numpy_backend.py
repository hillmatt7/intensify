"""NumPy backend implementation (fallback when JAX is unavailable)."""

from types import SimpleNamespace

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import scipy.optimize as opt
except ImportError:  # pragma: no cover
    opt = None

HAS_NUMPY = np is not None


def is_available() -> bool:
    return HAS_NUMPY


def _noop_random(name: str):
    def _f(*_a, **_k):
        raise RuntimeError(
            f"numpy backend random.{name} called but NumPy is not available."
        )

    return _f


if HAS_NUMPY:
    _random_ns = SimpleNamespace(
        PRNGKey=lambda seed: None,
        split=lambda key, num: [None] * num,
        uniform=lambda key=None, shape=None, low=0.0, high=1.0, size=None, dtype=None: (
            np.random.uniform(low, high, size if size is not None else shape).astype(dtype)
            if dtype
            else np.random.uniform(low, high, size if size is not None else shape)
        ),
        normal=lambda key=None, shape=None, loc=0.0, scale=1.0, size=None, dtype=None: (
            np.random.normal(loc, scale, size if size is not None else shape).astype(dtype)
            if dtype
            else np.random.normal(loc, scale, size if size is not None else shape)
        ),
        exponential=lambda key=None, shape=None, scale=1.0, size=None, dtype=None: (
            np.random.exponential(scale, size if size is not None else shape).astype(dtype)
            if dtype
            else np.random.exponential(scale, size if size is not None else shape)
        ),
        poisson=lambda key=None, lam=1.0, shape=None: np.random.poisson(lam=lam, size=shape),
    )
else:  # pragma: no cover
    _random_ns = SimpleNamespace(
        PRNGKey=_noop_random("PRNGKey"),
        split=_noop_random("split"),
        uniform=_noop_random("uniform"),
        normal=_noop_random("normal"),
        exponential=_noop_random("exponential"),
        poisson=_noop_random("poisson"),
    )

backend = SimpleNamespace(
    array=lambda *args, **kwargs: np.array(*args, **kwargs) if HAS_NUMPY else None,
    asarray=lambda *args, **kwargs: np.asarray(*args, **kwargs) if HAS_NUMPY else None,
    zeros=lambda *args, **kwargs: np.zeros(*args, **kwargs) if HAS_NUMPY else None,
    ones=lambda *args, **kwargs: np.ones(*args, **kwargs) if HAS_NUMPY else None,
    zeros_like=lambda *args, **kwargs: np.zeros_like(*args, **kwargs) if HAS_NUMPY else None,
    ones_like=lambda *args, **kwargs: np.ones_like(*args, **kwargs) if HAS_NUMPY else None,
    full=lambda *args, **kwargs: np.full(*args, **kwargs) if HAS_NUMPY else None,
    arange=lambda *args, **kwargs: np.arange(*args, **kwargs) if HAS_NUMPY else None,
    linspace=lambda *args, **kwargs: np.linspace(*args, **kwargs) if HAS_NUMPY else None,
    exp=np.exp if HAS_NUMPY else None,
    log=np.log if HAS_NUMPY else None,
    log1p=np.log1p if HAS_NUMPY else None,
    power=np.power if HAS_NUMPY else None,
    sqrt=np.sqrt if HAS_NUMPY else None,
    square=np.square if HAS_NUMPY else None,
    abs=np.abs if HAS_NUMPY else None,
    sign=np.sign if HAS_NUMPY else None,
    sin=np.sin if HAS_NUMPY else None,
    cos=np.cos if HAS_NUMPY else None,
    tan=np.tan if HAS_NUMPY else None,
    sum=np.sum if HAS_NUMPY else None,
    mean=np.mean if HAS_NUMPY else None,
    max=np.max if HAS_NUMPY else None,
    min=np.min if HAS_NUMPY else None,
    std=np.std if HAS_NUMPY else None,
    var=np.var if HAS_NUMPY else None,
    prod=np.prod if HAS_NUMPY else None,
    any=np.any if HAS_NUMPY else None,
    all=np.all if HAS_NUMPY else None,
    concatenate=np.concatenate if HAS_NUMPY else None,
    stack=np.stack if HAS_NUMPY else None,
    reshape=np.reshape if HAS_NUMPY else None,
    transpose=np.transpose if HAS_NUMPY else None,
    swapaxes=np.swapaxes if HAS_NUMPY else None,
    squeeze=np.squeeze if HAS_NUMPY else None,
    expand_dims=np.expand_dims if HAS_NUMPY else None,
    split=np.split if HAS_NUMPY else None,
    roll=np.roll if HAS_NUMPY else None,
    repeat=np.repeat if HAS_NUMPY else None,
    tile=np.tile if HAS_NUMPY else None,
    take=np.take if HAS_NUMPY else None,
    where=np.where if HAS_NUMPY else None,
    select=np.select if HAS_NUMPY else None,
    sort=np.sort if HAS_NUMPY else None,
    dot=np.dot if HAS_NUMPY else None,
    matmul=np.matmul if HAS_NUMPY else None,
    inner=np.inner if HAS_NUMPY else None,
    outer=np.outer if HAS_NUMPY else None,
    cross=np.cross if HAS_NUMPY else None,
    linalg=np.linalg if HAS_NUMPY else None,
    diff=np.diff if HAS_NUMPY else None,
    cumsum=np.cumsum if HAS_NUMPY else None,
    median=np.median if HAS_NUMPY else None,
    quantile=np.quantile if HAS_NUMPY else None,
    random=_random_ns,
    lax=SimpleNamespace(
        cond=lambda pred, true_fun, false_fun, operand: (
            true_fun(operand) if pred else false_fun(operand)
        ),
        scan=lambda f, init, xs, length=None: (init, []),
    ),
    jit=lambda f: f,
    grad=None,
    value_and_grad=None,
    vmap=None,
    pmap=None,
    dtype=np.dtype if HAS_NUMPY else None,
    finfo=np.finfo if HAS_NUMPY else None,
    iinfo=np.iinfo if HAS_NUMPY else None,
    isfinite=np.isfinite if HAS_NUMPY else None,
    isnan=np.isnan if HAS_NUMPY else None,
    isinf=np.isinf if HAS_NUMPY else None,
    allclose=np.allclose if HAS_NUMPY else None,
    array_equal=np.array_equal if HAS_NUMPY else None,
    scipy_optimize=opt,
    is_available=is_available,
)

__all__ = [
    "backend",
    "is_available",
]
