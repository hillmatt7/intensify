"""JAX backend implementation."""

from types import SimpleNamespace

# Attempt to import JAX
try:
    import jax
    import jax.lax
    import jax.numpy as jnp
    import jax.random as random
    from jax import config as jax_config
    from jax import grad, jit, pmap, value_and_grad, vmap
    HAS_JAX = True
except ImportError as e:  # pragma: no cover
    HAS_JAX = False
    jnp = None
    jax = None
    jax_config = None
    grad = None
    jit = None
    random = None
    vmap = None
    pmap = None
    IMPORT_ERROR = e


def enable_x64():
    """Enable 64-bit precision in JAX."""
    if HAS_JAX:
        jax_config.update("jax_enable_x64", True)


def is_available():
    return HAS_JAX


backend = SimpleNamespace(
    # Array types and constructors
    array=lambda *args, **kwargs: jnp.array(*args, **kwargs) if HAS_JAX else None,
    asarray=lambda *args, **kwargs: jnp.asarray(*args, **kwargs) if HAS_JAX else None,
    zeros=lambda *args, **kwargs: jnp.zeros(*args, **kwargs) if HAS_JAX else None,
    ones=lambda *args, **kwargs: jnp.ones(*args, **kwargs) if HAS_JAX else None,
    zeros_like=lambda *args, **kwargs: jnp.zeros_like(*args, **kwargs) if HAS_JAX else None,
    ones_like=lambda *args, **kwargs: jnp.ones_like(*args, **kwargs) if HAS_JAX else None,
    full=lambda *args, **kwargs: jnp.full(*args, **kwargs) if HAS_JAX else None,
    arange=lambda *args, **kwargs: jnp.arange(*args, **kwargs) if HAS_JAX else None,
    linspace=lambda *args, **kwargs: jnp.linspace(*args, **kwargs) if HAS_JAX else None,
    # Basic math ops
    exp=jnp.exp if HAS_JAX else None,
    log=jnp.log if HAS_JAX else None,
    log1p=jnp.log1p if HAS_JAX else None,
    sqrt=jnp.sqrt if HAS_JAX else None,
    square=jnp.square if HAS_JAX else None,
    power=jnp.power if HAS_JAX else None,
    abs=jnp.abs if HAS_JAX else None,
    sign=jnp.sign if HAS_JAX else None,
    sin=jnp.sin if HAS_JAX else None,
    cos=jnp.cos if HAS_JAX else None,
    tan=jnp.tan if HAS_JAX else None,
    # Reduction ops
    sum=jnp.sum if HAS_JAX else None,
    mean=jnp.mean if HAS_JAX else None,
    max=jnp.max if HAS_JAX else None,
    min=jnp.min if HAS_JAX else None,
    std=jnp.std if HAS_JAX else None,
    var=jnp.var if HAS_JAX else None,
    prod=jnp.prod if HAS_JAX else None,
    any=jnp.any if HAS_JAX else None,
    all=jnp.all if HAS_JAX else None,
    # Array manipulation
    concatenate=jnp.concatenate if HAS_JAX else None,
    stack=jnp.stack if HAS_JAX else None,
    reshape=jnp.reshape if HAS_JAX else None,
    transpose=jnp.transpose if HAS_JAX else None,
    swapaxes=jnp.swapaxes if HAS_JAX else None,
    squeeze=jnp.squeeze if HAS_JAX else None,
    expand_dims=jnp.expand_dims if HAS_JAX else None,
    split=jnp.split if HAS_JAX else None,
    roll=jnp.roll if HAS_JAX else None,
    repeat=jnp.repeat if HAS_JAX else None,
    tile=jnp.tile if HAS_JAX else None,
    # Indexing and slicing
    take=jnp.take if HAS_JAX else None,
    where=jnp.where if HAS_JAX else None,
    select=jnp.select if HAS_JAX else None,
    # Linear algebra
    dot=jnp.dot if HAS_JAX else None,
    matmul=jnp.matmul if HAS_JAX else None,
    inner=jnp.inner if HAS_JAX else None,
    outer=jnp.outer if HAS_JAX else None,
    cross=jnp.cross if HAS_JAX else None,
    linalg=jax.numpy.linalg if HAS_JAX else None,
    # Statistics
    median=jnp.median if HAS_JAX else None,
    quantile=jnp.quantile if HAS_JAX else None,
    # Array operations
    diff=jnp.diff if HAS_JAX else None,
    cumsum=jnp.cumsum if HAS_JAX else None,
    # Random
    random=random if HAS_JAX else None,
    PRNGKey=jax.random.PRNGKey if HAS_JAX else None,
    # Control flow and transformations
    lax=jax.lax if HAS_JAX else None,
    jit=jit if HAS_JAX else None,
    grad=grad if HAS_JAX else None,
    value_and_grad=value_and_grad if HAS_JAX else None,
    vmap=vmap if HAS_JAX else None,
    pmap=pmap if HAS_JAX else None,
    cond=jax.lax.cond if HAS_JAX else None,
    scan=jax.lax.scan if HAS_JAX else None,
    # Utility functions
    dtype=jnp.dtype if HAS_JAX else None,
    finfo=jnp.finfo if HAS_JAX else None,
    iinfo=jnp.iinfo if HAS_JAX else None,
    isfinite=jnp.isfinite if HAS_JAX else None,
    isnan=jnp.isnan if HAS_JAX else None,
    isinf=jnp.isinf if HAS_JAX else None,
    allclose=jnp.allclose if HAS_JAX else None,
    array_equal=jnp.array_equal if HAS_JAX else None,
    # Configuration
    config=jax_config if HAS_JAX else None,
    enable_x64=enable_x64,
    is_available=is_available,
    scipy_optimize=None,
)

__all__ = [
    "backend",
    "enable_x64",
    "is_available",
    "HAS_JAX",
]