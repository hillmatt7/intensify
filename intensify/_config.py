"""Global configuration for intensify."""


# Configuration defaults
_DEFAULTS = {
    "recursive_warning_threshold": 50_000,  # warn if N > 50k and using O(N^2) kernel
    "float64_auto_enable": True,  # automatically enable x64 for float64 inputs
    "warn_on_nonstationary": True,
    "performance_warning": True,
}

# Current config (mutable)
_CONFIG: dict[str, object] = _DEFAULTS.copy()


def get(key: str):
    """Get configuration value."""
    return _CONFIG.get(key, _DEFAULTS.get(key))


# Public aliases for package exports
config_get = get


def set_config(key: str, value: object) -> None:
    """Set configuration value."""
    if key not in _DEFAULTS:
        raise KeyError(f"Unknown config key '{key}'. Valid keys: {list(_DEFAULTS.keys())}")
    _CONFIG[key] = value


config_set = set_config


def reset() -> None:
    """Reset configuration to defaults."""
    _CONFIG.clear()
    _CONFIG.update(_DEFAULTS)


config_reset = reset