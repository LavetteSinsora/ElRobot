"""The one exception the config layer raises: a refusal at resolution or at
the startup checks, before anything moves."""


class ConfigError(Exception):
    """A startup refusal. Never raised after the run is ACTIVE."""


__all__ = ["ConfigError"]
