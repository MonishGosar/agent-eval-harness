"""Errors raised by the harness. These are user-facing and stay out of tracebacks in the CLI."""


class AgentEvalError(Exception):
    """Base error for configuration, dataset, and run failures the caller can fix."""


class ConfigError(AgentEvalError):
    """evals.yaml is missing a field, names an unknown metric, or points at a bad target."""


class DatasetError(AgentEvalError):
    """A golden file could not be parsed or failed validation."""


class AdapterError(AgentEvalError):
    """An adapter could not be constructed. Case-level agent failures are recorded, not raised."""
