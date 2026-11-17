"""Exceptions mapped to HTTP status codes by the API layer."""


class ErasmatchError(Exception):
    """Base class."""


class NotFoundError(ErasmatchError):
    """A university or course does not exist."""


class ProviderError(ErasmatchError):
    """An embedding or LLM provider failed or is unreachable."""


class ModelMismatchError(ErasmatchError):
    """Stored embeddings were produced by a different model than the one configured."""


class LLMDisabledError(ErasmatchError):
    """An operation needs an LLM but ERASMATCH_LLM=none."""
