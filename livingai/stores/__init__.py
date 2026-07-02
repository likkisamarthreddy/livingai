"""Optional storage backends for Living AI.

These backends require installing the matching extras::

    pip install livingai[redis]       # RedisStore
    pip install livingai[postgres]    # PostgresStore

Both backends implement the :class:`~livingai.storage.CheckpointStore` protocol
and are therefore drop-in replacements for the default
:class:`~livingai.storage.SQLiteStore`.
"""

__all__ = ["RedisStore", "PostgresStore"]


def __getattr__(name: str) -> object:  # pragma: no cover
    if name == "RedisStore":
        from .redis import RedisStore
        return RedisStore
    if name == "PostgresStore":
        from .postgres import PostgresStore
        return PostgresStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
