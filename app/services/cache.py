"""Cache management helpers for generation services."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any, Iterator


class GenerationCache:
    """Container for audio/token caches with context manager support."""

    def __init__(
        self,
        audio: MutableMapping[str, Any] | None = None,
        tokens: MutableMapping[str, Any] | None = None,
    ) -> None:
        self._audio = audio or {}
        self._tokens = tokens or {}

    @property
    def audio(self) -> MutableMapping[str, Any]:
        """Return the underlying audio cache mapping."""
        return self._audio

    @property
    def tokens(self) -> MutableMapping[str, Any]:
        """Return the underlying token cache mapping."""
        return self._tokens

    def clear(self) -> None:
        """Clear both audio and token caches."""
        self._audio.clear()
        self._tokens.clear()

    def __enter__(self) -> "GenerationCache":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> bool:
        self.clear()
        return False

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        """Iterate over audio cache entries for convenience."""
        return iter(self._audio.items())
