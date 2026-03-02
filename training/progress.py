from __future__ import annotations

from typing import Any, Iterable


try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None


class _NoOpProgress:
    def __iter__(self):
        return iter([])

    def update(self, n: int = 1) -> None:
        _ = n

    def close(self) -> None:
        return


def tqdm(iterable: Iterable[Any] | None = None, **kwargs):
    """Return tqdm iterator/bar when available, otherwise a no-op fallback."""
    if _tqdm is None:
        if iterable is None:
            return _NoOpProgress()
        return iterable
    if iterable is None:
        return _tqdm(**kwargs)
    return _tqdm(iterable, **kwargs)
