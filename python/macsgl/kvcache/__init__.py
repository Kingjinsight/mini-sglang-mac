from __future__ import annotations

from typing import Protocol

from macsgl.utils import Registry

from .base import BaseCacheHandle, BasePrefixCache, MatchResult, SizeInfo


class CacheManagerCreator(Protocol):
    def __call__(self, page_size: int) -> BasePrefixCache: ...


SUPPORTED_CACHE_MANAGER = Registry[CacheManagerCreator]("Cache Manager")


@SUPPORTED_CACHE_MANAGER.register("naive")
def create_naive_cache(page_size: int):
    from .naive_cache import NaivePrefixCache

    return NaivePrefixCache()


@SUPPORTED_CACHE_MANAGER.register("radix")
def create_radix_cache(page_size: int):
    from .radix_cache import RadixPrefixCache

    return RadixPrefixCache(page_size=page_size)


def create_prefix_cache(type: str, page_size: int) -> BasePrefixCache:
    return SUPPORTED_CACHE_MANAGER[type](page_size)


__all__ = [
    "create_prefix_cache",
    "BaseCacheHandle",
    "BasePrefixCache",
    "SizeInfo",
    "MatchResult",
    "SUPPORTED_CACHE_MANAGER",
]
