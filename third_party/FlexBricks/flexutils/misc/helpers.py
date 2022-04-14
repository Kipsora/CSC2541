__all__ = ['dict_walk', 'dict_lookup', 'map_lookup', 'map_walk']

from typing import Any, Mapping, Collection

from flexutils.misc import deprecated


def dict_lookup(data: Mapping[str, Any], *keys: str):
  for key in list(keys):
    if key in data:
      return data[key]


def dict_walk(data: Mapping[str, Any], path: Collection[str]):
  for key in path:
    data = data[key]
  return data


@deprecated("map_lookup is deprecated and replaced by dict_lookup.")
def map_lookup(data: Mapping[str, Any], *keys: str):
  return dict_lookup(data, *keys)


@deprecated("map_walk is deprecated and replaced by dict_walk.")
def map_walk(data: Mapping[str, Any], path: Collection[str]):
  return dict_walk(data, path)
