__all__ = ['NestedDictMerger', 'NestedDictifier']

import keyword
from typing import Sequence

from flexutils.misc.attr_dict import AttrDict


class NestedDictMerger(object):

  def __init__(self, merge_sequence=False):
    self._merge_sequence = merge_sequence

  def merge(self, a, b):
    if isinstance(a, dict) and isinstance(b, dict):
      result = dict()
      for k, v in a.items():
        if k not in b:
          result.setdefault(k, v)
        else:
          result.setdefault(k, self.merge(v, b[k]))
      for k, v in b.items():
        if k not in result:
          result.setdefault(k, v)
      return result
    elif self._merge_sequence:
      if isinstance(a, Sequence) and isinstance(b, Sequence):
        if len(a) == len(b):
          result = [self.merge(x, y) for x, y in zip(a, b)]
          return result
    return b


class NestedDictifier(object):

  def __init__(self, use_attr_dict=True):
    self._use_attr_dict = use_attr_dict

  def dictify(self, data):
    if isinstance(data, dict):
      items = []
      is_valid_attr_dict = self._use_attr_dict
      for k, v in data.items():
        if not (isinstance(k, str) and k.isidentifier() and
                not keyword.iskeyword(k)):
          is_valid_attr_dict = False
        items.append((k, self.dictify(v)))
      if is_valid_attr_dict:
        return AttrDict(items)
      else:
        return data.__class__(items)
    elif isinstance(data, Sequence) and not isinstance(data, str):
      return [self.dictify(e) for e in data]
    return data
