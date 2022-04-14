__all__ = ['StringMatcher', 'WildcardMatcher', 'RegularMatcher']

import abc
import fnmatch
import re
from typing import Union, List


class StringMatcher(object, metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def match(self, text: str, **kwargs):
    pass


class WildcardMatcher(StringMatcher):

  def __init__(self, patterns: Union[str, List[str]]):
    if isinstance(patterns, str):
      patterns = [patterns]
    self._patterns = patterns

  def match(self, text: str, **kwargs):
    for pattern in self._patterns:
      pattern = pattern.format(**kwargs)
      if fnmatch.fnmatch(text, pattern):
        return True
    return False


class RegularMatcher(StringMatcher):

  def __init__(self, patterns: Union[str, List[str]]):
    if isinstance(patterns, str):
      patterns = [patterns]
    self._patterns = patterns

  def match(self, text: str, **kwargs):
    for pattern in self._patterns:
      if re.match(pattern.format(**kwargs), text):
        return True
    return False
