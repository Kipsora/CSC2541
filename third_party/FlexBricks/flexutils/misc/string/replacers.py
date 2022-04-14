__all__ = ['StringReplacer', 'RegularReplacer']

import abc
import re
from typing import Dict


class StringReplacer(object, metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def substitute(self, text, **kwargs):
    pass


class RegularReplacer(StringReplacer):

  def __init__(self, patterns: Dict[str, str]):
    self._patterns = patterns

  def substitute(self, text, **kwargs):
    for source, target in self._patterns.items():
      source = source.format(**kwargs)
      target = target.format(**kwargs)

      if re.match(source, text):
        return re.sub(source, target, text)
    return None
