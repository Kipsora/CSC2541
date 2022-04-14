__all__ = ['attribute', 'condition', 'hint', 'classproperty']

import functools
from typing import Optional

HINT_KEY = "__hint__"


def attribute(key: str, value):

  def annotator(checker):
    setattr(checker, key, value)
    return checker

  return annotator


def hint(message: str):
  return attribute(HINT_KEY, message)


def condition(checker,
              message: Optional[str] = None,
              default_message_attribute=HINT_KEY):

  def wrapper(fn):

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
      if not checker():
        raise RuntimeError(message or
                           getattr(checker, default_message_attribute))
      return fn(*args, **kwargs)

    return wrapped

  return wrapper


class classproperty(property):
  """Descriptor to be used as decorator for @classmethods."""

  def __get__(self, obj, objtype=None):
    return self.fget.__get__(None, objtype)()  # pytype: disable=attribute-error
