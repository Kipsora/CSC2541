__all__ = ['Context']

import contextlib
from typing import Optional


class Context(object):
  """
  The class :class:`Context` is a convenient class to define arguments for other
  objects in FlexBricks. It is literally just a key-value storage that can be
  defaulted.
  """

  __default__: Optional['Context'] = None

  def __init__(self, **kwargs):
    self._kwargs = kwargs

  @contextlib.contextmanager
  def as_default(self):
    """
    A context manager that makes the context as default.
    """
    original_default = self.__class__.__default__
    try:
      self.__class__.__default__ = self
      yield self
    finally:
      self.__class__.__default__ = original_default

  @classmethod
  def default(cls) -> 'Context':
    """
    Retrieve the current default context.

    Returns:
      The current default context.
    """
    return cls.__default__

  def __setitem__(self, key, value):
    self._kwargs[key] = value

  def __getitem__(self, item):
    return self._kwargs[item]

  def __contains__(self, item):
    return item in self._kwargs

  def pop(self, key: str, default=None):
    return self._kwargs.pop(key, default)

  def list_keys(self):
    return self._kwargs.keys()

  def list_values(self):
    return self._kwargs.values()

  @classmethod
  def list_default_keys(cls):
    context = cls.default()
    if context is None:
      return list()
    return context.list_keys()

  @classmethod
  def list_default_values(cls):
    context = cls.default()
    if context is None:
      return list()
    return context.list_values()

  @classmethod
  def get_default_argument(cls, key: str, default=None, required=False):
    """
    Get an argument in the default context.

    Args:
      key: The key to the argument.
      default: The default value if the argument does not exist.
      required: Whether the argument is required. If it is ``True``, then an
        exception will be raised when the argument does not exist.

    Returns:
      The value of the argument.
    """
    context = cls.default()
    if context is None:
      if required:
        raise ValueError(f"Argument {key} has not been set")
      return None
    if required and key not in context:
      raise ValueError(f"Argument {key} has not been set")
    return context._kwargs.get(key, default)

  @classmethod
  def has_default_argument(cls, key: str):
    """
    Whether an argument exists in the default context.

    Args:
      key: The key to the argument.

    Returns:
      Whether the argument exists.
    """
    context = cls.default()
    if context is None:
      return False
    return key in context

  @classmethod
  def set_default_argument(cls, key: str, value):
    """
    Set the value to an argument in the default context.

    Args:
      key: The key to the argument.
      value: The value of the argument.
    """
    context = cls.default()
    if context is not None:
      context._kwargs[key] = value
    return value

  @classmethod
  @contextlib.contextmanager
  def switch_default_argument(cls, key: str, value):
    """
    A context manager that can modify an argument in the default context.

    Args:
      key: The key to the argument.
      value: The new value of the argument.
    """
    context = cls.default()
    has_argument = key in context
    original_value = None
    if has_argument:
      original_value = context[key]
    else:
      context[key] = value
    try:
      yield cls
    finally:
      if has_argument:
        context[key] = original_value
      else:
        context.pop(key)
