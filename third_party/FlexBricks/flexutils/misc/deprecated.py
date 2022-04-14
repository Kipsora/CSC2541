__all__ = ['deprecated']

import functools
import warnings


def deprecated(fn_or_docs):

  def wrapper(fn):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
      if not getattr(fn, "__deprecated__", False):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn(f"Deprecated: {fn.__name__}."
                      if not fn_or_docs else str(fn_or_docs),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
      setattr(fn, "__deprecated__", True)
      return fn(*args, **kwargs)

    return wrapped

  if callable(fn_or_docs):
    return wrapper(fn_or_docs)
  else:
    return wrapper
