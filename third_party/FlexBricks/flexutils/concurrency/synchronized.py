__all__ = ['synchronized']

import functools


def synchronized(fn_or_lock_member):
  if callable(fn_or_lock_member):
    return synchronized('_lock')(fn_or_lock_member)

  def wrapper(fn):

    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
      lock = getattr(self, fn_or_lock_member, None)
      if lock is None:
        return fn(self, *args, **kwargs)

      lock.acquire()
      try:
        return fn(self, *args, **kwargs)
      finally:
        lock.release()

    return wrapped

  return wrapper
