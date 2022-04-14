__all__ = ['TimeMeter', 'CPUTimeMeter']

import abc
import contextlib
import time


class TimeMeter(object, metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def prior(self):
    pass

  @abc.abstractmethod
  def after(self):
    pass

  @abc.abstractmethod
  def elapsed_time(self):
    pass

  @abc.abstractmethod
  def reset(self):
    pass

  @contextlib.contextmanager
  def activate(self):
    self.prior()
    try:
      yield self
    finally:
      self.after()


class CPUTimeMeter(TimeMeter):

  def __init__(self):
    self._elapsed_ns = 0

  def prior(self):
    self._elapsed_ns -= time.time_ns()

  def reset(self):
    self._elapsed_ns = 0

  def after(self):
    self._elapsed_ns += time.time_ns()

  def elapsed_time(self):
    return self._elapsed_ns / 1e6
