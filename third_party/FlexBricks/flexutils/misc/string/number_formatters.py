__all__ = ['NumberFormatter', 'TemplateNumberFormatter', 'TimeNumberFormatter']

import abc
import math


class NumberFormatter(object, metaclass=abc.ABCMeta):

  @abc.abstractmethod
  def __call__(self, value):
    pass


class TemplateNumberFormatter(NumberFormatter):

  def __init__(self, template: str):
    self._template = template

  def __call__(self, value):
    return self._template.format(value)


class TimeNumberFormatter(NumberFormatter):
  MILLISECOND = 1
  SECOND = 1000 * MILLISECOND
  MINUTE = 60 * SECOND
  HOUR = 60 * MINUTE
  DAY = 24 * HOUR

  def __init__(self, scale="second"):
    if isinstance(scale, str):
      scale = scale.lower()
      if scale == "millisecond":
        self._scale = self.MILLISECOND
      elif scale == "second":
        self._scale = self.SECOND
      elif scale == "minute":
        self._scale = self.MINUTE
      elif scale == "hour":
        self._scale = self.HOUR
      elif scale == "day":
        self._scale = self.DAY
      else:
        raise ValueError(f"Unsupported scale {scale}")
    else:
      self._scale = scale

  def __call__(self, value):
    value = float(value)
    value *= self._scale

    result = []
    if value >= self.DAY * 2:
      num_days = math.floor(value / self.DAY)
      value -= num_days * self.DAY
      result.append(f'{num_days}d')

    if value >= self.HOUR * 3:
      num_hours = math.floor(value / self.HOUR)
      value -= num_hours * self.HOUR
      result.append(f'{num_hours}h')

    if value >= self.MINUTE:
      num_minutes = math.floor(value / self.MINUTE)
      value -= num_minutes * self.MINUTE
      result.append(f'{num_minutes}m')

    if value <= 10 * self.SECOND:
      result.append(f"{value:.0f}ms")
    else:
      value /= self.SECOND
      result.append(f'{value:.2f}s')

    return ' '.join(result)
