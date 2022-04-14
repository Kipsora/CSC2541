__all__ = ['Inferer']

import abc
from typing import Optional, Iterable

from flexlearn.engine import Engine
from flexlearn.engine.callbacks import Callback


class Inferer(Engine, metaclass=abc.ABCMeta):

  def __init__(self, callback: Optional[Callback] = None):
    super().__init__(callback)
    self.batch_index = None

  def _prior_epoch(self, data_loader, **kwargs):
    self._callback.prior_epoch(self, data_loader=data_loader, **kwargs)

  def _after_epoch(self, data_loader, **kwargs):
    self._callback.after_epoch(self, data_loader=data_loader, **kwargs)

  def _prior_batch(self, inputs, **kwargs):
    self._callback.prior_batch(self, inputs=inputs, **kwargs)

  def _after_batch(self, inputs, outputs, **kwargs):
    self._callback.after_batch(self, inputs=inputs, outputs=outputs, **kwargs)

  def _prior_forward(self, **kwargs):
    self._callback.prior_forward(self, **kwargs)

  def _after_forward(self, **kwargs):
    self._callback.after_forward(self, **kwargs)

  @abc.abstractmethod
  def _run_inference(self, data_loader, **kwargs):
    pass

  def run(self, data_loader: Iterable, **kwargs):
    try:
      self._prior_all()
      self._prior_epoch(data_loader, **kwargs)
      self._run_inference(data_loader, **kwargs)
      self._after_epoch(data_loader, **kwargs)
    except BaseException as exception:
      self._on_exception(exception)
      raise
    finally:
      self._after_all()
