__all__ = ['ApplyPerEpochLRScheduler', 'ApplyPerBatchLRScheduler']

from typing import Optional

from flexlearn.context import Context
from flexlearn.engine import Engine
from flexlearn.engine.callbacks import Callback
from flexlearn.summary import Summary


class ApplyPerEpochLRScheduler(Callback):

  def __init__(self,
               scheduler,
               requires_loss: bool = False,
               summary: Optional[Summary] = None,
               field: Optional[str] = None):
    self._scheduler = scheduler
    self._requires_loss = requires_loss

    if self._requires_loss:
      try:
        self._summary = summary or Context.get_default_argument('summary',
                                                                required=True)
        if field is not None:
          raise ValueError("Argument loss_field is not set")
      except ValueError as exception:
        raise ValueError("Argument summary and loss_field must be given "
                         "if loss is required") from exception
    else:
      self._summary = None
      self._field = None

  def after_epoch(self, engine: Engine, **kwargs):
    if self._scheduler is None:
      return
    if self._field is not None:
      loss = self._summary[self._field].value()
      self._scheduler.step(loss)
    else:
      self._scheduler.step()


class ApplyPerBatchLRScheduler(Callback):

  def __init__(self, scheduler, requires_loss: bool = False):
    self._scheduler = scheduler
    self._requires_loss = requires_loss

  def after_batch(self, engine: Engine, **kwargs):
    if self._scheduler is None:
      return
    if self._requires_loss:
      objective = kwargs['outputs']['objective']
      loss = objective[None] if isinstance(objective, dict) else objective
      self._scheduler.step(loss)
    else:
      self._scheduler.step()
