__all__ = ['ComposeCallback']

from typing import Collection, TYPE_CHECKING

from flexlearn.engine.callbacks import Callback

Engine = None
if TYPE_CHECKING:
  from flexlearn.engine import Engine


class ComposeCallback(Callback):

  def __init__(self,
               callbacks: Collection[Callback],
               reverse_prior_callbacks: bool = False,
               reverse_after_callbacks: bool = False):
    self._callbacks = list(callbacks)
    self._reverse_after_callbacks = reverse_after_callbacks
    self._reverse_prior_callbacks = reverse_prior_callbacks

  def _prior_callbacks(self):
    if self._reverse_prior_callbacks:
      return reversed(self._callbacks)
    return self._callbacks

  def _after_callbacks(self):
    if self._reverse_after_callbacks:
      return reversed(self._callbacks)
    return self._callbacks

  def prior_all(self, engine: Engine, **kwargs):
    for callback in self._prior_callbacks():
      callback.prior_all(engine, **kwargs)

  def after_all(self, engine: Engine, **kwargs):
    for callback in self._after_callbacks():
      callback.after_all(engine, **kwargs)

  def prior_epoch(self, engine: Engine, **kwargs):
    for callback in self._prior_callbacks():
      callback.prior_epoch(engine, **kwargs)

  def after_epoch(self, engine: Engine, **kwargs):
    for callback in self._after_callbacks():
      callback.after_epoch(engine, **kwargs)

  def prior_batch(self, engine: Engine, **kwargs):
    for callback in self._prior_callbacks():
      callback.prior_batch(engine, **kwargs)

  def after_batch(self, engine: Engine, **kwargs):
    for callback in self._after_callbacks():
      callback.after_batch(engine, **kwargs)

  def prior_forward(self, engine: Engine, **kwargs):
    for callback in self._prior_callbacks():
      callback.prior_forward(engine, **kwargs)

  def after_forward(self, engine: Engine, **kwargs):
    for callback in self._after_callbacks():
      callback.after_forward(engine, **kwargs)

  def prior_backward(self, engine: Engine, **kwargs):
    for callback in self._prior_callbacks():
      callback.prior_backward(engine, **kwargs)

  def after_backward(self, engine: Engine, **kwargs):
    for callback in self._after_callbacks():
      callback.after_backward(engine, **kwargs)

  def prior_update(self, engine: Engine, **kwargs):
    for callback in self._prior_callbacks():
      callback.prior_update(engine, **kwargs)

  def after_update(self, engine: Engine, **kwargs):
    for callback in self._after_callbacks():
      callback.after_update(engine, **kwargs)

  def on_exception(self, engine: Engine, exception: BaseException, **kwargs):
    for callback in self._callbacks:
      callback.on_exception(engine, exception, **kwargs)
