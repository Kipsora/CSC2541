import abc
from typing import Optional, Dict, Any, Iterable

from flexlearn.engine import Engine
from flexlearn.engine.callbacks import Callback


class Trainer(Engine, metaclass=abc.ABCMeta):

  def __init__(self, callback: Optional[Callback] = None):
    super().__init__(callback)
    self.global_step = -1
    self.epoch_index = 0
    self.batch_index = None
    self.num_epochs = 0
    self._epoch_early_stop_flag = False
    self._batch_early_stop_flag = False

  def set_epoch_early_stop_flag(self, flag: bool):
    self._epoch_early_stop_flag = flag

  def set_batch_early_stop_flag(self, flag: bool):
    self._batch_early_stop_flag = flag

  def _prior_epoch(self, data_loader: Iterable):
    self._callback.prior_epoch(self, data_loader=data_loader)

  def _after_epoch(self, data_loader: Iterable):
    self._callback.after_epoch(self, data_loader=data_loader)

  def _prior_batch(self, inputs):
    self._callback.prior_batch(self, inputs=inputs)

  def _after_batch(self, inputs, outputs):
    self._callback.after_batch(self, inputs=inputs, outputs=outputs)

  def _prior_forward(self, sources, targets):
    self._callback.prior_forward(self, sources=sources, targets=targets)

  def _after_forward(self, sources, targets):
    self._callback.after_forward(self, sources=sources, targets=targets)

  def _prior_backward(self):
    self._callback.prior_backward(self)

  def _after_backward(self):
    self._callback.after_backward(self)

  def _prior_update(self):
    self._callback.prior_update(self)

  def _after_update(self):
    self._callback.after_update(self)

  @abc.abstractmethod
  def _run_batch(self, inputs, **kwargs):
    pass

  def state_dict(self) -> Dict[str, Any]:
    state = super().state_dict()
    state.update({
        "global_step": self.global_step,
        "epoch_index": self.epoch_index,
        "num_epochs": self.num_epochs
    })
    return state

  def load_state_dict(self, state: Dict[str, Any], **kwargs):
    super().load_state_dict(state)
    self.global_step = state.get("global_step", self.global_step)
    self.epoch_index = state.get("epoch_index", self.epoch_index)
    self.num_epochs = state.get("num_epochs", self.num_epochs)
    self.batch_index = None

  def _run_epoch(self, data_loader: Iterable, **kwargs):
    self.batch_index = -1
    for inputs in data_loader:
      if self._batch_early_stop_flag:
        break

      self.batch_index += 1
      self.global_step += 1

      self._prior_batch(inputs)
      outputs = self._run_batch(inputs, **kwargs)
      self._after_batch(inputs, outputs)
    self.batch_index = None

  def run(self,
          data_loader: Iterable,
          num_epochs: Optional[int] = None,
          **kwargs):
    if num_epochs is not None:
      self.num_epochs += num_epochs
    try:
      self._prior_all()
      while self.epoch_index < self.num_epochs:
        if self._epoch_early_stop_flag:
          break

        self.epoch_index += 1

        self._prior_epoch(data_loader)
        self._run_epoch(data_loader, **kwargs)
        self._after_epoch(data_loader)

        if self._batch_early_stop_flag:
          break
    except BaseException as exception:
      self._on_exception(exception)
      raise
    finally:
      self._after_all()
