__all__ = [
    'SynchronizeSummary', 'WriteSummaryToTensorBoard', 'RecordLearningRate',
    'RecordEstimatedToArrival', 'RecordTimeMeters', 'RecordModelWeightsPerEpoch'
]

from collections import deque
from functools import partial
from typing import Dict, Any, Optional, Union, Collection

import torch.cuda
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from flexlearn.context import Context
from flexlearn.distributed import ProcessGroup
from flexlearn.engine import Engine
from flexlearn.engine.callbacks import *
from flexlearn.meters import TimeMeter
from flexlearn.summary import SummaryReduction, SummaryItemType
from flexlearn.torch.summary import TorchSummary
from flexutils.misc import deprecated
from flexutils.misc.string import StringMatcher, filtered_join

EventOrTime = Union[torch.cuda.Event, float]


class SynchronizeSummary(Callback):

  def __init__(self,
               summary: Optional[TorchSummary] = None,
               *,
               use_zeroth_epoch: bool = False):
    self._summary = summary or Context.get_default_argument("summary",
                                                            required=True)
    self._use_zeroth_epoch = use_zeroth_epoch

  def prior_all(self, engine: Engine, **kwargs):
    epoch_index = self._get_attribute("epoch_index", engine, **kwargs)
    if self._use_zeroth_epoch and epoch_index == 0:
      self._synchronize_epoch(epoch_index)

  def after_epoch(self, engine: Engine, **kwargs):
    epoch_index = self._get_attribute("epoch_index", engine, **kwargs)
    self._synchronize_epoch(epoch_index)

  def after_batch(self, engine: Engine, **kwargs):
    batch_index = self._get_attribute("batch_index", engine, **kwargs)
    self._synchronize_batch(batch_index)

  def _synchronize_epoch(self, epoch_index):
    futures = list()
    for name in self._summary.names():
      storage = self._summary[name]
      if not storage or storage.indices[-1] != epoch_index or \
              storage.item_type != SummaryItemType.PER_EPOCH:
        continue

      current_futures = storage.synchronize(index=epoch_index,
                                            use_async_op=True)
      if current_futures is not None:
        futures.extend(current_futures)
    for future in futures:
      future.wait()

  def _synchronize_batch(self, batch_index):
    futures = list()
    for name in self._summary.names():
      storage = self._summary[name]
      if not storage or storage.indices[-1] != batch_index or \
              storage.item_type != SummaryItemType.PER_GLOBAL_STEP:
        continue
      current_futures = storage.synchronize(index=batch_index,
                                            use_async_op=True)
      if current_futures is not None:
        futures.extend(current_futures)
    for future in futures:
      future.wait()


class WriteSummaryToTensorBoard(WriteSummary):

  def __init__(self,
               summary: Optional[TorchSummary] = None,
               *,
               path: str,
               matcher: Optional[StringMatcher] = None,
               use_graph_writer: bool = False,
               use_zeroth_epoch: bool = False,
               use_unique_local: bool = True,
               use_strict_graph_trace: bool = True,
               process_group: Optional[ProcessGroup] = None):
    super().__init__(summary,
                     use_zeroth_epoch=use_zeroth_epoch,
                     use_unique_local=use_unique_local,
                     process_group=process_group)
    self._path = path
    self._writer: Optional[SummaryWriter] = None
    self._matcher = matcher
    self._use_graph_writer = use_graph_writer
    self._use_strict_graph_trace = use_strict_graph_trace

  def prior_forward(self, engine: Engine, **kwargs):
    super(WriteSummaryToTensorBoard, self).prior_batch(engine, **kwargs)

    # The writer should be ready since prior_all will be called
    # before prior_forward
    if self._use_graph_writer and self._writer is not None:
      # Only write model's graph once
      self._use_graph_writer = False

      model = self._get_attribute("model", engine, **kwargs)
      sources = self._get_attribute("sources", engine, **kwargs)
      self._writer.add_graph(model,
                             sources,
                             use_strict_trace=self._use_strict_graph_trace)

  def prior_all(self, engine: Engine, **kwargs):
    if not self._use_unique_local or ProcessGroup.is_local_master(
        self._process_group):
      # In prior_all callback the epoch_index has not been touched in engines
      purge_step = self._get_attribute("epoch_index", engine, **kwargs) + 1
      self._writer = SummaryWriter(log_dir=self._path, purge_step=purge_step)
    super().prior_all(engine, **kwargs)

  def _write(self, index, item_type):
    for name in sorted(self._summary.names()):
      if self._matcher is not None and not self._matcher.match(name):
        continue
      storage = self._summary[name]
      if storage and storage.indices[-1] == index and \
              storage.item_type == item_type:
        value = storage.value(index=index)
        if value.size == 1:
          self._writer.add_scalar(name, value.item(), global_step=index)
        else:
          self._writer.add_histogram(name, value, global_step=index)

  def _write_epoch(self, epoch_index):
    self._write(epoch_index, SummaryItemType.PER_EPOCH)

  def _write_batch(self, global_step):
    self._write(global_step, SummaryItemType.PER_GLOBAL_STEP)

  def _close(self):
    if self._writer is not None:
      self._writer.close()
      self._writer = None


class RecordLearningRate(Callback):

  def __init__(self,
               summary: Optional[TorchSummary] = None,
               *,
               optimizer: Optimizer,
               interval: str = 'per_batch',
               prefix: Optional[str] = None):
    self._optimizer = optimizer
    self._summary = summary or Context.get_default_argument("summary",
                                                            required=True)
    self._prefix = prefix or ""
    if interval not in ('per_batch', 'per_epoch'):
      raise ValueError("Invalid value of interval "
                       "(expected \"per_batch\" or \"per_epoch\")")
    if interval == 'per_batch':
      self.after_batch = self._record
    else:
      self.after_epoch = self._record

  def _record(self, engine, **kwargs):
    lrs = [
        torch.as_tensor(group['lr']) for group in self._optimizer.param_groups
    ]
    epoch_index = self._get_attribute("epoch_index", engine, **kwargs)
    if len(lrs) > 1:
      for index, lr in enumerate(lrs):
        name = filtered_join('/', [self._prefix, 'lr', f'group{index}'])
        self._summary.add_value(name, lr, index=epoch_index, reset=True)
    else:
      self._summary.add_value(filtered_join('/', [self._prefix, f'lr']),
                              lrs[0],
                              index=epoch_index,
                              reset=True)


@deprecated("RecordEstimatedToArrival is deprecated and replaced "
            "by RecordTimeMeters")
class RecordEstimatedToArrival(Callback):

  def __init__(self,
               time_meter: TimeMeter,
               summary: Optional[TorchSummary] = None,
               num_smoothing_epochs: int = 5,
               use_epoch_time: bool = True):
    self._summary = summary or Context.get_default_argument("summary",
                                                            required=True)
    self._history = deque(maxlen=num_smoothing_epochs)
    self._time_meter = time_meter
    self._use_epoch_time = use_epoch_time

  def prior_all(self, engine: Engine, **kwargs):
    super().prior_all(engine, **kwargs)
    self._history.clear()

  def prior_epoch(self, engine: Engine, **kwargs):
    super().prior_epoch(engine, **kwargs)
    self._time_meter.reset()
    self._time_meter.prior()

  def after_epoch(self, engine: Engine, **kwargs):
    self._time_meter.after()
    epoch_time = self._time_meter.elapsed_time() / 1e3
    self._history.append(epoch_time)
    epoch_index = self._get_attribute("epoch_index", engine, **kwargs)
    num_epochs = self._get_attribute("num_epochs", engine, **kwargs)
    averaged_epoch_time = sum(self._history) / len(self._history)
    estimated_time = averaged_epoch_time * (num_epochs - epoch_index)
    self._summary.add_value("eta", estimated_time, index=epoch_index)
    if self._use_epoch_time:
      self._summary.add_value("epoch_time", epoch_time, index=epoch_index)
    super().after_epoch(engine, **kwargs)


class RecordTimeMeters(Callback):

  def __init__(self,
               meters: Dict[str, TimeMeter],
               summary: Optional[TorchSummary] = None,
               prefix: Optional[str] = None,
               num_eta_smoothing_epochs: Optional[int] = 5):
    self._summary = summary or Context.get_default_argument("summary",
                                                            required=True)
    self._history = deque(maxlen=num_eta_smoothing_epochs) \
      if num_eta_smoothing_epochs is not None else None
    self._prefix = prefix

    self._meter_epoch = meters.get("epoch")

    self._meter_batch = meters.get("batch")
    self._meter_forward = meters.get("forward")
    self._meter_backward = meters.get("backward")
    self._meter_update = meters.get("update")

    self._valid_scopes = []
    if self._meter_batch is not None:
      self._valid_scopes.append('batch')
    if self._meter_forward is not None:
      self._valid_scopes.append('forward')
    if self._meter_backward is not None:
      self._valid_scopes.append('backward')
    if self._meter_update is not None:
      self._valid_scopes.append('update')

    for scope in self._valid_scopes:
      meter = getattr(self, f"_meter_{scope}")
      setattr(self, f"prior_{scope}", partial(self._prior_meter, meter))
      setattr(self, f"after_{scope}", partial(self._after_meter, meter))

    self._num_epochs = None
    self._epoch_index = None

  def prior_all(self, engine: Engine, **kwargs):
    if self._history is not None:
      self._history.clear()

  def prior_epoch(self, engine: Engine, **kwargs):
    self._epoch_index = self._get_attribute("epoch_index", engine, **kwargs)
    self._num_epochs = self._get_attribute("num_epochs", engine, **kwargs)

    for scope in self._valid_scopes:
      meter = getattr(self, f"_meter_{scope}")
      meter.reset()

    if self._meter_epoch is not None:
      self._meter_epoch.reset()
      self._meter_epoch.prior()

  def after_epoch(self, engine: Engine, **kwargs):
    if self._meter_epoch is not None:
      self._meter_epoch.after()
      epoch_time = self._meter_epoch.elapsed_time() / 1e3
      self._summary.add_value(filtered_join('/', [self._prefix, "time/epoch"]),
                              epoch_time,
                              index=self._epoch_index)

      if self._history is not None:
        self._history.append(epoch_time)
        estimated_time = sum(self._history) / len(self._history)
        estimated_time = estimated_time * (self._num_epochs - self._epoch_index)
        self._summary.add_value(filtered_join('/', [self._prefix, "time/eta"]),
                                estimated_time,
                                index=self._epoch_index)

    for scope in self._valid_scopes:
      meter: TimeMeter = getattr(self, f"_meter_{scope}", None)
      name = f"time/batch/{scope}" if scope != 'batch' else f'time/batch'
      name = filtered_join('/', [self._prefix, name])
      self._summary.add_value(name,
                              meter.elapsed_time() / 1e3,
                              index=self._epoch_index)

  @classmethod
  def _prior_meter(cls, meter: TimeMeter, engine: Engine, **kwargs):
    meter.prior()

  @classmethod
  def _after_meter(cls, meter: TimeMeter, engine: Engine, **kwargs):
    meter.after()


class RecordModelWeightsPerEpoch(Callback):

  def __init__(self,
               model: Optional[torch.nn.Module] = None,
               summary: Optional[TorchSummary] = None,
               record_weights: bool = True,
               record_gradients: bool = False,
               record_gradient_norm: bool = False,
               prefix: Optional[str] = None):
    self._summary = summary or Context.get_default_argument("summary",
                                                            required=True)
    self._model = model
    self._prefix = prefix
    self._record_weights = record_weights
    self._record_gradients = record_gradients
    self._record_gradient_norm = record_gradient_norm

  def after_epoch(self, engine: Engine, **kwargs):
    epoch_index = self._get_attribute("epoch_index", engine, **kwargs)
    model = self._model or self._get_attribute("model", engine, **kwargs) or \
            Context.get_default_argument("model", required=True)
    for name, param in model.named_parameters():
      if param.grad is not None:
        if self._record_gradients:
          path = filtered_join('/', [self._prefix, 'gradient', name])
          self._summary.add_value(path,
                                  param.grad,
                                  index=epoch_index,
                                  reduction=SummaryReduction.GLOBAL_MEAN)
        if self._record_gradient_norm:
          path = filtered_join('/', [self._prefix, 'gradient_norm', name])
          self._summary.add_value(path,
                                  param.grad.detach().norm(2),
                                  index=epoch_index,
                                  reduction=SummaryReduction.GLOBAL_MEAN)
      if self._record_weights:
        path = filtered_join('/', [self._prefix, 'weight', name])
        self._summary.add_value(path,
                                param,
                                index=epoch_index,
                                reduction=SummaryReduction.GLOBAL_MEAN)
