__all__ = [
    'SaveCheckpoint', 'SaveCheckpointOnEveryNEpochs',
    'SaveCheckpointOnBetterMetric'
]

import logging
from typing import Optional, TYPE_CHECKING

from flexlearn.checkpoints import Checkpoint
from flexlearn.context import Context
from flexlearn.distributed import ProcessGroup
from flexlearn.engine.callbacks import Callback
from flexlearn.summary import Summary

Engine = None
if TYPE_CHECKING:
  from flexlearn.engine import Engine


class SaveCheckpoint(Callback):
  """
  A convenient base class for implementing checkpoint saving callbacks.

  .. warning:: This class should not be directly used as it does not inherit any
    ``prior_`` or ``after_`` methods. Instead, the user should inherit from this
    class and implement their own logic.

  Args:
    checkpoint: The checkpoint for saving the epoch. It is required and will
      be retrieved from the default :class:`~flexlearn.context.Context`
      if not given.
    logger: A :class:`~logging.Logger` instance for printing messages. It is
      required and will be retrieved from the default
      :class:`~flexlearn.context.Context` if not given.
    process_group: The process group for determining whether the current
      process is the local master process. It will be retrieved from the default
      :class:`~flexlearn.context.Context` if not given. If it is not defined
      anyway, then it will be set as ``None`` and will assume the current
      process is the local master process.
    use_unique_local: Whether to save the checkpoint only for the local master
      process.
  """

  def __init__(self,
               checkpoint: Optional[Checkpoint] = None,
               logger: Optional[logging.Logger] = None,
               process_group: Optional[ProcessGroup] = None,
               use_unique_local: bool = True):
    self._process_group = process_group or Context.get_default_argument(
        "process_group")
    self._logger = logger or Context.get_default_argument("logger",
                                                          required=True)
    self._checkpoint = checkpoint or Context.get_default_argument("checkpoint",
                                                                  required=True)
    self._use_unique_local = use_unique_local

  def _save(self, index):
    if index != getattr(self._checkpoint, '__last_saved_index__', None):
      setattr(self._checkpoint, '__last_saved_index__', index)
      self._checkpoint.save(index=index)
    else:
      self._logger.info(f"{self.__class__.__name__} will not save the "
                        f"checkpoint since it's already saved by another "
                        f"callback (index={index})")


class SaveCheckpointOnEveryNEpochs(SaveCheckpoint):
  """
  Save the checkpoint on every ``N`` epochs.

  Args:
    frequency: The frequency for saving the checkpoint in terms of epochs.
    checkpoint: The checkpoint for saving the epoch. It is required and will
      be retrieved from the default :class:`~flexlearn.context.Context`
      if not given.
    logger: A :class:`~logging.Logger` instance for printing messages. It is
      required and will be retrieved from the default
      :class:`~flexlearn.context.Context` if not given.
    process_group: The process group for determining whether the current
      process is the local master process. It will be retrieved from the default
      :class:`~flexlearn.context.Context` if not given. If it is not defined
      anyway, then it will be set as ``None`` and will assume the current
      process is the local master process.
    use_unique_local: Whether to save the checkpoint only for the local master
      process.
  """

  def __init__(self,
               frequency: int,
               checkpoint: Optional[Checkpoint] = None,
               logger: Optional[logging.Logger] = None,
               process_group: Optional[ProcessGroup] = None,
               use_unique_local: bool = True):
    super().__init__(checkpoint, logger, process_group, use_unique_local)
    self._frequency = frequency

  def after_epoch(self, engine: Engine, **kwargs):
    if not self._use_unique_local or ProcessGroup.is_local_master(
        self._process_group):
      epoch_index = self._get_attribute("epoch_index", engine, **kwargs)
      if epoch_index % self._frequency == 0:
        self._save(epoch_index)


class SaveCheckpointOnBetterMetric(SaveCheckpoint):
  """
  Save the checkpoint when better metric results are achieved.

  Args:
    metric_field: The field in the :class:`~flexlearn.summary.Summary` to
      indicate the metric this callback should be monitoring.
    comparator: The comparator to determine whether a metric is better than
      another. Currently, only ``min`` and ``max`` is supported.
    summary: A :class:`~flexlearn.summary.Summary` instance in which the metric
      is monitored. It is required and will be retrieved from the default
      :class:`~flexlearn.context.Context` if not given.
    checkpoint: The checkpoint for saving the epoch. It is required and will
      be retrieved from the default :class:`~flexlearn.context.Context`
      if not given.
    logger: A :class:`~logging.Logger` instance for printing messages. It is
      required and will be retrieved from the default
      :class:`~flexlearn.context.Context` if not given.
    process_group: The process group for determining whether the current
      process is the local master process. It will be retrieved from the default
      :class:`~flexlearn.context.Context` if not given. If it is not defined
      anyway, then it will be set as ``None`` and will assume the current
      process is the local master process.
    use_unique_local: Whether to save the checkpoint only for the local master
      process.
    use_zeroth_epoch: Whether to save the checkpoint even on the zeroth epoch.
  """

  def __init__(self,
               metric_field: str,
               comparator="max",
               summary: Optional[Summary] = None,
               checkpoint: Optional[Checkpoint] = None,
               logger: Optional[logging.Logger] = None,
               process_group: Optional[ProcessGroup] = None,
               use_unique_local: bool = True,
               use_zeroth_epoch: bool = False):
    super().__init__(checkpoint, logger, process_group, use_unique_local)
    self._summary = summary or Context.get_default_argument("summary",
                                                            required=True)
    self._metric_field = metric_field
    self._use_zeroth_epoch = use_zeroth_epoch

    self._comparator = comparator
    if self._comparator not in ("min", "max"):
      raise ValueError(f"Unsupported comparator \"{self._comparator}\"")

  def _compare_and_update(self, value):
    metadata_value = self._checkpoint.metadata.get("best_metric")
    if self._comparator == "min":
      if metadata_value is None or value < metadata_value:
        self._checkpoint.metadata['best_metric'] = value
        return True
    else:
      if metadata_value is None or value > metadata_value:
        self._checkpoint.metadata['best_metric'] = value
        return True
    return False

  def prior_all(self, engine: Engine, **kwargs):
    if not self._use_unique_local or ProcessGroup.is_local_master(
        self._process_group):
      epoch_index = self._get_attribute("epoch_index", engine, **kwargs)

      # It is not necessary that the model is evaluated at the very first, but
      # we try our best to fetch the stored value
      if self._metric_field in self._summary and \
              epoch_index in self._summary[self._metric_field]:
        value = self._summary[self._metric_field].value(index=epoch_index)
        self._compare_and_update(value)

  def after_epoch(self, engine: Engine, **kwargs):
    if not self._use_unique_local or ProcessGroup.is_local_master(
        self._process_group):
      epoch_index = self._get_attribute("epoch_index", engine, **kwargs)
      value = self._summary[self._metric_field].value(index=epoch_index)
      if self._compare_and_update(value):
        self._save(epoch_index)
