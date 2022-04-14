__all__ = [
    'WriteSummary', 'WriteSummaryToLogger', 'WriteSummaryToJSON',
    'RecordBatchOutputs'
]

import abc
import logging
import os
from typing import Tuple, Collection, Optional, TYPE_CHECKING, Dict, Any, Union

from flexlearn.context import Context
from flexlearn.distributed import ProcessGroup
from flexlearn.engine.callbacks import Callback
from flexlearn.summary import Summary, SummaryReduction, SummaryItemType
from flexutils.io import serialize
from flexutils.io.file_system import ensure_directory
from flexutils.misc.string import *

Engine = None
if TYPE_CHECKING:
  from flexlearn.engine import Engine


class WriteSummary(Callback, metaclass=abc.ABCMeta):
  """
  A convenient base class for implementing summary writers.

  .. warning:: This class should not be directly used as it does not inherit any
    ``prior_`` or ``after_`` methods. Instead, the user should inherit from this
    class and implement their own logic.

  Args:
    summary: A :class:`~flexlearn.summary.Summary` instance in which the metric
      is monitored. It is required and will be retrieved from the default
      :class:`~flexlearn.context.Context` if not given.
    use_zeroth_epoch: Whether to write the summary even on the zeroth epoch.
    use_unique_local: Whether to write the summary only for the local
      master process.
    process_group: The process group for determining whether the current
      process is the local master process. It will be retrieved from the default
      :class:`~flexlearn.context.Context` if not given. If it is not defined
      anyway, then it will be set as ``None`` and will assume the current
      process is the local master process.
  """

  def __init__(self,
               summary: Optional[Summary] = None,
               *,
               use_zeroth_epoch: bool = False,
               use_unique_local: bool = True,
               process_group: Optional[ProcessGroup] = None):
    self._summary = summary or Context.get_default_argument("summary",
                                                            required=True)
    self._process_group = process_group or Context.get_default_argument(
        "process_group")
    self._use_zeroth_epoch = use_zeroth_epoch
    self._use_unique_local = use_unique_local

  def _write_epoch(self, epoch_index):
    pass

  def _write_batch(self, global_step):
    pass

  def _close(self):
    pass

  def prior_all(self, engine: Engine, **kwargs):
    if not self._use_unique_local or ProcessGroup.is_local_master(
        self._process_group):
      epoch_index = self._get_attribute("epoch_index", engine, **kwargs)
      if self._use_zeroth_epoch and epoch_index == 0:
          self._write_epoch(epoch_index)

  def after_epoch(self, engine: Engine, **kwargs):
    if not self._use_unique_local or ProcessGroup.is_local_master(
        self._process_group):
      self._write_epoch(self._get_attribute("epoch_index", engine, **kwargs))

  def after_batch(self, engine: Engine, **kwargs):
    if not self._use_unique_local or ProcessGroup.is_local_master(
        self._process_group):
      global_step = self._get_attribute("global_step", engine, **kwargs)
      self._write_batch(global_step)

  def on_exception(self, engine: Engine, exception: BaseException, **kwargs):
    self._close()

  def after_all(self, engine: Engine, **kwargs):
    self._close()


class WriteSummaryToLogger(WriteSummary):
  """
  Write the summary entries to a :class:`logging.Logger` instance.

  .. note::
    Only scalar values will be written.

  Args:
    summary: A :class:`~flexlearn.summary.Summary` instance in which the metric
      is monitored. It is required and will be retrieved from the default
      :class:`~flexlearn.context.Context` if not given.
    level: The logger level of the message.
    header: The header of the message.
    plugins: The plugins that retrieve the attributes used in the header.
    logger: A :class:`logging.Logger` instance to write the summary.
    value_format_rules: The format rules for the values in summary texts.
    use_zeroth_epoch: Whether to write the summary even on the zeroth epoch.
    use_unique_local: Whether to write the summary only for the local
      master process.
    process_group: The process group for determining whether the current
      process is the local master process. It will be retrieved from the default
      :class:`~flexlearn.context.Context` if not given. If it is not defined
      anyway, then it will be set as ``None`` and will assume the current
      process is the local master process.
  """
  FormatRule = Tuple[StringMatcher, NumberFormatter]

  def __init__(self,
               summary: Optional[Summary] = None,
               level=logging.INFO,
               *,
               header: Optional[str] = None,
               plugins: Optional[Union[str, Collection[str]]] = None,
               logger: Optional[logging.Logger] = None,
               value_format_rules: Optional[Collection[FormatRule]] = None,
               use_zeroth_epoch: bool = False,
               use_unique_local: bool = False,
               process_group: Optional[ProcessGroup] = None):
    super().__init__(summary,
                     use_zeroth_epoch=use_zeroth_epoch,
                     use_unique_local=use_unique_local,
                     process_group=process_group)
    if isinstance(plugins, str):
      plugins = {plugins}
    elif plugins is not None:
      plugins = set(plugins)
    else:
      plugins = set()
    self._plugins = plugins

    self._header = header or "Epoch {epoch_index}"

    self._logger = logger or Context.get_default_argument("logger",
                                                          required=True)
    self._level = level
    if value_format_rules is None:
      self._format_rules = [(None, TemplateNumberFormatter('{:.7g}'))]
    else:
      self._format_rules = list(value_format_rules)

    self._attributes = dict()

  def after_epoch(self, engine: Engine, **kwargs):
    super().after_epoch(engine, **kwargs)
    self._attributes.clear()
    for plugin in self._plugins:
      self._attributes[plugin] = self._get_attribute(plugin, engine, **kwargs)

  def _format_value(self, name, value):
    for matcher, formatter in self._format_rules:
      if matcher is not None and not matcher.match(name):
        continue
      return formatter(value)
    return str(value)

  def _build_message(self, epoch_index):
    headers = ['Field', 'Reduction', 'Value']
    rows = []
    for name in sorted(self._summary.names()):
      storage = self._summary[name]
      if storage.item_type != SummaryItemType.PER_EPOCH:
        continue

      value = storage.value()
      if value.size != 1:
        continue
      value = self._format_value(name, value.item())
      reduction = SummaryReduction.to_string(storage.reduction)

      if storage.indices[-1] != epoch_index:
        rows.append((name + "!", reduction, value))
      else:
        rows.append((name, reduction, value))

    self._attributes.setdefault("epoch_index", epoch_index)
    message = self._header.format(**self._attributes) + ":\n"
    message += format_table(headers, rows) + '\n'
    return message

  def _write_epoch(self, epoch_index):
    self._logger.log(level=self._level, msg=self._build_message(epoch_index))


class WriteSummaryToJSON(WriteSummary):
  """
  Write the summary entries to a JSON file named ``<epoch_index>.json``.

  .. note::
    Only scalar values will be written.

  Args:
    summary: A :class:`~flexlearn.summary.Summary` instance in which the metric
      is monitored. It is required and will be retrieved from the default
      :class:`~flexlearn.context.Context` if not given.
    path: The path to base directory of the JSON files to be written.
    use_zeroth_epoch: Whether to write the summary even on the zeroth epoch.
    use_unique_local: Whether to write the summary only for the local
      master process.
    process_group: The process group for determining whether the current
      process is the local master process. It will be retrieved from the default
      :class:`~flexlearn.context.Context` if not given. If it is not defined
      anyway, then it will be set as ``None`` and will assume the current
      process is the local master process.
  """

  def __init__(self,
               summary: Optional[Summary] = None,
               *,
               path: str,
               use_zeroth_epoch: bool = False,
               use_unique_local: bool = True,
               process_group: Optional[ProcessGroup] = None):
    super().__init__(summary,
                     use_unique_local=use_unique_local,
                     use_zeroth_epoch=use_zeroth_epoch,
                     process_group=process_group)
    self._path = path

  def _write_epoch(self, epoch_index):
    ensure_directory(self._path)
    data = dict()
    for name in sorted(self._summary.names()):
      storage = self._summary[name]
      if not storage or storage.indices[-1] != epoch_index or \
              storage.item_type != SummaryItemType.PER_EPOCH:
        continue
      value = storage.value(index=epoch_index)
      if value.size == 1:
        data[name] = value.item()
    serialize.dump_json(data, os.path.join(self._path, f"{epoch_index}.json"))


class RecordBatchOutputs(Callback):

  def __init__(self,
               summary: Optional[Summary] = None,
               plugins: Optional[Union[str, Collection[str]]] = None,
               *,
               batch_dim: int = 0,
               source_key: Optional[str] = None,
               reduction_map: Dict[str, Any],
               prefix: Optional[str] = None,
               use_suffix_matching: bool = False):
    self._summary = summary or Context.get_default_argument("summary",
                                                            required=True)

    if isinstance(plugins, str):
      plugins = {plugins}
    elif plugins is not None:
      plugins = set(plugins)
    else:
      plugins = set()
    self._plugins = plugins
    self._prefix = prefix if prefix else ""
    self._batch_dim = batch_dim
    self._reduction_map = self._normalize_reduction_map(reduction_map)
    self._use_suffix_matching = use_suffix_matching
    self._source_key = source_key or "source"

    self._attributes = None

  @classmethod
  def _normalize_reduction_map(cls, reduction_map: Dict[str, Any]):
    result = dict()
    for key, value in reduction_map.items():
      if isinstance(value, SummaryReduction):
        result.setdefault(key, value)
      elif isinstance(value, dict):
        result.setdefault(key, cls._normalize_reduction_map(value))
      else:
        result.setdefault(key, SummaryReduction.from_string(value))
    return result

  def prior_epoch(self, engine: Engine, **kwargs):
    self._attributes = dict()
    for plugin in self._plugins.union({"epoch_index"}):
      self._attributes[plugin] = self._get_attribute(plugin, engine, **kwargs)

  def _record(self,
              outputs: dict,
              reduction_map: dict,
              *,
              path: str,
              batch_size: int,
              has_matched=False):
    for key, value in outputs.items():
      if key is None and isinstance(value, dict):
        raise ValueError("The value corresponding to the key \"None\" should "
                         "not be a dict")
      if key in reduction_map:
        if isinstance(value, dict):
          self._record(value,
                       reduction_map=reduction_map[key],
                       path=filtered_join('/', [path, key]),
                       batch_size=batch_size,
                       has_matched=True)
        else:
          self._summary.add_value(filtered_join('/', [path, key]),
                                  value,
                                  count=batch_size,
                                  index=self._attributes["epoch_index"],
                                  reduction=reduction_map[key])
      elif not has_matched and self._use_suffix_matching:
        if isinstance(value, dict):
          self._record(value,
                       reduction_map,
                       path=filtered_join('/', [path, key]),
                       batch_size=batch_size,
                       has_matched=False)

  def after_batch(self, engine: Engine, **kwargs):
    outputs = kwargs.get('outputs')
    if outputs is None:
      return

    batch_size = kwargs['inputs'][self._source_key].shape[self._batch_dim]
    kwargs = {k: v for k, v in self._attributes.items() if k in self._plugins}
    prefix = self._prefix.format(**kwargs)
    self._record(outputs,
                 self._reduction_map,
                 path=prefix,
                 batch_size=batch_size)
