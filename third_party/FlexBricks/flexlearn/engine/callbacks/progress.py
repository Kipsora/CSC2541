__all__ = ['ShowEpochProgress']

import logging
import sys
from typing import Optional, Collection, Union, Tuple, TYPE_CHECKING

import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from flexlearn.context import Context
from flexlearn.distributed import ProcessGroup
from flexlearn.engine.callbacks import Callback
from flexlearn.summary import Summary
from flexutils.misc.string import *

Engine = None
if TYPE_CHECKING:
  from flexlearn.engine import Engine


class ShowEpochProgress(Callback):
  """
  Display the epoch progress bar.

  Args:
    header: The header in the description of the progress bar. It can have
      placeholders that are specified in the ``plugins`` argument.
    plugins: The values to be retrieved for displaying the progress bar.
      The values should either be defined in ``**kwargs`` or an attribute of
      the engine in the ``after_batch`` hook.
    field_formatter: The formatter of the field in the progress bar
      description. This is useful when the user want to show a shortened name
      in the progress bar (compared to the name in the summary).
    value_format_rules: The format rules for the values in the description of
      the progress bar. This argument is useful when the value is a floating
      number showing which could overflows the progress bar.
    summary: A :class:`~flexlearn.summary.Summary` instance in which the metric
      is monitored. It is required and will be retrieved from the default
      :class:`~flexlearn.context.Context` if not given.
    process_group: The process group for determining whether the current
      process is the local master process. It will be retrieved from the default
      :class:`~flexlearn.context.Context` if not given. If it is not defined
      anyway, then it will be set as ``None`` and will assume the current
      process is the local master process.
    managed_loggers: Loggers which will be taken care of when using the progress
      bar. This argument is useful when the loggers need to print to the console
      as the progress bar will conflict with the loggers raw print method.
    max_field_length: The maximum length of each field. The field whose length
      is larger than this argument will be shortened using ellipsis.
    use_unique_local: Whether to display the progress bar only for the local
      master process.

  Examples:
    To show a progress bar with the learning rate and a header:

      >>> from flexutils.misc.string import RegularReplacer
      >>> callback = ShowEpochProgress(
      >>>     header="Epoch {epoch_index}/{num_epochs}",
      >>>     plugins=["epoch_index", "num_epochs"],
      >>>     field_formatter=RegularReplacer({
      >>>         'lr': 'lr'
      >>>     }))
  """
  FormatRule = Tuple[Optional[StringMatcher], NumberFormatter]
  Loggers = Union[logging.Logger, Collection[logging.Logger]]

  def __init__(self,
               header: Optional[str] = None,
               plugins: Optional[Union[str, Collection[str]]] = None,
               field_formatter: Optional[StringReplacer] = None,
               value_format_rules: Optional[Collection[FormatRule]] = None,
               summary: Optional[Summary] = None,
               process_group: Optional[ProcessGroup] = None,
               managed_loggers: Optional[Loggers] = None,
               max_field_length=0,
               use_unique_local: bool = True):
    self._header = header

    if isinstance(plugins, str):
      plugins = {plugins}
    elif plugins is not None:
      plugins = set(plugins)
    else:
      plugins = set()
    self._plugins = plugins

    self._summary = summary or Context.get_default_argument("summary")
    self._process_group = process_group or Context.get_default_argument(
        "process_group")

    self._field_formatter = field_formatter
    if value_format_rules is None:
      self._value_format_rules = [(None, TemplateNumberFormatter('{:.3g}'))]
    else:
      self._value_format_rules = list(value_format_rules)
    self._bar: Optional[tqdm.tqdm] = None

    self._managed_loggers = []
    if managed_loggers is None:
      default_logger = Context.get_default_argument("logger")
      if default_logger is not None:
        self._managed_loggers.append(default_logger)
    elif isinstance(managed_loggers, logging.Logger):
      self._managed_loggers.append(managed_loggers)
    else:
      self._managed_loggers.extend(managed_loggers)

    self._switch_handler_context = None

    if 0 < max_field_length < 4:
      raise ValueError("max_field_length should be at least 4 to "
                       "show meaningful field name")
    self._max_field_length = max_field_length
    self._use_unique_local = use_unique_local

  def prior_epoch(self, engine: Engine, **kwargs):
    if not self._use_unique_local or ProcessGroup.is_local_master(
        self._process_group):
      data_loader = self._get_attribute("data_loader", engine, **kwargs)
      self._create_process_bar(total=len(data_loader))

  def _create_process_bar(self, total):
    self._bar = tqdm.tqdm(total=total, ncols=0, leave=False, desc="[Waiting]")
    self._switch_handler_context = logging_redirect_tqdm(
        self._managed_loggers, self._bar.__class__)
    # noinspection PyUnresolvedReferences
    self._switch_handler_context.__enter__()

  def _format_value(self, name, value):
    for matcher, formatter in self._value_format_rules:
      if matcher is not None and not matcher.match(name):
        continue
      return formatter(value)
    return str(value)

  def _update_progress_bar(self, engine: Optional[Engine], **kwargs):
    epoch_index = self._get_attribute("epoch_index", engine, **kwargs)
    kwargs = {
        k: self._get_attribute(k, engine, **kwargs) for k in self._plugins
    }
    description = []
    if self._summary is not None:
      for field in sorted(self._summary.names()):
        original_field = field
        if self._field_formatter is not None:
          field = self._field_formatter.substitute(field, **kwargs)
          if field is None:
            continue
        if self._max_field_length != 0:
          field = [
              field if len(field) <= self._max_field_length else
              field[:self._max_field_length - 3] + "..."
              for field in field.split("/")
          ]
          field = "/".join(field)
        value = self._summary[original_field].value(index=epoch_index)
        value = self._format_value(original_field, value)
        description.append(f'[{field}] = {value}')

    header = self._header.format(**kwargs)
    self._bar.set_description(": ".join(
        filter(bool, [header, ', '.join(description)])))
    self._bar.update()

  def _close_progress_bar(self):
    if self._switch_handler_context is not None:
      self._switch_handler_context.__exit__(*sys.exc_info())
      self._switch_handler_context = None

    if self._bar is not None:
      self._bar.close()
      self._bar = None

  def after_batch(self, engine: Engine, **kwargs):
    if not self._use_unique_local or ProcessGroup.is_local_master(
        self._process_group):
      kwargs = {
          k: self._get_attribute(k, engine, **kwargs) for k in self._plugins
      }
      self._update_progress_bar(engine, **kwargs)

  def after_epoch(self, engine: Engine, **kwargs):
    if not self._use_unique_local or ProcessGroup.is_local_master(
        self._process_group):
      self._close_progress_bar()

  def on_exception(self, engine: Engine, exception: BaseException, **kwargs):
    if not self._use_unique_local or ProcessGroup.is_local_master(
        self._process_group):
      self._close_progress_bar()
