__all__ = [
    'Summary', 'SummaryStorage', 'SummaryReduction', 'SummaryItem',
    'SummaryItemType'
]

import abc
import enum
from collections import deque
from typing import Optional, Dict, Deque

import numpy as np


class SummaryReduction(enum.Enum):
  """
  The reduction modes for summary storage across different nodes.
  """

  NONE = enum.auto()
  """
  The given value is the sum of values on the current node and after
  synchronized, we only get the sum of values on the current node
  (i.e. no synchronization). Note that `None` can also be used to indicate this 
  reduction mode.
  """

  SUM = enum.auto()
  """
  The given value is the sum of values on the current node and after
  synchronized, we should get the sum of values on all nodes.
  """

  MEAN = enum.auto()
  """
  The given value is the mean of values on the current node and after
  synchronized, we should get the mean of values on all nodes.
  """

  LOCAL_MEAN = enum.auto()
  """
  The given value is the mean of values on the current node and
  after synchronized, we only get the mean of values on the current node
  (i.e. no synchronization but still get the mean)
  """

  GLOBAL_MEAN = enum.auto()
  """
  The given value is the sum of values on the current node and
  after synchronized, we should get the mean of values on all nodes.
  """

  @classmethod
  def to_string(cls, reduction: Optional['SummaryReduction']):
    if reduction is None or reduction == SummaryReduction.NONE:
      return 'none'
    elif reduction == SummaryReduction.SUM:
      return 'sum'
    elif reduction == SummaryReduction.MEAN:
      return 'mean'
    elif reduction == SummaryReduction.LOCAL_MEAN:
      return 'local_mean'
    elif reduction == SummaryReduction.GLOBAL_MEAN:
      return 'global_mean'
    else:
      raise ValueError(f"Invalid reduction {reduction}")

  @classmethod
  def from_string(cls, reduction: Optional[str]):
    if reduction is None:
      return None
    reduction = reduction.lower()
    if reduction == 'none' or reduction == SummaryReduction.NONE:
      return None
    elif reduction == 'sum':
      return SummaryReduction.SUM
    elif reduction == 'mean':
      return SummaryReduction.MEAN
    elif reduction == 'global_mean':
      return SummaryReduction.GLOBAL_MEAN
    elif reduction == 'local_mean':
      return SummaryReduction.LOCAL_MEAN
    else:
      raise ValueError(f"Invalid reduction {reduction}")


class SummaryItem(object):

  def __init__(self, value, count):
    self.value = value
    self.count = count


class SummaryItemType(enum.Enum):
  PER_GLOBAL_STEP = enum.auto()
  """
  This summary item will be updated on every batch (global step).
  """

  PER_EPOCH = enum.auto()
  """
  This summary item will be updated on every epoch.
  """


class SummaryStorage(object, metaclass=abc.ABCMeta):

  def __init__(self, *, item_type: SummaryItemType, max_size: Optional[int],
               reduction: SummaryReduction):
    self._storage: Dict[int, SummaryItem] = dict()
    self._indices: Deque[int] = deque()
    self.item_type = item_type
    self.max_size = max_size
    self.reduction = reduction

  @abc.abstractmethod
  def _record_item(self, item: SummaryItem, value, count, reset: bool):
    pass

  @abc.abstractmethod
  def _create_item(self, value, count) -> SummaryItem:
    pass

  @property
  def indices(self):
    return self._indices

  def record(self,
             value,
             count=None,
             *,
             index: Optional[int] = None,
             reset: bool = False):
    if index is None:
      index = self._indices[-1]
    elif self._indices and index < self._indices[-1]:
      raise RuntimeError("Recording to previous steps is not allowed")

    if index not in self._storage:
      self._storage[index] = self._create_item(value, count)
      self._indices.append(index)

      while len(self._indices) > self.max_size:
        self._storage.pop(self._indices.popleft())

      return

    item = self._storage[index]
    self._record_item(item, value, count, reset=reset)

  @abc.abstractmethod
  def value(self, *, index: Optional[int] = None) -> np.ndarray:
    pass

  def __len__(self):
    return len(self._indices)

  def __bool__(self):
    return bool(self._indices)


class Summary(object, metaclass=abc.ABCMeta):

  def __init__(self):
    self._history: Dict[str, SummaryStorage] = dict()

  @abc.abstractmethod
  def create_storage(self, reduction: Optional[SummaryReduction],
                     item_type: SummaryItemType,
                     max_size: Optional[int]) -> SummaryStorage:
    pass

  def add_value(self,
                name: str,
                value,
                count=None,
                *,
                index: Optional[int] = None,
                reduction: Optional[SummaryReduction] = None,
                item_type: SummaryItemType = SummaryItemType.PER_EPOCH,
                max_size: Optional[int] = 1,
                reset: bool = False):
    """
    To register or add an entry to the `Summary` instance.

    Args:
      name: The name of the entry
      value: The value of the entry
      count: The count of the entry. This argument is useful when `value`
        is a sum and `count` can specify how many items are summed.
      index: The index of the value. Note that `Summary` is not limit to
        record per epoch value, so the index can be `epoch_index` or
        `batch_index` or others.
      reduction: A `SummaryReduction` instance. Please refer to
        `SummaryReduction` for more details.
      item_type: A `SummaryItemType` instance, indicating the type of the item.
      max_size: The maximum history items for the value.
      reset: whether to reset or accumulate the value.
    """

    if name not in self._history:
      storage = self.create_storage(reduction, item_type, max_size)
      self._history.setdefault(name, storage)
    storage = self._history[name]
    if storage.reduction != reduction or storage.max_size != max_size or \
            storage.item_type != item_type:
      raise ValueError(
          f"Different storage settings were found at record {name}")
    storage.record(value, count, index=index, reset=reset)

  def names(self):
    return self._history.keys()

  def storages(self):
    return self._history.values()

  def __getitem__(self, item):
    return self._history[item]

  def __contains__(self, item):
    return item in self._history
